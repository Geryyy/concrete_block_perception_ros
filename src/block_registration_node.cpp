#include <rclcpp/rclcpp.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>

#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/pose.hpp>

#include <cv_bridge/cv_bridge.h>
#include <open3d/Open3D.h>

#include "pcd_block_estimation/mask_projection.hpp"
#include "pcd_block_estimation/pose_estimation.hpp"
#include "pcd_block_estimation/template_utils.hpp"
#include "pcd_block_estimation/utils.hpp"
#include "pcd_block_estimation/yaml_utils.hpp"

#include "concrete_block_perception/srv/register_block.hpp"
#include "concrete_block_perception/io_utils.hpp"

using RegisterBlock = concrete_block_perception::srv::RegisterBlock;
using namespace pcd_block;
using namespace open3d;

class BlockRegistrationNode : public rclcpp::Node
{
public:
  BlockRegistrationNode()
  : Node("block_registration_node")
  {
    // ------------------------------------------------------------
    // Parameters
    // ------------------------------------------------------------
    const std::string default_calib_yaml =
      ament_index_cpp::get_package_share_directory(
      "concrete_block_perception") +
      "/config/calib_zed2i_to_seyond.yaml";

    const std::string default_template_dir =
      ament_index_cpp::get_package_share_directory(
      "concrete_block_perception") +
      "/config/templates";

    declare_parameter<std::string>("calib_yaml", default_calib_yaml);    // default: empty → must be provided
    declare_parameter<std::string>("template_dir", default_template_dir);
    declare_parameter<double>("dist_thresh", 0.02);
    declare_parameter<int>("min_inliers", 100);
    declare_parameter<double>("icp_dist", 0.04);
    declare_parameter<double>("angle_thresh_degree", 30.0);
    declare_parameter<int>("yaw_step", 30); // template rotation step size for local registration

    calib_yaml = get_parameter("calib_yaml").as_string();
    template_dir = get_parameter("template_dir").as_string();
    dist_thresh = get_parameter("dist_thresh").as_double();
    min_inliers = get_parameter("min_inliers").as_int();
    icp_dist = get_parameter("icp_dist").as_double();
    double angle_thresh_deg = get_parameter("angle_thresh_degree").as_double();
    angle_thresh = std::cos(angle_thresh_deg * M_PI / 180.0);
    yaw_step = get_parameter("yaw_step").as_int();

    if (calib_yaml.empty() || template_dir.empty()) {
      throw std::runtime_error("Missing parameters");
    }

    T_P_C_ = load_T_4x4(calib_yaml);
    K_ = load_camera_matrix(calib_yaml);
    templates_ = load_templates(template_dir);

    service_ = create_service<RegisterBlock>(
      "register_block_pose",
      std::bind(
        &BlockRegistrationNode::handle_request,
        this,
        std::placeholders::_1,
        std::placeholders::_2
      )
    );

    Z_WORLD = Eigen::Vector3d(0.0, -1.0, 0.0);  // world z-axis w.r.t. to camera camera frame (neg y-axis)

    RCLCPP_INFO(
      get_logger(),
      "Block registration service ready");
  }

private:
  // --------------------------------------------------------
  // Service callback
  // --------------------------------------------------------
  void handle_request(
    const std::shared_ptr<RegisterBlock::Request> req,
    std::shared_ptr<RegisterBlock::Response> res)
  {
    try {
      // -------------------------------
      // Convert mask
      // -------------------------------
      cv::Mat mask =
        cv_bridge::toCvCopy(
        req->mask, "mono8")->image;

      // -------------------------------
      // Convert point cloud
      // -------------------------------
      auto scene =
        std::make_shared<geometry::PointCloud>();


      if (scene->points_.empty()) {
        res->success = false;
        return;
      }

      // -------------------------------
      // Mask-based cutout
      // -------------------------------
      auto pts_sel = select_points_by_mask(
        scene->points_,
        mask,
        K_,
        T_P_C_
      );

      if (pts_sel.empty()) {
        res->success = false;
        return;
      }

      geometry::PointCloud cutout;
      cutout.points_ = pts_sel;
      cutout.EstimateNormals();

      // -------------------------------
      // Pose estimation
      // -------------------------------

      // global registration
      GlobalRegistrationResult globreg_result = compute_global_registration(
        cutout,
        Z_WORLD,
        angle_thresh,
        MAX_PLANES,
        dist_thresh,
        min_inliers
      );

      // local registration
      LocalRegistrationResult result = compute_local_registration(
        cutout,
        templates_,
        globreg_result,
        icp_dist,
        yaw_step
      );


      if (!result.success) {
        res->success = false;
        return;
      }

      // -------------------------------
      // Fill response
      // -------------------------------
      res->pose = to_ros_pose(
        result.icp.transformation_);

      res->fitness = result.icp.fitness_;
      res->rmse = result.icp.inlier_rmse_;
      res->success = true;
    } catch (const std::exception & e) {
      RCLCPP_ERROR(
        get_logger(),
        "Registration failed: %s", e.what());
      res->success = false;
    }
  }

  // --------------------------------------------------------
  // Members
  // --------------------------------------------------------
  rclcpp::Service<RegisterBlock>::SharedPtr service_;

  Eigen::Matrix4d T_P_C_;
  Eigen::Matrix3d K_;

  std::vector<TemplateData> templates_;

  Eigen::Vector3d Z_WORLD;

  // Fixed compile-time params
  const int MAX_PLANES = 3;

  std::string calib_yaml;
  std::string template_dir;
  double dist_thresh;
  int min_inliers;
  double icp_dist;
  double angle_thresh;
  int yaw_step;
};


int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  try {
    auto node =
      std::make_shared<BlockRegistrationNode>();

    rclcpp::spin(node);
  } catch (const std::exception & e) {
    RCLCPP_FATAL(
      rclcpp::get_logger("block_registration_node"),
      "Fatal error during startup: %s",
      e.what());
  }

  rclcpp::shutdown();
  return 0;
}
