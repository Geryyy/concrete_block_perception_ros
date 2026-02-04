#include <rclcpp/rclcpp.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>

#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/pose.hpp>

#include <cv_bridge/cv_bridge.h>
#include <open3d/Open3D.h>

#include <thread>
#include <filesystem>

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

#define LOG(logger, ...) RCLCPP_INFO(logger, __VA_ARGS__)

class BlockRegistrationNode : public rclcpp::Node
{
public:
  BlockRegistrationNode()
  : Node("block_registration_node")
  {
    // ------------------------------------------------------------
    // Defaults
    // ------------------------------------------------------------
    const std::string pkg_share =
      ament_index_cpp::get_package_share_directory(
      "concrete_block_perception");

    const std::string default_calib_yaml =
      pkg_share + "/config/calib_zed2i_to_seyond.yaml";

    const std::string default_template_dir =
      pkg_share + "/config/templates";

    // ------------------------------------------------------------
    // Parameters
    // ------------------------------------------------------------
    declare_parameter<std::string>("calib_yaml", "");
    declare_parameter<std::string>("template_dir", "");
    declare_parameter<double>("dist_thresh", 0.02);
    declare_parameter<int>("min_inliers", 100);
    declare_parameter<double>("icp_dist", 0.04);
    declare_parameter<double>("angle_thresh_degree", 30.0);
    declare_parameter<int>("yaw_step", 30);

    calib_yaml = get_parameter("calib_yaml").as_string();
    template_dir = get_parameter("template_dir").as_string();

    if (calib_yaml.empty()) {
      calib_yaml = default_calib_yaml;
    }
    if (template_dir.empty()) {
      template_dir = default_template_dir;
    }

    dist_thresh = get_parameter("dist_thresh").as_double();
    min_inliers = get_parameter("min_inliers").as_int();
    icp_dist = get_parameter("icp_dist").as_double();
    yaw_step = get_parameter("yaw_step").as_int();

    double angle_thresh_deg =
      get_parameter("angle_thresh_degree").as_double();
    angle_thresh = std::cos(angle_thresh_deg * M_PI / 180.0);

    // ------------------------------------------------------------
    // Validate files
    // ------------------------------------------------------------
    if (!std::filesystem::exists(calib_yaml)) {
      throw std::runtime_error(
              "Calibration YAML not found: " + calib_yaml);
    }
    if (!std::filesystem::exists(template_dir)) {
      throw std::runtime_error(
              "Template directory not found: " + template_dir);
    }

    // ------------------------------------------------------------
    // Load data
    // ------------------------------------------------------------
    T_P_C_ = load_T_4x4(calib_yaml);
    K_ = load_camera_matrix(calib_yaml);
    templates_ = load_templates(template_dir);

    Z_WORLD = Eigen::Vector3d(0.0, -1.0, 0.0);

    // ------------------------------------------------------------
    // Service (SERIALIZED!)
    // ------------------------------------------------------------
    service_cb_group_ =
      create_callback_group(
      rclcpp::CallbackGroupType::MutuallyExclusive);

    service_ = create_service<RegisterBlock>(
      "register_block_pose",
      std::bind(
        &BlockRegistrationNode::handle_request,
        this,
        std::placeholders::_1,
        std::placeholders::_2),
      rmw_qos_profile_services_default,
      service_cb_group_);


    // callbacks

    publish_debug_cutout_ =
      declare_parameter<bool>("debug.publish_cutout", true);

    if (publish_debug_cutout_) {
      debug_cutout_pub_ =
        create_publisher<sensor_msgs::msg::PointCloud2>(
        "debug/cutout_cloud", 1);

      RCLCPP_INFO(
        get_logger(),
        "Publishing debug cutout cloud on %s",
        debug_cutout_pub_->get_topic_name());
    }

    publish_debug_mask_ =
      declare_parameter<bool>("debug.publish_mask", true);

    if (publish_debug_mask_) {
      debug_mask_pub_ =
        create_publisher<sensor_msgs::msg::Image>(
        "debug/segmentation_mask", 1);

      RCLCPP_INFO(
        get_logger(),
        "Publishing debug segmentation mask on %s",
        debug_mask_pub_->get_topic_name());
    }


    RCLCPP_INFO(
      get_logger(),
      "Block registration service ready");
    RCLCPP_INFO(
      get_logger(),
      "  calib_yaml:   %s", calib_yaml.c_str());
    RCLCPP_INFO(
      get_logger(),
      "  template_dir: %s", template_dir.c_str());
  }

private:
  // --------------------------------------------------------
  // Service callback
  // --------------------------------------------------------
  void handle_request(
    const std::shared_ptr<RegisterBlock::Request> req,
    std::shared_ptr<RegisterBlock::Response> res)
  {
    const auto t0 = this->now();
    const auto tid =
      std::hash<std::thread::id>{}(std::this_thread::get_id());

    LOG(
      get_logger(),
      "handle_request() start [tid=%lu]", tid);

    try {
      // ------------------------------------------------------------
      // Mask
      // ------------------------------------------------------------
      cv::Mat mask =
        cv_bridge::toCvCopy(
        req->mask, "mono8")->image;

      // ------------------------------------------------------------
      // Debug: publish segmentation mask (scaled for visualization)
      // ------------------------------------------------------------
      if (publish_debug_mask_ && debug_mask_pub_) {
        cv::Mat mask_vis;
        mask.convertTo(mask_vis, CV_8UC1, 255.0);

        sensor_msgs::msg::Image::SharedPtr mask_msg =
          cv_bridge::CvImage(
          req->mask.header,
          "mono8",
          mask_vis).toImageMsg();

        debug_mask_pub_->publish(*mask_msg);

        LOG(
          get_logger(),
          "Published debug mask (%dx%d)",
          mask_vis.cols,
          mask_vis.rows);
      }


      // ------------------------------------------------------------
      // PointCloud2 → Open3D
      // ------------------------------------------------------------
      LOG(
        get_logger(),
        "Incoming cloud: width=%u height=%u data=%zu bytes",
        req->cloud.width,
        req->cloud.height,
        req->cloud.data.size());

      auto scene =
        pointcloud2_to_open3d(req->cloud);

      LOG(
        get_logger(),
        "Converted cloud: %zu points",
        scene->points_.size());

      if (scene->points_.empty()) {
        res->success = false;
        LOG(get_logger(), "Empty scene cloud after conversion");
        return;
      }

      // ------------------------------------------------------------
      // Mask-based cutout
      // ------------------------------------------------------------
      LOG(get_logger(), "Mask-based pcl cutout");

      auto pts_sel = select_points_by_mask(
        scene->points_,
        mask,
        K_,
        T_P_C_);

      if (pts_sel.empty()) {
        res->success = false;
        LOG(get_logger(), "No points after masking");
        return;
      }

      geometry::PointCloud cutout;
      cutout.points_ = pts_sel;
      cutout.EstimateNormals();

      // ------------------------------------------------------------
      // Debug: publish cutout point cloud
      // ------------------------------------------------------------
      if (publish_debug_cutout_ && debug_cutout_pub_) {
        sensor_msgs::msg::PointCloud2 msg =
          open3d_to_pointcloud2(
          cutout,
          req->cloud.header.frame_id,
          rclcpp::Time(req->cloud.header.stamp));

        msg.header = req->cloud.header; // preserve frame + timestamp
        debug_cutout_pub_->publish(msg);

        LOG(
          get_logger(),
          "Published cutout cloud (%zu points)",
          cutout.points_.size());
      }

      LOG(
        get_logger(),
        "Cutout stats: n=%zu (min required=%d)",
        cutout.points_.size(),
        min_inliers);

      // ------------------------------------------------------------
      // Global registration
      // ------------------------------------------------------------
      LOG(get_logger(), "Global registration");

      GlobalRegistrationResult globreg =
        compute_global_registration(
        cutout,
        Z_WORLD,
        angle_thresh,
        MAX_PLANES,
        dist_thresh,
        min_inliers);

      if (globreg.success == false) {
        res->success = false;
        return;
      }

      // ------------------------------------------------------------
      // Local registration
      // ------------------------------------------------------------
      LOG(get_logger(), "Local registration");

      LocalRegistrationResult result =
        compute_local_registration(
        cutout,
        templates_,
        globreg,
        icp_dist,
        yaw_step);

      if (!result.success) {
        res->success = false;
        LOG(get_logger(), "Local registration failed");
        return;
      }

      // TODO: convert to world frame!

      // ------------------------------------------------------------
      // Response
      // ------------------------------------------------------------
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

    LOG(
      get_logger(),
      "handle_request() finish (%.3f s)",
      (this->now() - t0).seconds());
  }


  // --------------------------------------------------------
  // Members
  // --------------------------------------------------------
  rclcpp::Service<RegisterBlock>::SharedPtr service_;
  rclcpp::CallbackGroup::SharedPtr service_cb_group_;

  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr debug_cutout_pub_;
  bool publish_debug_cutout_;

  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr debug_mask_pub_;
  bool publish_debug_mask_;

  Eigen::Matrix4d T_P_C_;
  Eigen::Matrix3d K_;
  std::vector<TemplateData> templates_;
  Eigen::Vector3d Z_WORLD;

  static constexpr int MAX_PLANES = 3;

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
