#include <rclcpp/rclcpp.hpp>

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

#include "pcd_block_estimation_msgs/srv/register_block.hpp"

using RegisterBlock = pcd_block_estimation_msgs::srv::RegisterBlock;
using namespace pcd_block;
using namespace open3d;

class BlockRegistrationNode : public rclcpp::Node
{
public:
  BlockRegistrationNode()
  : Node("block_registration_node")
  {
    declare_parameter<std::string>("calib_yaml");
    declare_parameter<std::string>("template_dir");

    const auto calib_yaml =
      get_parameter("calib_yaml").as_string();
    const auto template_dir =
      get_parameter("template_dir").as_string();

    if (calib_yaml.empty() || template_dir.empty()) {
      throw std::runtime_error("Missing parameters");
    }

    T_P_C_ = load_T_4x4(calib_yaml);
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

      io::ReadPointCloudFromPointCloud2(
        req->cloud, *scene);

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
      // Pose estimation (ICP yaw sweep)
      // -------------------------------
      Eigen::Vector3d center =
        compute_center(cutout);

      PoseResult result =
        estimate_pose_from_cutout(
        cutout,
        templates_,
        center
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

  std::vector<Template> templates_;
};
