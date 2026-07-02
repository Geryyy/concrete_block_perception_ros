#pragma once

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <std_msgs/msg/header.hpp>
#include <std_msgs/msg/string.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <visualization_msgs/msg/marker_array.hpp>

#include <opencv2/core.hpp>
#include <open3d/Open3D.h>
#include <Eigen/Dense>
#include <atomic>
#include <vector>
#include <string>

#include "concrete_block_perception_interfaces/action/register_block.hpp"
#include "concrete_block_perception/registration/block_registration_pipeline.hpp"
#include "concrete_block_perception/registration/registration_config.hpp"
#include "pcd_block_estimation/template_utils.hpp"

namespace concrete_block_perception
{

class RosDebugHelpers
{
public:
  RosDebugHelpers(
    rclcpp::Node & node,
    const BlockRegistrationConfig & cfg);

  void publishMask(
    const sensor_msgs::msg::Image & header_source,
    const cv::Mat & mask);

  void publishRegistrationDebug(
    const sensor_msgs::msg::PointCloud2 & cloud_source,
    const open3d::geometry::PointCloud & mask_cutout,
    const open3d::geometry::PointCloud & cleaned_cutout,
    const open3d::geometry::PointCloud & registration_cloud,
    int template_index,
    const Eigen::Matrix4d & T);

  void publishDiagnostics(
    const RegistrationOutput & output,
    const std::string & source);

  void publishCutoutDebug(
    const sensor_msgs::msg::PointCloud2 & cloud_source,
    const open3d::geometry::PointCloud & mask_cutout_world,
    const open3d::geometry::PointCloud & cutout_world);

  void publishGripperBoxes(
    const std_msgs::msg::Header & header_source);

  void dumpInput(
    const concrete_block_perception_interfaces::action::RegisterBlock::Goal & goal);

  void dumpFailurePackage(
    const sensor_msgs::msg::PointCloud2 & cloud,
    const sensor_msgs::msg::Image & mask,
    const open3d::geometry::PointCloud & cutout_world,
    const std::string & stage,
    const std::string & reason);

private:
  rclcpp::Node & node_;

  std::string world_frame_;

  bool publish_debug_cutout_{false};
  bool publish_debug_mask_{false};
  GripperFilterConfig gripper_filter_;
  bool dump_enabled_{false};
  std::string dump_dir_;

  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr debug_mask_cutout_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr debug_cleaned_cutout_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr debug_registration_cloud_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr debug_template_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr debug_mask_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr debug_diagnostics_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr debug_gripper_boxes_pub_;

  std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

  std::vector<pcd_block::TemplateData> templates_;
  std::atomic_size_t cutout_color_index_{0};
};

} // namespace concrete_block_perception
