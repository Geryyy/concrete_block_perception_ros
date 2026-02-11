#pragma once

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_ros/transform_broadcaster.h>

#include <opencv2/core.hpp>
#include <open3d/Open3D.h>
#include <Eigen/Dense>
#include <vector>
#include <string>

#include "concrete_block_perception/action/register_block.hpp"
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

  void publishVisualization(
    const sensor_msgs::msg::PointCloud2 & cloud_source,
    const open3d::geometry::PointCloud & scene,
    int template_index,
    const Eigen::Matrix4d & T);

  void dumpInput(
    const concrete_block_perception::action::RegisterBlock::Goal & goal);

private:
  rclcpp::Node & node_;

  std::string world_frame_;

  bool publish_debug_cutout_{false};
  bool publish_debug_mask_{false};
  bool dump_enabled_{false};
  std::string dump_dir_;

  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr debug_cutout_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr debug_template_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr debug_mask_pub_;

  std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

  std::vector<pcd_block::TemplateData> templates_;
};

} // namespace concrete_block_perception
