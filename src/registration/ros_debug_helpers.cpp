#include "concrete_block_perception/registration/ros_debug_helpers.hpp"

#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgcodecs.hpp>
#include <filesystem>
#include <iomanip>
#include <sstream>

#include "concrete_block_perception/utils/io_utils.hpp"
#include "pcd_block_estimation/utils.hpp"

using namespace open3d;

namespace concrete_block_perception
{

RosDebugHelpers::RosDebugHelpers(
  rclcpp::Node & node,
  const BlockRegistrationConfig & cfg)
: node_(node),
  world_frame_(cfg.world_frame),
  publish_debug_cutout_(cfg.publish_debug_cutout),
  publish_debug_mask_(cfg.publish_debug_mask),
  dump_enabled_(cfg.dump_enabled),
  dump_dir_(cfg.dump_dir),
  templates_(cfg.templates)   // IMPORTANT FIX
{
  if (publish_debug_cutout_) {
    debug_cutout_pub_ =
      node_.create_publisher<sensor_msgs::msg::PointCloud2>(
      "debug/cutout_cloud", 1);

    debug_template_pub_ =
      node_.create_publisher<sensor_msgs::msg::PointCloud2>(
      "debug/template_cloud", 1);

    tf_broadcaster_ =
      std::make_shared<tf2_ros::TransformBroadcaster>(node_);
  }

  if (publish_debug_mask_) {
    debug_mask_pub_ =
      node_.create_publisher<sensor_msgs::msg::Image>(
      "debug/segmentation_mask", 1);
  }

  if (dump_enabled_) {
    std::filesystem::create_directories(dump_dir_);
    RCLCPP_WARN(
      node_.get_logger(),
      "Dump ENABLED → writing data to %s",
      dump_dir_.c_str());
  }
}

void RosDebugHelpers::publishMask(
  const sensor_msgs::msg::Image & header_source,
  const cv::Mat & mask)
{
  if (!publish_debug_mask_ || !debug_mask_pub_) {
    return;
  }

  cv::Mat mask_vis;
  mask.convertTo(mask_vis, CV_8UC1, 255.0);

  auto msg =
    cv_bridge::CvImage(
    header_source.header,
    "mono8",
    mask_vis).toImageMsg();

  debug_mask_pub_->publish(*msg);
}

void RosDebugHelpers::publishVisualization(
  const sensor_msgs::msg::PointCloud2 & cloud_source,
  const open3d::geometry::PointCloud & scene,
  int template_index,
  const Eigen::Matrix4d & T)
{
  if (!publish_debug_cutout_ || !debug_cutout_pub_) {
    return;
  }

  const rclcpp::Time stamp(cloud_source.header.stamp);

  // ---------------- Scene (red) ----------------
  geometry::PointCloud scene_vis = scene;
  scene_vis.PaintUniformColor({1.0, 0.0, 0.0});

  debug_cutout_pub_->publish(
    open3d_to_pointcloud2_colored(
      scene_vis,
      world_frame_,
      stamp));

  // ---------------- Template (green) ----------------
  if (template_index >= 0 &&
    template_index < static_cast<int>(templates_.size()))
  {
    if (debug_template_pub_) {
      auto tpl =
        std::make_shared<geometry::PointCloud>(
        *templates_[template_index].pcd);

      tpl->Transform(T);
      tpl->PaintUniformColor({0.0, 1.0, 0.0});

      debug_template_pub_->publish(
        open3d_to_pointcloud2_colored(
          *tpl,
          world_frame_,
          stamp));
    }
  } else {
    RCLCPP_WARN(
      node_.get_logger(),
      "Invalid template index %d (size=%zu)",
      template_index,
      templates_.size());
  }

  // ---------------- TF frame ----------------
  if (tf_broadcaster_) {
    geometry_msgs::msg::TransformStamped tf;
    tf.header.stamp = stamp;
    tf.header.frame_id = world_frame_;
    tf.child_frame_id = "block_debug";

    Eigen::Quaterniond q(T.block<3, 3>(0, 0));

    tf.transform.translation.x = T(0, 3);
    tf.transform.translation.y = T(1, 3);
    tf.transform.translation.z = T(2, 3);
    tf.transform.rotation.x = q.x();
    tf.transform.rotation.y = q.y();
    tf.transform.rotation.z = q.z();
    tf.transform.rotation.w = q.w();

    tf_broadcaster_->sendTransform(tf);
  }
}

void RosDebugHelpers::dumpInput(
  const concrete_block_perception::action::RegisterBlock::Goal & goal)
{
  if (!dump_enabled_) {
    return;
  }

  const auto & stamp = goal.cloud.header.stamp;

  std::ostringstream base;
  base << stamp.sec << "_"
       << std::setw(9)
       << std::setfill('0')
       << stamp.nanosec;

  const std::string prefix =
    dump_dir_ + "/" + base.str();

  // ---------------- Dump mask ----------------
  try {
    auto mask =
      cv_bridge::toCvCopy(goal.mask, "mono8")->image;

    cv::imwrite(prefix + "_mask.png", mask);
  } catch (const std::exception & e) {
    RCLCPP_WARN(
      node_.get_logger(),
      "Failed to dump mask: %s",
      e.what());
  }

  // ---------------- Dump cloud ----------------
  try {
    auto cloud =
      pointcloud2_to_open3d(goal.cloud);

    open3d::io::WritePointCloud(
      prefix + "_cloud.ply",
      *cloud,
      false);
  } catch (const std::exception & e) {
    RCLCPP_WARN(
      node_.get_logger(),
      "Failed to dump cloud: %s",
      e.what());
  }
}

} // namespace concrete_block_perception
