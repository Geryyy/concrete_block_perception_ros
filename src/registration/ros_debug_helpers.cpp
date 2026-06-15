#include "concrete_block_perception/registration/ros_debug_helpers.hpp"

#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgcodecs.hpp>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <array>

#include "concrete_block_perception/utils/io_utils.hpp"
#include "pcd_block_estimation/utils.hpp"

using namespace open3d;

namespace concrete_block_perception
{
namespace
{

Eigen::Vector3d debug_cutout_color(size_t index)
{
  static const std::array<Eigen::Vector3d, 8> kColors = {
    Eigen::Vector3d{1.0, 0.0, 0.0},
    Eigen::Vector3d{0.0, 0.75, 1.0},
    Eigen::Vector3d{1.0, 0.55, 0.0},
    Eigen::Vector3d{0.85, 0.0, 1.0},
    Eigen::Vector3d{1.0, 1.0, 0.0},
    Eigen::Vector3d{0.0, 0.25, 1.0},
    Eigen::Vector3d{1.0, 0.0, 0.45},
    Eigen::Vector3d{0.0, 1.0, 0.6},
  };
  return kColors[index % kColors.size()];
}

}  // namespace

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
    const auto debug_qos = rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local();
    debug_mask_cutout_pub_ =
      node_.create_publisher<sensor_msgs::msg::PointCloud2>(
      "debug/registration/mask_cutout", debug_qos);

    debug_cleaned_cutout_pub_ =
      node_.create_publisher<sensor_msgs::msg::PointCloud2>(
      "debug/registration/cleaned_cutout", debug_qos);

    debug_registration_cloud_pub_ =
      node_.create_publisher<sensor_msgs::msg::PointCloud2>(
      "debug/registration/plane_cloud", debug_qos);

    debug_template_pub_ =
      node_.create_publisher<sensor_msgs::msg::PointCloud2>(
      "debug/registration/template", debug_qos);

    debug_diagnostics_pub_ =
      node_.create_publisher<std_msgs::msg::String>(
      "debug/registration/diagnostics", debug_qos);

    tf_broadcaster_ =
      std::make_shared<tf2_ros::TransformBroadcaster>(node_);
  }

  if (publish_debug_mask_) {
    const auto debug_image_qos = rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local();
    debug_mask_pub_ =
      node_.create_publisher<sensor_msgs::msg::Image>(
      "debug/registration/mask", debug_image_qos);
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

void RosDebugHelpers::publishDiagnostics(
  const RegistrationOutput & output,
  const std::string & source)
{
  if (!debug_diagnostics_pub_) {
    return;
  }

  std::ostringstream ss;
  ss << "{"
     << "\"source\":\"" << source << "\","
     << "\"success\":" << (output.success ? "true" : "false") << ","
     << "\"failure_stage\":\"" << output.failure_stage << "\","
     << "\"failure_reason\":\"" << output.failure_reason << "\","
     << "\"scene_points\":" << output.scene_points << ","
     << "\"mask_cutout_points\":" << output.mask_cutout_points << ","
     << "\"cleaned_cutout_points\":" << output.cleaned_cutout_points << ","
     << "\"registration_cloud_points\":" << output.registration_cloud_points << ","
     << "\"used_pose_prior\":" << (output.used_pose_prior ? "true" : "false") << ","
     << "\"prior_position_sigma_m\":" << output.prior_position_sigma_m << ","
     << "\"prior_orientation_sigma_rad\":" << output.prior_orientation_sigma_rad << ","
     << "\"template_index\":" << output.template_index << ","
     << "\"fitness\":" << output.fitness << ","
     << "\"rmse\":" << output.rmse
     << "}";

  std_msgs::msg::String msg;
  msg.data = ss.str();
  debug_diagnostics_pub_->publish(msg);
}

void RosDebugHelpers::publishRegistrationDebug(
  const sensor_msgs::msg::PointCloud2 & cloud_source,
  const open3d::geometry::PointCloud & mask_cutout,
  const open3d::geometry::PointCloud & cleaned_cutout,
  const open3d::geometry::PointCloud & registration_cloud,
  int template_index,
  const Eigen::Matrix4d & T)
{
  if (!publish_debug_cutout_) {
    return;
  }

  const rclcpp::Time stamp(cloud_source.header.stamp);

  // ---------------- Raw mask cutout ----------------
  if (debug_mask_cutout_pub_ && !mask_cutout.points_.empty()) {
    geometry::PointCloud mask_cutout_vis = mask_cutout;
    mask_cutout_vis.PaintUniformColor(debug_cutout_color(cutout_color_index_.fetch_add(1)));

    debug_mask_cutout_pub_->publish(
      open3d_to_pointcloud2_colored(
        mask_cutout_vis,
        world_frame_,
        stamp));
  }

  // ---------------- Cleaned cutout ----------------
  if (debug_cleaned_cutout_pub_ && !cleaned_cutout.points_.empty()) {
    geometry::PointCloud cutout_vis = cleaned_cutout;
    cutout_vis.PaintUniformColor(debug_cutout_color(cutout_color_index_.fetch_add(1)));

    debug_cleaned_cutout_pub_->publish(
      open3d_to_pointcloud2_colored(
        cutout_vis,
        world_frame_,
        stamp));
  }

  // ---------------- Plane cloud used for registration ----------------
  if (debug_registration_cloud_pub_ && !registration_cloud.points_.empty()) {
    geometry::PointCloud registration_vis = registration_cloud;
    registration_vis.PaintUniformColor(debug_cutout_color(cutout_color_index_.fetch_add(1)));

    debug_registration_cloud_pub_->publish(
      open3d_to_pointcloud2_colored(
        registration_vis,
        world_frame_,
        stamp));
  }

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
  } else if (template_index >= 0) {
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

void RosDebugHelpers::publishCutoutDebug(
  const sensor_msgs::msg::PointCloud2 & cloud_source,
  const open3d::geometry::PointCloud & mask_cutout_world,
  const open3d::geometry::PointCloud & cutout_world)
{
  if (!publish_debug_cutout_) {
    return;
  }

  const rclcpp::Time stamp(cloud_source.header.stamp);
  if (debug_mask_cutout_pub_ && !mask_cutout_world.points_.empty()) {
    geometry::PointCloud mask_cutout_vis = mask_cutout_world;
    mask_cutout_vis.PaintUniformColor(debug_cutout_color(cutout_color_index_.fetch_add(1)));
    debug_mask_cutout_pub_->publish(
      open3d_to_pointcloud2_colored(
        mask_cutout_vis,
        world_frame_,
        stamp));
  }

  if (debug_cleaned_cutout_pub_ && !cutout_world.points_.empty()) {
    geometry::PointCloud cutout_vis = cutout_world;
    cutout_vis.PaintUniformColor(debug_cutout_color(cutout_color_index_.fetch_add(1)));
    debug_cleaned_cutout_pub_->publish(
      open3d_to_pointcloud2_colored(
        cutout_vis,
        world_frame_,
        stamp));
  }
}

void RosDebugHelpers::dumpInput(
  const concrete_block_perception_interfaces::action::RegisterBlock::Goal & goal)
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

void RosDebugHelpers::dumpFailurePackage(
  const sensor_msgs::msg::PointCloud2 & cloud,
  const sensor_msgs::msg::Image & mask,
  const open3d::geometry::PointCloud & cutout_world,
  const std::string & stage,
  const std::string & reason)
{
  if (!dump_enabled_) {
    return;
  }

  const auto & stamp = cloud.header.stamp;
  std::ostringstream base;
  base << stamp.sec << "_"
       << std::setw(9)
       << std::setfill('0')
       << stamp.nanosec
       << "_fail";
  const std::string prefix = dump_dir_ + "/" + base.str();

  try {
    cv::Mat mask_img = cv_bridge::toCvCopy(mask, "mono8")->image;
    cv::imwrite(prefix + "_mask.png", mask_img);
  } catch (const std::exception & e) {
    RCLCPP_WARN(node_.get_logger(), "Failed to dump failure mask: %s", e.what());
  }

  try {
    auto cloud_o3d = pointcloud2_to_open3d(cloud);
    if (cloud_o3d) {
      open3d::io::WritePointCloud(prefix + "_cloud.ply", *cloud_o3d, false);
    }
  } catch (const std::exception & e) {
    RCLCPP_WARN(node_.get_logger(), "Failed to dump failure cloud: %s", e.what());
  }

  try {
    if (!cutout_world.points_.empty()) {
      open3d::io::WritePointCloud(prefix + "_cutout_world.ply", cutout_world, false);
    }
  } catch (const std::exception & e) {
    RCLCPP_WARN(node_.get_logger(), "Failed to dump failure cutout: %s", e.what());
  }

  try {
    std::ofstream meta(prefix + "_meta.yaml");
    if (meta.is_open()) {
      meta << "stage: \"" << stage << "\"\n";
      meta << "reason: \"" << reason << "\"\n";
      meta << "world_frame: \"" << world_frame_ << "\"\n";
      meta << "cloud_frame: \"" << cloud.header.frame_id << "\"\n";
      meta << "stamp:\n";
      meta << "  sec: " << stamp.sec << "\n";
      meta << "  nanosec: " << stamp.nanosec << "\n";
      meta << "mask:\n";
      meta << "  width: " << mask.width << "\n";
      meta << "  height: " << mask.height << "\n";
      meta << "cutout_world_points: " << cutout_world.points_.size() << "\n";
      meta.close();
    }
  } catch (const std::exception & e) {
    RCLCPP_WARN(node_.get_logger(), "Failed to dump failure metadata: %s", e.what());
  }

  RCLCPP_WARN(
    node_.get_logger(),
    "Failure package dumped: %s_[mask.png|cloud.ply|cutout_world.ply|meta.yaml]",
    prefix.c_str());
}

} // namespace concrete_block_perception
