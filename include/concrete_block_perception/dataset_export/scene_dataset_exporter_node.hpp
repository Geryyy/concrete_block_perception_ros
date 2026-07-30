#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

namespace concrete_block_perception
{

// Primitively subsamples a played-back rosbag into an unlabeled training-data
// snapshot every `export_interval_s` seconds of message time (not wall/sim
// clock), matched to the existing scene_discovery_capture/ file layout
// (cloud.pcd, rgb.png, tf.yaml, metadata.yaml) so it consumes without any new
// parsing code. No annotations/associations/candidates files are written: an
// absent annotations.yaml is the existing convention for "unlabeled".
class SceneDatasetExporterNode : public rclcpp::Node
{
public:
  explicit SceneDatasetExporterNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

private:
  using SyncPolicy = message_filters::sync_policies::ApproximateTime<
    sensor_msgs::msg::Image, sensor_msgs::msg::PointCloud2>;

  void cameraInfoCallback(sensor_msgs::msg::CameraInfo::ConstSharedPtr msg);
  void syncedCallback(
    const sensor_msgs::msg::Image::ConstSharedPtr & image,
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr & cloud);
  // Resolves the tf.yaml content without writing anything, so an incomplete
  // TF snapshot (see the .cpp for why this happens at the start of a bag) can
  // be discovered and the whole sample skipped before any file is created.
  struct TfSnapshot
  {
    std::string yaml;
    bool core_complete{false};
  };
  TfSnapshot resolveTfSnapshot(
    const std::string & image_frame, const std::string & cloud_frame,
    const builtin_interfaces::msg::Time & stamp);
  void writeCameraInfoYaml(const std::filesystem::path & path);

  double export_interval_s_;
  std::string world_frame_;
  std::filesystem::path output_dir_;
  std::string source_bag_;
  std::vector<std::string> extra_tf_frames_;

  message_filters::Subscriber<sensor_msgs::msg::Image> image_sub_;
  message_filters::Subscriber<sensor_msgs::msg::PointCloud2> cloud_sub_;
  std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_;
  sensor_msgs::msg::CameraInfo::ConstSharedPtr last_camera_info_;

  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  bool has_last_export_stamp_{false};
  int64_t last_export_stamp_ns_{0};
  std::size_t exported_count_{0};
};

}  // namespace concrete_block_perception
