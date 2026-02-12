#pragma once

#include <vision_msgs/msg/detection2_d_array.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <rclcpp/rclcpp.hpp>
#include <opencv2/core.hpp>

#include "concrete_block_perception/msg/tracked_detection_array.hpp"
#include "concrete_block_perception/detection/detection_params.hpp"

#include <unordered_map>
#include <vector>

namespace concrete_block_perception
{

class BlockDetectionTracker
{
public:
  explicit BlockDetectionTracker(const DetectionParams & params);

  msg::TrackedDetectionArray update(
    const vision_msgs::msg::Detection2DArray & detections,
    const sensor_msgs::msg::Image::SharedPtr & mask);

  void reset();   // optional but useful

private:
  struct DetectionTrack
  {
    uint32_t detection_id;
    vision_msgs::msg::Detection2D detection;
    sensor_msgs::msg::Image mask;

    uint32_t age = 0;
    uint32_t misses = 0;
    rclcpp::Time last_seen;
  };

  std::unordered_map<uint32_t, DetectionTrack> tracks_;
  uint32_t next_detection_id_ = 1;

  DetectionParams params_;

  // ---- helpers ----
  bool passesConfidenceThreshold(
    const vision_msgs::msg::Detection2D & det) const;

  bool passesSizeFilter(
    const vision_msgs::msg::Detection2D & det) const;

  bool isContained(
    const vision_msgs::msg::Detection2D & a,
    const vision_msgs::msg::Detection2D & b,
    double containment_ratio) const;

  double suppressionRadiusPx(
    const DetectionTrack & track) const;

  void pruneContainedTracks();

  cv::Rect toCvRect(
    const vision_msgs::msg::Detection2D & det) const;
};

}  // namespace concrete_block_perception
