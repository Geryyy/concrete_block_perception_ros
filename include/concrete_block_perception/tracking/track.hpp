#pragma once

#include <rclcpp/time.hpp>
#include "concrete_block_perception/tracking/damped_cv_kalman_filter.hpp"
#include "concrete_block_perception/msg/block.hpp"

namespace cbp::tracking
{

using Block = concrete_block_perception::msg::Block;

struct Track
{
  int id{-1};

  DampedCVKalmanFilter kf;   // owns state + covariance

  rclcpp::Time last_update;

  int age{0};
  int hits{0};
  int misses{0};
  bool confirmed{false};

  cv::Rect last_bbox;

  // semantic / world-model state
  std::string block_id;        // persistent ID
  int pose_status{Block::POSE_UNKNOWN};
  int task_status{Block::TASK_UNKNOWN};
  float confidence{0.0f};
};

}  // namespace cbp::tracking
