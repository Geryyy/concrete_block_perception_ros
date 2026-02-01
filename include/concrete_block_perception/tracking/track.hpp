#pragma once

#include <rclcpp/time.hpp>
#include "concrete_block_perception/tracking/damped_cv_kalman_filter.hpp"

namespace cbp::tracking
{

struct Track
{
  int id{-1};

  DampedCVKalmanFilter kf;   // owns state + covariance

  rclcpp::Time last_update;

  int age{0};
  int hits{0};
  int misses{0};
};

}  // namespace cbp::tracking
