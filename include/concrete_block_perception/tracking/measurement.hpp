#pragma once

#include <Eigen/Dense>
#include <rclcpp/time.hpp>

namespace cbp::tracking
{

struct Measurement
{
  // Position in world frame
  Eigen::Vector3d position{Eigen::Vector3d::Zero()};

  // Coarse yaw estimate (0.0 if unknown)
  double yaw{0.0};

  // Measurement covariance for [x y z yaw]
  Eigen::Matrix<double, 4, 4> R =
    Eigen::Matrix<double, 4, 4>::Zero();

  double confidence{1.0};
  rclcpp::Time stamp;
};

}  // namespace cbp::tracking
