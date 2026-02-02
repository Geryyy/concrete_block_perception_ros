#pragma once

#include <Eigen/Dense>
#include <rclcpp/time.hpp>
#include <sstream>
#include <iomanip>
#include <opencv2/core.hpp>

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

  cv::Rect bbox;             // or
  cv::Mat mask;            // binary mask

  std::string to_string() const
  {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4);

    oss << "Measurement {\n";
    oss << "  stamp      : " << stamp.seconds() << " s\n";
    oss << "  position   : ["
        << position.x() << ", "
        << position.y() << ", "
        << position.z() << "]\n";
    oss << "  yaw        : " << yaw << "\n";
    oss << "  confidence : " << confidence << "\n";
    oss << "  R          :\n";

    for (int i = 0; i < R.rows(); ++i) {
      oss << "    [ ";
      for (int j = 0; j < R.cols(); ++j) {
        oss << std::setw(8) << R(i, j);
        if (j < R.cols() - 1) {
          oss << ", ";
        }
      }
      oss << " ]\n";
    }

    oss << "}";

    return oss.str();
  }
};

}  // namespace cbp::tracking
