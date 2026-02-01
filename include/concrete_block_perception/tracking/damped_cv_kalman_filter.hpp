#pragma once

#include <Eigen/Dense>
#include <rclcpp/time.hpp>

namespace cbp::tracking
{

class DampedCVKalmanFilter
{
public:
  static constexpr int kStateDim = 7;
  static constexpr int kMeasDim = 4;

  DampedCVKalmanFilter();

  // Initialize from first observation
  void initialize(
    const Eigen::Vector4d & z,
    const Eigen::Matrix4d & R);

  // Predict step (Δt from timestamps)
  void predict(
    double dt,
    const Eigen::Matrix<double, kStateDim, kStateDim> & Q,
    double velocity_damping);

  // Update with measurement (coarse or ICP)
  void update(
    const Eigen::Vector4d & z,
    const Eigen::Matrix4d & R);

  // Accessors
  const Eigen::Vector<double, kStateDim> & x() const {return x_;}
  const Eigen::Matrix<double, kStateDim, kStateDim> & P() const {return P_;}

  Eigen::Vector<double, kStateDim> & x() {return x_;}
  Eigen::Matrix<double, kStateDim, kStateDim> & P() {return P_;}

private:
  void buildMotionModel(double dt, double velocity_damping);
  void buildMeasurementModel();

private:
  bool initialized_{false};

  Eigen::Vector<double, kStateDim> x_;
  Eigen::Matrix<double, kStateDim, kStateDim> P_;

  Eigen::Matrix<double, kStateDim, kStateDim> F_;
  Eigen::Matrix<double, kMeasDim, kStateDim> H_;
};

}  // namespace cbp::tracking
