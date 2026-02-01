#include "concrete_block_perception/tracking/damped_cv_kalman_filter.hpp"

namespace cbp::tracking
{

DampedCVKalmanFilter::DampedCVKalmanFilter()
{
  x_.setZero();
  P_.setIdentity();
  P_ *= 1.0;  // large initial uncertainty

  buildMeasurementModel();
}

void DampedCVKalmanFilter::initialize(
  const Eigen::Vector4d & z,
  const Eigen::Matrix4d & R)
{
  x_.setZero();
  x_.segment<4>(0) = z;

  P_.setZero();
  P_.block<4, 4>(0, 0) = R;
  P_.block<3, 3>(4, 4).diagonal().setConstant(0.5);  // unknown velocity

  initialized_ = true;
}

void DampedCVKalmanFilter::buildMotionModel(
  double dt,
  double velocity_damping)
{
  F_.setIdentity();

  // Position integration
  F_(0, 4) = dt;
  F_(1, 5) = dt;
  F_(2, 6) = dt;

  // Velocity damping
  F_(4, 4) = velocity_damping;
  F_(5, 5) = velocity_damping;
  F_(6, 6) = velocity_damping;
}

void DampedCVKalmanFilter::buildMeasurementModel()
{
  H_.setZero();
  H_.block<4, 4>(0, 0).setIdentity();
}

void DampedCVKalmanFilter::predict(
  double dt,
  const Eigen::Matrix<double, kStateDim, kStateDim> & Q,
  double velocity_damping)
{
  if (!initialized_) {
    return;
  }

  buildMotionModel(dt, velocity_damping);

  x_ = F_ * x_;
  P_ = F_ * P_ * F_.transpose() + Q;
}

void DampedCVKalmanFilter::update(
  const Eigen::Vector4d & z,
  const Eigen::Matrix4d & R)
{
  if (!initialized_) {
    initialize(z, R);
    return;
  }

  const Eigen::Vector4d y = z - H_ * x_;
  const Eigen::Matrix4d S = H_ * P_ * H_.transpose() + R;
  const Eigen::Matrix<double, kStateDim, kMeasDim> K =
    P_ * H_.transpose() * S.inverse();

  x_ = x_ + K * y;
  P_ = (Eigen::Matrix<double, kStateDim, kStateDim>::Identity() - K * H_) * P_;
}

}  // namespace cbp::tracking
