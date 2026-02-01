#include "concrete_block_perception/tracking/gating.hpp"

namespace cbp::tracking
{

GateResult Gating::gatePosition3D(
  const Eigen::Vector<double, 7> & x,
  const Eigen::Matrix<double, 7, 7> & P,
  const Eigen::Vector3d & z,
  const Eigen::Matrix3d & R,
  double chi2_gate)
{
  // Measurement model: z = [x y z]
  Eigen::Matrix<double, 3, 7> H;
  H.setZero();
  H.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();

  Eigen::Vector3d y = z - H * x;
  Eigen::Matrix3d S = H * P * H.transpose() + R;

  const double d2 = y.transpose() * S.inverse() * y;

  return {
    d2 <= chi2_gate,
    d2
  };
}

}  // namespace cbp::tracking
