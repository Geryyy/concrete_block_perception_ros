#pragma once

#include <Eigen/Dense>

namespace cbp::tracking
{

struct GateResult
{
  bool accepted{false};
  double mahalanobis_sq{0.0};
};

struct Gating
{
  static GateResult gatePosition3D(
    const Eigen::Vector<double, 7> & x,
    const Eigen::Matrix<double, 7, 7> & P,
    const Eigen::Vector3d & z,
    const Eigen::Matrix3d & R,
    double chi2_gate);
};

}  // namespace cbp::tracking
