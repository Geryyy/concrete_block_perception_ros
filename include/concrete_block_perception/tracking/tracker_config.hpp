#pragma once

#include <Eigen/Dense>

namespace cbp::tracking
{

struct TrackerConfig
{
  // ----------------------------
  // Assignment / lifecycle
  // ----------------------------
  double chi2_gate{7.815};   // χ² threshold (95%, 3 DoF)
  int min_hits{3};           // confirmations before publishing
  int max_misses{10};        // track deletion threshold

  // ----------------------------
  // Motion model
  // ----------------------------
  double velocity_damping{0.9};  // α ∈ (0,1), velocity decay

  // ----------------------------
  // Noise models
  // ----------------------------

  // State: [px py pz yaw vx vy vz] (7)
  Eigen::Matrix<double, 7, 7> Q =
    Eigen::Matrix<double, 7, 7>::Zero();

  // Measurement: [px py pz yaw] (4)
  Eigen::Matrix<double, 4, 4> R =
    Eigen::Matrix<double, 4, 4>::Zero();

  double deduplication_radius{0.1};
  double iou_thresh{0.3};
  double birth_suppression_radius{0.1};
};

}  // namespace cbp::tracking
