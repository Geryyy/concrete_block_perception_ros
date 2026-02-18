#pragma once

#include <Eigen/Dense>

namespace concrete_block_perception
{

struct PreprocessingParams
{
  size_t max_pts{500};
  int nb_neighbors{20};
  double std_dev{2.0};
};

struct GlobalRegistrationParams
{
  static constexpr int MAX_PLANES = 2;
  Eigen::Vector3d Z_WORLD{0, 0, 1};
  double dist_thresh{0.02};
  int min_inliers{100};
  double angle_thresh{0.9};
  double max_plane_center_dist{0.6};
  bool enable_plane_clipping{false};
};

struct LocalRegistrationParams
{
  double icp_dist{0.04};
};

}
