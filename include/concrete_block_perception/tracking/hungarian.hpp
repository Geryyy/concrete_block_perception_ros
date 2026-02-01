#pragma once

#include <Eigen/Dense>
#include <vector>

namespace cbp::tracking
{

/**
 * @brief Hungarian (Munkres) algorithm for minimum-cost assignment
 *
 * Cost matrix:
 *   rows    = tracks
 *   columns = measurements
 *
 * Output:
 *   assignment[i] = j  → track i assigned to measurement j
 *   assignment[i] = -1 → unassigned
 */
class Hungarian
{
public:
  static std::vector<int> solve(const Eigen::MatrixXd & cost);

private:
  static constexpr double INF = 1e12;
};

}  // namespace cbp::tracking
