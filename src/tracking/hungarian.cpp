#include "concrete_block_perception/tracking/hungarian.hpp"

#include <limits>
#include <algorithm>

namespace cbp::tracking
{

std::vector<int> Hungarian::solve(const Eigen::MatrixXd & cost)
{
  const int n = cost.rows();
  const int m = cost.cols();

  const int dim = std::max(n, m);

  // Pad to square matrix
  Eigen::MatrixXd C = Eigen::MatrixXd::Constant(dim, dim, INF);
  C.block(0, 0, n, m) = cost;

  // Hungarian internals
  std::vector<double> u(dim + 1), v(dim + 1);
  std::vector<int> p(dim + 1), way(dim + 1);

  for (int i = 1; i <= dim; ++i) {
    p[0] = i;
    int j0 = 0;

    std::vector<double> minv(dim + 1, INF);
    std::vector<bool> used(dim + 1, false);

    do {
      used[j0] = true;
      const int i0 = p[j0];
      int j1 = 0;
      double delta = INF;

      for (int j = 1; j <= dim; ++j) {
        if (used[j]) {
          continue;
        }
        const double cur = C(i0 - 1, j - 1) - u[i0] - v[j];
        if (cur < minv[j]) {
          minv[j] = cur;
          way[j] = j0;
        }
        if (minv[j] < delta) {
          delta = minv[j];
          j1 = j;
        }
      }

      for (int j = 0; j <= dim; ++j) {
        if (used[j]) {
          u[p[j]] += delta;
          v[j] -= delta;
        } else {
          minv[j] -= delta;
        }
      }

      j0 = j1;
    } while (p[j0] != 0);

    // Augmenting path
    do {
      const int j1 = way[j0];
      p[j0] = p[j1];
      j0 = j1;
    } while (j0);
  }

  // Extract assignment
  std::vector<int> assignment(n, -1);
  for (int j = 1; j <= dim; ++j) {
    if (p[j] <= n && j <= m) {
      const double c = cost(p[j] - 1, j - 1);
      if (c < INF) {
        assignment[p[j] - 1] = j - 1;
      }
    }
  }

  return assignment;
}

}  // namespace cbp::tracking
