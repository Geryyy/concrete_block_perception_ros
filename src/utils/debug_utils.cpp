#include "concrete_block_perception/debug_utils.hpp"


TicToc::TicToc()
: t0(Clock::now()), t_last(t0) {}

void TicToc::tic()
{
  t0 = Clock::now();
}

double TicToc::toc()
{
  const auto now = Clock::now();
  const double ms =
    std::chrono::duration<double, std::milli>(now - t_last).count();
  t_last = now;
  return ms;
}

double TicToc::total() const
{
  return std::chrono::duration<double, std::milli>(
    Clock::now() - t0).count();
}
