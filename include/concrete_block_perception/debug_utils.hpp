#pragma once

#include <chrono>

struct TicToc
{
  using Clock = std::chrono::steady_clock;

  TicToc();

  double toc();
  double total() const;

private:
  Clock::time_point t0;
  Clock::time_point t_last;
};
