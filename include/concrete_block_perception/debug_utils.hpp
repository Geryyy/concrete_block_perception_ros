#pragma once

#include <chrono>
#include <rclcpp/rclcpp.hpp>

struct TicToc
{
  using Clock = std::chrono::steady_clock;

  TicToc();

  void tic();
  double toc();
  double total() const;

private:
  Clock::time_point t0;
  Clock::time_point t_last;
};
