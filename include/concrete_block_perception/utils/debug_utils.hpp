#pragma once

#include <chrono>
#include <rclcpp/rclcpp.hpp>

#include <rclcpp_action/rclcpp_action.hpp>

#include <iomanip>
#include <sstream>

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


// ------------------------------------------------------------
// UUID formatting
// ------------------------------------------------------------
inline std::string goal_uuid_short(
  const rclcpp_action::GoalUUID & uuid,
  size_t n = 8)
{
  std::ostringstream oss;
  for (auto b : uuid) {
    oss << std::hex << std::setw(2) << std::setfill('0')
        << static_cast<int>(b);
  }
  auto s = oss.str();
  return s.substr(0, std::min(n, s.size()));
}

// ------------------------------------------------------------
// Goal-aware INFO logger
// ------------------------------------------------------------
inline void log_goal_info(
  const rclcpp::Logger & logger,
  const rclcpp_action::GoalUUID & uuid,
  const char * fmt,
  ...)
{
  // prepend [goal=xxxx]
  std::ostringstream prefix;
  prefix << "[goal=" << goal_uuid_short(uuid) << "] ";

  char buffer[1024];

  va_list args;
  va_start(args, fmt);
  vsnprintf(buffer, sizeof(buffer), fmt, args);
  va_end(args);

  RCLCPP_INFO(
    logger,
    "%s%s",
    prefix.str().c_str(),
    buffer);
}

inline void log_goal_warn(
  const rclcpp::Logger & logger,
  const rclcpp_action::GoalUUID & uuid,
  const char * fmt,
  ...)
{
  // prepend [goal=xxxx]
  std::ostringstream prefix;
  prefix << "[goal=" << goal_uuid_short(uuid) << "] ";

  char buffer[1024];

  va_list args;
  va_start(args, fmt);
  vsnprintf(buffer, sizeof(buffer), fmt, args);
  va_end(args);

  RCLCPP_WARN(
    logger,
    "%s%s",
    prefix.str().c_str(),
    buffer);
}

inline void log_goal_error(
  const rclcpp::Logger & logger,
  const rclcpp_action::GoalUUID & uuid,
  const char * fmt,
  ...)
{
  // prepend [goal=xxxx]
  std::ostringstream prefix;
  prefix << "[goal=" << goal_uuid_short(uuid) << "] ";

  char buffer[1024];

  va_list args;
  va_start(args, fmt);
  vsnprintf(buffer, sizeof(buffer), fmt, args);
  va_end(args);

  RCLCPP_ERROR(
    logger,
    "%s%s",
    prefix.str().c_str(),
    buffer);
}
