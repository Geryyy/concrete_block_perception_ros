#pragma once

#include <rclcpp/rclcpp.hpp>

// ------------------------------------------------------------
// Compile-time switch
// ------------------------------------------------------------
// Uncomment to force INFO instead of DEBUG
#define CBP_TRACKER_LOG_INFO

namespace cbp::tracking
{

inline rclcpp::Logger make_child_logger(
  rclcpp::Logger parent,
  const std::string & child)
{
  return parent.get_child(child);
}

}  // namespace cbp::tracking

// ------------------------------------------------------------
// Logging macros
// ------------------------------------------------------------
#ifdef CBP_TRACKER_LOG_INFO
  #define TRACKER_LOG(logger, ...) \
  RCLCPP_INFO(logger, __VA_ARGS__)
#else
  #define TRACKER_LOG(logger, ...) \
  RCLCPP_DEBUG(logger, __VA_ARGS__)
#endif

#define TRACKER_LOG_WARN(logger, ...) \
  RCLCPP_WARN(logger, __VA_ARGS__)

#define TRACKER_LOG_ERROR(logger, ...) \
  RCLCPP_ERROR(logger, __VA_ARGS__)
