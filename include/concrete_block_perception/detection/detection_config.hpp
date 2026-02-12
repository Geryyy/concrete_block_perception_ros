#pragma once

#include <rclcpp/rclcpp.hpp>
#include "concrete_block_perception/detection/detection_params.hpp"

namespace concrete_block_perception
{

class DetectionConfig
{
public:
  explicit DetectionConfig(rclcpp::Node & node);

  const DetectionParams & params() const {return params_;}

private:
  DetectionParams params_;
};

}  // namespace concrete_block_perception
