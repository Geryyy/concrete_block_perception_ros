#include "concrete_block_perception/detection/detection_config.hpp"

namespace concrete_block_perception
{

DetectionConfig::DetectionConfig(rclcpp::Node & node)
{
  params_.min_confidence =
    node.declare_parameter<double>(
    "detection.min_confidence", 0.4);

  params_.min_bbox_area =
    node.declare_parameter<double>(
    "detection.min_bbox_area", 500.0);

  params_.iou_threshold =
    node.declare_parameter<double>(
    "detection.iou_threshold", 0.3);

  params_.containment_ratio =
    node.declare_parameter<double>(
    "detection.containment_ratio", 0.9);

  params_.suppression_radius =
    node.declare_parameter<double>(
    "detection.suppression_radius", 0.5);

  params_.confirm_age =
    node.declare_parameter<int>(
    "detection.confirm_age", 2);

  params_.max_misses =
    node.declare_parameter<int>(
    "detection.max_misses", 5);

  params_.publish_debug_image =
    node.declare_parameter<bool>(
    "detection.publish_debug_image", true);
}

}  // namespace concrete_block_perception
