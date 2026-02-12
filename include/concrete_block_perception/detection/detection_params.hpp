#pragma once

namespace concrete_block_perception
{

struct DetectionParams
{
  double min_confidence = 0.4;
  double min_bbox_area = 500.0;
  double iou_threshold = 0.3;
  double containment_ratio = 0.9;
  double suppression_radius = 0.5;

  uint32_t confirm_age = 2;
  uint32_t max_misses = 5;

  bool publish_debug_image = true;
};

}  // namespace concrete_block_perception
