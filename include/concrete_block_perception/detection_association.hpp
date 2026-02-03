#pragma once

#include <unordered_map>
#include <vector>

#include <opencv2/opencv.hpp>
#include <vision_msgs/msg/detection2_d.hpp>

#include "concrete_block_perception/img_utils.hpp"

struct AssociationResult
{
  int detection_index;   // index into valid_indices
  double iou;
};

AssociationResult findBestAssociation(
  const vision_msgs::msg::Detection2D & track_det,
  const std::vector<vision_msgs::msg::Detection2D> & detections,
  const std::vector<size_t> & valid_indices,
  const std::vector<bool> & detection_used,
  const cv::Mat & full_mask,
  double iou_threshold);
