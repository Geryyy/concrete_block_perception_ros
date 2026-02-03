#pragma once

#include <opencv2/imgproc.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>
#include <vision_msgs/msg/detection2_d.hpp>
#include <vision_msgs/msg/pose2_d.hpp>


cv::Mat extract_mask_roi(
  const cv::Mat & full_mask,
  const vision_msgs::msg::Detection2D & det);
double bboxCenterDistance(
  const vision_msgs::msg::Detection2D & a,
  const vision_msgs::msg::Detection2D & b);
double bboxIoU(const cv::Rect & a, const cv::Rect & b);
double maskIoU(const cv::Mat & a, const cv::Mat & b);
cv::Rect toCvRect(const vision_msgs::msg::Detection2D & det);
