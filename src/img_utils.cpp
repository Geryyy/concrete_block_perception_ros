#include "concrete_block_perception/img_utils.hpp"

cv::Mat extract_mask_roi(
  const cv::Mat & full_mask,
  const vision_msgs::msg::Detection2D & det)
{
  // Create empty mask with same size/type
  cv::Mat det_mask = cv::Mat::zeros(
    full_mask.rows,
    full_mask.cols,
    full_mask.type()
  );

  const int cx = static_cast<int>(std::round(det.bbox.center.position.x));
  const int cy = static_cast<int>(std::round(det.bbox.center.position.y));
  const int w = static_cast<int>(std::round(det.bbox.size_x));
  const int h = static_cast<int>(std::round(det.bbox.size_y));

  int x0 = std::max(0, cx - w / 2);
  int y0 = std::max(0, cy - h / 2);
  int x1 = std::min(full_mask.cols, cx + w / 2);
  int y1 = std::min(full_mask.rows, cy + h / 2);

  if (x1 <= x0 || y1 <= y0) {
    return det_mask;  // valid, but empty
  }

  // Copy only the ROI into the same-sized mask
  full_mask(cv::Rect(x0, y0, x1 - x0, y1 - y0))
  .copyTo(det_mask(cv::Rect(x0, y0, x1 - x0, y1 - y0)));

  return det_mask;
}


double bboxIoU(const cv::Rect & a, const cv::Rect & b)
{
  const int x1 = std::max(a.x, b.x);
  const int y1 = std::max(a.y, b.y);
  const int x2 = std::min(a.x + a.width,  b.x + b.width);
  const int y2 = std::min(a.y + a.height, b.y + b.height);

  const int inter_area =
    std::max(0, x2 - x1) * std::max(0, y2 - y1);

  const int union_area =
    a.area() + b.area() - inter_area;

  if (union_area <= 0)
    return 0.0;

  return static_cast<double>(inter_area) / union_area;
}
