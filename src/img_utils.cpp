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
