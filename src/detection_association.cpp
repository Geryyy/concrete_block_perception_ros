#include "concrete_block_perception/detection_association.hpp"

AssociationResult findBestAssociation(
  const vision_msgs::msg::Detection2D & track_det,
  const std::vector<vision_msgs::msg::Detection2D> & detections,
  const std::vector<size_t> & valid_indices,
  const std::vector<bool> & detection_used,
  const cv::Mat & full_mask,
  double iou_threshold)
{
  AssociationResult best{-1, 0.0};

  for (size_t i = 0; i < valid_indices.size(); ++i) {
    if (detection_used[i]) {continue;}

    const auto & det = detections[valid_indices[i]];
    double iou = 0.0;

    // ---- mask IoU (preferred)
    if (!full_mask.empty()) {
      cv::Mat det_mask = extract_mask_roi(full_mask, det);
      cv::Mat track_mask = extract_mask_roi(full_mask, track_det);

      if (cv::countNonZero(det_mask) > 0 &&
        cv::countNonZero(track_mask) > 0)
      {
        iou = maskIoU(det_mask, track_mask);
      }
    }

    // ---- fallback: bbox IoU
    if (iou <= 0.0) {
      iou = bboxIoU(
        toCvRect(track_det),
        toCvRect(det)
      );
    }

    if (iou > best.iou) {
      best = {static_cast<int>(i), iou};
    }
  }

  if (best.iou < iou_threshold) {
    return {-1, 0.0};
  }

  return best;
}
