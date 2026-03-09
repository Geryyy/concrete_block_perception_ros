#include "concrete_block_perception/world_model/roi_processing.hpp"

#include <algorithm>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgproc.hpp>

#include "concrete_block_perception/utils/img_utils.hpp"

namespace cbp::world_model
{

sensor_msgs::msg::Image::SharedPtr buildRoiSegmentationInputImage(
  const sensor_msgs::msg::Image::ConstSharedPtr & image,
  const cv::Rect & roi_rect,
  const RoiInputConfig & roi_cfg)
{
  cv::Mat image_bgr = toCvBgr(*image);
  cv::Mat roi_image_full;
  if (roi_cfg.use_black_bg) {
    roi_image_full = cv::Mat::zeros(image_bgr.size(), image_bgr.type());
  } else {
    const int ksize = std::max(1, roi_cfg.blur_kernel_size | 1);
    cv::GaussianBlur(image_bgr, roi_image_full, cv::Size(ksize, ksize), 0.0);
  }
  image_bgr(roi_rect).copyTo(roi_image_full(roi_rect));

  auto out = cv_bridge::CvImage(image->header, "bgr8", roi_image_full).toImageMsg();
  return out;
}

bool roiSegmentationToFullMask(
  const sensor_msgs::msg::Image::ConstSharedPtr & image,
  const cv::Mat & roi_mask,
  const sensor_msgs::msg::Image::SharedPtr & roi_image_msg,
  const RoiInputConfig & roi_cfg,
  const RunSegmentationSyncFn & run_segmentation_sync,
  cv::Mat & full_seg_mask,
  size_t & detections_count,
  std::string & reason)
{
  ros2_yolos_cpp::srv::SegmentImage::Response::SharedPtr seg_response;
  if (!run_segmentation_sync(
      *roi_image_msg,
      roi_cfg.segmentation_timeout_s,
      seg_response,
      reason))
  {
    return false;
  }

  cv::Mat seg_mask_full = toCvMono(seg_response->mask);
  if (seg_mask_full.size() != roi_mask.size()) {
    reason = "segmentation mask/image size mismatch";
    return false;
  }

  full_seg_mask = cv::Mat::zeros(seg_mask_full.size(), CV_8UC1);
  seg_mask_full.copyTo(full_seg_mask, roi_mask);
  detections_count = seg_response->detections.detections.size();

  if (cv::countNonZero(full_seg_mask) == 0) {
    reason = "ROI segmentation produced empty mask";
    return false;
  }

  (void)image;
  return true;
}

sensor_msgs::msg::Image::SharedPtr buildRoiSegmentationDebugOverlay(
  const sensor_msgs::msg::Image::ConstSharedPtr & image,
  const cv::Rect & roi_rect,
  const cv::Mat & full_seg_mask)
{
  cv::Mat dbg = toCvBgr(*image);
  cv::rectangle(dbg, roi_rect, cv::Scalar(0, 0, 255), 2);
  overlayMask(dbg, full_seg_mask, cv::Scalar(255, 0, 0), 0.35);
  return cv_bridge::CvImage(image->header, "bgr8", dbg).toImageMsg();
}

}  // namespace cbp::world_model

