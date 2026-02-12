#include "concrete_block_perception/detection/block_detection_tracker.hpp"

#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include "concrete_block_perception/utils/img_utils.hpp"
#include "concrete_block_perception/utils/detection_association.hpp"

namespace concrete_block_perception
{

BlockDetectionTracker::BlockDetectionTracker(
  const DetectionParams & params)
: params_(params)
{
}

void BlockDetectionTracker::reset()
{
  tracks_.clear();
  next_detection_id_ = 1;
}

msg::TrackedDetectionArray BlockDetectionTracker::update(
  const vision_msgs::msg::Detection2DArray & msg,
  const sensor_msgs::msg::Image::SharedPtr & mask_msg)
{
  const rclcpp::Time now = msg.header.stamp;

  // -----------------------------
  // 1. Pre-filter detections
  // -----------------------------
  std::vector<size_t> valid_indices;
  valid_indices.reserve(msg.detections.size());

  for (size_t i = 0; i < msg.detections.size(); ++i) {
    const auto & det = msg.detections[i];

    if (!passesConfidenceThreshold(det)) {continue;}
    if (!passesSizeFilter(det)) {continue;}

    valid_indices.push_back(i);
  }

  // -----------------------------
  // 2. Prepare mask
  // -----------------------------
  cv::Mat full_mask;
  if (mask_msg) {
    full_mask =
      cv_bridge::toCvCopy(*mask_msg, "mono8")->image;
  }

  std::vector<bool> detection_used(valid_indices.size(), false);

  // -----------------------------
  // 3. Associate detections
  // -----------------------------
  for (auto & kv : tracks_) {
    auto & track = kv.second;

    AssociationResult assoc =
      findBestAssociation(
      track.detection,
      msg.detections,
      valid_indices,
      detection_used,
      full_mask,
      params_.iou_threshold);

    if (assoc.detection_index < 0) {
      continue;
    }

    const size_t det_idx =
      valid_indices[assoc.detection_index];

    track.detection = msg.detections[det_idx];
    track.detection.header.stamp = now;

    if (!full_mask.empty()) {
      cv::Mat det_mask =
        extract_mask_roi(full_mask, track.detection);

      auto mask_msg_out =
        cv_bridge::CvImage(
        std_msgs::msg::Header(),
        "mono8",
        det_mask).toImageMsg();

      mask_msg_out->header.stamp = now;
      mask_msg_out->header.frame_id =
        track.detection.header.frame_id;

      track.mask = *mask_msg_out;
    }

    track.age++;
    track.misses = 0;
    track.last_seen = now;

    detection_used[assoc.detection_index] = true;
  }

  // -----------------------------
  // 4. Create new tracks
  // -----------------------------
  for (size_t i = 0; i < valid_indices.size(); ++i) {

    if (detection_used[i]) {
      continue;
    }

    const auto & det =
      msg.detections[valid_indices[i]];

    bool suppressed = false;

    for (const auto & kv : tracks_) {
      const auto & prev = kv.second;

      if (prev.age < params_.confirm_age) {
        continue;
      }

      if (isContained(
          det, prev.detection,
          params_.containment_ratio))
      {
        suppressed = true;
        break;
      }

      double dist =
        bboxCenterDistance(det, prev.detection);

      if (dist < suppressionRadiusPx(prev)) {
        suppressed = true;
        break;
      }
    }

    if (suppressed) {
      continue;
    }

    DetectionTrack track;
    track.detection_id = next_detection_id_++;
    track.detection = det;
    track.age = 1;
    track.misses = 0;
    track.last_seen = now;

    if (!full_mask.empty()) {
      cv::Mat det_mask =
        extract_mask_roi(full_mask, det);

      auto mask_msg_out =
        cv_bridge::CvImage(
        std_msgs::msg::Header(),
        "mono8",
        det_mask).toImageMsg();

      mask_msg_out->header.stamp = now;
      mask_msg_out->header.frame_id =
        det.header.frame_id;

      track.mask = *mask_msg_out;
    }

    tracks_[track.detection_id] = track;
  }

  // -----------------------------
  // 5. Miss handling
  // -----------------------------
  for (auto it = tracks_.begin();
    it != tracks_.end(); )
  {
    if (it->second.last_seen != now) {
      it->second.misses++;
    }

    if (it->second.misses > params_.max_misses) {
      it = tracks_.erase(it);
    } else {
      ++it;
    }
  }

  pruneContainedTracks();

  // -----------------------------
  // 6. Output
  // -----------------------------
  msg::TrackedDetectionArray out;
  out.stamp = now;

  for (const auto & kv : tracks_) {
    const auto & track = kv.second;

    if (track.age < params_.confirm_age) {
      continue;
    }

    msg::TrackedDetection td;
    td.detection_id = track.detection_id;
    td.detection = track.detection;
    td.mask = track.mask;
    td.age = track.age;
    td.misses = track.misses;
    td.stamp = now;

    out.detections.push_back(td);
  }

  return out;
}

// ===================================================
// Helpers
// ===================================================

bool BlockDetectionTracker::passesConfidenceThreshold(
  const vision_msgs::msg::Detection2D & det) const
{
  for (const auto & r : det.results) {
    if (r.hypothesis.score >= params_.min_confidence) {
      return true;
    }
  }
  return false;
}

bool BlockDetectionTracker::passesSizeFilter(
  const vision_msgs::msg::Detection2D & det) const
{
  const auto & bbox = det.bbox;
  double area = bbox.size_x * bbox.size_y;
  return area >= params_.min_bbox_area;
}

cv::Rect BlockDetectionTracker::toCvRect(
  const vision_msgs::msg::Detection2D & det) const
{
  const auto & bbox = det.bbox;

  int cx = static_cast<int>(bbox.center.position.x);
  int cy = static_cast<int>(bbox.center.position.y);
  int w = static_cast<int>(bbox.size_x);
  int h = static_cast<int>(bbox.size_y);

  return cv::Rect(cx - w / 2, cy - h / 2, w, h);
}

bool BlockDetectionTracker::isContained(
  const vision_msgs::msg::Detection2D & a,
  const vision_msgs::msg::Detection2D & b,
  double containment_ratio) const
{
  cv::Rect ra = toCvRect(a);
  cv::Rect rb = toCvRect(b);

  cv::Rect inter = ra & rb;
  if (inter.area() <= 0) {
    return false;
  }

  double min_area =
    std::min(ra.area(), rb.area());

  if (min_area <= 0) {
    return false;
  }

  double overlap =
    static_cast<double>(inter.area()) / min_area;

  return overlap >= containment_ratio;
}

double BlockDetectionTracker::suppressionRadiusPx(
  const DetectionTrack & track) const
{
  const auto & bbox = track.detection.bbox;
  return params_.suppression_radius *
         std::max(bbox.size_x, bbox.size_y);
}

void BlockDetectionTracker::pruneContainedTracks()
{
  std::vector<uint32_t> to_delete;

  for (auto it1 = tracks_.begin();
    it1 != tracks_.end(); ++it1)
  {
    auto it2 = it1;
    ++it2;

    for (; it2 != tracks_.end(); ++it2) {
      const auto & t1 = it1->second;
      const auto & t2 = it2->second;

      if (t1.age < params_.confirm_age ||
        t2.age < params_.confirm_age)
      {
        continue;
      }

      if (isContained(
          t1.detection,
          t2.detection,
          params_.containment_ratio))
      {
        to_delete.push_back(it1->first);
      }
    }
  }

  for (auto id : to_delete) {
    tracks_.erase(id);
  }
}

}  // namespace concrete_block_perception
