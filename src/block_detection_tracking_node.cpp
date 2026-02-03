#include <rclcpp/rclcpp.hpp>

#include <cv_bridge/cv_bridge.h>
#include <vision_msgs/msg/detection2_d_array.hpp>
#include <sensor_msgs/msg/image.hpp>

#include "concrete_block_perception/msg/tracked_detection.hpp"
#include "concrete_block_perception/msg/tracked_detection_array.hpp"
#include "concrete_block_perception/img_utils.hpp"
#include "concrete_block_perception/detection_association.hpp"

#include <unordered_map>
#include <vector>

// ================================
// Logging macro
// ================================
// Change RCLCPP_DEBUG -> RCLCPP_INFO to globally raise verbosity
#define TRACK_LOG(logger, ...) RCLCPP_INFO(logger, __VA_ARGS__)


using vision_msgs::msg::Detection2DArray;
using sensor_msgs::msg::Image;
using concrete_block_perception::msg::TrackedDetection;
using concrete_block_perception::msg::TrackedDetectionArray;

struct DetectionTrack
{
  uint32_t detection_id;
  vision_msgs::msg::Detection2D detection;
  sensor_msgs::msg::Image mask;

  uint32_t age = 0;
  uint32_t misses = 0;

  rclcpp::Time last_seen;
};

struct PrevTrackSnapshot
{
  vision_msgs::msg::Detection2D detection;
  uint32_t age;
};

class BlockDetectionTrackingNode : public rclcpp::Node
{
public:
  BlockDetectionTrackingNode()
  : Node("block_detection_tracking_node")
  {

    // -----------------------------
    // Parameters
    // -----------------------------

    min_confidence_ = this->declare_parameter<double>(
      "min_confidence", 0.4);

    min_bbox_area_ = this->declare_parameter<double>(
      "min_bbox_area", 500.0);

    iou_threshold_ = this->declare_parameter<double>(
      "iou_threshold", 0.3);

    containment_ratio_ = this->declare_parameter<double>(
      "containment_ratio", 0.9);

    suppression_radius_ = this->declare_parameter<double>(
      "suppression_radius", 0.5);

    confirm_age_ = this->declare_parameter<int>(
      "confirm_age", 2);

    max_misses_ = this->declare_parameter<int>(
      "max_misses", 5);

    publish_debug_image_ = this->declare_parameter<bool>(
      "publish_debug_image", true);

    debug_topic_ = this->declare_parameter<std::string>(
      "debug_topic", "debug/tracked_detections_image");


    detections_sub_ = this->create_subscription<Detection2DArray>(
      "detections", rclcpp::SensorDataQoS(),
      std::bind(&BlockDetectionTrackingNode::detectionsCallback, this, std::placeholders::_1));

    masks_sub_ = this->create_subscription<Image>(
      "masks", rclcpp::SensorDataQoS(),
      std::bind(&BlockDetectionTrackingNode::maskCallback, this, std::placeholders::_1));

    tracked_pub_ = this->create_publisher<TrackedDetectionArray>(
      "tracked_detections", 10);

    if (publish_debug_image_) {
      debug_image_pub_ =
        this->create_publisher<sensor_msgs::msg::Image>(
        debug_topic_, 1);
    }

    TRACK_LOG(
      this->get_logger(),
      "Parameters: min_confidence=%.2f min_bbox_area=%.1f iou_threshold=%.2f "
      "confirm_age=%u max_misses=%u publish_debug_image=%s",
      min_confidence_,
      min_bbox_area_,
      iou_threshold_,
      confirm_age_,
      max_misses_,
      publish_debug_image_ ? "true" : "false");

    RCLCPP_INFO(this->get_logger(), "BlockDetectionTrackingNode started");
  }

private:
  // ================================
  // ROS Interfaces
  // ================================
  rclcpp::Subscription<Detection2DArray>::SharedPtr detections_sub_;
  rclcpp::Subscription<Image>::SharedPtr masks_sub_;
  rclcpp::Publisher<TrackedDetectionArray>::SharedPtr tracked_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr debug_image_pub_;


  // ================================
  // Internal State
  // ================================
  std::unordered_map<uint32_t, DetectionTrack> tracks_;
  uint32_t next_detection_id_ = 1;

  // Latest mask frame (simple version)
  sensor_msgs::msg::Image::SharedPtr last_mask_;

  // -----------------------------
  // Parameters
  // -----------------------------
  double min_confidence_;
  double min_bbox_area_;
  double iou_threshold_;
  double containment_ratio_;
  double suppression_radius_;
  uint32_t confirm_age_;
  uint32_t max_misses_;
  bool publish_debug_image_;
  std::string debug_topic_;

  // ================================
  // Callbacks
  // ================================
  void maskCallback(const Image::SharedPtr msg)
  {
    last_mask_ = msg;
    TRACK_LOG(
      this->get_logger(),
      "Received segmentation mask [%ux%u]",
      msg->width, msg->height);
  }

  void detectionsCallback(const Detection2DArray::SharedPtr msg)
  {
    // std::vector<vision_msgs::msg::Detection2D> prev_track_detections;
    // prev_track_detections.reserve(tracks_.size());

    // for (const auto & kv : tracks_) {
    //   prev_track_detections.push_back(kv.second.detection);
    // }

    std::vector<PrevTrackSnapshot> prev_tracks;
    prev_tracks.reserve(tracks_.size());

    for (const auto & kv : tracks_) {
      prev_tracks.push_back({kv.second.detection, kv.second.age});
    }

    TRACK_LOG(
      this->get_logger(),
      "Received %zu detections",
      msg->detections.size());

    const rclcpp::Time now = msg->header.stamp;

    // --------------------------------
    // 1. Pre-filter detections
    // --------------------------------
    std::vector<size_t> valid_indices;
    valid_indices.reserve(msg->detections.size());

    for (size_t i = 0; i < msg->detections.size(); ++i) {
      const auto & det = msg->detections[i];

      if (!passesConfidenceThreshold(det)) {continue;}
      if (!passesSizeFilter(det)) {continue;}

      valid_indices.push_back(i);
    }

    TRACK_LOG(
      this->get_logger(),
      "Pre-filtering: %zu -> %zu valid detections",
      msg->detections.size(),
      valid_indices.size());

    // --------------------------------
    // 2. Prepare mask (once)
    // --------------------------------
    cv::Mat full_mask;
    if (last_mask_) {
      full_mask =
        cv_bridge::toCvCopy(*last_mask_, "mono8")->image;
    }

    // --------------------------------
    // 3. Associate detections to tracks
    // --------------------------------
    std::unordered_map<uint32_t, bool> track_matched;
    for (const auto & kv : tracks_) {
      track_matched[kv.first] = false;
    }

    std::vector<bool> detection_used(valid_indices.size(), false);


    for (auto & kv : tracks_) {
      auto & track = kv.second;

      AssociationResult assoc =
        findBestAssociation(
        track.detection,
        msg->detections,
        valid_indices,
        detection_used,
        full_mask,
        iou_threshold_);

      if (assoc.detection_index >= 0) {
        TRACK_LOG(
          this->get_logger(),
          "Track %u matched with detection %zu (IoU=%.3f)",
          track.detection_id,
          valid_indices[assoc.detection_index],
          assoc.iou);
      } else {
        TRACK_LOG(
          this->get_logger(),
          "Track %u not matched (misses=%u)",
          track.detection_id,
          track.misses);
      }

      if (assoc.detection_index < 0) {
        continue;
      }

      const size_t det_idx = valid_indices[assoc.detection_index];

      // ---- update track
      track.detection = msg->detections[det_idx];

      if (!full_mask.empty()) {
        cv::Mat det_mask =
          extract_mask_roi(full_mask, track.detection);

        track.mask =
          *cv_bridge::CvImage(
          last_mask_->header,
          "mono8",
          det_mask).toImageMsg();
      } else {
        track.mask = sensor_msgs::msg::Image{};
      }

      track.age++;
      track.misses = 0;
      track.last_seen = now;

      track_matched[track.detection_id] = true;
      detection_used[assoc.detection_index] = true;
    }

    // --------------------------------
    // 4. Create new tracks
    // --------------------------------
    for (size_t i = 0; i < valid_indices.size(); ++i) {
      if (detection_used[i]) {
        continue;
      }

      const auto & det = msg->detections[valid_indices[i]];

      // --------------------------------
      // Containment suppression
      // --------------------------------
      bool contained = false;
      bool suppressed = false;

      for (const auto & prev : prev_tracks) {
        if (prev.age < confirm_age_) {
          continue;
        }

        // --- containment test ---
        if (isContained(det, prev.detection, containment_ratio_)) {
          TRACK_LOG(
            this->get_logger(),
            "Suppressed new track: detection %zu contained in previous track",
            valid_indices[i]);

          contained = true;
          break;
        }


        // --- radius-based suppression ---
        const double dist = bboxCenterDistance(det, prev.detection);

        const double suppress_radius =
          suppression_radius_ * std::max(
          prev.detection.bbox.size_x,
          prev.detection.bbox.size_y);

        if (dist < suppress_radius) {
          TRACK_LOG(
            this->get_logger(),
            "Suppressed new track: detection %zu near confirmed track "
            "(dist=%.1f < %.1f px)",
            valid_indices[i],
            dist,
            suppress_radius);

          suppressed = true;
          break;
        }


      }

      if (contained) {
        continue;
      }

      if (suppressed) {
        continue;
      }

      // --------------------------------
      // Create new track
      // --------------------------------
      DetectionTrack track;
      track.detection_id = next_detection_id_++;
      track.detection = det;

      TRACK_LOG(
        this->get_logger(),
        "Creating new track %u from detection %zu",
        track.detection_id,
        valid_indices[i]);

      if (!full_mask.empty()) {
        cv::Mat det_mask =
          extract_mask_roi(full_mask, det);

        track.mask =
          *cv_bridge::CvImage(
          last_mask_->header,
          "mono8",
          det_mask).toImageMsg();
      }

      track.age = 1;
      track.misses = 0;
      track.last_seen = now;

      tracks_[track.detection_id] = track;
    }


    // --------------------------------
    // 5. Handle missing tracks
    // --------------------------------

    for (auto it = tracks_.begin(); it != tracks_.end(); ) {
      if (it->second.last_seen != now) {
        it->second.misses++;
      }

      if (it->second.misses > max_misses_) {
        TRACK_LOG(
          this->get_logger(),
          "Deleting track %u after %u misses",
          it->second.detection_id,
          it->second.misses);

        it = tracks_.erase(it);
      } else {
        ++it;
      }
    }

    // --------------------------------
    // 5b. Prune contained tracks
    // --------------------------------
    pruneContainedTracks();

    // --------------------------------
    // 6. Publish tracked detections
    // --------------------------------
    TrackedDetectionArray out;
    out.stamp = now;

    for (const auto & kv : tracks_) {
      const auto & track = kv.second;

      if (track.age < confirm_age_) {
        continue; // confirmation gate

      }
      TrackedDetection td;
      td.detection_id = track.detection_id;
      td.detection = track.detection;
      td.mask = track.mask;
      td.age = track.age;
      td.misses = track.misses;
      td.stamp = now;

      out.detections.push_back(td);
    }

    TRACK_LOG(
      this->get_logger(),
      "Publishing %zu tracked detections (%zu total tracks)",
      out.detections.size(),
      tracks_.size());

    tracked_pub_->publish(out);

    // --------------------------------
    // 7. Publish debug overlay image
    // --------------------------------
    if (publish_debug_image_ &&
      debug_image_pub_ &&
      debug_image_pub_->get_subscription_count() > 0 &&
      last_mask_)
    {
      TRACK_LOG(
        this->get_logger(),
        "Publishing debug overlay image with %zu tracks",
        tracks_.size());


      // Convert mask to color image for visualization
      cv::Mat base;
      cv::cvtColor(
        cv_bridge::toCvCopy(*last_mask_, "mono8")->image,
        base,
        cv::COLOR_GRAY2BGR);

      // Draw only confirmed tracks
      for (const auto & kv : tracks_) {
        const auto & track = kv.second;
        //if (track.age < 2) {continue;} // tentative tracks

        drawTrackOverlay(base, track);
      }

      auto dbg_msg =
        cv_bridge::CvImage(
        last_mask_->header,
        "bgr8",
        base).toImageMsg();

      debug_image_pub_->publish(*dbg_msg);
    }

  }


  // ================================
  // Helper Functions
  // ================================
  bool passesConfidenceThreshold(
    const vision_msgs::msg::Detection2D & det) const
  {
    for (const auto & r : det.results) {
      if (r.hypothesis.score >= min_confidence_) {
        return true;
      }
    }
    return false;
  }


  bool passesSizeFilter(
    const vision_msgs::msg::Detection2D & det) const
  {
    const auto & bbox = det.bbox;
    const double area = bbox.size_x * bbox.size_y;
    return area >= min_bbox_area_;
  }

  cv::Scalar trackStateColor(const DetectionTrack & track)
  {
    // BGR format (OpenCV)
    if (track.age < confirm_age_) {
      // Tentative
      return cv::Scalar(100, 255, 255); // yellow
    }

    if (track.misses > 0) {
      // Stale / temporarily lost
      return cv::Scalar(0, 0, 255);   // red
    }

    // Confirmed & healthy
    return cv::Scalar(0, 255, 0);     // green
  }

  bool isContained(
    const vision_msgs::msg::Detection2D & a,
    const vision_msgs::msg::Detection2D & b,
    double containment_ratio = 0.7) // <- lower!
  {
    const cv::Rect ra = toCvRect(a);
    const cv::Rect rb = toCvRect(b);

    const cv::Rect inter = ra & rb;
    if (inter.area() <= 0) {
      return false;
    }

    const double min_area =
      static_cast<double>(std::min(ra.area(), rb.area()));

    if (min_area <= 0.0) {
      return false;
    }

    const double overlap_ratio =
      static_cast<double>(inter.area()) / min_area;

    // Additional center-in-box check
    const cv::Point center_a(
      ra.x + ra.width / 2,
      ra.y + ra.height / 2);

    const bool center_inside = rb.contains(center_a);

    return overlap_ratio >= containment_ratio || center_inside;
  }

  double suppressionRadiusPx(const DetectionTrack & track) const
  {
    const auto & bbox = track.detection.bbox;
    return suppression_radius_ * std::max(bbox.size_x, bbox.size_y);
  }

  void pruneContainedTracks()
  {
    std::vector<uint32_t> to_delete;

    for (auto it1 = tracks_.begin(); it1 != tracks_.end(); ++it1) {
      auto it2 = it1;
      ++it2;

      for (; it2 != tracks_.end(); ++it2) {
        const auto & t1 = it1->second;
        const auto & t2 = it2->second;

        // Only prune confirmed tracks
        if (t1.age < confirm_age_ || t2.age < confirm_age_) {
          continue;
        }

        const cv::Rect r1 = toCvRect(t1.detection);
        const cv::Rect r2 = toCvRect(t2.detection);

        // No overlap → skip
        if ((r1 & r2).area() <= 0) {
          continue;
        }

        const bool t1_in_t2 = isContained(t1.detection, t2.detection, containment_ratio_);
        const bool t2_in_t1 = isContained(t2.detection, t1.detection, containment_ratio_);

        if (!t1_in_t2 && !t2_in_t1) {
          continue;
        }

        // Decide which one to keep
        const double a1 = r1.area();
        const double a2 = r2.area();

        uint32_t erase_id;

        if (t1_in_t2 && !t2_in_t1) {
          erase_id = it1->first;
        } else if (t2_in_t1 && !t1_in_t2) {
          erase_id = it2->first;
        } else {
          // Mutual containment → resolve by area, then age
          if (a1 < a2) {
            erase_id = it1->first;
          } else if (a2 < a1) {
            erase_id = it2->first;
          } else {
            erase_id = (t1.age < t2.age) ? it1->first : it2->first;
          }
        }

        TRACK_LOG(
          this->get_logger(),
          "Pruned contained track %u (contained by another track)",
          erase_id);

        to_delete.push_back(erase_id);
      }
    }

    // Erase after iteration
    for (uint32_t id : to_delete) {
      tracks_.erase(id);
    }
  }


  void drawTrackOverlay(
    cv::Mat & image,
    const DetectionTrack & track)
  {
    const auto & bbox = track.detection.bbox;

    const int cx = static_cast<int>(std::round(bbox.center.position.x));
    const int cy = static_cast<int>(std::round(bbox.center.position.y));
    const int w = static_cast<int>(std::round(bbox.size_x));
    const int h = static_cast<int>(std::round(bbox.size_y));

    const cv::Rect rect(cx - w / 2, cy - h / 2, w, h);
    const cv::Scalar color = trackStateColor(track);

    // --------------------------------
    // Suppression radius (only for confirmed tracks)
    // --------------------------------
    if (track.age >= confirm_age_) {
      const int radius =
        static_cast<int>(suppressionRadiusPx(track));

      cv::circle(
        image,
        cv::Point(cx, cy),
        radius,
        cv::Scalar(255, 100, 100), // blue
        1,
        cv::LINE_AA);
    }

    // Bounding box
    cv::rectangle(image, rect, color, 2);

    // Label
    std::ostringstream ss;
    ss << "id=" << track.detection_id
       << " age=" << track.age
       << " miss=" << track.misses;

    cv::putText(
      image,
      ss.str(),
      cv::Point(rect.x, std::max(0, rect.y - 5)),
      cv::FONT_HERSHEY_SIMPLEX,
      0.5,
      color,
      1);
  }


};


int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<BlockDetectionTrackingNode>());
  rclcpp::shutdown();
  return 0;
}
