#include <memory>
#include <mutex>

#include <rclcpp/rclcpp.hpp>

#include "concrete_block_perception/srv/track_detections.hpp"
#include "concrete_block_perception/detection/detection_config.hpp"
#include "concrete_block_perception/detection/block_detection_tracker.hpp"

using concrete_block_perception::srv::TrackDetections;
using concrete_block_perception::BlockDetectionTracker;
using concrete_block_perception::DetectionConfig;

class BlockDetectionTrackingServiceNode : public rclcpp::Node
{
public:
  BlockDetectionTrackingServiceNode()
  : Node("block_detection_tracking_service_node")
  {
    // =========================================
    // Load parameters via DetectionConfig
    // =========================================
    DetectionConfig config(*this);

    tracker_ = std::make_unique<BlockDetectionTracker>(
      config.params());

    // =========================================
    // Create callback group (Reentrant)
    // =========================================
    callback_group_ =
      create_callback_group(
      rclcpp::CallbackGroupType::Reentrant);

    // =========================================
    // Create service (Humble-compatible)
    // =========================================
    service_ = create_service<TrackDetections>(
      "~/track",
      std::bind(
        &BlockDetectionTrackingServiceNode::handleRequest,
        this,
        std::placeholders::_1,
        std::placeholders::_2),
      rmw_qos_profile_services_default,
      callback_group_);

    RCLCPP_INFO(
      get_logger(),
      "BlockDetectionTrackingServiceNode ready.");
  }

private:
  // ==========================================================
  // Service callback
  // ==========================================================
  void handleRequest(
    const std::shared_ptr<TrackDetections::Request> request,
    std::shared_ptr<TrackDetections::Response> response)
  {
    std::lock_guard<std::mutex> lock(tracker_mutex_);

    RCLCPP_INFO(
      get_logger(),
      "Received tracking request with %zu detections",
      request->detections.detections.size());

    try {
      sensor_msgs::msg::Image::SharedPtr mask_ptr;

      if (!request->mask.data.empty()) {
        mask_ptr =
          std::make_shared<sensor_msgs::msg::Image>(
          request->mask);
      } else {
        mask_ptr = nullptr;
      }

      response->tracked =
        tracker_->update(
        request->detections,
        mask_ptr);

      RCLCPP_INFO(
        get_logger(),
        "Tracking completed: %zu tracked detections",
        response->tracked.detections.size());

    } catch (const std::exception & e) {
      RCLCPP_ERROR(
        get_logger(),
        "Tracking failed: %s",
        e.what());

      response->tracked =
        concrete_block_perception::msg::TrackedDetectionArray();
    }
  }

  // ==========================================================
  // Members
  // ==========================================================
  std::unique_ptr<BlockDetectionTracker> tracker_;
  std::mutex tracker_mutex_;

  rclcpp::Service<TrackDetections>::SharedPtr service_;
  rclcpp::CallbackGroup::SharedPtr callback_group_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  auto node =
    std::make_shared<BlockDetectionTrackingServiceNode>();

  rclcpp::executors::MultiThreadedExecutor exec(
    rclcpp::ExecutorOptions(), 2);

  exec.add_node(node);
  exec.spin();

  rclcpp::shutdown();
  return 0;
}
