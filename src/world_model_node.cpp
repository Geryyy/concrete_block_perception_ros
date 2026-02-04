#include <deque>
#include <limits>
#include <sstream>
#include <iomanip>
#include <rclcpp/rclcpp.hpp>

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include "concrete_block_perception/msg/tracked_detection_array.hpp"
#include "concrete_block_perception/msg/block.hpp"
#include "concrete_block_perception/msg/block_array.hpp"
#include "concrete_block_perception/srv/register_block.hpp"

#define WM_LOG(logger, ...) RCLCPP_INFO(logger, __VA_ARGS__)

using concrete_block_perception::msg::TrackedDetectionArray;
using concrete_block_perception::msg::Block;
using concrete_block_perception::msg::BlockArray;
using concrete_block_perception::srv::RegisterBlock;

class WorldModelNode : public rclcpp::Node
{
public:
  WorldModelNode()
  : Node("block_world_model_node")
  {
    // ----------------------------
    // Parameters
    // ----------------------------
    min_points_ =
      declare_parameter<int>("min_points", 50);

    min_fitness_ =
      declare_parameter<double>("min_fitness", 0.3);

    max_rmse_ =
      declare_parameter<double>("max_rmse", 0.05);

    object_class_ =
      declare_parameter<std::string>("object_class", "concrete_block");

    max_dt_ =
      declare_parameter<double>("sync.max_dt", 0.5);

    max_cloud_buffer_ =
      declare_parameter<int>("sync.cloud_buffer_size", 10);

    service_name_ =
      declare_parameter<std::string>(
      "registration.service_name", "/register_block_pose");

    // ----------------------------
    // Subscribers
    // ----------------------------
    det_sub_ = create_subscription<TrackedDetectionArray>(
      "tracked_detections",
      rclcpp::SensorDataQoS(),
      std::bind(&WorldModelNode::detectionsCallback, this, std::placeholders::_1));

    cloud_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      "points",
      rclcpp::SensorDataQoS(),
      std::bind(&WorldModelNode::cloudCallback, this, std::placeholders::_1));

    // ----------------------------
    // Client + Publishers
    // ----------------------------
    reg_client_ = create_client<RegisterBlock>(service_name_);

    world_pub_ =
      create_publisher<BlockArray>("block_world_model", 10);

    marker_pub_ =
      create_publisher<visualization_msgs::msg::MarkerArray>(
      "block_world_model_markers", 10);

    WM_LOG(
      get_logger(),
      "WorldModelNode started (max_dt=%.2fs buffer=%zu)",
      max_dt_, max_cloud_buffer_);
  }

private:
  // ==========================================================
  // Manual time sync
  // ==========================================================
  sensor_msgs::msg::PointCloud2::ConstSharedPtr
  findClosestCloud(const rclcpp::Time & t)
  {
    sensor_msgs::msg::PointCloud2::ConstSharedPtr best;
    double best_dt = std::numeric_limits<double>::max();

    for (const auto & c : cloud_buffer_) {
      const rclcpp::Time ct(c->header.stamp);
      const double dt = std::abs((ct - t).seconds());
      if (dt < best_dt) {
        best_dt = dt;
        best = c;
      }
    }

    if (!best || best_dt > max_dt_) {
      return nullptr;
    }

    return best;
  }

  // ==========================================================
  // Callbacks
  // ==========================================================
  void detectionsCallback(
    const TrackedDetectionArray::ConstSharedPtr msg)
  {
    const rclcpp::Time t(msg->stamp);

    // WM_LOG(
    //   get_logger(),
    //   "[DETECTIONS] t=%.6f n=%zu",
    //   t.seconds(), msg->detections.size());

    auto cloud = findClosestCloud(t);
    if (cloud) {
      process(msg, cloud);
    } else {
      det_buffer_.push_back(msg);
    }
  }

  void cloudCallback(
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg)
  {
    const rclcpp::Time t(msg->header.stamp);

    // WM_LOG(
    //   get_logger(),
    //   "[CLOUD] t=%.6f",
    //   t.seconds());

    cloud_buffer_.push_back(msg);
    while (cloud_buffer_.size() > max_cloud_buffer_) {
      cloud_buffer_.pop_front();
    }

    for (auto it = det_buffer_.begin(); it != det_buffer_.end(); ) {
      auto cloud = findClosestCloud(rclcpp::Time((*it)->stamp));
      if (cloud) {
        process(*it, cloud);
        it = det_buffer_.erase(it);
      } else {
        ++it;
      }
    }
  }

  // ==========================================================
  // Processing
  // ==========================================================
  void process(
    const TrackedDetectionArray::ConstSharedPtr & detections,
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr & cloud)
  {
    const rclcpp::Time td(detections->stamp);
    const rclcpp::Time tc(cloud->header.stamp);

    std::string invalid_reason;
    if (!isCloudValid(*cloud, invalid_reason)) {
      WM_LOG(
        get_logger(),
        "PROCESS abort: invalid cloud (%s)",
        invalid_reason.c_str());
      return;
    }

    const size_t num_points = cloud->width * cloud->height;

    WM_LOG(
      get_logger(),
      "PROCESS start: Δt=%.3f sec, detections=%zu, cloud_points=%zu, fields=%zu",
      std::abs((tc - td).seconds()),
      detections->detections.size(),
      num_points,
      cloud->fields.size());


    BlockArray out;
    out.header = cloud->header;

    if (detections->detections.empty()) {
      WM_LOG(get_logger(), "No detections → publishing empty BlockArray");
    }

    size_t idx = 0;
    for (const auto & td_det : detections->detections) {
      WM_LOG(
        get_logger(),
        "Detection[%zu]: id=%u, mask size=%ux%u",
        idx,
        td_det.detection_id,
        td_det.mask.width,
        td_det.mask.height);

      // ------------------------------------------------------------
      // Service availability
      // ------------------------------------------------------------
      if (!reg_client_->wait_for_service(std::chrono::milliseconds(3000))) {
        WM_LOG(
          get_logger(),
          "Service register_block_pose not available (skipping all detections)");
        return; // NOTE: this aborts the *entire* callback
      }

      auto req = std::make_shared<RegisterBlock::Request>();
      req->mask = td_det.mask;
      req->cloud = *cloud;
      req->object_class = object_class_;

      WM_LOG(
        get_logger(),
        "Sending registration request (object_class=%s)",
        object_class_.c_str());

      auto future = reg_client_->async_send_request(req);

      // ------------------------------------------------------------
      // Service response timeout
      // ------------------------------------------------------------
      if (future.wait_for(std::chrono::milliseconds(3000)) !=
        std::future_status::ready)
      {
        WM_LOG(
          get_logger(),
          "Registration timeout (>200 ms) for detection %u",
          td_det.detection_id);
        ++idx;
        continue;
      }

      const auto res = future.get();

      WM_LOG(
        get_logger(),
        "Registration response: success=%d, fitness=%.4f, rmse=%.4f",
        res->success,
        res->fitness,
        res->rmse);

      // ------------------------------------------------------------
      // Quality gating
      // ------------------------------------------------------------
      if (!res->success) {
        WM_LOG(
          get_logger(),
          "Rejected detection %u: registration failed",
          td_det.detection_id);
        ++idx;
        continue;
      }

      if (res->fitness < min_fitness_) {
        WM_LOG(
          get_logger(),
          "Rejected detection %u: fitness %.3f < %.3f",
          td_det.detection_id,
          res->fitness,
          min_fitness_);
        ++idx;
        continue;
      }

      if (res->rmse > max_rmse_) {
        WM_LOG(
          get_logger(),
          "Rejected detection %u: rmse %.3f > %.3f",
          td_det.detection_id,
          res->rmse,
          max_rmse_);
        ++idx;
        continue;
      }

      // ------------------------------------------------------------
      // Accepted block
      // ------------------------------------------------------------
      Block b;
      b.id = "block_" + std::to_string(td_det.detection_id);
      b.pose = res->pose;
      b.confidence = res->fitness;
      b.last_seen = tc;

      out.blocks.push_back(b);

      WM_LOG(
        get_logger(),
        "Accepted block %s (fitness=%.3f, rmse=%.3f)",
        b.id.c_str(),
        b.confidence,
        res->rmse);

      ++idx;
    }

    WM_LOG(
      get_logger(),
      "Publishing BlockArray: %zu blocks",
      out.blocks.size());

    world_pub_->publish(out);
    publishMarkers(out, cloud->header.frame_id, tc);
  }


  // ==========================================================
  // Markers
  // ==========================================================
  void publishMarkers(
    const BlockArray & blocks,
    const std::string & frame,
    const rclcpp::Time & stamp)
  {
    visualization_msgs::msg::MarkerArray arr;
    int id = 0;

    for (const auto & b : blocks.blocks) {
      arr.markers.push_back(makeBlockMarker(b, frame, id, stamp));
      arr.markers.push_back(makeTextMarker(b, frame, id, stamp));
      ++id;
    }

    marker_pub_->publish(arr);
  }

  visualization_msgs::msg::Marker makeBlockMarker(
    const Block & b,
    const std::string & frame,
    int id,
    const rclcpp::Time & stamp) const
  {
    visualization_msgs::msg::Marker m;
    m.header.frame_id = frame;
    m.header.stamp = stamp;
    m.ns = "blocks";
    m.id = id;
    m.type = visualization_msgs::msg::Marker::CUBE;
    m.action = visualization_msgs::msg::Marker::ADD;
    m.pose = b.pose;
    m.scale.x = 0.4;
    m.scale.y = 0.2;
    m.scale.z = 0.15;
    m.color.r = 0.1f;
    m.color.g = 0.8f;
    m.color.b = 0.1f;
    m.color.a = 0.8f;
    return m;
  }

  visualization_msgs::msg::Marker makeTextMarker(
    const Block & b,
    const std::string & frame,
    int id,
    const rclcpp::Time & stamp) const
  {
    visualization_msgs::msg::Marker m;
    m.header.frame_id = frame;
    m.header.stamp = stamp;
    m.ns = "block_labels";
    m.id = id;
    m.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
    m.action = visualization_msgs::msg::Marker::ADD;
    m.pose = b.pose;
    m.pose.position.z += 0.2;
    m.scale.z = 0.08;
    m.color.r = 1.0f;
    m.color.g = 1.0f;
    m.color.b = 1.0f;
    m.color.a = 1.0f;

    std::ostringstream ss;
    ss << b.id << " fit=" << std::fixed << std::setprecision(2) << b.confidence;
    m.text = ss.str();
    return m;
  }

private:
  // ----------------------------
  // Helpers
  // ----------------------------
  bool isCloudValid(
    const sensor_msgs::msg::PointCloud2 & cloud,
    std::string & reason) const
  {
    if (cloud.data.empty()) {
      reason = "cloud.data is empty";
      return false;
    }

    if (cloud.width == 0 || cloud.height == 0) {
      reason = "width or height is zero";
      return false;
    }

    if (cloud.point_step == 0) {
      reason = "point_step is zero";
      return false;
    }

    if (cloud.row_step == 0) {
      reason = "row_step is zero";
      return false;
    }

    if (cloud.fields.empty()) {
      reason = "no PointFields";
      return false;
    }

    return true;
  }

  // ----------------------------
  // Buffers
  // ----------------------------
  std::deque<TrackedDetectionArray::ConstSharedPtr> det_buffer_;
  std::deque<sensor_msgs::msg::PointCloud2::ConstSharedPtr> cloud_buffer_;

  // ----------------------------
  // ROS
  // ----------------------------
  rclcpp::Subscription<TrackedDetectionArray>::SharedPtr det_sub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
  rclcpp::Client<RegisterBlock>::SharedPtr reg_client_;
  rclcpp::Publisher<BlockArray>::SharedPtr world_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;

  // ----------------------------
  // Params
  // ----------------------------
  int min_points_;
  double min_fitness_;
  double max_rmse_;
  double max_dt_;
  size_t max_cloud_buffer_;
  std::string service_name_;
  std::string object_class_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<WorldModelNode>());
  rclcpp::shutdown();
  return 0;
}
