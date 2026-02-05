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
#include <rclcpp_action/rclcpp_action.hpp>
#include "concrete_block_perception/action/register_block.hpp"

#define WM_LOG(logger, ...) RCLCPP_INFO(logger, __VA_ARGS__)

using concrete_block_perception::msg::TrackedDetectionArray;
using concrete_block_perception::msg::Block;
using concrete_block_perception::msg::BlockArray;

class WorldModelNode : public rclcpp::Node
{
  using RegisterBlockAction =
    concrete_block_perception::action::RegisterBlock;

  using GoalHandleRegisterBlock =
    rclcpp_action::ClientGoalHandle<RegisterBlockAction>;

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
    action_client_ =
      rclcpp_action::create_client<RegisterBlockAction>(
      this,
      "register_block");

    WM_LOG(get_logger(), "Waiting for register_block action server...");

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

      if (!action_client_->wait_for_action_server(
          std::chrono::seconds(3)))
      {
        WM_LOG(
          get_logger(),
          "Action server register_block not available");
        return;
      }

      RegisterBlockAction::Goal goal;
      goal.mask = td_det.mask;
      goal.cloud = *cloud;
      goal.object_class = object_class_;

      WM_LOG(
        get_logger(),
        "Sending RegisterBlock goal (id=%u)",
        td_det.detection_id);

      auto send_goal_options =
        rclcpp_action::Client<RegisterBlockAction>::SendGoalOptions();

      auto goal_handle_future =
        action_client_->async_send_goal(goal, send_goal_options);

      if (goal_handle_future.wait_for(std::chrono::seconds(1)) !=
        std::future_status::ready)
      {
        WM_LOG(get_logger(), "Goal rejected or timed out");
        continue;
      }

      auto goal_handle = goal_handle_future.get();
      if (!goal_handle) {
        WM_LOG(get_logger(), "Goal was rejected");
        continue;
      }

      auto result_future =
        action_client_->async_get_result(goal_handle);

      if (result_future.wait_for(std::chrono::seconds(5)) !=
        std::future_status::ready)
      {
        WM_LOG(
          get_logger(),
          "Action result timeout for detection %u",
          td_det.detection_id);
        continue;
      }

      const auto wrapped_result = result_future.get();
      const auto & res = wrapped_result.result;

      WM_LOG(
        get_logger(),
        "Action result: success=%d fitness=%.3f rmse=%.3f",
        res->success,
        res->fitness,
        res->rmse);

      if (!res->success ||
        res->fitness < min_fitness_ ||
        res->rmse > max_rmse_)
      {
        continue;
      }

      Block b;
      b.id = "block_" + std::to_string(td_det.detection_id);
      b.pose = res->pose;
      b.confidence = res->fitness;
      b.last_seen = tc;

      out.blocks.push_back(b);
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
  rclcpp_action::Client<RegisterBlockAction>::SharedPtr action_client_;

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
