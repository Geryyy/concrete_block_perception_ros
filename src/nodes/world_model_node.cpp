#include <deque>
#include <unordered_map>
#include <mutex>
#include <atomic>
#include <sstream>
#include <iomanip>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include "concrete_block_perception/msg/tracked_detection_array.hpp"
#include "concrete_block_perception/msg/block.hpp"
#include "concrete_block_perception/msg/block_array.hpp"
#include "concrete_block_perception/action/register_block.hpp"

#define WM_LOG(logger, ...) RCLCPP_INFO(logger, __VA_ARGS__)

using concrete_block_perception::msg::TrackedDetectionArray;
using concrete_block_perception::msg::Block;
using concrete_block_perception::msg::BlockArray;

class WorldModelNode : public rclcpp::Node
{
  using RegisterBlock = concrete_block_perception::action::RegisterBlock;
  using GoalHandleRegisterBlock = rclcpp_action::ClientGoalHandle<RegisterBlock>;

  struct FrameContext
  {
    std_msgs::msg::Header header;
    std::vector<Block> blocks;
    std::atomic<size_t> pending{0};
  };

public:
  WorldModelNode()
  : Node("block_world_model_node")
  {
    // ----------------------------
    // Parameters
    // ----------------------------
    min_fitness_ = declare_parameter<double>("min_fitness", 0.3);
    max_rmse_ = declare_parameter<double>("max_rmse", 0.05);
    max_dt_ = declare_parameter<double>("sync.max_dt", 0.5);
    max_cloud_buffer_ =
      declare_parameter<int>("sync.cloud_buffer_size", 10);
    object_class_ =
      declare_parameter<std::string>("object_class", "concrete_block");
    action_name_ =
      declare_parameter<std::string>("registration.action_name", "register_block");

    // ----------------------------
    // ROS interfaces
    // ----------------------------
    det_sub_ = create_subscription<TrackedDetectionArray>(
      "tracked_detections",
      rclcpp::SensorDataQoS(),
      std::bind(&WorldModelNode::detectionsCallback, this, std::placeholders::_1));

    cloud_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      "points",
      rclcpp::SensorDataQoS(),
      std::bind(&WorldModelNode::cloudCallback, this, std::placeholders::_1));

    world_pub_ =
      create_publisher<BlockArray>("block_world_model", 10);

    marker_pub_ =
      create_publisher<visualization_msgs::msg::MarkerArray>(
      "block_world_model_markers", 10);

    action_client_ =
      rclcpp_action::create_client<RegisterBlock>(this, action_name_);

    WM_LOG(get_logger(), "Waiting for action server '%s'...", action_name_.c_str());
    if (!action_client_->wait_for_action_server(std::chrono::seconds(5))) {
      throw std::runtime_error("RegisterBlock action server not available");
    }

    WM_LOG(get_logger(), "WorldModelNode ready");
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
    auto cloud = findClosestCloud(rclcpp::Time(msg->stamp));
    if (!cloud) {
      det_buffer_.push_back(msg);
      return;
    }
    RCLCPP_INFO(get_logger(), "detection buffer size: %ld", det_buffer_.size());

    processFrame(msg, cloud);
  }

  void cloudCallback(
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg)
  {
    cloud_buffer_.push_back(msg);
    while (cloud_buffer_.size() > max_cloud_buffer_) {
      cloud_buffer_.pop_front();
    }

    RCLCPP_INFO(get_logger(), "cloud buffer size: %ld", cloud_buffer_.size());

    for (auto it = det_buffer_.begin(); it != det_buffer_.end(); ) {
      auto cloud = findClosestCloud(rclcpp::Time((*it)->stamp));
      if (cloud) {
        processFrame(*it, cloud);
        it = det_buffer_.erase(it);
      } else {
        ++it;
      }
    }
  }

  // ==========================================================
  // Frame processing (non-blocking)
  // ==========================================================
  void processFrame(
    const TrackedDetectionArray::ConstSharedPtr & detections,
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr & cloud)
  {
    if (detections->detections.empty()) {
      publishEmpty(cloud->header);
      return;
    }

    auto ctx = std::make_shared<FrameContext>();
    ctx->header = cloud->header;
    ctx->pending.store(detections->detections.size());

    const uint64_t frame_id = ++frame_counter_;

    WM_LOG(
      get_logger(),
      "[FRAME %lu] detections=%zu pending=%zu",
      frame_id,
      detections->detections.size(),
      ctx->pending.load());


    {
      std::lock_guard<std::mutex> lock(frames_mutex_);
      frames_[frame_id] = ctx;
    }

    for (const auto & det : detections->detections) {
      RegisterBlock::Goal goal;
      goal.mask = det.mask;
      goal.cloud = *cloud;
      goal.object_class = object_class_;

      auto options =
        rclcpp_action::Client<RegisterBlock>::SendGoalOptions();

      options.result_callback =
        [this, frame_id, det](const auto & result) {
          handleResult(frame_id, det, result);
        };

      action_client_->async_send_goal(goal, options);
    }
  }

  // ==========================================================
  // Action result handling
  // ==========================================================
  void handleResult(
    uint64_t frame_id,
    const TrackedDetectionArray::_detections_type::value_type & det,
    const rclcpp_action::ClientGoalHandle<RegisterBlock>::WrappedResult & result)
  {
    std::lock_guard<std::mutex> lock(frames_mutex_);
    auto it = frames_.find(frame_id);
    if (it == frames_.end()) {return;}

    auto & ctx = it->second;

    RCLCPP_DEBUG(
      get_logger(),
      "[FRAME %lu] det=%u success=%d fit=%.3f rmse=%.3f pending=%zu",
      frame_id,
      det.detection_id,
      result.code == rclcpp_action::ResultCode::SUCCEEDED,
      result.result ? result.result->fitness : -1.0,
      result.result ? result.result->rmse : -1.0,
      ctx->pending.load());

    if (result.code == rclcpp_action::ResultCode::SUCCEEDED) {
      const auto & res = result.result;
      if (res->success &&
        res->fitness >= min_fitness_ &&
        res->rmse <= max_rmse_)
      {

        Block b;
        b.id = "block_" + std::to_string(det.detection_id);
        b.pose = res->pose;
        b.confidence = res->fitness;
        b.last_seen = ctx->header.stamp;
        ctx->blocks.push_back(b);
      }
    }

    if (ctx->pending.fetch_sub(1) == 1) {
      publishFrame(*ctx);
      frames_.erase(it);
    }
  }

  // ==========================================================
  // Publishing
  // ==========================================================
  void publishFrame(const FrameContext & ctx)
  {
    BlockArray out;
    out.header = ctx.header;
    out.blocks = ctx.blocks;
    world_pub_->publish(out);
    publishMarkers(out, ctx.header.frame_id, ctx.header.stamp);
  }

  void publishEmpty(const std_msgs::msg::Header & header)
  {
    BlockArray out;
    out.header = header;
    world_pub_->publish(out);
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
      arr.markers.push_back(makeBlockMarker(b, frame, id++, stamp));
      arr.markers.push_back(makeTextMarker(b, frame, id++, stamp));
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

  // ==========================================================
  // Internal state
  // ==========================================================

  std::deque<TrackedDetectionArray::ConstSharedPtr> det_buffer_;
  std::deque<sensor_msgs::msg::PointCloud2::ConstSharedPtr> cloud_buffer_;

  std::unordered_map<uint64_t, std::shared_ptr<FrameContext>> frames_;
  std::mutex frames_mutex_;
  std::atomic<uint64_t> frame_counter_{0};

  // ROS
  rclcpp::Subscription<TrackedDetectionArray>::SharedPtr det_sub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
  rclcpp_action::Client<RegisterBlock>::SharedPtr action_client_;
  rclcpp::Publisher<BlockArray>::SharedPtr world_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;

  // Params
  double min_fitness_;
  double max_rmse_;
  double max_dt_;
  size_t max_cloud_buffer_;
  std::string object_class_;
  std::string action_name_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  auto node = std::make_shared<WorldModelNode>();
  rclcpp::executors::MultiThreadedExecutor exec(
    rclcpp::ExecutorOptions(), 4);
  exec.add_node(node);
  exec.spin();

  rclcpp::shutdown();
  return 0;
}
