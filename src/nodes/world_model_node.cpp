#include <deque>
#include <unordered_map>
#include <mutex>
#include <atomic>
#include <thread>
#include <sstream>
#include <iomanip>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>

#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include "ros2_yolos_cpp/srv/segment_image.hpp"
#include "concrete_block_perception/srv/track_detections.hpp"
#include "concrete_block_perception/action/register_block.hpp"
#include "concrete_block_perception/msg/block.hpp"
#include "concrete_block_perception/msg/block_array.hpp"
#include "concrete_block_perception/utils/img_utils.hpp"

#define WM_LOG(logger, ...) RCLCPP_INFO(logger, __VA_ARGS__)

using concrete_block_perception::msg::Block;
using concrete_block_perception::msg::BlockArray;

class WorldModelNode : public rclcpp::Node
{
  using SegmentSrv = ros2_yolos_cpp::srv::SegmentImage;
  using TrackSrv = concrete_block_perception::srv::TrackDetections;
  using RegisterBlock = concrete_block_perception::action::RegisterBlock;
  using GoalHandleRegisterBlock =
    rclcpp_action::ClientGoalHandle<RegisterBlock>;

  using SyncPolicy =
    message_filters::sync_policies::ApproximateTime<
    sensor_msgs::msg::Image,
    sensor_msgs::msg::PointCloud2>;

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
    // ================================
    // Parameters
    // ================================
    min_fitness_ = declare_parameter<double>("min_fitness", 0.3);
    max_rmse_ = declare_parameter<double>("max_rmse", 0.05);
    object_class_ =
      declare_parameter<std::string>("object_class", "concrete_block");

    // ================================
    // Debug publishers
    // ================================
    det_debug_pub_ =
      create_publisher<sensor_msgs::msg::Image>(
      "debug/detection_overlay", 1);

    track_debug_pub_ =
      create_publisher<sensor_msgs::msg::Image>(
      "debug/tracking_overlay", 1);

    // reg_cutout_pub_ =
    //   create_publisher<sensor_msgs::msg::PointCloud2>(
    //   "debug/registration_cutout", 1);

    // reg_template_pub_ =
    //   create_publisher<sensor_msgs::msg::PointCloud2>(
    //   "debug/registration_template", 1);

    world_pub_ =
      create_publisher<BlockArray>("block_world_model", 10);

    marker_pub_ =
      create_publisher<visualization_msgs::msg::MarkerArray>(
      "block_world_model_markers", 10);

    // ================================
    // Sync image + cloud
    // ================================
    image_sub_.subscribe(this, "image");
    cloud_sub_.subscribe(this, "points");

    sync_ =
      std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
      SyncPolicy(10), image_sub_, cloud_sub_);

    sync_->registerCallback(
      std::bind(
        &WorldModelNode::syncCallback,
        this,
        std::placeholders::_1,
        std::placeholders::_2));

    // ================================
    // Services
    // ================================
    segment_client_ =
      create_client<SegmentSrv>(
      "/yolos_segmentor_service/segment");

    track_client_ =
      create_client<TrackSrv>(
      "/block_detection_tracking_node/track");

    // ================================
    // Action client
    // ================================
    action_client_ =
      rclcpp_action::create_client<RegisterBlock>(
      this, "register_block");

    action_client_->wait_for_action_server();

    WM_LOG(get_logger(), "WorldModelNode ready");
  }

private:
  // ==========================================================
  // Sync callback (non-blocking)
  // ==========================================================
  void syncCallback(
    const sensor_msgs::msg::Image::ConstSharedPtr image,
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr cloud)
  {
    if (busy_) {
      // RCLCPP_INFO(get_logger(), "Dropping frame (busy)");
      return;
    }

    busy_ = true;

    std::thread(
      [this, image, cloud]() {
        processFrame(image, cloud);
        busy_ = false;
      }).detach();
  }

  // ==========================================================
  // Pipeline
  // ==========================================================
  void processFrame(
    const sensor_msgs::msg::Image::ConstSharedPtr & image,
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr & cloud)
  {
    try {

      // ==========================
      // 1️⃣ Segmentation
      // ==========================
      RCLCPP_INFO(get_logger(), "Processing frame %lu", frame_counter_ + 1);

      RCLCPP_INFO(get_logger(), "Requesting segmentation...");
      auto seg_req =
        std::make_shared<SegmentSrv::Request>();
      seg_req->image = *image;
      seg_req->return_debug = false;

      auto seg_res =
        segment_client_->async_send_request(seg_req).get();

      if (!seg_res->success) {return;}

      publishDetectionOverlay(
        image, seg_res->detections, seg_res->mask);

      // ==========================
      // 2️⃣ Tracking
      // ==========================
      RCLCPP_INFO(get_logger(), "Requesting tracking...");
      auto track_req =
        std::make_shared<TrackSrv::Request>();

      track_req->detections = seg_res->detections;
      track_req->mask = seg_res->mask;

      auto track_res =
        track_client_->async_send_request(track_req).get();

      publishTrackingOverlay(
        image, track_res->tracked);

      if (track_res->tracked.detections.empty()) {
        publishEmpty(cloud->header);
        return;
      }

      // ==========================
      // 3️⃣ Registration
      // ==========================
      RCLCPP_INFO(
        get_logger(), "Requesting registration for %zu detections...",
        track_res->tracked.detections.size());
      auto ctx = std::make_shared<FrameContext>();
      ctx->header = cloud->header;
      ctx->pending.store(
        track_res->tracked.detections.size());

      uint64_t frame_id = ++frame_counter_;

      {
        std::lock_guard<std::mutex> lock(frames_mutex_);
        frames_[frame_id] = ctx;
      }

      for (const auto & det :
        track_res->tracked.detections)
      {
        RCLCPP_INFO(
          get_logger(), "Requesting registration for detection %u (confidence: %.2f)...",
          det.detection_id, det.detection.results[0].hypothesis.score);

        RegisterBlock::Goal goal;
        goal.mask = det.mask;
        goal.cloud = *cloud;
        goal.object_class = object_class_;

        auto options =
          rclcpp_action::Client<RegisterBlock>::SendGoalOptions();

        options.result_callback =
          [this, frame_id, det](
          const auto & result)
          {
            handleResult(frame_id, det, result);
          };

        action_client_->async_send_goal(goal, options);
      }

    } catch (const std::exception & e) {
      RCLCPP_ERROR(
        get_logger(),
        "Pipeline error: %s", e.what());
    }
  }

  // ==========================================================
  // Registration Result
  // ==========================================================
  void handleResult(
    uint64_t frame_id,
    const concrete_block_perception::msg::TrackedDetection & det,
    const rclcpp_action::ClientGoalHandle<RegisterBlock>::WrappedResult & result)
  {
    RCLCPP_INFO(
      get_logger(),
      "Received registration result for frame %lu, detection %u",
      frame_id, det.detection_id);

    std::lock_guard<std::mutex> lock(frames_mutex_);

    auto it = frames_.find(frame_id);
    if (it == frames_.end()) {return;}

    auto & ctx = it->second;

    if (result.code ==
      rclcpp_action::ResultCode::SUCCEEDED)
    {
      const auto & res = result.result;

      if (res->success &&
        res->fitness >= min_fitness_ &&
        res->rmse <= max_rmse_)
      {
        Block b;
        b.id = "block_" +
          std::to_string(det.detection_id);
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
  // Debug overlays
  // ==========================================================
  void publishDetectionOverlay(
    const sensor_msgs::msg::Image::ConstSharedPtr & image,
    const vision_msgs::msg::Detection2DArray & detections,
    const sensor_msgs::msg::Image & mask_msg)
  {
    cv::Mat img = toCvBgr(*image);

    if (!mask_msg.data.empty()) {
      cv::Mat mask = toCvMono(mask_msg);
      overlayMask(img, mask, cv::Scalar(255, 255, 0), 0.35); // light blue
    } else {
      RCLCPP_WARN(get_logger(), "No mask provided for detection overlay");
    }

    drawDetectionBoxes(img, detections, cv::Scalar(0, 255, 0)); // green

    auto out =
      cv_bridge::CvImage(
      image->header,
      "bgr8",
      img).toImageMsg();

    det_debug_pub_->publish(*out);
  }

  void publishTrackingOverlay(
    const sensor_msgs::msg::Image::ConstSharedPtr & image,
    const concrete_block_perception::msg::TrackedDetectionArray & tracked)
  {
    cv::Mat img = toCvBgr(*image);

    for (const auto & t : tracked.detections) {

      if (!t.mask.data.empty()) {
        cv::Mat mask = toCvMono(t.mask);
        overlayMask(img, mask, cv::Scalar(255, 255, 0), 0.35); // light blue
      } else {
        RCLCPP_WARN(
          get_logger(), "No mask provided for tracking overlay (detection %u)", t.detection_id);
      }
    }

    drawTrackingBoxes(img, tracked, cv::Scalar(0, 0, 255)); // red

    auto out =
      cv_bridge::CvImage(
      image->header,
      "bgr8",
      img).toImageMsg();

    track_debug_pub_->publish(*out);
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
  }

  void publishEmpty(const std_msgs::msg::Header & header)
  {
    BlockArray out;
    out.header = header;
    world_pub_->publish(out);
  }

  // ==========================================================
  // Members
  // ==========================================================
  message_filters::Subscriber<sensor_msgs::msg::Image> image_sub_;
  message_filters::Subscriber<sensor_msgs::msg::PointCloud2> cloud_sub_;
  std::shared_ptr<
    message_filters::Synchronizer<SyncPolicy>> sync_;

  rclcpp::Client<SegmentSrv>::SharedPtr segment_client_;
  rclcpp::Client<TrackSrv>::SharedPtr track_client_;
  rclcpp_action::Client<RegisterBlock>::SharedPtr action_client_;

  rclcpp::Publisher<BlockArray>::SharedPtr world_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;

  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr det_debug_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr track_debug_pub_;
  // rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr reg_cutout_pub_;
  // rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr reg_template_pub_;

  std::unordered_map<uint64_t, std::shared_ptr<FrameContext>> frames_;
  std::mutex frames_mutex_;
  std::atomic<uint64_t> frame_counter_{0};
  std::atomic<bool> busy_{false};

  double min_fitness_;
  double max_rmse_;
  std::string object_class_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  auto node =
    std::make_shared<WorldModelNode>();

  rclcpp::executors::MultiThreadedExecutor exec(
    rclcpp::ExecutorOptions(), 4);

  exec.add_node(node);
  exec.spin();

  rclcpp::shutdown();
  return 0;
}
