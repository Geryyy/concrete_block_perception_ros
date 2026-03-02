#include <deque>
#include <unordered_map>
#include <mutex>
#include <atomic>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <cstdint>

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
#include "concrete_block_perception/msg/tracked_detection_array.hpp"
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

  enum class PipelineMode
  {
    kSegment,
    kTrack,
    kRegister,
    kFull
  };

  struct FrameContext
  {
    std_msgs::msg::Header header;
    std::vector<Block> blocks;
    std::mutex blocks_mutex;
    std::atomic<size_t> pending{0};
    std::atomic<bool> finished{false};

    // ---- timing ----
    std::chrono::steady_clock::time_point t_start;
    std::chrono::steady_clock::time_point t_after_seg;
    std::chrono::steady_clock::time_point t_after_track;
    std::chrono::steady_clock::time_point t_after_reg;
  };

  public:
  WorldModelNode()
  : Node("block_world_model_node")
  {
    // ================================
    // Parameters: gating + commissioning
    // ================================
    const std::string mode_str =
      declare_parameter<std::string>("pipeline_mode", "full");
    pipeline_mode_ = parseMode(mode_str);

    min_fitness_ = declare_parameter<double>("min_fitness", 0.3);
    max_rmse_ = declare_parameter<double>("max_rmse", 0.05);
    object_class_ =
      declare_parameter<std::string>("object_class", "concrete_block");
    max_sync_delta_s_ =
      declare_parameter<double>("sync.max_delta_s", 0.06);
    min_reregister_s_ =
      declare_parameter<double>("registration.min_reregister_s", 2.0);
    register_every_frame_ =
      declare_parameter<bool>("registration.register_every_frame", false);
    object_timeout_s_ =
      declare_parameter<double>("world_model.object_timeout_s", 10.0);
    debug_detection_overlay_enabled_ =
      declare_parameter<bool>("debug.publish_detection_overlay", true);
    debug_tracking_overlay_enabled_ =
      declare_parameter<bool>("debug.publish_tracking_overlay", true);
    perf_log_timing_enabled_ =
      declare_parameter<bool>("perf.log_timing", true);
    perf_log_every_n_frames_ =
      declare_parameter<int>("perf.log_every_n_frames", 20);
    if (perf_log_every_n_frames_ < 1) {
      perf_log_every_n_frames_ = 1;
    }

    // ================================
    // Debug publishers
    // ================================
    if (debug_detection_overlay_enabled_) {
      det_debug_pub_ =
        create_publisher<sensor_msgs::msg::Image>(
        "debug/detection_overlay", 1);
    }

    if (debug_tracking_overlay_enabled_) {
      track_debug_pub_ =
        create_publisher<sensor_msgs::msg::Image>(
        "debug/tracking_overlay", 1);
    }

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
    tracked_pub_ =
      create_publisher<concrete_block_perception::msg::TrackedDetectionArray>(
      "tracked_detections", 10);

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

    if (!segment_client_->wait_for_service(std::chrono::seconds(2))) {
      RCLCPP_WARN(
        get_logger(),
        "Segmentation service not available at startup.");
    }

    if (needsTracking() &&
      !track_client_->wait_for_service(std::chrono::seconds(2)))
    {
      RCLCPP_WARN(
        get_logger(),
        "Tracking service not available at startup.");
    }

    if (needsRegistration() &&
      !action_client_->wait_for_action_server(std::chrono::seconds(2)))
    {
      RCLCPP_WARN(
        get_logger(),
        "Registration action not available at startup.");
    }

    WM_LOG(
      get_logger(),
      "WorldModelNode ready | mode=%s",
      mode_str.c_str());
  }

private:
  PipelineMode parseMode(const std::string & mode_str)
  {
    if (mode_str == "segment") {
      return PipelineMode::kSegment;
    }
    if (mode_str == "track") {
      return PipelineMode::kTrack;
    }
    if (mode_str == "register") {
      return PipelineMode::kRegister;
    }
    return PipelineMode::kFull;
  }

  bool needsTracking() const
  {
    return pipeline_mode_ == PipelineMode::kTrack ||
           pipeline_mode_ == PipelineMode::kRegister ||
           pipeline_mode_ == PipelineMode::kFull;
  }

  bool needsRegistration() const
  {
    return pipeline_mode_ == PipelineMode::kRegister ||
           pipeline_mode_ == PipelineMode::kFull;
  }

  bool shouldRegister(
    const concrete_block_perception::msg::TrackedDetection & det,
    const rclcpp::Time & stamp)
  {
    if (register_every_frame_) {
      last_registration_stamp_[det.detection_id] = stamp;
      return true;
    }

    if (det.age <= 1) {
      last_registration_stamp_[det.detection_id] = stamp;
      return true;
    }

    auto it = last_registration_stamp_.find(det.detection_id);
    if (it == last_registration_stamp_.end()) {
      last_registration_stamp_[det.detection_id] = stamp;
      return true;
    }

    if ((stamp - it->second).seconds() >= min_reregister_s_) {
      it->second = stamp;
      return true;
    }

    return false;
  }

  void resetBusy()
  {
    busy_.store(false);
  }

  void cleanupFrame(uint64_t frame_id)
  {
    std::lock_guard<std::mutex> lock(frames_mutex_);
    frames_.erase(frame_id);
  }

  void publishWorldMarkers(
    const std_msgs::msg::Header & header,
    const std::vector<Block> & blocks)
  {
    visualization_msgs::msg::MarkerArray ma;
    int marker_id = 0;
    for (const auto & b : blocks) {
      visualization_msgs::msg::Marker m;
      m.header = header;
      m.ns = "cbp_blocks";
      m.id = marker_id++;
      m.type = visualization_msgs::msg::Marker::CUBE;
      m.action = visualization_msgs::msg::Marker::ADD;
      m.pose = b.pose;
      m.scale.x = 0.4;
      m.scale.y = 0.2;
      m.scale.z = 0.2;
      m.color.r = 0.1f;
      m.color.g = 0.8f;
      m.color.b = 0.2f;
      m.color.a = 0.6f;
      ma.markers.push_back(std::move(m));
    }
    marker_pub_->publish(ma);
  }

  void publishPersistentWorld(const std_msgs::msg::Header & header)
  {
    BlockArray out;
    out.header = header;

    const rclcpp::Time now_stamp(header.stamp);

    {
      std::lock_guard<std::mutex> lock(persistent_world_mutex_);
      for (auto it = persistent_world_.begin(); it != persistent_world_.end();) {
        const rclcpp::Time seen(it->second.last_seen);
        if ((now_stamp - seen).seconds() > object_timeout_s_) {
          it = persistent_world_.erase(it);
          continue;
        }
        out.blocks.push_back(it->second);
        ++it;
      }
    }

    world_pub_->publish(out);
    publishWorldMarkers(header, out.blocks);
  }

  void maybeFinalizeFrame(uint64_t frame_id)
  {
    std::shared_ptr<FrameContext> ctx;
    {
      std::lock_guard<std::mutex> lock(frames_mutex_);
      auto it = frames_.find(frame_id);
      if (it == frames_.end()) {
        return;
      }
      ctx = it->second;
    }

    if (ctx->pending.load() != 0) {
      return;
    }

    bool expected = false;
    if (!ctx->finished.compare_exchange_strong(expected, true)) {
      return;
    }

    ctx->t_after_reg = std::chrono::steady_clock::now();

    auto seg_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(
      ctx->t_after_seg - ctx->t_start).count();

    auto track_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(
      ctx->t_after_track - ctx->t_after_seg).count();

    auto reg_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(
      ctx->t_after_reg - ctx->t_after_track).count();

    auto total_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(
      ctx->t_after_reg - ctx->t_start).count();

    recordTiming(seg_ms, track_ms, reg_ms, total_ms);

    if (pipeline_mode_ == PipelineMode::kFull) {
      publishPersistentWorld(ctx->header);
    } else {
      publishFrame(*ctx);
    }

    cleanupFrame(frame_id);
    resetBusy();
  }

  // ==========================================================
  // Sync callback (non-blocking)
  // ==========================================================
  void syncCallback(
    const sensor_msgs::msg::Image::ConstSharedPtr image,
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr cloud)
  {
    bool expected = false;
    if (!busy_.compare_exchange_strong(expected, true)) {
      dropped_busy_frames_.fetch_add(1);
      return;
    }

    const rclcpp::Time t_img(image->header.stamp);
    const rclcpp::Time t_cloud(cloud->header.stamp);
    const double dt = std::abs((t_img - t_cloud).seconds());

    if (dt > max_sync_delta_s_) {
      dropped_sync_frames_.fetch_add(1);
      RCLCPP_WARN(
        get_logger(),
        "Dropped frame pair due to sync delta %.4f s (> %.4f s)",
        dt,
        max_sync_delta_s_);
      resetBusy();
      return;
    }

    processFrame(image, cloud);
  }

  // ==========================================================
  // Pipeline (asynchronous chain)
  // ==========================================================
  void processFrame(
    const sensor_msgs::msg::Image::ConstSharedPtr & image,
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr & cloud)
  {
    const auto t_start = std::chrono::steady_clock::now();

    if (!segment_client_->service_is_ready()) {
      RCLCPP_WARN(get_logger(), "Segmentation service unavailable.");
      resetBusy();
      return;
    }

    auto seg_req =
      std::make_shared<SegmentSrv::Request>();
    seg_req->image = *image;
    seg_req->return_debug = false;

    segment_client_->async_send_request(
      seg_req,
      [this, image, cloud, t_start](
        rclcpp::Client<SegmentSrv>::SharedFuture seg_future)
      {
        try {
          auto seg_res = seg_future.get();
          auto t_after_seg = std::chrono::steady_clock::now();

          if (!seg_res || !seg_res->success) {
            publishEmpty(cloud->header);
            resetBusy();
            return;
          }

          if (debug_detection_overlay_enabled_) {
            publishDetectionOverlay(
              image, seg_res->detections, seg_res->mask);
          }

          if (pipeline_mode_ == PipelineMode::kSegment) {
            auto total_ms =
              std::chrono::duration_cast<std::chrono::milliseconds>(
              t_after_seg - t_start).count();
            recordTiming(total_ms, 0, 0, total_ms);
            publishEmpty(cloud->header);
            resetBusy();
            return;
          }

          if (!track_client_->service_is_ready()) {
            RCLCPP_WARN(get_logger(), "Tracking service unavailable.");
            resetBusy();
            return;
          }

          auto track_req =
            std::make_shared<TrackSrv::Request>();
          track_req->detections = seg_res->detections;
          track_req->mask = seg_res->mask;

          track_client_->async_send_request(
            track_req,
            [this, image, cloud, t_start, t_after_seg](
              rclcpp::Client<TrackSrv>::SharedFuture track_future)
            {
              try {
                auto track_res = track_future.get();
                auto t_after_track = std::chrono::steady_clock::now();

                if (!track_res) {
                  publishEmpty(cloud->header);
                  resetBusy();
                  return;
                }

                if (debug_tracking_overlay_enabled_) {
                  publishTrackingOverlay(
                    image, track_res->tracked);
                }
                tracked_pub_->publish(track_res->tracked);

                if (track_res->tracked.detections.empty()) {
                  auto seg_ms =
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                    t_after_seg - t_start).count();
                  auto track_ms =
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                    t_after_track - t_after_seg).count();
                  auto total_ms =
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                    t_after_track - t_start).count();
                  recordTiming(seg_ms, track_ms, 0, total_ms);
                  publishEmpty(cloud->header);
                  resetBusy();
                  return;
                }

                if (pipeline_mode_ == PipelineMode::kTrack) {
                  auto seg_ms =
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                    t_after_seg - t_start).count();
                  auto track_ms =
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                    t_after_track - t_after_seg).count();
                  auto total_ms =
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                    t_after_track - t_start).count();
                  recordTiming(seg_ms, track_ms, 0, total_ms);
                  publishEmpty(cloud->header);
                  resetBusy();
                  return;
                }

                if (!action_client_->action_server_is_ready()) {
                  RCLCPP_WARN(get_logger(), "Registration action unavailable.");
                  resetBusy();
                  return;
                }

                auto ctx = std::make_shared<FrameContext>();
                ctx->t_start = t_start;
                ctx->t_after_seg = t_after_seg;
                ctx->t_after_track = t_after_track;
                ctx->header = cloud->header;
                ctx->pending.store(0);
                ctx->finished.store(false);

                const uint64_t frame_id = ++frame_counter_;
                {
                  std::lock_guard<std::mutex> lock(frames_mutex_);
                  frames_[frame_id] = ctx;
                }

                size_t registered = 0;
                const rclcpp::Time frame_stamp(cloud->header.stamp);

                for (const auto & det : track_res->tracked.detections) {
                  if (!shouldRegister(det, frame_stamp)) {
                    continue;
                  }

                  if (det.mask.data.empty()) {
                    continue;
                  }

                  RegisterBlock::Goal goal;
                  goal.mask = det.mask;
                  goal.cloud = *cloud;
                  goal.object_class = object_class_;

                  auto options =
                    rclcpp_action::Client<RegisterBlock>::SendGoalOptions();

                  ctx->pending.fetch_add(1);
                  ++registered;

                  options.goal_response_callback =
                    [this, frame_id](
                    GoalHandleRegisterBlock::SharedPtr goal_handle)
                    {
                      if (!goal_handle) {
                        std::lock_guard<std::mutex> lock(frames_mutex_);
                        auto it = frames_.find(frame_id);
                        if (it != frames_.end()) {
                          it->second->pending.fetch_sub(1);
                        }
                        maybeFinalizeFrame(frame_id);
                      }
                    };

                  options.result_callback =
                    [this, frame_id, det](
                    const auto & result)
                    {
                      handleResult(frame_id, det, result);
                    };

                  action_client_->async_send_goal(goal, options);
                }

                if (registered == 0) {
                  auto seg_ms =
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                    t_after_seg - t_start).count();
                  auto track_ms =
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                    t_after_track - t_after_seg).count();
                  auto total_ms =
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                    t_after_track - t_start).count();
                  recordTiming(seg_ms, track_ms, 0, total_ms);
                  if (pipeline_mode_ == PipelineMode::kFull) {
                    publishPersistentWorld(cloud->header);
                  } else {
                    publishEmpty(cloud->header);
                  }
                  cleanupFrame(frame_id);
                  resetBusy();
                  return;
                }

              } catch (const std::exception & e) {
                RCLCPP_ERROR(get_logger(), "Tracking stage failed: %s", e.what());
                resetBusy();
              }
            });

        } catch (const std::exception & e) {
          RCLCPP_ERROR(get_logger(), "Segmentation stage failed: %s", e.what());
          resetBusy();
        }
      });
  }

  // ==========================================================
  // Registration Result
  // ==========================================================
  void handleResult(
    uint64_t frame_id,
    const concrete_block_perception::msg::TrackedDetection & det,
    const rclcpp_action::ClientGoalHandle<RegisterBlock>::WrappedResult & result)
  {
    std::shared_ptr<FrameContext> ctx;
    {
      std::lock_guard<std::mutex> lock(frames_mutex_);
      auto it = frames_.find(frame_id);
      if (it == frames_.end()) {
        return;
      }
      ctx = it->second;
    }

    if (result.code ==
      rclcpp_action::ResultCode::SUCCEEDED)
    {
      const auto & res = result.result;

      if (res &&
        res->success &&
        res->fitness >= min_fitness_ &&
        res->rmse <= max_rmse_)
      {
        Block b;
        b.id = "block_" +
          std::to_string(det.detection_id);
        b.pose = res->pose;
        b.confidence = res->fitness;
        b.last_seen = ctx->header.stamp;
        b.pose_status = Block::POSE_PRECISE;
        b.task_status = Block::TASK_FREE;
        {
          std::lock_guard<std::mutex> lock(ctx->blocks_mutex);
          ctx->blocks.push_back(b);
        }

        if (pipeline_mode_ == PipelineMode::kFull) {
          std::lock_guard<std::mutex> lock(persistent_world_mutex_);
          persistent_world_[b.id] = b;
        }
      }
    }

    ctx->pending.fetch_sub(1);
    maybeFinalizeFrame(frame_id);
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

    drawCircle(img, tracked, cv::Scalar(255, 0, 255), 2, cv::LINE_AA); // magenta circle

    auto out =
      cv_bridge::CvImage(
      image->header,
      "bgr8",
      img).toImageMsg();

    track_debug_pub_->publish(*out);
  }

  void recordTiming(
    int64_t seg_ms,
    int64_t track_ms,
    int64_t reg_ms,
    int64_t total_ms)
  {
    if (!perf_log_timing_enabled_) {
      return;
    }

    std::lock_guard<std::mutex> lock(perf_mutex_);
    perf_timing_count_++;
    perf_seg_sum_ms_ += seg_ms;
    perf_track_sum_ms_ += track_ms;
    perf_reg_sum_ms_ += reg_ms;
    perf_total_sum_ms_ += total_ms;

    const uint64_t n = perf_timing_count_;
    if ((n % static_cast<uint64_t>(perf_log_every_n_frames_)) != 0U) {
      return;
    }

    const double avg_seg =
      static_cast<double>(perf_seg_sum_ms_) / static_cast<double>(n);
    const double avg_track =
      static_cast<double>(perf_track_sum_ms_) / static_cast<double>(n);
    const double avg_reg =
      static_cast<double>(perf_reg_sum_ms_) / static_cast<double>(n);
    const double avg_total =
      static_cast<double>(perf_total_sum_ms_) / static_cast<double>(n);

    RCLCPP_INFO(
      get_logger(),
      "Timing avg over %llu frames | seg %.1f ms | track %.1f ms | reg %.1f ms | total %.1f ms | dropped busy %llu | dropped sync %llu",
      static_cast<unsigned long long>(n),
      avg_seg,
      avg_track,
      avg_reg,
      avg_total,
      static_cast<unsigned long long>(dropped_busy_frames_.load()),
      static_cast<unsigned long long>(dropped_sync_frames_.load()));
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
    publishWorldMarkers(out.header, out.blocks);
  }

  void publishEmpty(const std_msgs::msg::Header & header)
  {
    BlockArray out;
    out.header = header;
    world_pub_->publish(out);
    publishWorldMarkers(out.header, out.blocks);
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
  rclcpp::Publisher<concrete_block_perception::msg::TrackedDetectionArray>::SharedPtr tracked_pub_;

  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr det_debug_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr track_debug_pub_;

  std::unordered_map<uint64_t, std::shared_ptr<FrameContext>> frames_;
  std::mutex frames_mutex_;
  std::atomic<uint64_t> frame_counter_{0};
  std::atomic<bool> busy_{false};
  std::atomic<uint64_t> dropped_busy_frames_{0};
  std::atomic<uint64_t> dropped_sync_frames_{0};

  std::unordered_map<uint32_t, rclcpp::Time> last_registration_stamp_;
  std::unordered_map<std::string, Block> persistent_world_;
  std::mutex persistent_world_mutex_;

  PipelineMode pipeline_mode_{PipelineMode::kFull};
  double min_fitness_;
  double max_rmse_;
  std::string object_class_;
  double max_sync_delta_s_{0.06};
  double min_reregister_s_{2.0};
  bool register_every_frame_{false};
  double object_timeout_s_{10.0};
  bool debug_detection_overlay_enabled_{true};
  bool debug_tracking_overlay_enabled_{true};
  bool perf_log_timing_enabled_{true};
  int perf_log_every_n_frames_{20};

  std::mutex perf_mutex_;
  uint64_t perf_timing_count_{0};
  int64_t perf_seg_sum_ms_{0};
  int64_t perf_track_sum_ms_{0};
  int64_t perf_reg_sum_ms_{0};
  int64_t perf_total_sum_ms_{0};
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
