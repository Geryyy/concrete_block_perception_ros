#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <future>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include "concrete_block_perception/action/register_block.hpp"
#include "concrete_block_perception/msg/block.hpp"
#include "concrete_block_perception/msg/block_array.hpp"
#include "concrete_block_perception/srv/get_coarse_blocks.hpp"
#include "concrete_block_perception/srv/run_pose_estimation.hpp"
#include "concrete_block_perception/srv/set_perception_mode.hpp"
#include "concrete_block_perception/utils/img_utils.hpp"
#include "concrete_block_perception/utils/world_model_utils.hpp"
#include "ros2_yolos_cpp/srv/segment_image.hpp"

#define WM_LOG(logger, ...) RCLCPP_INFO(logger, __VA_ARGS__)

using concrete_block_perception::msg::Block;
using concrete_block_perception::msg::BlockArray;

namespace cbpwm = cbp::world_model;

class WorldModelNode : public rclcpp::Node
{
  using SegmentSrv = ros2_yolos_cpp::srv::SegmentImage;
  using SetModeSrv = concrete_block_perception::srv::SetPerceptionMode;
  using GetCoarseSrv = concrete_block_perception::srv::GetCoarseBlocks;
  using RunPoseSrv = concrete_block_perception::srv::RunPoseEstimation;
  using RegisterBlock = concrete_block_perception::action::RegisterBlock;
  using GoalHandleRegisterBlock = rclcpp_action::ClientGoalHandle<RegisterBlock>;

  using SyncPolicy = message_filters::sync_policies::ApproximateTime<
    sensor_msgs::msg::Image,
    sensor_msgs::msg::PointCloud2>;

  struct OneShotRequest
  {
    uint64_t sequence{0};
    cbpwm::OneShotMode mode{cbpwm::OneShotMode::kNone};
    std::string target_block_id;
    bool enable_debug{true};
    double registration_timeout_s{3.0};
  };

public:
  WorldModelNode()
  : Node("block_world_model_node")
  {
    run_pose_cb_group_ = create_callback_group(rclcpp::CallbackGroupType::Reentrant);
    action_client_cb_group_ = create_callback_group(rclcpp::CallbackGroupType::Reentrant);

    const std::string pipeline_mode_str = declare_parameter<std::string>("pipeline_mode", "full");
    pipeline_mode_ = cbpwm::parsePipelineMode(pipeline_mode_str);

    const std::string perception_mode_str = declare_parameter<std::string>("perception_mode", "FULL");
    min_fitness_ = declare_parameter<double>("min_fitness", 0.3);
    max_rmse_ = declare_parameter<double>("max_rmse", 0.05);
    object_class_ = declare_parameter<std::string>("object_class", "concrete_block");
    world_frame_ = declare_parameter<std::string>("world_frame", "world");
    max_sync_delta_s_ = declare_parameter<double>("sync.max_delta_s", 0.06);
    object_timeout_s_ = declare_parameter<double>("world_model.object_timeout_s", 10.0);
    debug_detection_overlay_enabled_ = declare_parameter<bool>("debug.publish_detection_overlay", true);
    perf_log_timing_enabled_ = declare_parameter<bool>("perf.log_timing", true);
    perf_log_every_n_frames_ = declare_parameter<int>("perf.log_every_n_frames", 20);
    const double marker_refresh_period_s =
      declare_parameter<double>("world_model.marker_refresh_period_s", 0.5);

    // Keep for launch-file compatibility; no longer used in one-shot flow.
    (void)declare_parameter<std::string>("calib_yaml", "");

    if (perf_log_every_n_frames_ < 1) {
      perf_log_every_n_frames_ = 1;
    }

    if (debug_detection_overlay_enabled_) {
      det_debug_pub_ = create_publisher<sensor_msgs::msg::Image>("debug/detection_overlay", 1);
    }

    world_pub_ = create_publisher<BlockArray>("block_world_model", 10);
    marker_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(
      "block_world_model_markers", 10);

    image_sub_.subscribe(this, "image");
    cloud_sub_.subscribe(this, "points");
    sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
      SyncPolicy(10), image_sub_, cloud_sub_);
    sync_->registerCallback(
      std::bind(
        &WorldModelNode::syncCallback,
        this,
        std::placeholders::_1,
        std::placeholders::_2));

    segment_client_ = create_client<SegmentSrv>("/yolos_segmentor_service/segment");
    action_client_ = rclcpp_action::create_client<RegisterBlock>(
      this, "register_block", action_client_cb_group_);

    if (!applyPerceptionMode(perception_mode_str)) {
      RCLCPP_WARN(
        get_logger(),
        "Unsupported startup perception_mode '%s'; keeping pipeline_mode '%s'.",
        perception_mode_str.c_str(),
        pipeline_mode_str.c_str());
    }

    if (pipeline_mode_ != cbpwm::PipelineMode::kIdle &&
      !segment_client_->wait_for_service(std::chrono::seconds(2)))
    {
      RCLCPP_WARN(get_logger(), "Segmentation service not available at startup.");
    }

    if (needsRegistration() &&
      !action_client_->wait_for_action_server(std::chrono::seconds(2)))
    {
      RCLCPP_WARN(get_logger(), "Registration action not available at startup.");
    }

    set_mode_srv_ = create_service<SetModeSrv>(
      "~/set_mode",
      std::bind(
        &WorldModelNode::handleSetMode,
        this,
        std::placeholders::_1,
        std::placeholders::_2));

    get_coarse_srv_ = create_service<GetCoarseSrv>(
      "~/get_coarse_blocks",
      std::bind(
        &WorldModelNode::handleGetCoarseBlocks,
        this,
        std::placeholders::_1,
        std::placeholders::_2));

    run_pose_srv_ = create_service<RunPoseSrv>(
      "~/run_pose_estimation",
      std::bind(
        &WorldModelNode::handleRunPoseEstimation,
        this,
        std::placeholders::_1,
        std::placeholders::_2),
      rmw_qos_profile_services_default,
      run_pose_cb_group_);

    marker_refresh_timer_ = create_wall_timer(
      std::chrono::duration<double>(marker_refresh_period_s),
      [this]() {
        const BlockArray snapshot = latestWorldSnapshot();
        if (!snapshot.blocks.empty()) {
          publishWorldMarkers(snapshot.header, snapshot.blocks);
        }
      });

    WM_LOG(
      get_logger(),
      "WorldModelNode ready | pipeline_mode=%s | perception_mode=%s",
      cbpwm::pipelineModeToString(pipeline_mode_),
      cbpwm::perceptionModeToString(perception_mode_));
  }

private:
  void resetPerfCounters()
  {
    std::lock_guard<std::mutex> lock(perf_mutex_);
    perf_timing_count_ = 0;
    perf_seg_sum_ms_ = 0;
    perf_track_sum_ms_ = 0;
    perf_reg_sum_ms_ = 0;
    perf_total_sum_ms_ = 0;
    dropped_busy_frames_.store(0);
    dropped_sync_frames_.store(0);
  }

  bool applyPerceptionMode(const std::string & mode)
  {
    cbpwm::PerceptionModeConfig config;
    if (!cbpwm::resolvePerceptionModeConfig(mode, config)) {
      return false;
    }

    {
      std::lock_guard<std::mutex> lock(mode_mutex_);
      perception_mode_ = config.perception_mode;
      pipeline_mode_ = config.pipeline_mode;
      resetPerfCounters();
    }

    RCLCPP_INFO(get_logger(), "%s", config.log_message);
    return true;
  }

  bool needsRegistration() const
  {
    return pipeline_mode_ == cbpwm::PipelineMode::kRegister ||
           pipeline_mode_ == cbpwm::PipelineMode::kFull;
  }

  std::string resolveGraspedBlockId()
  {
    std::lock_guard<std::mutex> lock(persistent_world_mutex_);
    rclcpp::Time newest_time(0, 0, get_clock()->get_clock_type());
    std::string best_id;
    for (const auto & kv : persistent_world_) {
      if (kv.second.task_status != Block::TASK_MOVE) {
        continue;
      }
      const rclcpp::Time seen(kv.second.last_seen, get_clock()->get_clock_type());
      if (best_id.empty() || seen > newest_time) {
        newest_time = seen;
        best_id = kv.first;
      }
    }
    return best_id;
  }

  void resetBusy()
  {
    busy_.store(false);
  }

  void updateLatestWorldCache(const BlockArray & out)
  {
    std::lock_guard<std::mutex> lock(latest_world_mutex_);
    latest_world_ = out;
  }

  BlockArray latestWorldSnapshot()
  {
    std::lock_guard<std::mutex> lock(latest_world_mutex_);
    return latest_world_;
  }

  bool runRegistrationSync(
    uint32_t detection_id,
    const sensor_msgs::msg::Image & mask,
    const sensor_msgs::msg::PointCloud2 & cloud,
    const std_msgs::msg::Header & header,
    double timeout_s,
    Block & out_block,
    std::string & reason)
  {
    RegisterBlock::Goal goal;
    goal.mask = mask;
    goal.cloud = cloud;
    goal.object_class = object_class_;

    std::mutex reg_mutex;
    std::condition_variable reg_cv;
    bool done = false;
    bool goal_rejected = false;
    rclcpp_action::ResultCode result_code = rclcpp_action::ResultCode::UNKNOWN;
    RegisterBlock::Result::SharedPtr action_result;

    rclcpp_action::Client<RegisterBlock>::SendGoalOptions options;
    options.goal_response_callback =
      [&reg_mutex, &reg_cv, &done, &goal_rejected](GoalHandleRegisterBlock::SharedPtr goal_handle)
      {
        if (goal_handle) {
          return;
        }
        std::lock_guard<std::mutex> lock(reg_mutex);
        goal_rejected = true;
        done = true;
        reg_cv.notify_all();
      };
    options.result_callback =
      [&reg_mutex, &reg_cv, &done, &result_code, &action_result](
      const GoalHandleRegisterBlock::WrappedResult & wrapped)
      {
        std::lock_guard<std::mutex> lock(reg_mutex);
        result_code = wrapped.code;
        action_result = wrapped.result;
        done = true;
        reg_cv.notify_all();
      };

    (void)action_client_->async_send_goal(goal, options);

    {
      std::unique_lock<std::mutex> lock(reg_mutex);
      const bool completed = reg_cv.wait_for(
        lock,
        std::chrono::duration<double>(timeout_s),
        [&done]() {return done;});
      if (!completed) {
        reason = "action timeout";
        return false;
      }
      if (goal_rejected) {
        reason = "goal rejected";
        return false;
      }
    }

    if (result_code != rclcpp_action::ResultCode::SUCCEEDED) {
      reason = "result code " + std::to_string(static_cast<int>(result_code));
      return false;
    }

    if (!action_result) {
      reason = "empty result response";
      return false;
    }
    if (!action_result->success) {
      reason = "registration reported success=false";
      return false;
    }
    if (action_result->fitness < min_fitness_ || action_result->rmse > max_rmse_) {
      reason = "thresholds failed (fitness=" + std::to_string(action_result->fitness) +
        ", rmse=" + std::to_string(action_result->rmse) + ")";
      return false;
    }

    out_block.id = "block_" + std::to_string(detection_id);
    out_block.pose = action_result->pose;
    out_block.confidence = action_result->fitness;
    out_block.last_seen = header.stamp;
    out_block.pose_status = Block::POSE_PRECISE;
    out_block.task_status = Block::TASK_FREE;
    return true;
  }

  void completeOneShotRequest(uint64_t sequence, bool success, const std::string & message)
  {
    std::lock_guard<std::mutex> lock(one_shot_mutex_);
    if (sequence == 0 || sequence != active_one_shot_.sequence) {
      return;
    }
    active_one_shot_ = OneShotRequest{};
    one_shot_done_sequence_ = sequence;
    one_shot_last_success_ = success;
    one_shot_last_message_ = message;
    one_shot_cv_.notify_all();
    (void)applyPerceptionMode("IDLE");
  }

  void handleRunPoseEstimation(
    const std::shared_ptr<RunPoseSrv::Request> request,
    std::shared_ptr<RunPoseSrv::Response> response)
  {
    const cbpwm::OneShotMode run_mode = cbpwm::parseOneShotMode(request->mode);
    if (run_mode == cbpwm::OneShotMode::kNone) {
      response->success = false;
      response->message = "Unsupported mode: " + request->mode;
      return;
    }

    OneShotRequest run_request;
    run_request.mode = run_mode;
    run_request.target_block_id = request->target_block_id;
    run_request.enable_debug = request->enable_debug;
    run_request.registration_timeout_s =
      request->timeout_s > 0.0f ? static_cast<double>(request->timeout_s) : 3.0;

    if (run_mode == cbpwm::OneShotMode::kRefineGrasped && run_request.target_block_id.empty()) {
      run_request.target_block_id = resolveGraspedBlockId();
      if (run_request.target_block_id.empty()) {
        response->success = false;
        response->message = "No grasped block found (TASK_MOVE) and no target_block_id provided.";
        return;
      }
    }

    {
      std::lock_guard<std::mutex> lock(one_shot_mutex_);
      if (active_one_shot_.mode != cbpwm::OneShotMode::kNone) {
        response->success = false;
        response->message = "Another one-shot request is already running.";
        return;
      }
      run_request.sequence = ++one_shot_sequence_counter_;
      active_one_shot_ = run_request;
    }

    {
      std::lock_guard<std::mutex> lock(mode_mutex_);
      debug_detection_overlay_enabled_ = request->enable_debug;
      pipeline_mode_ = cbpwm::PipelineMode::kFull;
    }

    RCLCPP_INFO(
      get_logger(),
      "Scheduled one-shot pose estimation: mode=%s target=%s",
      cbpwm::oneShotModeToString(run_mode),
      run_request.target_block_id.empty() ? "<all>" : run_request.target_block_id.c_str());

    const double timeout_s = request->timeout_s > 0.0f ? request->timeout_s : 5.0;
    {
      std::unique_lock<std::mutex> lock(one_shot_mutex_);
      const bool done = one_shot_cv_.wait_for(
        lock,
        std::chrono::duration<double>(timeout_s),
        [this, &run_request]() {
          return one_shot_done_sequence_ >= run_request.sequence;
        });
      if (!done) {
        if (active_one_shot_.sequence == run_request.sequence) {
          active_one_shot_ = OneShotRequest{};
        }
        (void)applyPerceptionMode("IDLE");
        response->success = false;
        response->message = "Timed out waiting for one-shot result.";
        response->blocks = latestWorldSnapshot();
        return;
      }
      response->success = one_shot_last_success_;
      response->message = one_shot_last_message_;
    }

    response->blocks = latestWorldSnapshot();
  }

  void handleSetMode(
    const std::shared_ptr<SetModeSrv::Request> request,
    std::shared_ptr<SetModeSrv::Response> response)
  {
    if (!applyPerceptionMode(request->mode)) {
      response->success = false;
      response->message = "Unsupported mode: " + request->mode;
      return;
    }

    {
      std::lock_guard<std::mutex> lock(mode_mutex_);
      debug_detection_overlay_enabled_ = request->enable_debug;
    }

    response->success = true;
    response->message = "Mode applied: " + request->mode;
  }

  void handleGetCoarseBlocks(
    const std::shared_ptr<GetCoarseSrv::Request> request,
    std::shared_ptr<GetCoarseSrv::Response> response)
  {
    (void)request;
    response->success = true;
    response->blocks = latestWorldSnapshot();
    response->message = "ok";
    RCLCPP_INFO(
      get_logger(),
      "GetCoarseBlocks -> %zu blocks (stamp=%u.%u)",
      response->blocks.blocks.size(),
      response->blocks.header.stamp.sec,
      response->blocks.header.stamp.nanosec);
  }

  void publishWorldMarkers(const std_msgs::msg::Header & header, const std::vector<Block> & blocks)
  {
    auto marker_header = header;
    marker_header.stamp = rclcpp::Time(0, 0, get_clock()->get_clock_type());
    const auto markers = cbpwm::buildWorldMarkers(marker_header, blocks, world_frame_);
    marker_pub_->publish(markers);

    if (!blocks.empty()) {
      const auto & b0 = blocks.front();
      RCLCPP_INFO_THROTTLE(
        get_logger(),
        *get_clock(),
        2000,
        "Published markers: %zu blocks in frame '%s' (first: id=%s pos=[%.3f, %.3f, %.3f])",
        blocks.size(),
        world_frame_.c_str(),
        b0.id.c_str(),
        b0.pose.position.x,
        b0.pose.position.y,
        b0.pose.position.z);
    }
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
    updateLatestWorldCache(out);
    publishWorldMarkers(out.header, out.blocks);
  }

  void publishDetectionOverlay(
    const sensor_msgs::msg::Image::ConstSharedPtr & image,
    const vision_msgs::msg::Detection2DArray & detections,
    const sensor_msgs::msg::Image & mask_msg)
  {
    cv::Mat img = toCvBgr(*image);

    if (!mask_msg.data.empty()) {
      cv::Mat mask = toCvMono(mask_msg);
      overlayMask(img, mask, cv::Scalar(255, 255, 0), 0.35);
    }

    drawDetectionBoxes(img, detections, cv::Scalar(0, 255, 0));
    auto out = cv_bridge::CvImage(image->header, "bgr8", img).toImageMsg();
    det_debug_pub_->publish(*out);
  }

  void recordTiming(int64_t seg_ms, int64_t track_ms, int64_t reg_ms, int64_t total_ms)
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

    const double avg_seg = static_cast<double>(perf_seg_sum_ms_) / static_cast<double>(n);
    const double avg_track = static_cast<double>(perf_track_sum_ms_) / static_cast<double>(n);
    const double avg_reg = static_cast<double>(perf_reg_sum_ms_) / static_cast<double>(n);
    const double avg_total = static_cast<double>(perf_total_sum_ms_) / static_cast<double>(n);

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

  void syncCallback(
    const sensor_msgs::msg::Image::ConstSharedPtr image,
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr cloud)
  {
    {
      std::lock_guard<std::mutex> lock(one_shot_mutex_);
      if (active_one_shot_.mode == cbpwm::OneShotMode::kNone) {
        return;
      }
    }

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

  void processFrame(
    const sensor_msgs::msg::Image::ConstSharedPtr & image,
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr & cloud)
  {
    const auto t_start = std::chrono::steady_clock::now();

    OneShotRequest run_request;
    {
      std::lock_guard<std::mutex> lock(one_shot_mutex_);
      run_request = active_one_shot_;
    }

    if (run_request.mode == cbpwm::OneShotMode::kNone ||
      pipeline_mode_ == cbpwm::PipelineMode::kIdle)
    {
      resetBusy();
      return;
    }

    if (!segment_client_->service_is_ready()) {
      RCLCPP_WARN(get_logger(), "Segmentation service unavailable.");
      resetBusy();
      return;
    }

    auto seg_req = std::make_shared<SegmentSrv::Request>();
    seg_req->image = *image;
    seg_req->return_debug = false;

    segment_client_->async_send_request(
      seg_req,
      [this, image, cloud, t_start, run_request](rclcpp::Client<SegmentSrv>::SharedFuture seg_future) {
        try {
          auto seg_res = seg_future.get();
          const auto t_after_seg = std::chrono::steady_clock::now();
          const size_t seg_detection_count =
            seg_res ? seg_res->detections.detections.size() : 0U;

          if (!seg_res || !seg_res->success) {
            publishPersistentWorld(cloud->header);
            completeOneShotRequest(run_request.sequence, false, "Segmentation failed.");
            resetBusy();
            return;
          }

          RCLCPP_INFO(
            get_logger(),
            "One-shot %s: segmentation detections=%zu",
            cbpwm::oneShotModeToString(run_request.mode),
            seg_detection_count);

          if (debug_detection_overlay_enabled_ && det_debug_pub_) {
            publishDetectionOverlay(image, seg_res->detections, seg_res->mask);
          }

          const auto t_after_track = t_after_seg;
          const size_t tracked_count = seg_detection_count;
          RCLCPP_INFO(
            get_logger(),
            "One-shot %s: detections=%zu tracked=%zu (tracker bypassed)",
            cbpwm::oneShotModeToString(run_request.mode),
            seg_detection_count,
            tracked_count);

          if (seg_res->detections.detections.empty()) {
            const auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
              t_after_seg - t_start).count();
            recordTiming(total_ms, 0, 0, total_ms);
            publishPersistentWorld(cloud->header);

            if (run_request.mode == cbpwm::OneShotMode::kSceneDiscovery) {
              completeOneShotRequest(
                run_request.sequence,
                true,
                "Scene discovery finished (detections=0, tracked=0, registrations=0).");
            } else {
              completeOneShotRequest(
                run_request.sequence,
                false,
                "Requested block was not detected.");
            }
            resetBusy();
            return;
          }

          std::vector<std::pair<uint32_t, sensor_msgs::msg::Image>> candidates;
          candidates.reserve(seg_res->detections.detections.size());

          cv::Mat full_mask = toCvMono(seg_res->mask);
          const auto & detections = seg_res->detections.detections;
          for (size_t i = 0; i < detections.size(); ++i) {
            const auto & det = detections[i];
            const uint32_t detection_id = static_cast<uint32_t>(i + 1U);
            const std::string det_id = "block_" + std::to_string(detection_id);

            if ((run_request.mode == cbpwm::OneShotMode::kRefineBlock ||
              run_request.mode == cbpwm::OneShotMode::kRefineGrasped) &&
              !run_request.target_block_id.empty() &&
              det_id != run_request.target_block_id)
            {
              continue;
            }

            cv::Mat det_mask = extract_mask_roi(full_mask, det);
            if (det_mask.empty() || cv::countNonZero(det_mask) == 0) {
              continue;
            }

            auto mask_msg = cv_bridge::CvImage(
              seg_res->mask.header,
              "mono8",
              det_mask).toImageMsg();
            candidates.emplace_back(detection_id, *mask_msg);
          }

          const size_t registration_candidates = candidates.size();
          RCLCPP_INFO(
            get_logger(),
            "One-shot %s: detections=%zu tracked=%zu registration_candidates=%zu",
            cbpwm::oneShotModeToString(run_request.mode),
            seg_detection_count,
            tracked_count,
            registration_candidates);

          if (registration_candidates == 0) {
            const auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
              t_after_track - t_start).count();
            recordTiming(total_ms, 0, 0, total_ms);
            publishPersistentWorld(cloud->header);

            if (run_request.mode == cbpwm::OneShotMode::kSceneDiscovery) {
              completeOneShotRequest(
                run_request.sequence,
                true,
                "Scene discovery finished (detections=" + std::to_string(seg_detection_count) +
                ", tracked=" + std::to_string(tracked_count) +
                ", registration_candidates=0, registrations=0).");
            } else {
              completeOneShotRequest(
                run_request.sequence,
                false,
                "Requested block was not available for registration "
                "(detections=" + std::to_string(seg_detection_count) +
                ", tracked=" + std::to_string(tracked_count) + ").");
            }

            resetBusy();
            return;
          }

          if (!action_client_->action_server_is_ready()) {
            RCLCPP_WARN(get_logger(), "Registration action unavailable.");
            completeOneShotRequest(run_request.sequence, false, "Registration action unavailable.");
            resetBusy();
            return;
          }

          const auto t_reg_start = std::chrono::steady_clock::now();
          size_t registrations_ok = 0;
          {
            std::lock_guard<std::mutex> lock(persistent_world_mutex_);
            for (const auto & candidate : candidates) {
              Block block;
              std::string reason;
              const bool ok = runRegistrationSync(
                candidate.first,
                candidate.second,
                *cloud,
                cloud->header,
                run_request.registration_timeout_s,
                block,
                reason);

              if (ok) {
                persistent_world_[block.id] = block;
                ++registrations_ok;
                RCLCPP_INFO(get_logger(), "Registration accepted for %s", block.id.c_str());
              } else {
                RCLCPP_WARN(
                  get_logger(),
                  "Registration rejected for block_%u: %s",
                  candidate.first,
                  reason.c_str());
              }
            }
          }

          const auto t_reg_end = std::chrono::steady_clock::now();
          const auto seg_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            t_after_seg - t_start).count();
          const auto track_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            t_after_track - t_after_seg).count();
          const auto reg_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            t_reg_end - t_reg_start).count();
          const auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            t_reg_end - t_start).count();
          recordTiming(seg_ms, track_ms, reg_ms, total_ms);

          publishPersistentWorld(cloud->header);
          completeOneShotRequest(
            run_request.sequence,
            registrations_ok > 0 || run_request.mode == cbpwm::OneShotMode::kSceneDiscovery,
            "Pose estimation finished (detections=" + std::to_string(seg_detection_count) +
            ", tracked=" + std::to_string(tracked_count) +
            ", registration_candidates=" + std::to_string(registration_candidates) +
            ", registrations=" + std::to_string(registrations_ok) + ").");

          resetBusy();
        } catch (const std::exception & e) {
          RCLCPP_ERROR(get_logger(), "Segmentation stage failed: %s", e.what());
          completeOneShotRequest(run_request.sequence, false, "Segmentation stage exception.");
          resetBusy();
        }
      });
  }

  message_filters::Subscriber<sensor_msgs::msg::Image> image_sub_;
  message_filters::Subscriber<sensor_msgs::msg::PointCloud2> cloud_sub_;
  std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;

  rclcpp::Client<SegmentSrv>::SharedPtr segment_client_;
  rclcpp_action::Client<RegisterBlock>::SharedPtr action_client_;
  rclcpp::Service<SetModeSrv>::SharedPtr set_mode_srv_;
  rclcpp::Service<GetCoarseSrv>::SharedPtr get_coarse_srv_;
  rclcpp::Service<RunPoseSrv>::SharedPtr run_pose_srv_;
  rclcpp::CallbackGroup::SharedPtr run_pose_cb_group_;
  rclcpp::CallbackGroup::SharedPtr action_client_cb_group_;
  rclcpp::TimerBase::SharedPtr marker_refresh_timer_;

  rclcpp::Publisher<BlockArray>::SharedPtr world_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr det_debug_pub_;

  std::unordered_map<std::string, Block> persistent_world_;
  std::mutex persistent_world_mutex_;

  BlockArray latest_world_;
  std::mutex latest_world_mutex_;

  std::mutex mode_mutex_;
  cbpwm::PerceptionMode perception_mode_{cbpwm::PerceptionMode::kSceneScan};
  cbpwm::PipelineMode pipeline_mode_{cbpwm::PipelineMode::kFull};

  std::atomic<bool> busy_{false};
  std::atomic<uint64_t> dropped_busy_frames_{0};
  std::atomic<uint64_t> dropped_sync_frames_{0};

  double min_fitness_{0.3};
  double max_rmse_{0.05};
  std::string object_class_;
  std::string world_frame_{"world"};
  double max_sync_delta_s_{0.06};
  double object_timeout_s_{10.0};
  bool debug_detection_overlay_enabled_{true};
  bool perf_log_timing_enabled_{true};
  int perf_log_every_n_frames_{20};

  std::mutex perf_mutex_;
  uint64_t perf_timing_count_{0};
  int64_t perf_seg_sum_ms_{0};
  int64_t perf_track_sum_ms_{0};
  int64_t perf_reg_sum_ms_{0};
  int64_t perf_total_sum_ms_{0};

  std::mutex one_shot_mutex_;
  std::condition_variable one_shot_cv_;
  OneShotRequest active_one_shot_;
  uint64_t one_shot_sequence_counter_{0};
  uint64_t one_shot_done_sequence_{0};
  bool one_shot_last_success_{false};
  std::string one_shot_last_message_;

};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  auto node = std::make_shared<WorldModelNode>();
  rclcpp::executors::MultiThreadedExecutor exec(rclcpp::ExecutorOptions(), 4);
  exec.add_node(node);
  exec.spin();

  rclcpp::shutdown();
  return 0;
}
