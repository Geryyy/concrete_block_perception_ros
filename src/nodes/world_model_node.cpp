#include <deque>
#include <unordered_map>
#include <mutex>
#include <condition_variable>
#include <future>
#include <atomic>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <cctype>
#include <filesystem>

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
#include "concrete_block_perception/srv/set_perception_mode.hpp"
#include "concrete_block_perception/srv/get_coarse_blocks.hpp"
#include "concrete_block_perception/srv/run_pose_estimation.hpp"
#include "concrete_block_perception/action/register_block.hpp"
#include "concrete_block_perception/msg/block.hpp"
#include "concrete_block_perception/msg/block_array.hpp"
#include "concrete_block_perception/msg/tracked_detection_array.hpp"
#include "concrete_block_perception/utils/img_utils.hpp"
#include "concrete_block_perception/utils/io_utils.hpp"
#include "pcd_block_estimation/mask_projection.hpp"
#include "pcd_block_estimation/yaml_utils.hpp"

#define WM_LOG(logger, ...) RCLCPP_INFO(logger, __VA_ARGS__)

using concrete_block_perception::msg::Block;
using concrete_block_perception::msg::BlockArray;

class WorldModelNode : public rclcpp::Node
{
  using SegmentSrv = ros2_yolos_cpp::srv::SegmentImage;
  using SetModeSrv = concrete_block_perception::srv::SetPerceptionMode;
  using GetCoarseSrv = concrete_block_perception::srv::GetCoarseBlocks;
  using RunPoseSrv = concrete_block_perception::srv::RunPoseEstimation;
  using RegisterBlock = concrete_block_perception::action::RegisterBlock;
  using GoalHandleRegisterBlock =
    rclcpp_action::ClientGoalHandle<RegisterBlock>;

  using SyncPolicy =
    message_filters::sync_policies::ApproximateTime<
    sensor_msgs::msg::Image,
    sensor_msgs::msg::PointCloud2>;

  enum class PipelineMode
  {
    kIdle,
    kSegment,
    kTrack,
    kRegister,
    kFull
  };

  enum class PerceptionMode
  {
    kIdle,
    kSceneScan,
    kPreGrasp,
    kGraspExecute,
    kTransport,
    kPreAssembly,
    kAssemblyExecute
  };

  enum class OneShotMode
  {
    kNone,
    kSceneDiscovery,
    kRefineBlock,
    kRefineGrasped
  };

  struct OneShotRequest
  {
    uint64_t sequence{0};
    OneShotMode mode{OneShotMode::kNone};
    std::string target_block_id;
    bool enable_debug{true};
    double registration_timeout_s{3.0};
  };

  struct FrameContext
  {
    std_msgs::msg::Header header;
    std::vector<Block> blocks;
    std::mutex blocks_mutex;
    std::atomic<size_t> pending{0};
    std::atomic<bool> finished{false};
    uint64_t one_shot_sequence{0};
    OneShotMode one_shot_mode{OneShotMode::kNone};
    std::string one_shot_target;
    size_t seg_detection_count{0};
    size_t tracked_count{0};
    size_t registration_candidates{0};

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
    run_pose_cb_group_ = create_callback_group(rclcpp::CallbackGroupType::Reentrant);
    action_client_cb_group_ = create_callback_group(rclcpp::CallbackGroupType::Reentrant);

    // ================================
    // Parameters: gating + commissioning
    // ================================
    const std::string mode_str =
      declare_parameter<std::string>("pipeline_mode", "full");
    pipeline_mode_ = parseMode(mode_str);
    const std::string perception_mode_str =
      declare_parameter<std::string>("perception_mode", "FULL");

    min_fitness_ = declare_parameter<double>("min_fitness", 0.3);
    max_rmse_ = declare_parameter<double>("max_rmse", 0.05);
    object_class_ =
      declare_parameter<std::string>("object_class", "concrete_block");
    const std::string calib_yaml =
      declare_parameter<std::string>("calib_yaml", "");
    world_frame_ =
      declare_parameter<std::string>("world_frame", "world");
    max_sync_delta_s_ =
      declare_parameter<double>("sync.max_delta_s", 0.06);
    object_timeout_s_ =
      declare_parameter<double>("world_model.object_timeout_s", 10.0);
    debug_detection_overlay_enabled_ =
      declare_parameter<bool>("debug.publish_detection_overlay", true);
    perf_log_timing_enabled_ =
      declare_parameter<bool>("perf.log_timing", true);
    perf_log_every_n_frames_ =
      declare_parameter<int>("perf.log_every_n_frames", 20);
    const double marker_refresh_period_s =
      declare_parameter<double>("world_model.marker_refresh_period_s", 0.5);
    if (perf_log_every_n_frames_ < 1) {
      perf_log_every_n_frames_ = 1;
    }

    if (!calib_yaml.empty() && std::filesystem::exists(calib_yaml)) {
      try {
        coarse_T_P_C_ = pcd_block::load_T_4x4(calib_yaml);
        coarse_K_ = pcd_block::load_camera_matrix(calib_yaml);
        coarse_projection_ready_ = true;
      } catch (const std::exception & e) {
        RCLCPP_WARN(
          get_logger(),
          "Failed to load coarse projection calibration from %s: %s",
          calib_yaml.c_str(),
          e.what());
      }
    } else {
      RCLCPP_WARN(
        get_logger(),
        "calib_yaml missing or not found (%s); coarse pose fallback will be limited.",
        calib_yaml.c_str());
    }

    // ================================
    // Debug publishers
    // ================================
    if (debug_detection_overlay_enabled_) {
      det_debug_pub_ =
        create_publisher<sensor_msgs::msg::Image>(
        "debug/detection_overlay", 1);
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

    action_client_ =
      rclcpp_action::create_client<RegisterBlock>(
      this, "register_block", action_client_cb_group_);

    // Apply BT-facing mode before startup service/action waits.
    if (!applyPerceptionMode(perception_mode_str)) {
      RCLCPP_WARN(
        get_logger(),
        "Unsupported startup perception_mode '%s'; keeping pipeline_mode '%s'.",
        perception_mode_str.c_str(),
        mode_str.c_str());
    }

    if (pipeline_mode_ != PipelineMode::kIdle &&
      !segment_client_->wait_for_service(std::chrono::seconds(2)))
    {
      RCLCPP_WARN(
        get_logger(),
        "Segmentation service not available at startup.");
    }

    if (needsRegistration() &&
      !action_client_->wait_for_action_server(std::chrono::seconds(2)))
    {
      RCLCPP_WARN(
        get_logger(),
        "Registration action not available at startup.");
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
        BlockArray snapshot;
        {
          std::lock_guard<std::mutex> lock(latest_world_mutex_);
          snapshot = latest_world_;
        }
        if (snapshot.blocks.empty()) {
          return;
        }
        publishWorldMarkers(snapshot.header, snapshot.blocks);
      });

    WM_LOG(
      get_logger(),
      "WorldModelNode ready | pipeline_mode=%s | perception_mode=%s",
      pipelineModeToString(pipeline_mode_),
      perceptionModeToString(perception_mode_));
  }

private:
  static const char * pipelineModeToString(PipelineMode mode)
  {
    switch (mode) {
      case PipelineMode::kIdle:
        return "idle";
      case PipelineMode::kSegment:
        return "segment";
      case PipelineMode::kTrack:
        return "track";
      case PipelineMode::kRegister:
        return "register";
      case PipelineMode::kFull:
      default:
        return "full";
    }
  }

  static const char * perceptionModeToString(PerceptionMode mode)
  {
    switch (mode) {
      case PerceptionMode::kIdle:
        return "IDLE";
      case PerceptionMode::kSceneScan:
        return "SCENE_SCAN";
      case PerceptionMode::kPreGrasp:
        return "PRE_GRASP";
      case PerceptionMode::kGraspExecute:
        return "GRASP_EXECUTE";
      case PerceptionMode::kTransport:
        return "TRANSPORT";
      case PerceptionMode::kPreAssembly:
        return "PRE_ASSEMBLY";
      case PerceptionMode::kAssemblyExecute:
      default:
        return "ASSEMBLY_EXECUTE";
    }
  }

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

  PipelineMode parseMode(const std::string & mode_str)
  {
    if (mode_str == "idle") {
      return PipelineMode::kIdle;
    }
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

  static std::string normalizeMode(std::string mode)
  {
    std::transform(mode.begin(), mode.end(), mode.begin(), [](unsigned char c) {
      return static_cast<char>(std::toupper(c));
    });
    return mode;
  }

  bool applyPerceptionMode(const std::string & mode)
  {
    const std::string m = normalizeMode(mode);
    std::lock_guard<std::mutex> lock(mode_mutex_);

    if (m == "IDLE") {
      perception_mode_ = PerceptionMode::kIdle;
      pipeline_mode_ = PipelineMode::kIdle;
      registration_on_demand_ = false;
      refine_request_pending_ = false;
      resetPerfCounters();
      RCLCPP_INFO(get_logger(), "Perception mode set: IDLE (no processing)");
      return true;
    }
    if (m == "SCENE_SCAN") {
      perception_mode_ = PerceptionMode::kSceneScan;
      pipeline_mode_ = PipelineMode::kTrack;
      registration_on_demand_ = false;
      resetPerfCounters();
      RCLCPP_INFO(get_logger(), "Perception mode set: SCENE_SCAN (track + coarse world publish)");
      return true;
    }
    if (m == "PRE_GRASP") {
      perception_mode_ = PerceptionMode::kPreGrasp;
      pipeline_mode_ = PipelineMode::kFull;
      registration_on_demand_ = true;
      resetPerfCounters();
      RCLCPP_INFO(get_logger(), "Perception mode set: PRE_GRASP (registration on demand)");
      return true;
    }
    if (m == "GRASP_EXECUTE") {
      perception_mode_ = PerceptionMode::kGraspExecute;
      pipeline_mode_ = PipelineMode::kTrack;
      registration_on_demand_ = false;
      resetPerfCounters();
      RCLCPP_INFO(get_logger(), "Perception mode set: GRASP_EXECUTE (track only)");
      return true;
    }
    if (m == "TRANSPORT") {
      perception_mode_ = PerceptionMode::kTransport;
      pipeline_mode_ = PipelineMode::kSegment;
      registration_on_demand_ = false;
      resetPerfCounters();
      RCLCPP_INFO(get_logger(), "Perception mode set: TRANSPORT (segment only)");
      return true;
    }
    if (m == "PRE_ASSEMBLY") {
      perception_mode_ = PerceptionMode::kPreAssembly;
      pipeline_mode_ = PipelineMode::kFull;
      registration_on_demand_ = true;
      resetPerfCounters();
      RCLCPP_INFO(get_logger(), "Perception mode set: PRE_ASSEMBLY (registration on demand)");
      return true;
    }
    if (m == "ASSEMBLY_EXECUTE") {
      perception_mode_ = PerceptionMode::kAssemblyExecute;
      pipeline_mode_ = PipelineMode::kTrack;
      registration_on_demand_ = false;
      resetPerfCounters();
      RCLCPP_INFO(get_logger(), "Perception mode set: ASSEMBLY_EXECUTE (track only)");
      return true;
    }

    // Backward compatible fallbacks to old pipeline mode names.
    if (m == "SEGMENT") {
      pipeline_mode_ = PipelineMode::kSegment;
      registration_on_demand_ = false;
      resetPerfCounters();
      return true;
    }
    if (m == "TRACK") {
      pipeline_mode_ = PipelineMode::kTrack;
      registration_on_demand_ = false;
      resetPerfCounters();
      return true;
    }
    if (m == "REGISTER") {
      pipeline_mode_ = PipelineMode::kRegister;
      registration_on_demand_ = false;
      resetPerfCounters();
      return true;
    }
    if (m == "FULL") {
      pipeline_mode_ = PipelineMode::kFull;
      registration_on_demand_ = false;
      resetPerfCounters();
      return true;
    }

    return false;
  }

  static const char * oneShotModeToString(OneShotMode mode)
  {
    switch (mode) {
      case OneShotMode::kSceneDiscovery:
        return "SCENE_DISCOVERY";
      case OneShotMode::kRefineBlock:
        return "REFINE_BLOCK";
      case OneShotMode::kRefineGrasped:
        return "REFINE_GRASPED";
      case OneShotMode::kNone:
      default:
        return "NONE";
    }
  }

  static OneShotMode parseOneShotMode(const std::string & mode)
  {
    const std::string m = normalizeMode(mode);
    if (m == "SCENE_DISCOVERY") {
      return OneShotMode::kSceneDiscovery;
    }
    if (m == "REFINE_BLOCK") {
      return OneShotMode::kRefineBlock;
    }
    if (m == "REFINE_GRASPED") {
      return OneShotMode::kRefineGrasped;
    }
    return OneShotMode::kNone;
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

    auto send_goal_future = action_client_->async_send_goal(goal);
    {
      std::lock_guard<std::mutex> lock(action_send_futures_mutex_);
      action_send_futures_.push_back(send_goal_future);
      if (action_send_futures_.size() > 128U) {
        action_send_futures_.erase(action_send_futures_.begin());
      }
    }

    if (send_goal_future.wait_for(std::chrono::duration<double>(timeout_s)) !=
      std::future_status::ready)
    {
      reason = "goal response timeout";
      return false;
    }

    auto goal_handle = send_goal_future.get();
    if (!goal_handle) {
      reason = "goal rejected";
      return false;
    }

    auto result_future = action_client_->async_get_result(goal_handle);
    if (result_future.wait_for(std::chrono::duration<double>(timeout_s)) !=
      std::future_status::ready)
    {
      reason = "result timeout";
      return false;
    }

    auto wrapped = result_future.get();
    if (wrapped.code != rclcpp_action::ResultCode::SUCCEEDED) {
      reason = "result code " + std::to_string(static_cast<int>(wrapped.code));
      return false;
    }

    const auto & res = wrapped.result;
    if (!res) {
      reason = "empty result response";
      return false;
    }
    if (!res->success) {
      reason = "registration reported success=false";
      return false;
    }
    if (res->fitness < min_fitness_ || res->rmse > max_rmse_) {
      reason = "thresholds failed (fitness=" + std::to_string(res->fitness) +
        ", rmse=" + std::to_string(res->rmse) + ")";
      return false;
    }

    out_block.id = "block_" + std::to_string(detection_id);
    out_block.pose = res->pose;
    out_block.confidence = res->fitness;
    out_block.last_seen = header.stamp;
    out_block.pose_status = Block::POSE_PRECISE;
    out_block.task_status = Block::TASK_FREE;
    return true;
  }

  void completeOneShotRequest(
    uint64_t sequence,
    bool success,
    const std::string & message)
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
    const OneShotMode run_mode = parseOneShotMode(request->mode);
    if (run_mode == OneShotMode::kNone) {
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

    if (run_mode == OneShotMode::kRefineGrasped &&
      run_request.target_block_id.empty())
    {
      run_request.target_block_id = resolveGraspedBlockId();
      if (run_request.target_block_id.empty()) {
        response->success = false;
        response->message = "No grasped block found (TASK_MOVE) and no target_block_id provided.";
        return;
      }
    }

    {
      std::lock_guard<std::mutex> lock(one_shot_mutex_);
      if (active_one_shot_.mode != OneShotMode::kNone) {
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
      pipeline_mode_ = PipelineMode::kFull;
      registration_on_demand_ = true;
      if (run_mode == OneShotMode::kSceneDiscovery) {
        refine_target_block_id_.clear();
      } else {
        refine_target_block_id_ = run_request.target_block_id;
      }
      refine_request_pending_ = true;
    }

    RCLCPP_INFO(
      get_logger(),
      "Scheduled one-shot pose estimation: mode=%s target=%s",
      oneShotModeToString(run_mode),
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
      if (!request->target_block_id.empty()) {
        refine_target_block_id_ = request->target_block_id;
        refine_request_pending_ = true;
      }
    }

    response->success = true;
    response->message = "Mode applied: " + request->mode;
  }

  void handleGetCoarseBlocks(
    const std::shared_ptr<GetCoarseSrv::Request> request,
    std::shared_ptr<GetCoarseSrv::Response> response)
  {
    (void)request;
    std::lock_guard<std::mutex> lock(coarse_blocks_mutex_);
    response->success = true;
    response->blocks = latest_coarse_blocks_;
    response->message = "ok";
    RCLCPP_INFO(
      get_logger(),
      "GetCoarseBlocks -> %zu blocks (stamp=%u.%u)",
      response->blocks.blocks.size(),
      response->blocks.header.stamp.sec,
      response->blocks.header.stamp.nanosec);
  }

  bool needsRegistration() const
  {
    return pipeline_mode_ == PipelineMode::kRegister ||
           pipeline_mode_ == PipelineMode::kFull;
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
    // Visual convention: block should appear rotated +90 deg about Y.
    // Swapping X/Z extents achieves the expected orientation appearance.
    constexpr double kMarkerWidthM = 0.9;
    constexpr double kMarkerHeightM = 0.6;
    constexpr double kMarkerDepthM = 0.6;

    visualization_msgs::msg::MarkerArray ma;
    auto marker_header = header;
    marker_header.frame_id = world_frame_;
    marker_header.stamp = rclcpp::Time(0, 0, get_clock()->get_clock_type());
    int marker_id = 0;
    for (const auto & b : blocks) {
      visualization_msgs::msg::Marker m;
      m.header = marker_header;
      m.ns = "cbp_blocks";
      m.id = marker_id++;
      if (b.pose_status == Block::POSE_COARSE) {
        m.type = visualization_msgs::msg::Marker::SPHERE;
      } else {
        m.type = visualization_msgs::msg::Marker::CUBE;
      }
      m.action = visualization_msgs::msg::Marker::ADD;
      m.pose = b.pose;
      m.scale.x = kMarkerWidthM;
      m.scale.y = kMarkerHeightM;
      m.scale.z = kMarkerDepthM;
      // Encode state by marker color for quick RViz inspection.
      if (b.task_status == Block::TASK_REMOVED) {
        m.color.r = 0.9f;
        m.color.g = 0.1f;
        m.color.b = 0.1f;
      } else if (b.task_status == Block::TASK_PLACED) {
        m.color.r = 0.1f;
        m.color.g = 0.9f;
        m.color.b = 0.9f;
      } else if (b.task_status == Block::TASK_MOVE) {
        m.color.r = 0.2f;
        m.color.g = 0.4f;
        m.color.b = 1.0f;
      } else if (b.pose_status == Block::POSE_PRECISE) {
        m.color.r = 0.1f;
        m.color.g = 0.8f;
        m.color.b = 0.2f;
      } else if (b.pose_status == Block::POSE_COARSE) {
        m.color.r = 1.0f;
        m.color.g = 0.8f;
        m.color.b = 0.1f;
      } else {
        m.color.r = 0.5f;
        m.color.g = 0.5f;
        m.color.b = 0.5f;
      }
      m.color.a = 0.6f;
      ma.markers.push_back(std::move(m));

      visualization_msgs::msg::Marker label;
      label.header = marker_header;
      label.ns = "cbp_block_ids";
      label.id = marker_id++;
      label.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
      label.action = visualization_msgs::msg::Marker::ADD;
      label.pose = b.pose;
      label.pose.position.z += 0.7;
      label.scale.z = 0.2;
      label.color.r = 1.0f;
      label.color.g = 1.0f;
      label.color.b = 1.0f;
      label.color.a = 0.95f;
      label.text = b.id;
      ma.markers.push_back(std::move(label));
    }
    marker_pub_->publish(ma);
    if (!blocks.empty()) {
      const auto & b0 = blocks.front();
      RCLCPP_INFO_THROTTLE(
        get_logger(),
        *get_clock(),
        2000,
        "Published markers: %zu blocks in frame '%s' (first: id=%s pos=[%.3f, %.3f, %.3f])",
        blocks.size(),
        marker_header.frame_id.c_str(),
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
    publishWorldMarkers(header, out.blocks);
  }

  static float bestScore(const vision_msgs::msg::Detection2D & det)
  {
    float best = 0.0f;
    for (const auto & r : det.results) {
      best = std::max(best, static_cast<float>(r.hypothesis.score));
    }
    return best;
  }

  BlockArray buildCoarseBlocks(
    const concrete_block_perception::msg::TrackedDetectionArray & tracked,
    const std_msgs::msg::Header & header,
    const sensor_msgs::msg::PointCloud2 & cloud) const
  {
    BlockArray out;
    out.header = header;
    out.blocks.reserve(tracked.detections.size());
    auto scene_cloud = pointcloud2_to_open3d(cloud);
    const auto & pts = scene_cloud->points_;

    for (const auto & det : tracked.detections) {
      Block b;
      b.id = "block_" + std::to_string(det.detection_id);
      b.pose_status = Block::POSE_COARSE;
      b.task_status = Block::TASK_FREE;
      b.confidence = bestScore(det.detection);
      b.last_seen = det.stamp;
      b.pose.orientation.w = 1.0;  // coarse pose placeholder

      if (coarse_projection_ready_ && !det.mask.data.empty() && !pts.empty()) {
        try {
          cv::Mat mask = toCvMono(det.mask);
          auto selected = pcd_block::select_points_by_mask(
            pts, mask, coarse_K_, coarse_T_P_C_, 0.1);
          if (!selected.empty()) {
            Eigen::Vector3d c(0.0, 0.0, 0.0);
            for (const auto & p : selected) {
              c += p;
            }
            c /= static_cast<double>(selected.size());
            b.pose.position.x = c.x();
            b.pose.position.y = c.y();
            b.pose.position.z = c.z();
          }
        } catch (const std::exception &) {
          // Keep default coarse pose if projection/mask conversion fails.
        }
      }

      out.blocks.push_back(std::move(b));
    }

    return out;
  }

  void publishCoarseBlocks(
    const concrete_block_perception::msg::TrackedDetectionArray & tracked,
    const std_msgs::msg::Header & header,
    const sensor_msgs::msg::PointCloud2 & cloud)
  {
    auto coarse = buildCoarseBlocks(tracked, header, cloud);
    {
      std::lock_guard<std::mutex> lock(coarse_blocks_mutex_);
      latest_coarse_blocks_ = coarse;
    }
    world_pub_->publish(coarse);
    updateLatestWorldCache(coarse);
    publishWorldMarkers(coarse.header, coarse.blocks);
    RCLCPP_INFO(
      get_logger(),
      "Published coarse world: %zu blocks (stamp=%u.%u)",
      coarse.blocks.size(),
      coarse.header.stamp.sec,
      coarse.header.stamp.nanosec);
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

    completeOneShotRequest(
      ctx->one_shot_sequence,
      true,
      "Pose estimation finished (detections=" +
      std::to_string(ctx->seg_detection_count) +
      ", tracked=" + std::to_string(ctx->tracked_count) +
      ", registration_candidates=" + std::to_string(ctx->registration_candidates) +
      ", registrations=" + std::to_string(ctx->blocks.size()) + ").");

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
    {
      std::lock_guard<std::mutex> lock(one_shot_mutex_);
      if (active_one_shot_.mode == OneShotMode::kNone) {
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

  // ==========================================================
  // Pipeline (asynchronous chain)
  // ==========================================================
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

    if (run_request.mode == OneShotMode::kNone || pipeline_mode_ == PipelineMode::kIdle) {
      resetBusy();
      return;
    }

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
      [this, image, cloud, t_start, run_request](
        rclcpp::Client<SegmentSrv>::SharedFuture seg_future)
      {
        try {
          auto seg_res = seg_future.get();
          auto t_after_seg = std::chrono::steady_clock::now();
          const size_t seg_detection_count =
            seg_res ? seg_res->detections.detections.size() : 0U;

          if (!seg_res || !seg_res->success) {
            publishEmpty(cloud->header);
            completeOneShotRequest(
              run_request.sequence, false,
              "Segmentation failed.");
            resetBusy();
            return;
          }

          RCLCPP_INFO(
            get_logger(),
            "One-shot %s: segmentation detections=%zu",
            oneShotModeToString(run_request.mode),
            seg_detection_count);

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

          const auto t_after_track = t_after_seg;
          const size_t tracked_count = seg_detection_count;

          RCLCPP_INFO(
            get_logger(),
            "One-shot %s: detections=%zu tracked=%zu (tracker bypassed)",
            oneShotModeToString(run_request.mode),
            seg_detection_count,
            tracked_count);

          if (seg_res->detections.detections.empty()) {
            auto total_ms =
              std::chrono::duration_cast<std::chrono::milliseconds>(
              t_after_seg - t_start).count();
            recordTiming(total_ms, 0, 0, total_ms);
            publishPersistentWorld(cloud->header);
            if (run_request.mode == OneShotMode::kSceneDiscovery) {
              completeOneShotRequest(
                run_request.sequence, true,
                "Scene discovery finished (detections=0, tracked=0, registrations=0).");
            } else {
              completeOneShotRequest(
                run_request.sequence, false,
                "Requested block was not detected.");
            }
            resetBusy();
            return;
          }

          std::string refine_target;
          if (run_request.mode == OneShotMode::kRefineBlock ||
            run_request.mode == OneShotMode::kRefineGrasped)
          {
            refine_target = run_request.target_block_id;
          }

          if (!action_client_->action_server_is_ready()) {
            RCLCPP_WARN(get_logger(), "Registration action unavailable.");
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
            if (!refine_target.empty() && det_id != refine_target) {
              continue;
            }

            cv::Mat det_mask = extract_mask_roi(full_mask, det);
            if (det_mask.empty() || cv::countNonZero(det_mask) == 0) {
              continue;
            }

            auto mask_msg = cv_bridge::CvImage(
              seg_res->mask.header, "mono8", det_mask).toImageMsg();

            candidates.emplace_back(detection_id, *mask_msg);
          }
          const size_t registered = candidates.size();

          RCLCPP_INFO(
            get_logger(),
            "One-shot %s: detections=%zu tracked=%zu registration_candidates=%zu",
            oneShotModeToString(run_request.mode),
            seg_detection_count,
            tracked_count,
            registered);

          if (registered == 0) {
            auto total_ms =
              std::chrono::duration_cast<std::chrono::milliseconds>(
              t_after_track - t_start).count();
            recordTiming(total_ms, 0, 0, total_ms);
            publishPersistentWorld(cloud->header);
            if (run_request.mode == OneShotMode::kSceneDiscovery) {
              completeOneShotRequest(
                run_request.sequence, true,
                "Scene discovery finished (detections=" +
                std::to_string(seg_detection_count) +
                ", tracked=" + std::to_string(tracked_count) +
                ", registration_candidates=0, registrations=0).");
            } else {
              completeOneShotRequest(
                run_request.sequence, false,
                "Requested block was not available for registration "
                "(detections=" + std::to_string(seg_detection_count) +
                ", tracked=" + std::to_string(tracked_count) + ").");
            }
            resetBusy();
            return;
          }

          const auto t_reg_start = std::chrono::steady_clock::now();
          size_t registrations_ok = 0;
          {
            std::lock_guard<std::mutex> lock(persistent_world_mutex_);
            for (const auto & c : candidates) {
              Block b;
              std::string reason;
              const bool ok = runRegistrationSync(
                c.first, c.second, *cloud, cloud->header,
                run_request.registration_timeout_s, b, reason);
              if (ok) {
                persistent_world_[b.id] = b;
                ++registrations_ok;
                RCLCPP_INFO(
                  get_logger(),
                  "Registration accepted for %s",
                  b.id.c_str());
              } else {
                RCLCPP_WARN(
                  get_logger(),
                  "Registration rejected for block_%u: %s",
                  c.first, reason.c_str());
              }
            }
          }
          const auto t_reg_end = std::chrono::steady_clock::now();
          const auto seg_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(
            t_after_seg - t_start).count();
          const auto track_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(
            t_after_track - t_after_seg).count();
          const auto reg_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(
            t_reg_end - t_reg_start).count();
          const auto total_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(
            t_reg_end - t_start).count();
          recordTiming(seg_ms, track_ms, reg_ms, total_ms);

          publishPersistentWorld(cloud->header);
          completeOneShotRequest(
            run_request.sequence,
            registrations_ok > 0 || run_request.mode == OneShotMode::kSceneDiscovery,
            "Pose estimation finished (detections=" +
            std::to_string(seg_detection_count) +
            ", tracked=" + std::to_string(tracked_count) +
            ", registration_candidates=" + std::to_string(registered) +
            ", registrations=" + std::to_string(registrations_ok) + ").");
          resetBusy();
          return;

        } catch (const std::exception & e) {
          RCLCPP_ERROR(get_logger(), "Segmentation stage failed: %s", e.what());
          completeOneShotRequest(
            run_request.sequence, false,
            "Segmentation stage exception.");
          resetBusy();
        }
      });
  }

  // ==========================================================
  // Registration Result
  // ==========================================================
  void handleResult(
    uint64_t frame_id,
    uint32_t detection_id,
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

    if (result.code == rclcpp_action::ResultCode::SUCCEEDED) {
      const auto & res = result.result;

      if (res &&
        res->success &&
        res->fitness >= min_fitness_ &&
        res->rmse <= max_rmse_)
      {
        Block b;
        b.id = "block_" +
          std::to_string(detection_id);
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
        RCLCPP_INFO(
          get_logger(),
          "Registration accepted for block_%u (fitness=%.3f rmse=%.4f)",
          detection_id,
          static_cast<double>(res->fitness),
          static_cast<double>(res->rmse));
      } else if (res) {
        RCLCPP_WARN(
          get_logger(),
          "Registration rejected for block_%u (success=%s fitness=%.3f rmse=%.4f thresholds: min_fitness=%.3f max_rmse=%.4f)",
          detection_id,
          res->success ? "true" : "false",
          static_cast<double>(res->fitness),
          static_cast<double>(res->rmse),
          min_fitness_,
          max_rmse_);
      } else {
        RCLCPP_WARN(get_logger(), "Registration failed for block_%u: empty result payload", detection_id);
      }
    } else {
      RCLCPP_WARN(
        get_logger(),
        "Registration action did not succeed for block_%u (result_code=%d)",
        detection_id,
        static_cast<int>(result.code));
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
    updateLatestWorldCache(out);
    publishWorldMarkers(out.header, out.blocks);
  }

  void publishEmpty(const std_msgs::msg::Header & header)
  {
    BlockArray out;
    out.header = header;
    world_pub_->publish(out);
    updateLatestWorldCache(out);
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

  std::unordered_map<uint64_t, std::shared_ptr<FrameContext>> frames_;
  std::mutex frames_mutex_;
  std::atomic<uint64_t> frame_counter_{0};
  std::atomic<bool> busy_{false};
  std::atomic<uint64_t> dropped_busy_frames_{0};
  std::atomic<uint64_t> dropped_sync_frames_{0};

  std::unordered_map<std::string, Block> persistent_world_;
  std::mutex persistent_world_mutex_;
  BlockArray latest_world_;
  std::mutex latest_world_mutex_;
  BlockArray latest_coarse_blocks_;
  std::mutex coarse_blocks_mutex_;
  std::mutex mode_mutex_;
  PerceptionMode perception_mode_{PerceptionMode::kSceneScan};
  bool registration_on_demand_{false};
  bool refine_request_pending_{false};
  std::string refine_target_block_id_;
  bool coarse_projection_ready_{false};
  Eigen::Matrix3d coarse_K_{Eigen::Matrix3d::Identity()};
  Eigen::Matrix4d coarse_T_P_C_{Eigen::Matrix4d::Identity()};

  PipelineMode pipeline_mode_{PipelineMode::kFull};
  double min_fitness_;
  double max_rmse_;
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
  std::mutex action_send_futures_mutex_;
  std::vector<std::shared_future<GoalHandleRegisterBlock::SharedPtr>> action_send_futures_;
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
