#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <filesystem>
#include <future>
#include <limits>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <cv_bridge/cv_bridge.h>
#include <Eigen/Dense>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2/exceptions.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <visualization_msgs/msg/marker_array.hpp>

#include "concrete_block_perception/action/register_block.hpp"
#include "concrete_block_perception/msg/block.hpp"
#include "concrete_block_perception/msg/block_array.hpp"
#include "concrete_block_perception/srv/get_coarse_blocks.hpp"
#include "concrete_block_perception/srv/register_block.hpp"
#include "concrete_block_perception/srv/run_pose_estimation.hpp"
#include "concrete_block_perception/srv/set_block_task_status.hpp"
#include "concrete_block_perception/srv/set_perception_mode.hpp"
#include "concrete_block_perception/utils/coarse_pose_utils.hpp"
#include "concrete_block_perception/utils/img_utils.hpp"
#include "concrete_block_perception/utils/world_model_utils.hpp"
#include "concrete_block_perception/world_model/registration_flow.hpp"
#include "concrete_block_perception/world_model/roi_processing.hpp"
#include "concrete_block_perception/world_model/refine_flow.hpp"
#include "concrete_block_perception/world_model/roi_refinement.hpp"
#include "concrete_block_perception/world_model/scene_discovery_flow.hpp"
#include "concrete_block_perception/world_model/state_manager.hpp"
#include "ros2_yolos_cpp/srv/segment_image.hpp"

#define WM_LOG(logger, ...) RCLCPP_INFO(logger, __VA_ARGS__)

using concrete_block_perception::msg::Block;
using concrete_block_perception::msg::BlockArray;

namespace cbpwm = cbp::world_model;

class WorldModelNode : public rclcpp::Node
{
  using SegmentSrv = ros2_yolos_cpp::srv::SegmentImage;
  using SetModeSrv = concrete_block_perception::srv::SetPerceptionMode;
  using SetBlockTaskStatusSrv = concrete_block_perception::srv::SetBlockTaskStatus;
  using GetCoarseSrv = concrete_block_perception::srv::GetCoarseBlocks;
  using RegisterBlockSrv = concrete_block_perception::srv::RegisterBlock;
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

  struct CameraIntrinsics
  {
    bool valid{false};
    double fx{0.0};
    double fy{0.0};
    double cx{0.0};
    double cy{0.0};
    uint32_t width{0};
    uint32_t height{0};
  };

  struct PoseFusionConfig
  {
    bool enabled{true};
    std::string mode{"position_from_registration_orientation_from_fk"};
    double max_translation_jump_m{0.35};
    double max_z_delta_m{0.25};
    bool debug_log{true};
  };

  struct StartupParameters
  {
    std::string pipeline_mode_str{"full"};
    std::string perception_mode_str{"FULL"};
    double marker_refresh_period_s{0.5};
    std::vector<double> tcp_to_block_xyz{0.0, 0.0, 0.0};
    std::vector<double> tcp_to_block_rpy{0.0, 0.0, 0.0};
    std::vector<double> refine_grasped_roi_size_m{0.60, 0.40};
    std::vector<double> refine_block_roi_size_m{1.20, 1.00};
  };

public:
  WorldModelNode()
  : Node("block_world_model_node")
  {
    run_pose_cb_group_ = create_callback_group(rclcpp::CallbackGroupType::Reentrant);
    action_client_cb_group_ = create_callback_group(rclcpp::CallbackGroupType::Reentrant);
    const StartupParameters startup = loadStartupParameters();
    const std::string & pipeline_mode_str = startup.pipeline_mode_str;
    const std::string & perception_mode_str = startup.perception_mode_str;

    if (perf_log_every_n_frames_ < 1) {
      perf_log_every_n_frames_ = 1;
    }

    if (debug_detection_overlay_enabled_) {
      det_debug_pub_ = create_publisher<sensor_msgs::msg::Image>("debug/detection_overlay", 1);
    }
    if (debug_refine_grasped_roi_input_enabled_) {
      refine_grasped_roi_input_pub_ =
        create_publisher<sensor_msgs::msg::Image>("debug/refine_grasped_roi_input", 1);
    }
    camera_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
      refine_grasped_camera_info_topic_,
      rclcpp::SensorDataQoS(),
      std::bind(&WorldModelNode::cameraInfoCallback, this, std::placeholders::_1));

    const double tx = getVectorComponent(
      startup.tcp_to_block_xyz, 0, 0.0, "refine_grasped.tcp_to_block.xyz");
    const double ty = getVectorComponent(
      startup.tcp_to_block_xyz, 1, 0.0, "refine_grasped.tcp_to_block.xyz");
    const double tz = getVectorComponent(
      startup.tcp_to_block_xyz, 2, 0.0, "refine_grasped.tcp_to_block.xyz");
    const double rr = getVectorComponent(
      startup.tcp_to_block_rpy, 0, 0.0, "refine_grasped.tcp_to_block.rpy");
    const double rp = getVectorComponent(
      startup.tcp_to_block_rpy, 1, 0.0, "refine_grasped.tcp_to_block.rpy");
    const double ry = getVectorComponent(
      startup.tcp_to_block_rpy, 2, 0.0, "refine_grasped.tcp_to_block.rpy");
    const Eigen::Matrix3d rot_tcp_block =
      (Eigen::AngleAxisd(ry, Eigen::Vector3d::UnitZ()) *
      Eigen::AngleAxisd(rp, Eigen::Vector3d::UnitY()) *
      Eigen::AngleAxisd(rr, Eigen::Vector3d::UnitX())).toRotationMatrix();
    T_tcp_block_ = Eigen::Matrix4d::Identity();
    T_tcp_block_.block<3, 3>(0, 0) = rot_tcp_block;
    T_tcp_block_.block<3, 1>(0, 3) = Eigen::Vector3d(tx, ty, tz);

    roi_size_x_m_ = getVectorComponent(
      startup.refine_grasped_roi_size_m, 0, 0.60, "refine_grasped.roi_size_m");
    roi_size_y_m_ = getVectorComponent(
      startup.refine_grasped_roi_size_m, 1, 0.40, "refine_grasped.roi_size_m");
    refine_block_roi_size_x_m_ =
      getVectorComponent(startup.refine_block_roi_size_m, 0, 1.20, "refine_block.roi_size_m");
    refine_block_roi_size_y_m_ =
      getVectorComponent(startup.refine_block_roi_size_m, 1, 1.00, "refine_block.roi_size_m");
    normalizeConfiguration();

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
    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    segment_client_ = create_client<SegmentSrv>(
      "/yolos_segmentor_service/segment",
      rmw_qos_profile_services_default,
      action_client_cb_group_);
    register_srv_client_ = create_client<RegisterBlockSrv>(
      "/register_block_pose",
      rmw_qos_profile_services_default,
      action_client_cb_group_);
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

    set_block_task_status_srv_ = create_service<SetBlockTaskStatusSrv>(
      "~/set_block_task_status",
      std::bind(
        &WorldModelNode::handleSetBlockTaskStatus,
        this,
        std::placeholders::_1,
        std::placeholders::_2));

    marker_refresh_timer_ = create_wall_timer(
      std::chrono::duration<double>(startup.marker_refresh_period_s),
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
    if (refine_grasped_use_fk_roi_) {
      WM_LOG(
        get_logger(),
        "REFINE_GRASPED FK+ROI enabled | tcp_frame=%s camera_frame_override=%s camera_info_topic=%s roi_size=[%.2f, %.2f]m",
        refine_grasped_tcp_frame_.c_str(),
        refine_grasped_camera_frame_.empty() ? "<image.header.frame_id>" : refine_grasped_camera_frame_.c_str(),
        refine_grasped_camera_info_topic_.c_str(),
        roi_size_x_m_,
        roi_size_y_m_);
      RCLCPP_INFO(
        get_logger(),
        "REFINE_GRASPED segmentation input: background=%s blur_kernel=%d seg_timeout=%.2fs",
        refine_grasped_use_black_bg_ ? "black" : "blur",
        refine_grasped_blur_kernel_size_,
        refine_grasped_segmentation_timeout_s_);
      RCLCPP_INFO(
        get_logger(),
        "REFINE_GRASPED pose fusion: enabled=%s mode=%s max_jump=%.3fm max_z_delta=%.3fm debug_log=%s",
        refine_grasped_pose_fusion_.enabled ? "true" : "false",
        refine_grasped_pose_fusion_.mode.c_str(),
        refine_grasped_pose_fusion_.max_translation_jump_m,
        refine_grasped_pose_fusion_.max_z_delta_m,
        refine_grasped_pose_fusion_.debug_log ? "true" : "false");
    }
    if (refine_block_use_pose_roi_) {
      RCLCPP_INFO(
        get_logger(),
        "REFINE_BLOCK pose+ROI enabled | roi_size=[%.2f, %.2f]m depth=[%.2f, %.2f]m seg_timeout=%.2fs",
        refine_block_roi_size_x_m_,
        refine_block_roi_size_y_m_,
        refine_block_min_depth_m_,
        refine_block_max_depth_m_,
        refine_block_segmentation_timeout_s_);
    }
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
    return cbpwm::resolveGraspedBlockId(persistent_world_, *get_clock());
  }

  cbpwm::AssociationConfig associationConfig() const
  {
    cbpwm::AssociationConfig cfg;
    cfg.association_max_distance_m = association_max_distance_m_;
    cfg.association_max_age_s = association_max_age_s_;
    cfg.min_update_confidence = min_update_confidence_;
    return cfg;
  }

  StartupParameters loadStartupParameters()
  {
    StartupParameters params;

    params.pipeline_mode_str = declare_parameter<std::string>("pipeline_mode", "full");
    pipeline_mode_ = cbpwm::parsePipelineMode(params.pipeline_mode_str);
    params.perception_mode_str = declare_parameter<std::string>("perception_mode", "FULL");

    min_fitness_ = declare_parameter<double>("min_fitness", 0.3);
    max_rmse_ = declare_parameter<double>("max_rmse", 0.05);
    object_class_ = declare_parameter<std::string>("object_class", "concrete_block");
    world_frame_ = declare_parameter<std::string>("world_frame", "world");
    max_sync_delta_s_ = declare_parameter<double>("sync.max_delta_s", 0.06);
    object_timeout_s_ = declare_parameter<double>("world_model.object_timeout_s", 10.0);
    association_max_distance_m_ =
      declare_parameter<double>("world_model.association_max_distance_m", 0.45);
    association_max_age_s_ =
      declare_parameter<double>("world_model.association_max_age_s", 20.0);
    min_update_confidence_ =
      declare_parameter<double>("world_model.min_update_confidence", 0.25);
    refine_target_max_distance_m_ =
      declare_parameter<double>("world_model.refine_target_max_distance_m", 1.2);
    scene_discovery_coarse_fallback_enabled_ =
      declare_parameter<bool>("world_model.scene_discovery_coarse_fallback.enable", true);
    scene_discovery_coarse_fallback_min_points_ =
      declare_parameter<int>("world_model.scene_discovery_coarse_fallback.min_points", 120);
    coarse_surface_square_ratio_thresh_ = declare_parameter<double>(
      "world_model.scene_discovery_coarse_fallback.surface_shape.square_ratio_thresh", 1.35);
    coarse_front_center_offset_square_m_ = declare_parameter<double>(
      "world_model.scene_discovery_coarse_fallback.center_offset.square_m", 0.45);
    coarse_front_center_offset_rect_m_ = declare_parameter<double>(
      "world_model.scene_discovery_coarse_fallback.center_offset.rect_m", 0.30);

    debug_detection_overlay_enabled_ =
      declare_parameter<bool>("debug.publish_detection_overlay", true);
    debug_refine_grasped_roi_input_enabled_ =
      declare_parameter<bool>("debug.publish_refine_grasped_roi_input", true);
    perf_log_timing_enabled_ = declare_parameter<bool>("perf.log_timing", true);
    perf_log_every_n_frames_ = declare_parameter<int>("perf.log_every_n_frames", 20);
    params.marker_refresh_period_s =
      declare_parameter<double>("world_model.marker_refresh_period_s", 0.5);

    refine_grasped_use_fk_roi_ = declare_parameter<bool>("refine_grasped.use_fk_roi", true);
    refine_grasped_tcp_frame_ =
      declare_parameter<std::string>("refine_grasped.tcp_frame", "elastic/K8_tool_center_point");
    refine_grasped_camera_frame_ =
      declare_parameter<std::string>("refine_grasped.camera_frame", "");
    refine_grasped_camera_info_topic_ = declare_parameter<std::string>(
      "refine_grasped.camera_info_topic", "/zed2i/warped/left/camera_info");
    refine_grasped_min_depth_m_ = declare_parameter<double>("refine_grasped.min_depth_m", 0.5);
    refine_grasped_max_depth_m_ = declare_parameter<double>("refine_grasped.max_depth_m", 30.0);
    refine_grasped_segmentation_timeout_s_ =
      declare_parameter<double>("refine_grasped.segmentation_timeout_s", 3.0);
    refine_grasped_use_black_bg_ = declare_parameter<bool>(
      "refine_grasped.segmentation_input.use_black_background", false);
    refine_grasped_blur_kernel_size_ =
      declare_parameter<int>("refine_grasped.segmentation_input.blur_kernel_size", 31);

    refine_block_use_pose_roi_ = declare_parameter<bool>("refine_block.use_pose_roi", false);
    params.refine_block_roi_size_m =
      declare_parameter<std::vector<double>>("refine_block.roi_size_m", {1.20, 1.00});
    refine_block_min_depth_m_ = declare_parameter<double>("refine_block.min_depth_m", 0.5);
    refine_block_max_depth_m_ = declare_parameter<double>("refine_block.max_depth_m", 30.0);
    refine_block_segmentation_timeout_s_ =
      declare_parameter<double>("refine_block.segmentation_timeout_s", 3.0);
    refine_block_use_black_bg_ = declare_parameter<bool>(
      "refine_block.segmentation_input.use_black_background", false);
    refine_block_blur_kernel_size_ =
      declare_parameter<int>("refine_block.segmentation_input.blur_kernel_size", 31);

    refine_grasped_pose_fusion_.enabled =
      declare_parameter<bool>("refine_grasped.pose_fusion.enable", true);
    refine_grasped_pose_fusion_.mode = declare_parameter<std::string>(
      "refine_grasped.pose_fusion.mode",
      "position_from_registration_orientation_from_fk");
    refine_grasped_pose_fusion_.max_translation_jump_m = declare_parameter<double>(
      "refine_grasped.pose_fusion.max_translation_jump_m", 0.35);
    refine_grasped_pose_fusion_.max_z_delta_m = declare_parameter<double>(
      "refine_grasped.pose_fusion.max_z_delta_m", 0.25);
    refine_grasped_pose_fusion_.debug_log =
      declare_parameter<bool>("refine_grasped.pose_fusion.debug_log", true);

    params.tcp_to_block_xyz =
      declare_parameter<std::vector<double>>("refine_grasped.tcp_to_block.xyz", {0.0, 0.0, 0.0});
    params.tcp_to_block_rpy =
      declare_parameter<std::vector<double>>("refine_grasped.tcp_to_block.rpy", {0.0, 0.0, 0.0});
    params.refine_grasped_roi_size_m =
      declare_parameter<std::vector<double>>("refine_grasped.roi_size_m", {0.60, 0.40});

    // Keep for launch-file compatibility; no longer used in one-shot flow.
    (void)declare_parameter<std::string>("calib_yaml", "");
    return params;
  }

  double getVectorComponent(
    const std::vector<double> & values,
    size_t index,
    double fallback,
    const char * param_name)
  {
    if (index < values.size()) {
      return values[index];
    }
    RCLCPP_WARN(
      get_logger(),
      "Parameter '%s' expected at least %zu entries, got %zu. Using fallback %.3f for index %zu.",
      param_name,
      index + 1,
      values.size(),
      fallback,
      index);
    return fallback;
  }

  void normalizeConfiguration()
  {
    auto clamp_min = [this](double & value, double min_value, const char * name) {
        if (value < min_value) {
          RCLCPP_WARN(get_logger(), "Invalid %s=%.3f, clamping to %.3f", name, value, min_value);
          value = min_value;
        }
      };
    auto normalize_blur = [this](int & kernel, const char * name) {
        if (kernel < 1) {
          RCLCPP_WARN(get_logger(), "Invalid %s=%d, clamping to 1", name, kernel);
          kernel = 1;
        }
        if ((kernel % 2) == 0) {
          RCLCPP_WARN(get_logger(), "Invalid %s=%d (must be odd), incrementing to %d", name, kernel, kernel + 1);
          kernel += 1;
        }
      };
    auto clamp_min_i = [this](int & value, int min_value, const char * name) {
        if (value < min_value) {
          RCLCPP_WARN(get_logger(), "Invalid %s=%d, clamping to %d", name, value, min_value);
          value = min_value;
        }
      };

    clamp_min(min_fitness_, 0.0, "min_fitness");
    clamp_min(max_rmse_, 0.0, "max_rmse");
    clamp_min_i(scene_discovery_coarse_fallback_min_points_, 1, "scene_discovery_coarse_fallback.min_points");
    clamp_min(coarse_surface_square_ratio_thresh_, 1.0, "scene_discovery_coarse_fallback.surface_shape.square_ratio_thresh");
    clamp_min(coarse_front_center_offset_square_m_, 0.0, "scene_discovery_coarse_fallback.center_offset.square_m");
    clamp_min(coarse_front_center_offset_rect_m_, 0.0, "scene_discovery_coarse_fallback.center_offset.rect_m");

    clamp_min(roi_size_x_m_, 0.01, "refine_grasped.roi_size_m[0]");
    clamp_min(roi_size_y_m_, 0.01, "refine_grasped.roi_size_m[1]");
    clamp_min(refine_block_roi_size_x_m_, 0.01, "refine_block.roi_size_m[0]");
    clamp_min(refine_block_roi_size_y_m_, 0.01, "refine_block.roi_size_m[1]");
    if (refine_grasped_min_depth_m_ > refine_grasped_max_depth_m_) {
      RCLCPP_WARN(
        get_logger(),
        "Invalid refine_grasped depth range [%.3f, %.3f], swapping bounds",
        refine_grasped_min_depth_m_, refine_grasped_max_depth_m_);
      std::swap(refine_grasped_min_depth_m_, refine_grasped_max_depth_m_);
    }
    if (refine_block_min_depth_m_ > refine_block_max_depth_m_) {
      RCLCPP_WARN(
        get_logger(),
        "Invalid refine_block depth range [%.3f, %.3f], swapping bounds",
        refine_block_min_depth_m_, refine_block_max_depth_m_);
      std::swap(refine_block_min_depth_m_, refine_block_max_depth_m_);
    }
    normalize_blur(
      refine_grasped_blur_kernel_size_, "refine_grasped.segmentation_input.blur_kernel_size");
    normalize_blur(refine_block_blur_kernel_size_, "refine_block.segmentation_input.blur_kernel_size");
  }

  bool buildCoarseBlockFromMaskAndCloud(
    uint32_t detection_id,
    const sensor_msgs::msg::Image & mask_msg,
    const sensor_msgs::msg::PointCloud2 & cloud_msg,
    const std_msgs::msg::Header & header,
    const Eigen::Vector3d * camera_origin_world,
    Block & out_block,
    std::string & reason) const
  {
    cbpwm::CoarsePoseInput in;
    in.detection_id = detection_id;
    in.header = header;
    in.mask = toCvMono(mask_msg);
    in.camera_origin_world = camera_origin_world;

    cbpwm::CoarsePoseConfig cfg;
    cfg.min_points = scene_discovery_coarse_fallback_min_points_;
    cfg.square_ratio_thresh = coarse_surface_square_ratio_thresh_;
    cfg.front_center_offset_square_m = coarse_front_center_offset_square_m_;
    cfg.front_center_offset_rect_m = coarse_front_center_offset_rect_m_;
    cfg.min_confidence = std::max(0.3F, static_cast<float>(min_update_confidence_));

    return cbpwm::buildCoarseBlockFromOrganizedCloud(in, cloud_msg, cfg, out_block, reason);
  }

  bool buildCoarseBlockFromCloudCentroid(
    uint32_t detection_id,
    const sensor_msgs::msg::Image & mask_msg,
    const sensor_msgs::msg::PointCloud2 & cutout_cloud_msg,
    const std_msgs::msg::Header & header,
    const Eigen::Vector3d * camera_origin_world,
    Block & out_block,
    std::string & reason) const
  {
    cbpwm::CoarsePoseInput in;
    in.detection_id = detection_id;
    in.header = header;
    in.mask = toCvMono(mask_msg);
    in.camera_origin_world = camera_origin_world;

    cbpwm::CoarsePoseConfig cfg;
    cfg.min_points = scene_discovery_coarse_fallback_min_points_;
    cfg.square_ratio_thresh = coarse_surface_square_ratio_thresh_;
    cfg.front_center_offset_square_m = coarse_front_center_offset_square_m_;
    cfg.front_center_offset_rect_m = coarse_front_center_offset_rect_m_;
    cfg.min_confidence = std::max(0.3F, static_cast<float>(min_update_confidence_));

    return cbpwm::buildCoarseBlockFromCutoutCloud(in, cutout_cloud_msg, cfg, out_block, reason);
  }

  bool runRegistrationServiceCutoutSync(
    const sensor_msgs::msg::Image & mask,
    const sensor_msgs::msg::PointCloud2 & cloud,
    double timeout_s,
    sensor_msgs::msg::PointCloud2 & out_cutout_cloud,
    std::string & reason)
  {
    if (!register_srv_client_ || !register_srv_client_->service_is_ready()) {
      reason = "register_block_pose service unavailable";
      return false;
    }

    auto req = std::make_shared<RegisterBlockSrv::Request>();
    req->mask = mask;
    req->cloud = cloud;
    req->object_class = object_class_;

    auto future = register_srv_client_->async_send_request(req);
    const auto ret = future.wait_for(std::chrono::duration<double>(timeout_s));
    if (ret != std::future_status::ready) {
      reason = "register_block_pose service timeout";
      return false;
    }
    const auto res = future.get();
    if (!res) {
      reason = "empty register_block_pose response";
      return false;
    }
    if (res->cutout_cloud.data.empty()) {
      reason = "register_block_pose returned empty cutout_cloud";
      return false;
    }
    out_cutout_cloud = res->cutout_cloud;
    reason = "cutout_cloud received from register_block_pose";
    return true;
  }


  bool lookupFrameOriginInWorld(
    const std_msgs::msg::Header & stamped_header,
    Eigen::Vector3d & origin_world,
    std::string & reason)
  {
    if (!tf_buffer_) {
      reason = "TF buffer unavailable";
      return false;
    }
    std::string source_frame = stamped_header.frame_id;
    if (source_frame.empty()) {
      if (!resolveCameraFrame(stamped_header, source_frame, reason)) {
        return false;
      }
    }
    try {
      const auto tf_w_s = tf_buffer_->lookupTransform(
        world_frame_, source_frame, stamped_header.stamp, tf2::durationFromSec(0.1));
      origin_world =
        Eigen::Vector3d(tf_w_s.transform.translation.x, tf_w_s.transform.translation.y,
        tf_w_s.transform.translation.z);
      return true;
    } catch (const tf2::TransformException & ex) {
      reason = std::string("TF lookup failed for camera origin (") + world_frame_ + " <- " +
        source_frame + "): " + ex.what();
      return false;
    }
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
    std::string & reason,
    const std::string & object_class_override = "")
  {
    RegisterBlock::Goal goal;
    goal.mask = mask;
    goal.cloud = cloud;
    goal.object_class = object_class_override.empty() ? object_class_ : object_class_override;

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

  bool runSegmentationSync(
    const sensor_msgs::msg::Image & image,
    double timeout_s,
    SegmentSrv::Response::SharedPtr & out_response,
    std::string & reason)
  {
    if (!segment_client_ || !segment_client_->service_is_ready()) {
      reason = "segmentation service unavailable";
      return false;
    }

    auto seg_req = std::make_shared<SegmentSrv::Request>();
    seg_req->image = image;
    seg_req->return_debug = false;

    auto future = segment_client_->async_send_request(seg_req);
    const auto ret = future.wait_for(std::chrono::duration<double>(timeout_s));
    if (ret != std::future_status::ready) {
      reason = "segmentation timeout";
      return false;
    }

    out_response = future.get();
    if (!out_response) {
      reason = "empty segmentation response";
      return false;
    }
    if (!out_response->success) {
      reason = "segmentation returned success=false";
      return false;
    }

    return true;
  }

  bool trySceneDiscoveryCoarseFallback(
    uint32_t detection_id,
    const std::string & registration_reason,
    const sensor_msgs::msg::Image & mask,
    const sensor_msgs::msg::PointCloud2 & cloud,
    const cbpwm::SceneFlowRequest & run_request,
    const Eigen::Vector3d * camera_origin_world,
    cbpwm::RegistrationCounters & counters)
  {
    if (run_request.mode != cbpwm::OneShotMode::kSceneDiscovery ||
      !scene_discovery_coarse_fallback_enabled_)
    {
      return false;
    }

    Block coarse_block;
    std::string coarse_reason;
    const bool coarse_ok = buildCoarseBlockFromMaskAndCloud(
      detection_id,
      mask,
      cloud,
      cloud.header,
      camera_origin_world,
      coarse_block,
      coarse_reason);
    if (!coarse_ok) {
      sensor_msgs::msg::PointCloud2 cutout_cloud;
      std::string cutout_reason;
      const bool got_cutout = runRegistrationServiceCutoutSync(
        mask, cloud, run_request.registration_timeout_s, cutout_cloud, cutout_reason);
      if (!got_cutout) {
        RCLCPP_WARN(
          get_logger(),
          "Registration rejected for block_%u (%s). Coarse fallback unavailable: %s",
          detection_id,
          registration_reason.c_str(),
          cutout_reason.c_str());
        return false;
      }

      const bool coarse_from_cutout_ok = buildCoarseBlockFromCloudCentroid(
        detection_id,
        mask,
        cutout_cloud,
        cutout_cloud.header,
        camera_origin_world,
        coarse_block,
        coarse_reason);
      if (!coarse_from_cutout_ok) {
        RCLCPP_WARN(
          get_logger(),
          "Registration rejected for block_%u (%s). Coarse fallback unavailable: %s",
          detection_id,
          registration_reason.c_str(),
          coarse_reason.c_str());
        return false;
      }
    }

    std::lock_guard<std::mutex> lock(persistent_world_mutex_);
    std::string assigned_id;
    std::string upsert_reason;
    const bool upsert_ok = cbpwm::upsertRegisteredBlock(
      persistent_world_,
      world_block_counter_,
      coarse_block,
      run_request.mode,
      run_request.target_block_id,
      cloud.header,
      *get_clock(),
      associationConfig(),
      assigned_id,
      upsert_reason);
    if (!upsert_ok) {
      RCLCPP_WARN(
        get_logger(),
        "Registration rejected for block_%u (%s). Coarse fallback rejected: %s",
        detection_id,
        registration_reason.c_str(),
        upsert_reason.c_str());
      return false;
    }

    ++counters.coarse_upserts_ok;
    RCLCPP_WARN(
      get_logger(),
      "Registration rejected for block_%u (%s). Coarse fallback accepted: incoming=%s assigned=%s",
      detection_id,
      registration_reason.c_str(),
      coarse_block.id.c_str(),
      assigned_id.c_str());
    return true;
  }

  

  static Eigen::Matrix4d transformToEigen(const geometry_msgs::msg::TransformStamped & tf)
  {
    Eigen::Quaterniond q(
      tf.transform.rotation.w,
      tf.transform.rotation.x,
      tf.transform.rotation.y,
      tf.transform.rotation.z);
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.block<3, 3>(0, 0) = q.normalized().toRotationMatrix();
    T(0, 3) = tf.transform.translation.x;
    T(1, 3) = tf.transform.translation.y;
    T(2, 3) = tf.transform.translation.z;
    return T;
  }

  void cameraInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg)
  {
    if (msg->k.size() < 9) {
      return;
    }
    CameraIntrinsics intr;
    intr.fx = msg->k[0];
    intr.fy = msg->k[4];
    intr.cx = msg->k[2];
    intr.cy = msg->k[5];
    intr.width = msg->width;
    intr.height = msg->height;
    intr.valid = intr.fx > 0.0 && intr.fy > 0.0;

    if (!intr.valid) {
      return;
    }
    std::lock_guard<std::mutex> lock(camera_info_mutex_);
    camera_intrinsics_ = intr;
    camera_info_frame_id_ = msg->header.frame_id;
  }

  bool lookupPredictedGraspedPose(
    const std_msgs::msg::Header & header,
    Eigen::Vector3d & p_world,
    Eigen::Vector3d & p_camera,
    Eigen::Quaterniond & q_world,
    std::string & reason)
  {
    if (!tf_buffer_) {
      reason = "TF buffer not initialized";
      return false;
    }

    try {
      const auto tf_world_tcp = tf_buffer_->lookupTransform(
        world_frame_,
        refine_grasped_tcp_frame_,
        rclcpp::Time(header.stamp),
        rclcpp::Duration::from_seconds(0.2));
      std::string camera_frame = refine_grasped_camera_frame_;
      if (camera_frame.empty()) {
        camera_frame = header.frame_id;
      }
      if (camera_frame.empty()) {
        std::lock_guard<std::mutex> lock(camera_info_mutex_);
        camera_frame = camera_info_frame_id_;
      }
      if (camera_frame.empty()) {
        reason = "camera frame unresolved (no override, image frame, or camera_info frame)";
        return false;
      }

      const auto tf_camera_world = tf_buffer_->lookupTransform(
        camera_frame,
        world_frame_,
        rclcpp::Time(header.stamp),
        rclcpp::Duration::from_seconds(0.2));

      const Eigen::Matrix4d T_world_tcp = transformToEigen(tf_world_tcp);
      const Eigen::Matrix4d T_camera_world = transformToEigen(tf_camera_world);
      const Eigen::Matrix4d T_world_block_pred = T_world_tcp * T_tcp_block_;
      p_world = T_world_block_pred.block<3, 1>(0, 3);
      q_world = Eigen::Quaterniond(T_world_block_pred.block<3, 3>(0, 0)).normalized();
      const Eigen::Vector4d p_block_world_h(
        p_world.x(),
        p_world.y(),
        p_world.z(),
        1.0);
      const Eigen::Vector4d p_block_camera_h = T_camera_world * p_block_world_h;

      p_camera = p_block_camera_h.head<3>();
      return true;
    } catch (const tf2::TransformException & ex) {
      reason = std::string("TF lookup failed: ") + ex.what();
      return false;
    }
  }

  bool resolveCameraFrame(
    const std_msgs::msg::Header & header,
    std::string & camera_frame,
    std::string & reason)
  {
    camera_frame = refine_grasped_camera_frame_;
    if (camera_frame.empty()) {
      camera_frame = header.frame_id;
    }
    if (camera_frame.empty()) {
      std::lock_guard<std::mutex> lock(camera_info_mutex_);
      camera_frame = camera_info_frame_id_;
    }
    if (camera_frame.empty()) {
      reason = "camera frame unresolved (no override, image frame, or camera_info frame)";
      return false;
    }
    return true;
  }

  bool worldPointToCamera(
    const std_msgs::msg::Header & header,
    const Eigen::Vector3d & p_world,
    Eigen::Vector3d & p_camera,
    std::string & reason)
  {
    if (!tf_buffer_) {
      reason = "TF buffer not initialized";
      return false;
    }

    std::string camera_frame;
    if (!resolveCameraFrame(header, camera_frame, reason)) {
      return false;
    }

    try {
      const auto tf_camera_world = tf_buffer_->lookupTransform(
        camera_frame,
        world_frame_,
        rclcpp::Time(header.stamp),
        rclcpp::Duration::from_seconds(0.2));
      const Eigen::Matrix4d T_camera_world = transformToEigen(tf_camera_world);
      const Eigen::Vector4d p_world_h(p_world.x(), p_world.y(), p_world.z(), 1.0);
      const Eigen::Vector4d p_camera_h = T_camera_world * p_world_h;
      p_camera = p_camera_h.head<3>();
      return true;
    } catch (const tf2::TransformException & ex) {
      reason = std::string("TF lookup failed: ") + ex.what();
      return false;
    }
  }

  cbpwm::RefineFlowRuntime makeRefineFlowRuntime()
  {
    cbpwm::RefineFlowRuntime rt;
    rt.logger = get_logger();
    rt.registration_ready = [this]() {
        return action_client_ && action_client_->action_server_is_ready();
      };
    rt.publish_persistent_world = [this](const std_msgs::msg::Header & header) {
        publishPersistentWorld(header);
      };
    rt.complete_one_shot = [this](uint64_t sequence, bool success, const std::string & message) {
        completeOneShotRequest(sequence, success, message);
      };
    rt.reset_busy = [this]() {resetBusy();};
    rt.record_timing = [this](int64_t seg_ms, int64_t track_ms, int64_t reg_ms, int64_t total_ms) {
        recordTiming(seg_ms, track_ms, reg_ms, total_ms);
      };
    rt.publish_debug_overlay = [this](const sensor_msgs::msg::Image & image_msg) {
        if (det_debug_pub_) {
          det_debug_pub_->publish(image_msg);
        }
      };
    rt.publish_roi_input = [this](const sensor_msgs::msg::Image & image_msg) {
        if (refine_grasped_roi_input_pub_) {
          refine_grasped_roi_input_pub_->publish(image_msg);
        }
      };
    rt.get_expected_target =
      [this](const std::string & target_id, Block & out_target) {
        std::lock_guard<std::mutex> lock(persistent_world_mutex_);
        const auto it = persistent_world_.find(target_id);
        if (it == persistent_world_.end()) {
          return false;
        }
        out_target = it->second;
        return true;
      };
    rt.get_projection_intrinsics = [this](cbpwm::ProjectionIntrinsics & out_intr) {
        std::lock_guard<std::mutex> lock(camera_info_mutex_);
        out_intr.valid = camera_intrinsics_.valid;
        out_intr.fx = camera_intrinsics_.fx;
        out_intr.fy = camera_intrinsics_.fy;
        out_intr.cx = camera_intrinsics_.cx;
        out_intr.cy = camera_intrinsics_.cy;
        out_intr.width = camera_intrinsics_.width;
        out_intr.height = camera_intrinsics_.height;
        return out_intr.valid;
      };
    rt.lookup_predicted_grasped_pose =
      [this](
      const std_msgs::msg::Header & header,
      Eigen::Vector3d & p_world,
      Eigen::Vector3d & p_camera,
      Eigen::Quaterniond & q_world,
      std::string & reason) {
        return lookupPredictedGraspedPose(header, p_world, p_camera, q_world, reason);
      };
    rt.world_point_to_camera =
      [this](
      const std_msgs::msg::Header & header,
      const Eigen::Vector3d & p_world,
      Eigen::Vector3d & p_camera,
      std::string & reason) {
        return worldPointToCamera(header, p_world, p_camera, reason);
      };
    rt.run_segmentation_sync =
      [this](
      const sensor_msgs::msg::Image & in_image,
      double timeout_s,
      SegmentSrv::Response::SharedPtr & out_response,
      std::string & out_reason) {
        return runSegmentationSync(in_image, timeout_s, out_response, out_reason);
      };
    rt.run_registration_sync =
      [this](
      uint32_t detection_id,
      const sensor_msgs::msg::Image & mask,
      const sensor_msgs::msg::PointCloud2 & cloud,
      const std_msgs::msg::Header & header,
      double timeout_s,
      Block & out_block,
      std::string & out_reason,
      const std::string & object_class_override) {
        return runRegistrationSync(
          detection_id,
          mask,
          cloud,
          header,
          timeout_s,
          out_block,
          out_reason,
          object_class_override);
      };
    rt.upsert_block = [](Block &, std::string &, std::string &) {
        return false;
      };
    return rt;
  }

  void processRefineGraspedWithFkRoi(
    const sensor_msgs::msg::Image::ConstSharedPtr & image,
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr & cloud,
    const OneShotRequest & run_request,
    const std::chrono::steady_clock::time_point & t_start)
  {
    cbpwm::RefineRequest req;
    req.sequence = run_request.sequence;
    req.target_block_id = run_request.target_block_id;
    req.registration_timeout_s = run_request.registration_timeout_s;

    cbpwm::RefineGraspedConfig cfg;
    cfg.roi_cfg.roi_size_x_m = roi_size_x_m_;
    cfg.roi_cfg.roi_size_y_m = roi_size_y_m_;
    cfg.roi_cfg.min_depth_m = refine_grasped_min_depth_m_;
    cfg.roi_cfg.max_depth_m = refine_grasped_max_depth_m_;
    cfg.roi_cfg.segmentation_timeout_s = refine_grasped_segmentation_timeout_s_;
    cfg.roi_cfg.use_black_bg = refine_grasped_use_black_bg_;
    cfg.roi_cfg.blur_kernel_size = refine_grasped_blur_kernel_size_;
    cfg.debug_detection_overlay_enabled = debug_detection_overlay_enabled_;
    cfg.debug_refine_grasped_roi_input_enabled = debug_refine_grasped_roi_input_enabled_;
    cfg.object_class = object_class_;
    cfg.pose_fusion.enabled = refine_grasped_pose_fusion_.enabled;
    cfg.pose_fusion.mode = refine_grasped_pose_fusion_.mode;
    cfg.pose_fusion.max_translation_jump_m = refine_grasped_pose_fusion_.max_translation_jump_m;
    cfg.pose_fusion.max_z_delta_m = refine_grasped_pose_fusion_.max_z_delta_m;
    cfg.pose_fusion.debug_log = refine_grasped_pose_fusion_.debug_log;

    auto rt = makeRefineFlowRuntime();
    rt.upsert_block = [this, &run_request, cloud](Block & block, std::string & assigned_id, std::string & reason) {
        std::lock_guard<std::mutex> lock(persistent_world_mutex_);
        return cbpwm::upsertRegisteredBlock(
          persistent_world_,
          world_block_counter_,
          block,
          run_request.mode,
          run_request.target_block_id,
          cloud->header,
          *get_clock(),
          associationConfig(),
          assigned_id,
          reason);
      };

    cbpwm::processRefineGraspedWithFkRoi(req, cfg, rt, image, cloud, t_start);
  }

  bool tryProcessRefineBlockWithPoseRoi(
    const sensor_msgs::msg::Image::ConstSharedPtr & image,
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr & cloud,
    const OneShotRequest & run_request,
    const std::chrono::steady_clock::time_point & t_start)
  {
    cbpwm::RefineRequest req;
    req.sequence = run_request.sequence;
    req.target_block_id = run_request.target_block_id;
    req.registration_timeout_s = run_request.registration_timeout_s;

    cbpwm::RefineBlockConfig cfg;
    cfg.use_pose_roi = refine_block_use_pose_roi_;
    cfg.roi_cfg.roi_size_x_m = refine_block_roi_size_x_m_;
    cfg.roi_cfg.roi_size_y_m = refine_block_roi_size_y_m_;
    cfg.roi_cfg.min_depth_m = refine_block_min_depth_m_;
    cfg.roi_cfg.max_depth_m = refine_block_max_depth_m_;
    cfg.roi_cfg.segmentation_timeout_s = refine_block_segmentation_timeout_s_;
    cfg.roi_cfg.use_black_bg = refine_block_use_black_bg_;
    cfg.roi_cfg.blur_kernel_size = refine_block_blur_kernel_size_;
    cfg.refine_target_max_distance_m = refine_target_max_distance_m_;
    cfg.debug_detection_overlay_enabled = debug_detection_overlay_enabled_;

    auto rt = makeRefineFlowRuntime();
    rt.upsert_block = [this, &run_request, cloud](Block & block, std::string & assigned_id, std::string & reason) {
        std::lock_guard<std::mutex> lock(persistent_world_mutex_);
        return cbpwm::upsertRegisteredBlock(
          persistent_world_,
          world_block_counter_,
          block,
          run_request.mode,
          run_request.target_block_id,
          cloud->header,
          *get_clock(),
          associationConfig(),
          assigned_id,
          reason);
      };

    return cbpwm::tryProcessRefineBlockWithPoseRoi(req, cfg, rt, image, cloud, t_start);
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

  static bool isKnownTaskStatus(int32_t task_status)
  {
    return task_status == Block::TASK_UNKNOWN ||
           task_status == Block::TASK_FREE ||
           task_status == Block::TASK_MOVE ||
           task_status == Block::TASK_PLACED ||
           task_status == Block::TASK_REMOVED;
  }

  void handleSetBlockTaskStatus(
    const std::shared_ptr<SetBlockTaskStatusSrv::Request> request,
    std::shared_ptr<SetBlockTaskStatusSrv::Response> response)
  {
    const std::string block_id = request->block_id;
    const int32_t target_task_status = request->task_status;

    if (block_id.empty()) {
      response->success = false;
      response->message = "block_id must not be empty.";
      RCLCPP_WARN(get_logger(), "SetBlockTaskStatus rejected: %s", response->message.c_str());
      return;
    }
    if (!isKnownTaskStatus(target_task_status)) {
      response->success = false;
      response->message = "Unsupported task_status: " + std::to_string(target_task_status);
      RCLCPP_WARN(get_logger(), "SetBlockTaskStatus rejected: %s", response->message.c_str());
      return;
    }

    std_msgs::msg::Header publish_header;
    publish_header.stamp = now();
    {
      const auto snapshot = latestWorldSnapshot();
      publish_header.frame_id = snapshot.header.frame_id.empty() ? world_frame_ : snapshot.header.frame_id;
    }

    int32_t prev_task_status = Block::TASK_UNKNOWN;
    {
      std::lock_guard<std::mutex> lock(persistent_world_mutex_);
      const auto it = persistent_world_.find(block_id);
      if (it == persistent_world_.end()) {
        response->success = false;
        response->message = "Unknown block_id: " + block_id;
        RCLCPP_WARN(get_logger(), "SetBlockTaskStatus rejected: %s", response->message.c_str());
        return;
      }

      prev_task_status = it->second.task_status;
      if (!cbpwm::isValidTaskTransition(prev_task_status, target_task_status)) {
        response->success = false;
        response->message =
          std::string("Invalid task transition: ") +
          cbpwm::taskStatusToString(prev_task_status) + " -> " +
          cbpwm::taskStatusToString(target_task_status);
        RCLCPP_WARN(
          get_logger(),
          "SetBlockTaskStatus rejected for block '%s': %s",
          block_id.c_str(),
          response->message.c_str());
        return;
      }

      it->second.task_status = target_task_status;
      // Keep block alive and reflect semantic state change immediately.
      it->second.last_seen = publish_header.stamp;
    }

    publishPersistentWorld(publish_header);

    response->success = true;
    response->message =
      std::string("Updated block '") + block_id + "' task_status to " +
      cbpwm::taskStatusToString(target_task_status);
    RCLCPP_INFO(
      get_logger(),
      "SetBlockTaskStatus applied: block '%s' %s -> %s",
      block_id.c_str(),
      cbpwm::taskStatusToString(prev_task_status),
      cbpwm::taskStatusToString(target_task_status));
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

  void handleOneShotSegmentationResponse(
    rclcpp::Client<SegmentSrv>::SharedFuture seg_future,
    const sensor_msgs::msg::Image::ConstSharedPtr & image,
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr & cloud,
    std::chrono::steady_clock::time_point t_start,
    const OneShotRequest & run_request)
  {
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

      auto candidates = cbpwm::buildRegistrationCandidates(
        *seg_res, run_request.mode, run_request.target_block_id);

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
      cbpwm::RegistrationCounters counters;
      Eigen::Vector3d camera_origin_world = Eigen::Vector3d::Zero();
      std::string camera_origin_reason;
      const bool have_camera_origin = lookupFrameOriginInWorld(
        cloud->header, camera_origin_world, camera_origin_reason);
      if (!have_camera_origin) {
        RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 3000,
          "Scene-discovery coarse center offset disabled for this frame: %s",
          camera_origin_reason.c_str());
      }
      cbpwm::SceneFlowRequest scene_req;
      scene_req.mode = run_request.mode;
      scene_req.target_block_id = run_request.target_block_id;
      scene_req.registration_timeout_s = run_request.registration_timeout_s;
      scene_req.refine_target_max_distance_m = refine_target_max_distance_m_;

      cbpwm::SceneFlowRuntime scene_rt;
      scene_rt.logger = get_logger();
      scene_rt.run_registration =
        [this](
        uint32_t detection_id,
        const sensor_msgs::msg::Image & mask,
        const sensor_msgs::msg::PointCloud2 & in_cloud,
        const std_msgs::msg::Header & header,
        double timeout_s,
        Block & out_block,
        std::string & out_reason) {
          return runRegistrationSync(
            detection_id,
            mask,
            in_cloud,
            header,
            timeout_s,
            out_block,
            out_reason);
        };
      scene_rt.upsert_block =
        [this, &run_request, cloud](Block & block, std::string & assigned_id, std::string & reason) {
          std::lock_guard<std::mutex> lock(persistent_world_mutex_);
          return cbpwm::upsertRegisteredBlock(
            persistent_world_,
            world_block_counter_,
            block,
            run_request.mode,
            run_request.target_block_id,
            cloud->header,
            *get_clock(),
            associationConfig(),
            assigned_id,
            reason);
        };
      scene_rt.get_expected_target =
        [this](const std::string & target_id, Block & out_target) {
          std::lock_guard<std::mutex> lock(persistent_world_mutex_);
          const auto it = persistent_world_.find(target_id);
          if (it == persistent_world_.end()) {
            return false;
          }
          out_target = it->second;
          return true;
        };
      scene_rt.try_coarse_fallback =
        [this, have_camera_origin, &camera_origin_world](
        uint32_t detection_id,
        const std::string & registration_reason,
        const sensor_msgs::msg::Image & mask,
        const sensor_msgs::msg::PointCloud2 & in_cloud,
        const cbpwm::SceneFlowRequest & request,
        cbpwm::RegistrationCounters & out_counters) {
          return trySceneDiscoveryCoarseFallback(
            detection_id,
            registration_reason,
            mask,
            in_cloud,
            request,
            have_camera_origin ? &camera_origin_world : nullptr,
            out_counters);
        };

      cbpwm::processRegistrationCandidates(
        candidates,
        *cloud,
        scene_req,
        scene_rt,
        counters);

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
        counters.registrations_ok > 0 || run_request.mode == cbpwm::OneShotMode::kSceneDiscovery,
        "Pose estimation finished (detections=" + std::to_string(seg_detection_count) +
        ", tracked=" + std::to_string(tracked_count) +
        ", registration_candidates=" + std::to_string(registration_candidates) +
        ", registrations=" + std::to_string(counters.registrations_ok) +
        ", coarse_registrations=" + std::to_string(counters.coarse_upserts_ok) + ").");

      resetBusy();
    } catch (const std::exception & e) {
      RCLCPP_ERROR(get_logger(), "Segmentation stage failed: %s", e.what());
      completeOneShotRequest(run_request.sequence, false, "Segmentation stage exception.");
      resetBusy();
    }
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

    if (run_request.mode == cbpwm::OneShotMode::kRefineGrasped && refine_grasped_use_fk_roi_) {
      processRefineGraspedWithFkRoi(image, cloud, run_request, t_start);
      return;
    }
    if (run_request.mode == cbpwm::OneShotMode::kRefineBlock &&
      tryProcessRefineBlockWithPoseRoi(image, cloud, run_request, t_start))
    {
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
        handleOneShotSegmentationResponse(seg_future, image, cloud, t_start, run_request);
      });
  }

  message_filters::Subscriber<sensor_msgs::msg::Image> image_sub_;
  message_filters::Subscriber<sensor_msgs::msg::PointCloud2> cloud_sub_;
  std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_;

  rclcpp::Client<SegmentSrv>::SharedPtr segment_client_;
  rclcpp::Client<RegisterBlockSrv>::SharedPtr register_srv_client_;
  rclcpp_action::Client<RegisterBlock>::SharedPtr action_client_;
  rclcpp::Service<SetModeSrv>::SharedPtr set_mode_srv_;
  rclcpp::Service<SetBlockTaskStatusSrv>::SharedPtr set_block_task_status_srv_;
  rclcpp::Service<GetCoarseSrv>::SharedPtr get_coarse_srv_;
  rclcpp::Service<RunPoseSrv>::SharedPtr run_pose_srv_;
  rclcpp::CallbackGroup::SharedPtr run_pose_cb_group_;
  rclcpp::CallbackGroup::SharedPtr action_client_cb_group_;
  rclcpp::TimerBase::SharedPtr marker_refresh_timer_;

  rclcpp::Publisher<BlockArray>::SharedPtr world_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr det_debug_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr refine_grasped_roi_input_pub_;

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
  double association_max_distance_m_{0.45};
  double association_max_age_s_{20.0};
  double min_update_confidence_{0.25};
  double refine_target_max_distance_m_{1.2};
  bool debug_detection_overlay_enabled_{true};
  bool debug_refine_grasped_roi_input_enabled_{true};
  bool perf_log_timing_enabled_{true};
  int perf_log_every_n_frames_{20};
  uint64_t world_block_counter_{0};

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

  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  std::mutex camera_info_mutex_;
  CameraIntrinsics camera_intrinsics_;
  std::string camera_info_frame_id_;
  bool refine_grasped_use_fk_roi_{true};
  std::string refine_grasped_tcp_frame_{"elastic/K8_tool_center_point"};
  std::string refine_grasped_camera_frame_{};
  std::string refine_grasped_camera_info_topic_{"/zed2i/warped/left/camera_info"};
  Eigen::Matrix4d T_tcp_block_{Eigen::Matrix4d::Identity()};
  double roi_size_x_m_{0.60};
  double roi_size_y_m_{0.40};
  double refine_grasped_min_depth_m_{0.5};
  double refine_grasped_max_depth_m_{30.0};
  double refine_grasped_segmentation_timeout_s_{3.0};
  bool refine_grasped_use_black_bg_{false};
  int refine_grasped_blur_kernel_size_{31};
  bool refine_block_use_pose_roi_{false};
  double refine_block_roi_size_x_m_{1.20};
  double refine_block_roi_size_y_m_{1.00};
  double refine_block_min_depth_m_{0.5};
  double refine_block_max_depth_m_{30.0};
  double refine_block_segmentation_timeout_s_{3.0};
  bool refine_block_use_black_bg_{false};
  int refine_block_blur_kernel_size_{31};
  bool scene_discovery_coarse_fallback_enabled_{true};
  int scene_discovery_coarse_fallback_min_points_{120};
  double coarse_surface_square_ratio_thresh_{1.35};
  double coarse_front_center_offset_square_m_{0.45};
  double coarse_front_center_offset_rect_m_{0.30};
  PoseFusionConfig refine_grasped_pose_fusion_;

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
