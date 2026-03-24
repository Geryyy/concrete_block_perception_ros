#include "concrete_block_perception/nodes/perception_orchestrator_node.hpp"

#include "concrete_block_perception/world_model/config_loader.hpp"
#include "concrete_block_perception/utils/world_model_utils.hpp"

#include <tf2/LinearMath/Quaternion.h>

#define WM_LOG(logger, ...) RCLCPP_INFO(logger, __VA_ARGS__)

PerceptionOrchestratorNode::PerceptionOrchestratorNode()
: Node("block_world_model_node")
{
    run_pose_cb_group_ = create_callback_group(rclcpp::CallbackGroupType::Reentrant);
    action_client_cb_group_ = create_callback_group(rclcpp::CallbackGroupType::Reentrant);
    auto startup = cbpwm::loadWorldModelConfig(*this);
    cbpwm::normalizeWorldModelConfig(get_logger(), startup);
    pipeline_mode_ = cbpwm::parsePipelineMode(startup.pipeline_mode_str);
    const std::string & pipeline_mode_str = startup.pipeline_mode_str;
    const std::string & perception_mode_str = startup.perception_mode_str;
    runtime_cfg_.min_fitness = startup.min_fitness;
    runtime_cfg_.max_rmse = startup.max_rmse;
    object_class_ = startup.object_class;
    world_frame_ = startup.world_frame;
    runtime_cfg_.max_sync_delta_s = startup.max_sync_delta_s;
    runtime_cfg_.object_timeout_s = startup.object_timeout_s;
    runtime_cfg_.association_max_distance_m = startup.association_max_distance_m;
    runtime_cfg_.association_max_age_s = startup.association_max_age_s;
    runtime_cfg_.min_update_confidence = startup.min_update_confidence;
    runtime_cfg_.refine_target_max_distance_m = startup.refine_target_max_distance_m;
    scene_discovery_coarse_fallback_enabled_ = startup.scene_discovery_coarse_fallback_enabled;
    scene_discovery_coarse_fallback_min_points_ = startup.scene_discovery_coarse_fallback_min_points;
    coarse_surface_square_ratio_thresh_ = startup.coarse_surface_square_ratio_thresh;
    coarse_front_center_offset_square_m_ = startup.coarse_front_center_offset_square_m;
    coarse_front_center_offset_rect_m_ = startup.coarse_front_center_offset_rect_m;
    debug_detection_overlay_enabled_ = startup.debug_detection_overlay_enabled;
    debug_refine_grasped_roi_input_enabled_ = startup.debug_refine_grasped_roi_input_enabled;
    perf_log_timing_enabled_ = startup.perf_log_timing_enabled;
    perf_log_every_n_frames_ = startup.perf_log_every_n_frames;
    refine_grasped_use_fk_roi_ = startup.refine_grasped_use_fk_roi;
    refine_grasped_tcp_frame_ = startup.refine_grasped_tcp_frame;
    refine_grasped_camera_frame_ = startup.refine_grasped_camera_frame;
    refine_grasped_camera_info_topic_ = startup.refine_grasped_camera_info_topic;
    refine_grasped_min_depth_m_ = startup.refine_grasped_min_depth_m;
    refine_grasped_max_depth_m_ = startup.refine_grasped_max_depth_m;
    refine_grasped_segmentation_timeout_s_ = startup.refine_grasped_segmentation_timeout_s;
    refine_grasped_use_black_bg_ = startup.refine_grasped_use_black_bg;
    refine_grasped_blur_kernel_size_ = startup.refine_grasped_blur_kernel_size;
    refine_grasped_pose_fusion_.enabled = startup.refine_grasped_pose_fusion.enabled;
    refine_grasped_pose_fusion_.mode = startup.refine_grasped_pose_fusion.mode;
    refine_grasped_pose_fusion_.max_translation_jump_m =
      startup.refine_grasped_pose_fusion.max_translation_jump_m;
    refine_grasped_pose_fusion_.max_z_delta_m = startup.refine_grasped_pose_fusion.max_z_delta_m;
    refine_grasped_pose_fusion_.debug_log = startup.refine_grasped_pose_fusion.debug_log;
    refine_block_use_pose_roi_ = startup.refine_block_use_pose_roi;
    refine_block_min_depth_m_ = startup.refine_block_min_depth_m;
    refine_block_max_depth_m_ = startup.refine_block_max_depth_m;
    refine_block_segmentation_timeout_s_ = startup.refine_block_segmentation_timeout_s;
    refine_block_use_black_bg_ = startup.refine_block_use_black_bg;
    refine_block_blur_kernel_size_ = startup.refine_block_blur_kernel_size;

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
      std::bind(&PerceptionOrchestratorNode::cameraInfoCallback, this, std::placeholders::_1));

    const double tx = cbpwm::vectorComponent(
      get_logger(),
      startup.refine_grasped_tcp_to_block_xyz,
      0,
      0.0,
      "refine_grasped.tcp_to_block.xyz");
    const double ty = cbpwm::vectorComponent(
      get_logger(),
      startup.refine_grasped_tcp_to_block_xyz,
      1,
      0.0,
      "refine_grasped.tcp_to_block.xyz");
    const double tz = cbpwm::vectorComponent(
      get_logger(),
      startup.refine_grasped_tcp_to_block_xyz,
      2,
      0.0,
      "refine_grasped.tcp_to_block.xyz");
    const double rr = cbpwm::vectorComponent(
      get_logger(),
      startup.refine_grasped_tcp_to_block_rpy,
      0,
      0.0,
      "refine_grasped.tcp_to_block.rpy");
    const double rp = cbpwm::vectorComponent(
      get_logger(),
      startup.refine_grasped_tcp_to_block_rpy,
      1,
      0.0,
      "refine_grasped.tcp_to_block.rpy");
    const double ry = cbpwm::vectorComponent(
      get_logger(),
      startup.refine_grasped_tcp_to_block_rpy,
      2,
      0.0,
      "refine_grasped.tcp_to_block.rpy");
    const Eigen::Matrix3d rot_tcp_block =
      (Eigen::AngleAxisd(ry, Eigen::Vector3d::UnitZ()) *
      Eigen::AngleAxisd(rp, Eigen::Vector3d::UnitY()) *
      Eigen::AngleAxisd(rr, Eigen::Vector3d::UnitX())).toRotationMatrix();
    T_tcp_block_ = Eigen::Matrix4d::Identity();
    T_tcp_block_.block<3, 3>(0, 0) = rot_tcp_block;
    T_tcp_block_.block<3, 1>(0, 3) = Eigen::Vector3d(tx, ty, tz);

    roi_size_x_m_ = cbpwm::vectorComponent(
      get_logger(), startup.refine_grasped_roi_size_m, 0, 0.60, "refine_grasped.roi_size_m");
    roi_size_y_m_ = cbpwm::vectorComponent(
      get_logger(), startup.refine_grasped_roi_size_m, 1, 0.40, "refine_grasped.roi_size_m");
    refine_block_roi_size_x_m_ =
      cbpwm::vectorComponent(
      get_logger(), startup.refine_block_roi_size_m, 0, 1.20, "refine_block.roi_size_m");
    refine_block_roi_size_y_m_ =
      cbpwm::vectorComponent(
      get_logger(), startup.refine_block_roi_size_m, 1, 1.00, "refine_block.roi_size_m");

    world_pub_ = create_publisher<BlockArray>("block_world_model", 10);
    marker_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(
      "block_world_model_markers", 10);

    image_sub_.subscribe(this, "image");
    cloud_sub_.subscribe(this, "points");
    sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
      SyncPolicy(10), image_sub_, cloud_sub_);
    sync_->registerCallback(
      std::bind(
        &PerceptionOrchestratorNode::syncCallback,
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
        &PerceptionOrchestratorNode::handleSetMode,
        this,
        std::placeholders::_1,
        std::placeholders::_2));

    get_coarse_srv_ = create_service<GetCoarseSrv>(
      "~/get_coarse_blocks",
      std::bind(
        &PerceptionOrchestratorNode::handleGetCoarseBlocks,
        this,
        std::placeholders::_1,
        std::placeholders::_2));

    run_pose_srv_ = create_service<RunPoseSrv>(
      "~/run_pose_estimation",
      std::bind(
        &PerceptionOrchestratorNode::handleRunPoseEstimation,
        this,
        std::placeholders::_1,
        std::placeholders::_2),
      rmw_qos_profile_services_default,
      run_pose_cb_group_);

    set_block_task_status_srv_ = create_service<SetBlockTaskStatusSrv>(
      "~/set_block_task_status",
      std::bind(
        &PerceptionOrchestratorNode::handleSetBlockTaskStatus,
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

    initializeSeededWorld(startup);

    WM_LOG(
      get_logger(),
      "PerceptionOrchestratorNode ready | pipeline_mode=%s | perception_mode=%s",
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

void PerceptionOrchestratorNode::initializeSeededWorld(const cbpwm::WorldModelConfig & startup)
{
  if (startup.initial_blocks.empty()) {
    return;
  }
  constexpr double kDegToRad = 3.14159265358979323846 / 180.0;

  const auto stamp = now();
  std_msgs::msg::Header header;
  header.stamp = stamp;
  header.frame_id = world_frame_;

  {
    std::lock_guard<std::mutex> lock(persistent_world_mutex_);
    for (const auto & cfg_block : startup.initial_blocks) {
      Block block;
      block.id = cfg_block.id;
      block.pose_status = cfg_block.pose_status;
      block.task_status = cfg_block.task_status;
      block.pose.position.x = cfg_block.position[0];
      block.pose.position.y = cfg_block.position[1];
      block.pose.position.z = cfg_block.position[2];
      tf2::Quaternion quat;
      quat.setRPY(0.0, 0.0, cfg_block.yaw_deg * kDegToRad);
      block.pose.orientation.x = quat.x();
      block.pose.orientation.y = quat.y();
      block.pose.orientation.z = quat.z();
      block.pose.orientation.w = quat.w();
      block.confidence = static_cast<float>(cfg_block.confidence);
      block.last_seen = stamp;
      persistent_world_[block.id] = block;
      seeded_block_ids_.insert(block.id);
    }
  }

  publishPersistentWorld(header);
  RCLCPP_INFO(
    get_logger(),
    "Seeded world model with %zu startup block(s).",
    startup.initial_blocks.size());
}

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  auto node = std::make_shared<PerceptionOrchestratorNode>();
  rclcpp::executors::MultiThreadedExecutor exec(rclcpp::ExecutorOptions(), 4);
  exec.add_node(node);
  exec.spin();

  rclcpp::shutdown();
  return 0;
}
