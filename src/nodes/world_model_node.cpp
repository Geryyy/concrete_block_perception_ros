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
#include "concrete_block_perception/world_model/config_loader.hpp"
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

public:
  WorldModelNode()
  : Node("block_world_model_node")
  {
    run_pose_cb_group_ = create_callback_group(rclcpp::CallbackGroupType::Reentrant);
    action_client_cb_group_ = create_callback_group(rclcpp::CallbackGroupType::Reentrant);
    auto startup = cbpwm::loadWorldModelConfig(*this);
    cbpwm::normalizeWorldModelConfig(get_logger(), startup);
    pipeline_mode_ = cbpwm::parsePipelineMode(startup.pipeline_mode_str);
    const std::string & pipeline_mode_str = startup.pipeline_mode_str;
    const std::string & perception_mode_str = startup.perception_mode_str;
    min_fitness_ = startup.min_fitness;
    max_rmse_ = startup.max_rmse;
    object_class_ = startup.object_class;
    world_frame_ = startup.world_frame;
    max_sync_delta_s_ = startup.max_sync_delta_s;
    object_timeout_s_ = startup.object_timeout_s;
    association_max_distance_m_ = startup.association_max_distance_m;
    association_max_age_s_ = startup.association_max_age_s;
    min_update_confidence_ = startup.min_update_confidence;
    refine_target_max_distance_m_ = startup.refine_target_max_distance_m;
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
      std::bind(&WorldModelNode::cameraInfoCallback, this, std::placeholders::_1));

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
#include "world_model_node_helpers.inc"
#include "world_model_node_runtime.inc"
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
