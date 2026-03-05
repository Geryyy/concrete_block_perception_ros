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
#include "concrete_block_perception/srv/run_pose_estimation.hpp"
#include "concrete_block_perception/srv/set_block_task_status.hpp"
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
  using SetBlockTaskStatusSrv = concrete_block_perception::srv::SetBlockTaskStatus;
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
    association_max_distance_m_ =
      declare_parameter<double>("world_model.association_max_distance_m", 0.45);
    association_max_age_s_ =
      declare_parameter<double>("world_model.association_max_age_s", 20.0);
    min_update_confidence_ =
      declare_parameter<double>("world_model.min_update_confidence", 0.25);
    refine_target_max_distance_m_ =
      declare_parameter<double>("world_model.refine_target_max_distance_m", 1.2);
    debug_detection_overlay_enabled_ = declare_parameter<bool>("debug.publish_detection_overlay", true);
    perf_log_timing_enabled_ = declare_parameter<bool>("perf.log_timing", true);
    perf_log_every_n_frames_ = declare_parameter<int>("perf.log_every_n_frames", 20);
    const double marker_refresh_period_s =
      declare_parameter<double>("world_model.marker_refresh_period_s", 0.5);
    refine_grasped_use_fk_roi_ = declare_parameter<bool>("refine_grasped.use_fk_roi", true);
    refine_grasped_tcp_frame_ =
      declare_parameter<std::string>("refine_grasped.tcp_frame", "elastic/K8_tool_center_point");
    refine_grasped_camera_frame_ =
      declare_parameter<std::string>("refine_grasped.camera_frame", "seyond");
    refine_grasped_camera_info_topic_ =
      declare_parameter<std::string>(
      "refine_grasped.camera_info_topic", "/zed2i/warped/left/camera_info");
    refine_grasped_min_depth_m_ =
      declare_parameter<double>("refine_grasped.min_depth_m", 0.5);
    refine_grasped_max_depth_m_ =
      declare_parameter<double>("refine_grasped.max_depth_m", 30.0);

    const auto tcp_to_block_xyz =
      declare_parameter<std::vector<double>>("refine_grasped.tcp_to_block.xyz", {0.0, 0.0, 0.0});
    const auto tcp_to_block_rpy =
      declare_parameter<std::vector<double>>("refine_grasped.tcp_to_block.rpy", {0.0, 0.0, 0.0});
    const auto roi_size_m =
      declare_parameter<std::vector<double>>("refine_grasped.roi_size_m", {0.60, 0.40});

    // Keep for launch-file compatibility; no longer used in one-shot flow.
    (void)declare_parameter<std::string>("calib_yaml", "");

    if (perf_log_every_n_frames_ < 1) {
      perf_log_every_n_frames_ = 1;
    }

    if (debug_detection_overlay_enabled_) {
      det_debug_pub_ = create_publisher<sensor_msgs::msg::Image>("debug/detection_overlay", 1);
    }
    camera_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
      refine_grasped_camera_info_topic_,
      rclcpp::SensorDataQoS(),
      std::bind(&WorldModelNode::cameraInfoCallback, this, std::placeholders::_1));

    const double tx = (tcp_to_block_xyz.size() >= 1) ? tcp_to_block_xyz[0] : 0.0;
    const double ty = (tcp_to_block_xyz.size() >= 2) ? tcp_to_block_xyz[1] : 0.0;
    const double tz = (tcp_to_block_xyz.size() >= 3) ? tcp_to_block_xyz[2] : 0.0;
    const double rr = (tcp_to_block_rpy.size() >= 1) ? tcp_to_block_rpy[0] : 0.0;
    const double rp = (tcp_to_block_rpy.size() >= 2) ? tcp_to_block_rpy[1] : 0.0;
    const double ry = (tcp_to_block_rpy.size() >= 3) ? tcp_to_block_rpy[2] : 0.0;
    const Eigen::Matrix3d rot_tcp_block =
      (Eigen::AngleAxisd(ry, Eigen::Vector3d::UnitZ()) *
      Eigen::AngleAxisd(rp, Eigen::Vector3d::UnitY()) *
      Eigen::AngleAxisd(rr, Eigen::Vector3d::UnitX())).toRotationMatrix();
    T_tcp_block_ = Eigen::Matrix4d::Identity();
    T_tcp_block_.block<3, 3>(0, 0) = rot_tcp_block;
    T_tcp_block_.block<3, 1>(0, 3) = Eigen::Vector3d(tx, ty, tz);

    roi_size_x_m_ = (roi_size_m.size() >= 1) ? roi_size_m[0] : 0.60;
    roi_size_y_m_ = (roi_size_m.size() >= 2) ? roi_size_m[1] : 0.40;

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

    set_block_task_status_srv_ = create_service<SetBlockTaskStatusSrv>(
      "~/set_block_task_status",
      std::bind(
        &WorldModelNode::handleSetBlockTaskStatus,
        this,
        std::placeholders::_1,
        std::placeholders::_2));

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
    if (refine_grasped_use_fk_roi_) {
      WM_LOG(
        get_logger(),
        "REFINE_GRASPED FK+ROI enabled | tcp_frame=%s camera_frame=%s camera_info_topic=%s roi_size=[%.2f, %.2f]m",
        refine_grasped_tcp_frame_.c_str(),
        refine_grasped_camera_frame_.c_str(),
        refine_grasped_camera_info_topic_.c_str(),
        roi_size_x_m_,
        roi_size_y_m_);
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

  std::string nextWorldBlockId()
  {
    world_block_counter_++;
    return "wm_block_" + std::to_string(world_block_counter_);
  }

  static double blockDistance(const Block & a, const Block & b)
  {
    const double dx = a.pose.position.x - b.pose.position.x;
    const double dy = a.pose.position.y - b.pose.position.y;
    const double dz = a.pose.position.z - b.pose.position.z;
    return std::sqrt(dx * dx + dy * dy + dz * dz);
  }

  bool upsertRegisteredBlock(
    Block incoming,
    const OneShotRequest & run_request,
    const std_msgs::msg::Header & header,
    std::string & assigned_id,
    std::string & reason)
  {
    const rclcpp::Time now_stamp(header.stamp, get_clock()->get_clock_type());
    if (incoming.confidence < min_update_confidence_) {
      reason = "confidence below min_update_confidence";
      return false;
    }

    std::string forced_id;
    if ((run_request.mode == cbpwm::OneShotMode::kRefineBlock ||
      run_request.mode == cbpwm::OneShotMode::kRefineGrasped) &&
      !run_request.target_block_id.empty())
    {
      forced_id = run_request.target_block_id;
    }

    const Block * best_match = nullptr;
    double best_dist = std::numeric_limits<double>::infinity();
    std::string best_id;

    for (const auto & kv : persistent_world_) {
      const auto & existing = kv.second;
      const rclcpp::Time seen(existing.last_seen, get_clock()->get_clock_type());
      const double age_s = (now_stamp - seen).seconds();
      if (age_s > association_max_age_s_) {
        continue;
      }

      const double dist = blockDistance(incoming, existing);
      if (!cbpwm::shouldAssociateByDistance(
          dist, association_max_distance_m_, incoming.confidence, min_update_confidence_))
      {
        continue;
      }
      if (dist < best_dist) {
        best_dist = dist;
        best_match = &existing;
        best_id = kv.first;
      }
    }

    if (!forced_id.empty()) {
      assigned_id = forced_id;
    } else if (best_match != nullptr) {
      assigned_id = best_id;
    } else {
      assigned_id = nextWorldBlockId();
    }

    auto it = persistent_world_.find(assigned_id);
    if (it != persistent_world_.end()) {
      const auto & previous = it->second;
      const rclcpp::Time prev_stamp(previous.last_seen, get_clock()->get_clock_type());
      const rclcpp::Time incoming_stamp(incoming.last_seen, get_clock()->get_clock_type());
      if (incoming_stamp < prev_stamp) {
        reason = "stale update (incoming older than stored state)";
        return false;
      }

      if (!cbpwm::isValidTaskTransition(previous.task_status, incoming.task_status)) {
        reason = std::string("invalid task transition ") +
          cbpwm::taskStatusToString(previous.task_status) + " -> " +
          cbpwm::taskStatusToString(incoming.task_status);
        return false;
      }

      if (previous.task_status != Block::TASK_UNKNOWN) {
        incoming.task_status = previous.task_status;
      }
    } else {
      incoming.task_status = Block::TASK_FREE;
    }

    incoming.id = assigned_id;
    incoming.pose_status = Block::POSE_PRECISE;
    persistent_world_[assigned_id] = incoming;
    return true;
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
  }

  bool lookupPredictedGraspedPose(
    const std_msgs::msg::Header & header,
    Eigen::Vector3d & p_camera,
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
      const auto tf_camera_world = tf_buffer_->lookupTransform(
        refine_grasped_camera_frame_,
        world_frame_,
        rclcpp::Time(header.stamp),
        rclcpp::Duration::from_seconds(0.2));

      const Eigen::Matrix4d T_world_tcp = transformToEigen(tf_world_tcp);
      const Eigen::Matrix4d T_camera_world = transformToEigen(tf_camera_world);
      const Eigen::Matrix4d T_world_block_pred = T_world_tcp * T_tcp_block_;
      const Eigen::Vector4d p_block_world_h(
        T_world_block_pred(0, 3),
        T_world_block_pred(1, 3),
        T_world_block_pred(2, 3),
        1.0);
      const Eigen::Vector4d p_block_camera_h = T_camera_world * p_block_world_h;

      p_camera = p_block_camera_h.head<3>();
      return true;
    } catch (const tf2::TransformException & ex) {
      reason = std::string("TF lookup failed: ") + ex.what();
      return false;
    }
  }

  bool buildRoiMaskFromPrediction(
    const sensor_msgs::msg::Image::ConstSharedPtr & image,
    const Eigen::Vector3d & p_camera,
    cv::Mat & roi_mask,
    cv::Rect & roi_rect,
    std::string & reason)
  {
    CameraIntrinsics intr;
    {
      std::lock_guard<std::mutex> lock(camera_info_mutex_);
      intr = camera_intrinsics_;
    }
    if (!intr.valid) {
      reason = "camera_info not received or invalid intrinsics";
      return false;
    }

    const double z = p_camera.z();
    if (!std::isfinite(z) || z < refine_grasped_min_depth_m_ || z > refine_grasped_max_depth_m_) {
      reason = "predicted block depth out of bounds: z=" + std::to_string(z);
      return false;
    }

    const double u = (intr.fx * p_camera.x() / z) + intr.cx;
    const double v = (intr.fy * p_camera.y() / z) + intr.cy;
    if (!std::isfinite(u) || !std::isfinite(v)) {
      reason = "projected image point is invalid";
      return false;
    }

    const int roi_w_px = std::max(1, static_cast<int>(std::lround(intr.fx * roi_size_x_m_ / z)));
    const int roi_h_px = std::max(1, static_cast<int>(std::lround(intr.fy * roi_size_y_m_ / z)));
    cv::Rect requested_roi(
      static_cast<int>(std::lround(u)) - roi_w_px / 2,
      static_cast<int>(std::lround(v)) - roi_h_px / 2,
      roi_w_px,
      roi_h_px);
    const cv::Rect image_rect(0, 0, static_cast<int>(image->width), static_cast<int>(image->height));
    roi_rect = requested_roi & image_rect;
    if (roi_rect.width < 2 || roi_rect.height < 2) {
      reason = "ROI outside image or too small after clamping";
      return false;
    }

    roi_mask = cv::Mat::zeros(image_rect.height, image_rect.width, CV_8UC1);
    cv::rectangle(roi_mask, roi_rect, cv::Scalar(255), cv::FILLED);
    return true;
  }

  void publishRoiDebugOverlay(
    const sensor_msgs::msg::Image::ConstSharedPtr & image,
    const cv::Rect & roi_rect)
  {
    if (!det_debug_pub_) {
      return;
    }
    cv::Mat img = toCvBgr(*image);
    cv::rectangle(img, roi_rect, cv::Scalar(0, 0, 255), 2);
    cv::circle(
      img,
      cv::Point(roi_rect.x + roi_rect.width / 2, roi_rect.y + roi_rect.height / 2),
      4,
      cv::Scalar(0, 0, 255),
      cv::FILLED);
    auto out = cv_bridge::CvImage(image->header, "bgr8", img).toImageMsg();
    det_debug_pub_->publish(*out);
  }

  void processRefineGraspedWithFkRoi(
    const sensor_msgs::msg::Image::ConstSharedPtr & image,
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr & cloud,
    const OneShotRequest & run_request,
    const std::chrono::steady_clock::time_point & t_start)
  {
    if (!action_client_->action_server_is_ready()) {
      completeOneShotRequest(run_request.sequence, false, "Registration action unavailable.");
      resetBusy();
      return;
    }

    Eigen::Vector3d p_camera = Eigen::Vector3d::Zero();
    std::string reason;
    if (!lookupPredictedGraspedPose(cloud->header, p_camera, reason)) {
      RCLCPP_WARN(get_logger(), "REFINE_GRASPED FK+ROI failed: %s", reason.c_str());
      publishPersistentWorld(cloud->header);
      completeOneShotRequest(
        run_request.sequence,
        false,
        "REFINE_GRASPED FK prediction failed: " + reason);
      resetBusy();
      return;
    }

    cv::Mat roi_mask;
    cv::Rect roi_rect;
    if (!buildRoiMaskFromPrediction(image, p_camera, roi_mask, roi_rect, reason)) {
      RCLCPP_WARN(get_logger(), "REFINE_GRASPED FK+ROI failed: %s", reason.c_str());
      publishPersistentWorld(cloud->header);
      completeOneShotRequest(
        run_request.sequence,
        false,
        "REFINE_GRASPED ROI construction failed: " + reason);
      resetBusy();
      return;
    }

    if (debug_detection_overlay_enabled_ && det_debug_pub_) {
      publishRoiDebugOverlay(image, roi_rect);
    }

    auto roi_mask_msg = cv_bridge::CvImage(image->header, "mono8", roi_mask).toImageMsg();
    Block block;
    bool registration_ok = false;
    std::string reg_reason;
    const auto t_reg_start = std::chrono::steady_clock::now();
    registration_ok = runRegistrationSync(
      1U,
      *roi_mask_msg,
      *cloud,
      cloud->header,
      run_request.registration_timeout_s,
      block,
      reg_reason);
    const auto t_reg_end = std::chrono::steady_clock::now();

    size_t registrations_ok = 0;
    if (registration_ok) {
      std::lock_guard<std::mutex> lock(persistent_world_mutex_);
      std::string assigned_id;
      std::string upsert_reason;
      const bool upsert_ok = upsertRegisteredBlock(
        block,
        run_request,
        cloud->header,
        assigned_id,
        upsert_reason);
      if (upsert_ok) {
        ++registrations_ok;
        RCLCPP_INFO(
          get_logger(),
          "REFINE_GRASPED FK+ROI accepted: target=%s assigned=%s roi=[x=%d y=%d w=%d h=%d]",
          run_request.target_block_id.c_str(),
          assigned_id.c_str(),
          roi_rect.x,
          roi_rect.y,
          roi_rect.width,
          roi_rect.height);
      } else {
        RCLCPP_WARN(
          get_logger(),
          "REFINE_GRASPED FK+ROI rejected after association checks: %s",
          upsert_reason.c_str());
        reg_reason = upsert_reason;
      }
    } else {
      RCLCPP_WARN(
        get_logger(),
        "REFINE_GRASPED FK+ROI registration failed: %s",
        reg_reason.c_str());
    }

    const auto reg_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
      t_reg_end - t_reg_start).count();
    const auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
      t_reg_end - t_start).count();
    recordTiming(0, 0, reg_ms, total_ms);

    publishPersistentWorld(cloud->header);
    completeOneShotRequest(
      run_request.sequence,
      registrations_ok > 0,
      registrations_ok > 0 ?
      ("Pose estimation finished (mode=REFINE_GRASPED_FK_ROI, registration_candidates=1, registrations=1).") :
      ("Pose estimation finished (mode=REFINE_GRASPED_FK_ROI, registration_candidates=1, registrations=0, reason=" +
      reg_reason + ")."));
    resetBusy();
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
              run_request.target_block_id.rfind("block_", 0) == 0 &&
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
          const bool targeted_refine =
            (run_request.mode == cbpwm::OneShotMode::kRefineBlock ||
            run_request.mode == cbpwm::OneShotMode::kRefineGrasped) &&
            !run_request.target_block_id.empty() &&
            run_request.target_block_id.rfind("block_", 0) != 0;

          bool have_expected_target = false;
          Block expected_target;
          if (targeted_refine) {
            std::lock_guard<std::mutex> lock(persistent_world_mutex_);
            const auto it = persistent_world_.find(run_request.target_block_id);
            if (it != persistent_world_.end()) {
              expected_target = it->second;
              have_expected_target = true;
            }
          }

          if (targeted_refine && have_expected_target) {
            Block best_block;
            bool best_valid = false;
            double best_dist = std::numeric_limits<double>::infinity();

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

              if (!ok) {
                RCLCPP_WARN(
                  get_logger(),
                  "Registration rejected for block_%u: %s",
                  candidate.first,
                  reason.c_str());
                continue;
              }

              const double dist = blockDistance(block, expected_target);
              if (dist < best_dist) {
                best_dist = dist;
                best_block = block;
                best_valid = true;
              }
            }

            if (best_valid && best_dist <= refine_target_max_distance_m_) {
              std::lock_guard<std::mutex> lock(persistent_world_mutex_);
              std::string assigned_id;
              std::string upsert_reason;
              const bool upsert_ok = upsertRegisteredBlock(
                best_block,
                run_request,
                cloud->header,
                assigned_id,
                upsert_reason);
              if (upsert_ok) {
                ++registrations_ok;
                RCLCPP_INFO(
                  get_logger(),
                  "Targeted refine accepted: target=%s assigned=%s dist=%.3f m",
                  run_request.target_block_id.c_str(),
                  assigned_id.c_str(),
                  best_dist);
              } else {
                RCLCPP_WARN(
                  get_logger(),
                  "Targeted refine rejected after association checks: %s",
                  upsert_reason.c_str());
              }
            } else {
              RCLCPP_WARN(
                get_logger(),
                "Targeted refine failed for '%s': no candidate within %.3f m of expected pose.",
                run_request.target_block_id.c_str(),
                refine_target_max_distance_m_);
            }
          } else {
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
                std::lock_guard<std::mutex> lock(persistent_world_mutex_);
                std::string assigned_id;
                std::string upsert_reason;
                const bool upsert_ok = upsertRegisteredBlock(
                  block,
                  run_request,
                  cloud->header,
                  assigned_id,
                  upsert_reason);
                if (upsert_ok) {
                  ++registrations_ok;
                  RCLCPP_INFO(
                    get_logger(),
                    "Registration accepted and associated: incoming=%s assigned=%s",
                    block.id.c_str(),
                    assigned_id.c_str());
                } else {
                  RCLCPP_WARN(
                    get_logger(),
                    "Registration rejected after association checks for %s: %s",
                    block.id.c_str(),
                    upsert_reason.c_str());
                }
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
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_;

  rclcpp::Client<SegmentSrv>::SharedPtr segment_client_;
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
  bool refine_grasped_use_fk_roi_{true};
  std::string refine_grasped_tcp_frame_{"elastic/K8_tool_center_point"};
  std::string refine_grasped_camera_frame_{"seyond"};
  std::string refine_grasped_camera_info_topic_{"/zed2i/warped/left/camera_info"};
  Eigen::Matrix4d T_tcp_block_{Eigen::Matrix4d::Identity()};
  double roi_size_x_m_{0.60};
  double roi_size_y_m_{0.40};
  double refine_grasped_min_depth_m_{0.5};
  double refine_grasped_max_depth_m_{30.0};

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
