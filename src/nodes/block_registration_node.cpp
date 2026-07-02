#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>

#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>

#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>

#include <cv_bridge/cv_bridge.h>

#include <chrono>
#include <cmath>
#include <thread>

#include "concrete_block_perception_interfaces/action/register_block.hpp"
#include "concrete_block_perception_interfaces/srv/extract_mask_cutout.hpp"
#include "concrete_block_perception_interfaces/srv/register_block.hpp"
#include "concrete_block_perception/utils/io_utils.hpp"

#include "concrete_block_perception/registration/block_registration_pipeline.hpp"
#include "concrete_block_perception/registration/registration_config.hpp"
#include "concrete_block_perception/registration/ros_debug_helpers.hpp"

#include "pcd_block_estimation/utils.hpp"

using namespace concrete_block_perception;

class BlockRegistrationNode : public rclcpp::Node
{
  using RegisterBlockAction =
    concrete_block_perception_interfaces::action::RegisterBlock;

  using GoalHandleRegisterBlock =
    rclcpp_action::ServerGoalHandle<RegisterBlockAction>;
  using RegisterBlockSrv = concrete_block_perception_interfaces::srv::RegisterBlock;
  using ExtractMaskCutoutSrv = concrete_block_perception_interfaces::srv::ExtractMaskCutout;

public:
  BlockRegistrationNode()
  : Node("block_registration_node")
  {
    config_ = load_registration_config(*this);

    pipeline_ =
      std::make_unique<BlockRegistrationPipeline>(
      config_.T_P_C,
      config_.K,
      config_.templates,
      config_.preproc,
      config_.glob,
      config_.local,
      get_logger(),
      config_.verbose_logs);

    debug_ =
      std::make_unique<RosDebugHelpers>(*this, config_);

    tf_buffer_ =
      std::make_shared<tf2_ros::Buffer>(this->get_clock());

    tf_listener_ =
      std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    // Reentrant callback group for multithreading
    action_cb_group_ =
      this->create_callback_group(
      rclcpp::CallbackGroupType::Reentrant);

    action_server_ =
      rclcpp_action::create_server<RegisterBlockAction>(
      this,
      "register_block",
      std::bind(
        &BlockRegistrationNode::handle_goal,
        this,
        std::placeholders::_1,
        std::placeholders::_2),
      std::bind(
        &BlockRegistrationNode::handle_cancel,
        this,
        std::placeholders::_1),
      std::bind(
        &BlockRegistrationNode::handle_accepted,
        this,
        std::placeholders::_1),
      rcl_action_server_get_default_options(),
      action_cb_group_);

    register_service_ = create_service<RegisterBlockSrv>(
      "register_block_pose",
      std::bind(
        &BlockRegistrationNode::handle_register_service,
        this,
        std::placeholders::_1,
        std::placeholders::_2));
    cutout_service_ = create_service<ExtractMaskCutoutSrv>(
      "extract_mask_cutout",
      std::bind(
        &BlockRegistrationNode::handle_cutout_service,
        this,
        std::placeholders::_1,
        std::placeholders::_2));

    RCLCPP_INFO(
      get_logger(),
      "Block registration node ready");
  }

private:
  bool runPipelineFromRosInputs(
    const sensor_msgs::msg::PointCloud2 & cloud,
    const sensor_msgs::msg::Image & mask_msg,
    const std::string & object_class,
    const RegisterBlockAction::Goal * action_goal,
    bool allow_fk_seed,
    cv::Mat & mask_cv,
    RegistrationOutput & output)
  {
    geometry_msgs::msg::TransformStamped tf_cloud;
    if (!lookupCloudTransform(cloud, tf_cloud)) {
      return false;
    }

    auto scene_ptr = pointcloud2_to_open3d(cloud);
    if (!scene_ptr || scene_ptr->points_.empty()) {
      RCLCPP_WARN(get_logger(), "Empty scene cloud.");
      return false;
    }

    mask_cv = cv_bridge::toCvCopy(mask_msg, "mono8")->image;

    RegistrationInput input;
    input.scene = *scene_ptr;
    input.mask = mask_cv;
    input.T_world_cloud = transformToEigen(tf_cloud);

    if (action_goal && action_goal->use_pose_prior) {
      Eigen::Matrix4d prior_pose_world = Eigen::Matrix4d::Identity();
      if (poseToEigen(action_goal->prior_pose, prior_pose_world)) {
        input.has_pose_prior_world = true;
        input.pose_prior_world = prior_pose_world;
        input.prior_position_sigma_m = action_goal->prior_position_sigma_m;
        input.prior_orientation_sigma_rad = action_goal->prior_orientation_sigma_rad;
        input.has_translation_seed_world = true;
        input.translation_seed_world = prior_pose_world.block<3, 1>(0, 3);
        if (config_.verbose_logs) {
          RCLCPP_INFO(
            get_logger(),
            "Using action pose prior: position=[%.3f %.3f %.3f] sigma=[%.3f m %.3f rad]",
            input.translation_seed_world.x(),
            input.translation_seed_world.y(),
            input.translation_seed_world.z(),
            input.prior_position_sigma_m,
            input.prior_orientation_sigma_rad);
        }
      } else {
        RCLCPP_WARN(get_logger(), "Ignoring invalid action pose prior quaternion.");
      }
    }

    if (allow_fk_seed && config_.local.use_fk_translation_seed && shouldUseFkSeedForGoal(object_class)) {
      if (!resolveFkPoseSeed(cloud.header, input.fk_pose_seed_world)) {
        RCLCPP_WARN(
          get_logger(),
          "FK pose seed requested but unavailable; falling back to global registration.");
      } else {
        input.has_fk_pose_seed = true;
      }
    } else if (!allow_fk_seed && config_.local.use_fk_translation_seed) {
      RCLCPP_DEBUG(
        get_logger(),
        "FK translation seed is action-mode gated; skipping for register_block_pose service call.");
    }

    output = pipeline_->run(input);
    return true;
  }

  void handle_register_service(
    const std::shared_ptr<RegisterBlockSrv::Request> request,
    std::shared_ptr<RegisterBlockSrv::Response> response)
  {
    const auto start_time = std::chrono::steady_clock::now();

    debug_->publishGripperBoxes(request->cloud.header);

    if (request->cloud.data.empty() || request->mask.data.empty()) {
      response->success = false;
      return;
    }

    cv::Mat mask;
    RegistrationOutput output;
    if (!runPipelineFromRosInputs(
        request->cloud, request->mask, request->object_class, nullptr, false, mask, output))
    {
      response->success = false;
      return;
    }
    debug_->publishDiagnostics(output, "service");
    response->cutout_cloud = open3d_to_pointcloud2_colored(
      output.debug_cleaned_cutout,
      config_.world_frame,
      rclcpp::Time(request->cloud.header.stamp));
    if (!output.success) {
      maybeDumpFailurePackage(request->cloud, request->mask, output);
      response->success = false;
      return;
    }

    debug_->publishMask(request->mask, mask);
    debug_->publishRegistrationDebug(
      request->cloud,
      output.debug_mask_cutout,
      output.debug_cleaned_cutout,
      output.debug_registration_cloud,
      output.template_index,
      output.T_world_block);

    response->pose = to_ros_pose(output.T_world_block);
    response->fitness = output.fitness;
    response->rmse = output.rmse;
    response->success = true;

    const auto end_time = std::chrono::steady_clock::now();
    const float total_ms =
      std::chrono::duration<float, std::milli>(end_time - start_time).count();
    RCLCPP_INFO(get_logger(), "Registration completed in %.2f ms", total_ms);
  }

  void handle_cutout_service(
    const std::shared_ptr<ExtractMaskCutoutSrv::Request> request,
    std::shared_ptr<ExtractMaskCutoutSrv::Response> response)
  {
    const auto start_time = std::chrono::steady_clock::now();

    debug_->publishGripperBoxes(request->cloud.header);

    if (request->cloud.data.empty() || request->mask.data.empty()) {
      response->success = false;
      response->reason = "empty cloud or mask";
      return;
    }

    geometry_msgs::msg::TransformStamped tf_cloud;
    if (!lookupCloudTransform(request->cloud, tf_cloud)) {
      response->success = false;
      response->reason = "cloud transform lookup failed";
      return;
    }

    auto scene_ptr = pointcloud2_to_open3d(request->cloud);
    if (!scene_ptr || scene_ptr->points_.empty()) {
      response->success = false;
      response->reason = "empty scene cloud";
      return;
    }

    cv::Mat mask;
    try {
      mask = cv_bridge::toCvCopy(request->mask, "mono8")->image;
    } catch (const std::exception & e) {
      response->success = false;
      response->reason = std::string("mask conversion failed: ") + e.what();
      return;
    }

    RegistrationInput input;
    input.scene = *scene_ptr;
    input.mask = mask;
    input.T_world_cloud = transformToEigen(tf_cloud);

    open3d::geometry::PointCloud cutout_world;
    open3d::geometry::PointCloud mask_cutout_world;
    std::string reason;
    if (!pipeline_->extractMaskCutout(input, cutout_world, reason, &mask_cutout_world)) {
      response->success = false;
      response->reason = reason;
      return;
    }

    response->cutout_cloud = open3d_to_pointcloud2_colored(
      cutout_world,
      config_.world_frame,
      rclcpp::Time(request->cloud.header.stamp));
    response->success = true;
    response->reason = reason;
    debug_->publishCutoutDebug(request->cloud, mask_cutout_world, cutout_world);

    const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - start_time).count();
    RCLCPP_INFO_THROTTLE(
      get_logger(), *get_clock(), 1000,
      "Cutout-only service: points=%zu elapsed=%ld ms",
      cutout_world.points_.size(),
      elapsed_ms);
  }

  rclcpp_action::GoalResponse
  handle_goal(
    const rclcpp_action::GoalUUID &,
    std::shared_ptr<const RegisterBlockAction::Goal> goal)
  {
    if (goal->cloud.data.empty()) {
      RCLCPP_WARN(
        get_logger(),
        "Rejecting goal: empty cloud");
      return rclcpp_action::GoalResponse::REJECT;
    }

    if (goal->mask.data.empty()) {
      RCLCPP_WARN(
        get_logger(),
        "Rejecting goal: empty mask");
      return rclcpp_action::GoalResponse::REJECT;
    }

    return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
  }

  rclcpp_action::CancelResponse
  handle_cancel(
    const std::shared_ptr<GoalHandleRegisterBlock>)
  {
    RCLCPP_WARN(
      get_logger(),
      "RegisterBlock goal cancelled");
    return rclcpp_action::CancelResponse::ACCEPT;
  }

  void handle_accepted(
    const std::shared_ptr<GoalHandleRegisterBlock> goal_handle)
  {
    std::thread(
      [this, goal_handle]() {
        execute(goal_handle);
      }).detach();
  }

  void execute(
    const std::shared_ptr<GoalHandleRegisterBlock> goal_handle)
  {
    const auto start_time = std::chrono::steady_clock::now();

    const auto goal = goal_handle->get_goal();
    auto result =
      std::make_shared<RegisterBlockAction::Result>();

    auto publish_feedback =
      [&](const std::string & stage, float progress)
      {
        RegisterBlockAction::Feedback fb;
        fb.stage = stage;
        fb.progress = progress;

        const auto now =
          std::chrono::steady_clock::now();

        fb.elapsed_ms =
          std::chrono::duration<float, std::milli>(
          now - start_time).count();

        goal_handle->publish_feedback(
          std::make_shared<RegisterBlockAction::Feedback>(fb));
      };

    // TF lookup
    publish_feedback("tf_lookup", 0.1f);

    // Conversion
    publish_feedback("conversion", 0.2f);

    cv::Mat mask;

    // Dump input
    debug_->dumpInput(*goal);
    debug_->publishGripperBoxes(goal->cloud.header);

    // Run registration
    publish_feedback("registration", 0.6f);
    RegistrationOutput output;
    if (!runPipelineFromRosInputs(goal->cloud, goal->mask, goal->object_class, goal.get(), true, mask, output)) {
      result->success = false;
      goal_handle->abort(result);
      return;
    }
    debug_->publishDiagnostics(output, "action");

    // Always publish debug cutout/template attempt for offline diagnosis, even on failure.
    debug_->publishMask(goal->mask, mask);
    debug_->publishRegistrationDebug(
      goal->cloud,
      output.debug_mask_cutout,
      output.debug_cleaned_cutout,
      output.debug_registration_cloud,
      output.template_index,
      output.T_world_block);

    if (!output.success) {
      maybeDumpFailurePackage(goal->cloud, goal->mask, output);
      RCLCPP_WARN(
        get_logger(),
        "Registration failed.");

      result->success = false;
      goal_handle->abort(result);
      return;
    }

    // Debug feedback
    publish_feedback("visualization", 0.9f);

    // Fill result
    result->pose =
      to_ros_pose(output.T_world_block);

    result->fitness = output.fitness;
    result->rmse = output.rmse;
    result->success = true;

    publish_feedback("done", 1.0f);
    goal_handle->succeed(result);

    const auto end_time =
      std::chrono::steady_clock::now();

    const float total_ms =
      std::chrono::duration<float, std::milli>(
      end_time - start_time).count();

    RCLCPP_INFO(
      get_logger(),
      "Registration completed in %.2f ms",
      total_ms);
  }

  bool poseToEigen(
    const geometry_msgs::msg::Pose & pose,
    Eigen::Matrix4d & out_pose_world) const
  {
    Eigen::Quaterniond q(
      pose.orientation.w,
      pose.orientation.x,
      pose.orientation.y,
      pose.orientation.z);

    if (!std::isfinite(pose.position.x) || !std::isfinite(pose.position.y) ||
      !std::isfinite(pose.position.z) ||
      !std::isfinite(q.w()) || !std::isfinite(q.x()) ||
      !std::isfinite(q.y()) || !std::isfinite(q.z()) ||
      q.norm() < 1.0e-6)
    {
      return false;
    }

    q.normalize();
    out_pose_world = Eigen::Matrix4d::Identity();
    out_pose_world.block<3, 3>(0, 0) = q.toRotationMatrix();
    out_pose_world(0, 3) = pose.position.x;
    out_pose_world(1, 3) = pose.position.y;
    out_pose_world(2, 3) = pose.position.z;
    return true;
  }

  bool lookupCloudTransform(
    const sensor_msgs::msg::PointCloud2 & cloud,
    geometry_msgs::msg::TransformStamped & tf_out)
  {
    try {
      tf_out =
        tf_buffer_->lookupTransform(
        config_.world_frame,
        cloud.header.frame_id,
        rclcpp::Time(cloud.header.stamp),
        rclcpp::Duration::from_seconds(0.5));

      return true;
    } catch (const tf2::TransformException & ex) {
      RCLCPP_WARN(
        get_logger(),
        "TF lookup failed: %s",
        ex.what());
      return false;
    }
  }

  bool resolveFkPoseSeed(
    const std_msgs::msg::Header & header,
    Eigen::Matrix4d & out_pose_world)
  {
    try {
      const auto tf_world_tcp =
        tf_buffer_->lookupTransform(
        config_.world_frame,
        config_.fk_seed_tcp_frame,
        rclcpp::Time(header.stamp),
        rclcpp::Duration::from_seconds(0.2));

      const Eigen::Matrix4d T_world_tcp = transformToEigen(tf_world_tcp);

      Eigen::Matrix4d T_tcp_block = Eigen::Matrix4d::Identity();
      T_tcp_block.block<3, 1>(0, 3) = config_.fk_seed_tcp_to_block_xyz;
      const Eigen::Vector3d & rpy = config_.fk_seed_tcp_to_block_rpy;
      T_tcp_block.block<3, 3>(0, 0) =
        (Eigen::AngleAxisd(rpy.z(), Eigen::Vector3d::UnitZ()) *
        Eigen::AngleAxisd(rpy.y(), Eigen::Vector3d::UnitY()) *
        Eigen::AngleAxisd(rpy.x(), Eigen::Vector3d::UnitX()))
        .toRotationMatrix();

      out_pose_world = T_world_tcp * T_tcp_block;
      return true;
    } catch (const tf2::TransformException & ex) {
      RCLCPP_WARN(
        get_logger(),
        "FK pose seed TF lookup failed (%s <- %s): %s",
        config_.world_frame.c_str(),
        config_.fk_seed_tcp_frame.c_str(),
        ex.what());
      return false;
    }
  }

  bool shouldUseFkSeedForGoal(const std::string & object_class) const
  {
    return object_class.find("#REFINE_GRASPED") != std::string::npos;
  }

  void maybeDumpFailurePackage(
    const sensor_msgs::msg::PointCloud2 & cloud,
    const sensor_msgs::msg::Image & mask,
    const RegistrationOutput & output)
  {
    if (!config_.dump_enabled || !config_.dump_failure_package || !debug_) {
      return;
    }
    const auto & dump_cloud =
      output.debug_registration_cloud.points_.empty() ?
      output.debug_cleaned_cutout : output.debug_registration_cloud;
    debug_->dumpFailurePackage(cloud, mask, dump_cloud, output.failure_stage, output.failure_reason);
  }

  BlockRegistrationConfig config_;
  std::unique_ptr<BlockRegistrationPipeline> pipeline_;
  std::unique_ptr<RosDebugHelpers> debug_;

  rclcpp_action::Server<RegisterBlockAction>::SharedPtr action_server_;
  rclcpp::Service<RegisterBlockSrv>::SharedPtr register_service_;
  rclcpp::Service<ExtractMaskCutoutSrv>::SharedPtr cutout_service_;
  rclcpp::CallbackGroup::SharedPtr action_cb_group_;

  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  auto node =
    std::make_shared<BlockRegistrationNode>();

  rclcpp::executors::MultiThreadedExecutor exec(
    rclcpp::ExecutorOptions(),
    std::thread::hardware_concurrency());

  exec.add_node(node);
  exec.spin();

  rclcpp::shutdown();
  return 0;
}
