#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>

#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>

#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>

#include <cv_bridge/cv_bridge.h>

#include <chrono>
#include <thread>

#include "concrete_block_perception/action/register_block.hpp"
#include "concrete_block_perception/utils/io_utils.hpp"

#include "concrete_block_perception/registration/block_registration_pipeline.hpp"
#include "concrete_block_perception/registration/registration_config.hpp"
#include "concrete_block_perception/registration/ros_debug_helpers.hpp"

#include "pcd_block_estimation/utils.hpp"

namespace concrete_block_perception
{

class BlockRegistrationNode : public rclcpp::Node
{
  using RegisterBlockAction =
    concrete_block_perception::action::RegisterBlock;

  using GoalHandleRegisterBlock =
    rclcpp_action::ServerGoalHandle<RegisterBlockAction>;

public:
  explicit BlockRegistrationNode(
    const rclcpp::NodeOptions & options)
  : Node("block_registration_node", options)
  {
    config_ = load_registration_config(*this);

    pipeline_ =
      std::make_unique<BlockRegistrationPipeline>(
      config_.T_P_C,
      config_.K,
      config_.templates,
      config_.preproc,
      config_.glob,
      config_.local);

    debug_ =
      std::make_unique<RosDebugHelpers>(*this, config_);

    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());

    tf_buffer_->setUsingDedicatedThread(true);

    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(
      *tf_buffer_,
      this,
      false // do not spin thread automatically
    );

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

    RCLCPP_INFO(
      get_logger(),
      "Block registration node ready (component)");
  }

private:
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
    execute(goal_handle);
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

    geometry_msgs::msg::TransformStamped tf_cloud;

    if (!lookupCloudTransform(*goal, tf_cloud)) {
      RCLCPP_WARN(
        get_logger(),
        "TF lookup failed.");

      result->success = false;
      goal_handle->abort(result);
      return;
    }

    // Conversion
    publish_feedback("conversion", 0.2f);

    auto scene_ptr =
      pointcloud2_to_open3d(goal->cloud);

    if (!scene_ptr || scene_ptr->points_.empty()) {
      RCLCPP_WARN(
        get_logger(),
        "Empty scene cloud.");

      result->success = false;
      goal_handle->abort(result);
      return;
    }

    cv::Mat mask =
      cv_bridge::toCvCopy(
      goal->mask,
      "mono8")->image;

    // Dump input
    debug_->dumpInput(*goal);

    // Prepare pipeline input
    RegistrationInput input;
    input.scene = *scene_ptr;
    input.mask = mask;
    input.T_world_cloud =
      transformToEigen(tf_cloud);

    // Run registration
    publish_feedback("registration", 0.6f);

    auto output =
      pipeline_->run(input);

    if (!output.success) {
      RCLCPP_WARN(
        get_logger(),
        "Registration failed.");

      result->success = false;
      goal_handle->abort(result);
      return;
    }

    // Debug visualization
    publish_feedback("visualization", 0.9f);

    debug_->publishMask(goal->mask, mask);

    debug_->publishVisualization(
      goal->cloud,
      output.debug_scene,
      output.template_index,
      output.T_world_block);

    // Fill result
    result->pose =
      to_ros_pose(output.T_world_block);

    result->fitness = output.fitness;
    result->rmse = output.rmse;
    result->success = true;

    goal_handle->succeed(result);

    publish_feedback("done", 1.0f);

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

  bool lookupCloudTransform(
    const RegisterBlockAction::Goal & goal,
    geometry_msgs::msg::TransformStamped & tf_out)
  {
    try {
      tf_out =
        tf_buffer_->lookupTransform(
        config_.world_frame,
        goal.cloud.header.frame_id,
        rclcpp::Time(goal.cloud.header.stamp),
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

  BlockRegistrationConfig config_;
  std::unique_ptr<BlockRegistrationPipeline> pipeline_;
  std::unique_ptr<RosDebugHelpers> debug_;

  rclcpp_action::Server<RegisterBlockAction>::SharedPtr action_server_;
  rclcpp::CallbackGroup::SharedPtr action_cb_group_;

  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
};

}  // namespace concrete_block_perception

#include "rclcpp_components/register_node_macro.hpp"

RCLCPP_COMPONENTS_REGISTER_NODE(
  concrete_block_perception::BlockRegistrationNode
)
