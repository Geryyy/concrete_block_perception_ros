#include <rclcpp/rclcpp.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>

#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/pose.hpp>

#include <tf2_ros/transform_listener.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/buffer.h>
#include <tf2/LinearMath/Quaternion.h>
#include <geometry_msgs/msg/transform_stamped.hpp>

#include <cv_bridge/cv_bridge.h>
#include <open3d/Open3D.h>

#include <thread>
#include <filesystem>

#include "pcd_block_estimation/mask_projection.hpp"
#include "pcd_block_estimation/pose_estimation.hpp"
#include "pcd_block_estimation/template_utils.hpp"
#include "pcd_block_estimation/utils.hpp"
#include "pcd_block_estimation/yaml_utils.hpp"

#include <rclcpp_action/rclcpp_action.hpp>
#include <rcl_action/action_server.h>
#include "concrete_block_perception/action/register_block.hpp"
#include "concrete_block_perception/io_utils.hpp"
#include "concrete_block_perception/debug_utils.hpp"

using namespace pcd_block;
using namespace open3d;

#define LOG(logger, ...) RCLCPP_INFO(logger, __VA_ARGS__)

class BlockRegistrationNode : public rclcpp::Node
{
  using RegisterBlockAction =
    concrete_block_perception::action::RegisterBlock;

  using GoalHandleRegisterBlock =
    rclcpp_action::ServerGoalHandle<RegisterBlockAction>;

public:
  BlockRegistrationNode()
  : Node("block_registration_node")
  {
    // ------------------------------------------------------------
    // Defaults
    // ------------------------------------------------------------
    const std::string pkg_share =
      ament_index_cpp::get_package_share_directory(
      "concrete_block_perception");

    const std::string default_calib_yaml =
      pkg_share + "/config/calib_zed2i_to_seyond.yaml";

    const std::string default_template_dir =
      pkg_share + "/config/templates";

    // ------------------------------------------------------------
    // Parameters
    // ------------------------------------------------------------
    declare_parameter<std::string>("calib_yaml", "");
    declare_parameter<std::string>("template_dir", "");
    declare_parameter<std::string>("world_frame", "world");
    declare_parameter<double>("dist_thresh", 0.02);
    declare_parameter<int>("min_inliers", 100);
    declare_parameter<double>("icp_dist", 0.04);
    declare_parameter<double>("angle_thresh_degree", 30.0);
    declare_parameter<int>("yaw_step", 30);

    calib_yaml = get_parameter("calib_yaml").as_string();
    template_dir = get_parameter("template_dir").as_string();

    if (calib_yaml.empty()) {
      calib_yaml = default_calib_yaml;
    }
    if (template_dir.empty()) {
      template_dir = default_template_dir;
    }

    world_frame_ = get_parameter("world_frame").as_string();
    dist_thresh = get_parameter("dist_thresh").as_double();
    min_inliers = get_parameter("min_inliers").as_int();
    icp_dist = get_parameter("icp_dist").as_double();
    yaw_step = get_parameter("yaw_step").as_int();

    double angle_thresh_deg =
      get_parameter("angle_thresh_degree").as_double();
    angle_thresh = std::cos(angle_thresh_deg * M_PI / 180.0);

    // ------------------------------------------------------------
    // Validate files
    // ------------------------------------------------------------
    if (!std::filesystem::exists(calib_yaml)) {
      throw std::runtime_error(
              "Calibration YAML not found: " + calib_yaml);
    }
    if (!std::filesystem::exists(template_dir)) {
      throw std::runtime_error(
              "Template directory not found: " + template_dir);
    }

    // ------------------------------------------------------------
    // Load data
    // ------------------------------------------------------------
    T_P_C_ = load_T_4x4(calib_yaml);
    K_ = load_camera_matrix(calib_yaml);
    templates_ = load_templates(template_dir);

    // Z_WORLD = Eigen::Vector3d(0.0, -1.0, 0.0);
    Z_WORLD = Eigen::Vector3d(0.0, 0.0, 1.0);

    // ------------------------------------------------------------
    // Service (SERIALIZED!)
    // ------------------------------------------------------------
    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ =
      std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
    tf_broadcaster_ =
      std::make_shared<tf2_ros::TransformBroadcaster>(this);


    action_cb_group_ =
      create_callback_group(
      rclcpp::CallbackGroupType::Reentrant);


    rcl_action_server_options_t action_opts =
      rcl_action_server_get_default_options();

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
      action_opts,
      action_cb_group_);


    // callbacks

    publish_debug_cutout_ =
      declare_parameter<bool>("debug.publish_cutout", true);

    if (publish_debug_cutout_) {
      debug_cutout_pub_ =
        create_publisher<sensor_msgs::msg::PointCloud2>(
        "debug/cutout_cloud", 1);

      debug_template_pub_ =
        create_publisher<sensor_msgs::msg::PointCloud2>(
        "debug/template_cloud", 1);


      RCLCPP_INFO(
        get_logger(),
        "Publishing debug cutout cloud on %s",
        debug_cutout_pub_->get_topic_name());

      RCLCPP_INFO(
        get_logger(),
        "Publishing template cloud on %s",
        debug_template_pub_->get_topic_name());

      RCLCPP_INFO(
        get_logger(),
        "Publishing debug block frame as tf");
    }

    publish_debug_mask_ =
      declare_parameter<bool>("debug.publish_mask", true);

    if (publish_debug_mask_) {
      debug_mask_pub_ =
        create_publisher<sensor_msgs::msg::Image>(
        "debug/segmentation_mask", 1);

      RCLCPP_INFO(
        get_logger(),
        "Publishing debug segmentation mask on %s",
        debug_mask_pub_->get_topic_name());
    }

    // TODO: generate templates if not existant


    RCLCPP_INFO(
      get_logger(),
      "Block registration action ready");
    RCLCPP_INFO(
      get_logger(),
      "  calib_yaml:   %s", calib_yaml.c_str());
    RCLCPP_INFO(
      get_logger(),
      "  template_dir: %s", template_dir.c_str());
  }

private:
  // --------------------------------------------------------
  // Action callback
  // --------------------------------------------------------

  rclcpp_action::GoalResponse
  handle_goal(
    const rclcpp_action::GoalUUID &,
    std::shared_ptr<const RegisterBlockAction::Goal> goal)
  {
    RCLCPP_INFO(get_logger(), "Received RegisterBlock goal");
    (void)goal;
    if (goal->cloud.data.empty()) {
      RCLCPP_WARN(get_logger(), "Rejecting goal: empty cloud");
      return rclcpp_action::GoalResponse::REJECT;
    }

    if (goal->mask.data.empty()) {
      RCLCPP_WARN(get_logger(), "Rejecting goal: empty mask");
      return rclcpp_action::GoalResponse::REJECT;
    }
    return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
  }

  rclcpp_action::CancelResponse
  handle_cancel(
    const std::shared_ptr<GoalHandleRegisterBlock> goal_handle)
  {
    RCLCPP_WARN(get_logger(), "RegisterBlock goal cancelled");
    (void)goal_handle;
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
    const auto goal = goal_handle->get_goal();
    auto result =
      std::make_shared<RegisterBlockAction::Result>();

    TicToc tt;

    LOG(get_logger(), "execute() start");

    geometry_msgs::msg::TransformStamped tf_cloud;
    if (!lookupCloudTransform(*goal, tf_cloud)) {
      result->success = false;
      goal_handle->abort(result);
      return;
    }
    if (check_cancel(goal_handle, result)) {return;}

    cv::Mat mask = convertMask(*goal);
    publishDebugMask(*goal, mask);
    if (check_cancel(goal_handle, result)) {return;}

    auto scene = convertCloud(*goal);
    if (scene->points_.empty()) {
      result->success = false;
      goal_handle->abort(result);
      return;
    }
    if (check_cancel(goal_handle, result)) {return;}

    geometry::PointCloud cutout;
    if (!computeCutout(*scene, mask, cutout)) {
      result->success = false;
      goal_handle->abort(result);
      return;
    }
    if (check_cancel(goal_handle, result)) {return;}

    preprocessCutout(cutout, tf_cloud);

    GlobalRegistrationResult glob;
    if (!runGlobalRegistration(cutout, glob)) {
      result->success = false;
      goal_handle->abort(result);
      return;
    }

    LocalRegistrationResult reg;
    if (!runLocalRegistration(cutout, glob, reg)) {
      result->success = false;
      goal_handle->abort(result);
      return;
    }

    publishDebugVisualization(*goal, cutout, reg);

    result->pose = to_ros_pose(reg.icp.transformation_);
    result->fitness = reg.icp.fitness_;
    result->rmse = reg.icp.inlier_rmse_;
    result->success = true;

    goal_handle->succeed(result);

    RCLCPP_INFO(
      get_logger(),
      "EXEC DONE in %.1f ms | pts=%zu",
      tt.total(),
      cutout.points_.size());
  }

  // --------------------------------------------------------
  // Helpers
  // --------------------------------------------------------
  bool check_cancel(
    const std::shared_ptr<GoalHandleRegisterBlock> & goal_handle,
    std::shared_ptr<RegisterBlockAction::Result> & result)
  {
    if (goal_handle->is_canceling()) {
      result->success = false;
      goal_handle->canceled(result);
      RCLCPP_WARN(get_logger(), "Goal canceled");
      return true;
    }
    return false;
  }

  bool lookupCloudTransform(
    const RegisterBlockAction::Goal & goal,
    geometry_msgs::msg::TransformStamped & tf_out)
  {
    try {
      tf_out =
        tf_buffer_->lookupTransform(
        world_frame_,
        goal.cloud.header.frame_id,
        rclcpp::Time(goal.cloud.header.stamp),
        rclcpp::Duration::from_seconds(0.5));
      return true;
    } catch (const tf2::TransformException & ex) {
      RCLCPP_WARN(
        get_logger(),
        "TF lookup failed (%s -> %s): %s",
        goal.cloud.header.frame_id.c_str(),
        world_frame_.c_str(),
        ex.what());
      return false;
    }
  }

  cv::Mat convertMask(
    const RegisterBlockAction::Goal & goal)
  {
    return cv_bridge::toCvCopy(
      goal.mask, "mono8")->image;
  }

  void publishDebugMask(
    const RegisterBlockAction::Goal & goal,
    const cv::Mat & mask)
  {
    if (!publish_debug_mask_ || !debug_mask_pub_) {return;}

    cv::Mat mask_vis;
    mask.convertTo(mask_vis, CV_8UC1, 255.0);

    auto msg =
      cv_bridge::CvImage(
      goal.mask.header,
      "mono8",
      mask_vis).toImageMsg();

    debug_mask_pub_->publish(*msg);
  }

  std::shared_ptr<geometry::PointCloud>
  convertCloud(const RegisterBlockAction::Goal & goal)
  {
    return pointcloud2_to_open3d(goal.cloud);
  }

  bool computeCutout(
    const geometry::PointCloud & scene,
    const cv::Mat & mask,
    geometry::PointCloud & cutout)
  {
    auto pts =
      select_points_by_mask(
      scene.points_, mask, K_, T_P_C_);

    if (pts.empty()) {
      RCLCPP_WARN(get_logger(), "No points after masking");
      return false;
    }

    cutout.points_ = pts;
    cutout.EstimateNormals();
    return true;
  }

  void preprocessCutout(
    geometry::PointCloud & cutout,
    const geometry_msgs::msg::TransformStamped & tf)
  {
    Eigen::Matrix4d T = transformToEigen(tf);

    cutout.RemoveStatisticalOutliers(20, 2.0);
    cutout.Transform(T);
    cutout.RemoveStatisticalOutliers(20, 2.0);
    cutout.EstimateNormals();
  }

  bool runGlobalRegistration(
    const geometry::PointCloud & cutout,
    GlobalRegistrationResult & out)
  {
    out = compute_global_registration(
      cutout,
      Z_WORLD,
      angle_thresh,
      MAX_PLANES,
      dist_thresh,
      min_inliers);

    return out.success;
  }

  bool runLocalRegistration(
    const geometry::PointCloud & cutout,
    const GlobalRegistrationResult & glob,
    LocalRegistrationResult & out)
  {
    out = compute_local_registration(
      cutout,
      templates_,
      glob,
      icp_dist,
      yaw_step);

    return out.success;
  }

  void publishDebugVisualization(
    const RegisterBlockAction::Goal & goal,
    const geometry::PointCloud & cutout,
    const LocalRegistrationResult & reg)
  {
    if (!publish_debug_cutout_) {return;}

    const rclcpp::Time stamp(goal.cloud.header.stamp);

    // Cutout (red)
    geometry::PointCloud cutout_vis = cutout;
    cutout_vis.PaintUniformColor({1, 0, 0});
    debug_cutout_pub_->publish(
      open3d_to_pointcloud2_colored(
        cutout_vis, world_frame_, stamp));

    // Template (green)
    auto tpl =
      std::make_shared<geometry::PointCloud>(
      *templates_[reg.template_index].pcd);
    tpl->Transform(reg.icp.transformation_);
    tpl->PaintUniformColor({0, 1, 0});

    debug_template_pub_->publish(
      open3d_to_pointcloud2_colored(
        *tpl, world_frame_, stamp));

    // TF frame
    geometry_msgs::msg::TransformStamped tf;
    tf.header.stamp = stamp;
    tf.header.frame_id = world_frame_;
    tf.child_frame_id = "block_debug";

    Eigen::Matrix4d T = reg.icp.transformation_;
    Eigen::Quaterniond q(T.block<3, 3>(0, 0));

    tf.transform.translation.x = T(0, 3);
    tf.transform.translation.y = T(1, 3);
    tf.transform.translation.z = T(2, 3);
    tf.transform.rotation.x = q.x();
    tf.transform.rotation.y = q.y();
    tf.transform.rotation.z = q.z();
    tf.transform.rotation.w = q.w();

    tf_broadcaster_->sendTransform(tf);
  }


  // --------------------------------------------------------
  // Members
  // --------------------------------------------------------
  rclcpp::CallbackGroup::SharedPtr action_cb_group_;
  rclcpp_action::Server<RegisterBlockAction>::SharedPtr action_server_;

  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr debug_cutout_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr debug_template_pub_;
  bool publish_debug_cutout_;

  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr debug_mask_pub_;
  bool publish_debug_mask_;

  Eigen::Matrix4d T_P_C_;
  Eigen::Matrix3d K_;
  std::vector<TemplateData> templates_;
  Eigen::Vector3d Z_WORLD;

  static constexpr int MAX_PLANES = 3;

  std::string calib_yaml;
  std::string template_dir;
  double dist_thresh;
  int min_inliers;
  double icp_dist;
  double angle_thresh;
  int yaw_step;
  std::string world_frame_;

  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  auto node =
    std::make_shared<BlockRegistrationNode>();

  rclcpp::executors::MultiThreadedExecutor exec(
    rclcpp::ExecutorOptions(),  // default options
    std::thread::hardware_concurrency());

  exec.add_node(node);
  exec.spin();

  rclcpp::shutdown();
  return 0;
}
