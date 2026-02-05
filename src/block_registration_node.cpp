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


struct PreprocessingParams
{
  size_t max_pts = 200;
  // open3D RemoveStatisticalOutliers
  int nb_neighbors = 20;
  double std_dev = 2.0;
};
struct GlobalRegistrationParams
{
  static constexpr int MAX_PLANES = 3;
  Eigen::Vector3d Z_WORLD = Eigen::Vector3d(0.0, 0.0, 1.0);
  double dist_thresh;
  int min_inliers;
  double angle_thresh;
};

struct LocalRegistrationParams
{
  double icp_dist;
  int yaw_step;
};

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

    const std::string config_dir =
      pkg_share + "/config";

    // ------------------------------------------------------------
    // Parameters
    // ------------------------------------------------------------
    declare_parameter<std::string>("calib_yaml", "");
    declare_parameter<std::string>("world_frame", "world");

    declare_parameter<std::string>("template.dir", "/templates");
    declare_parameter<std::string>("template.cad_name", "ConcreteBlock.ply");
    declare_parameter<int>("template.n_points", 2000);
    declare_parameter<double>("template.angle_deg", 15.0);

    declare_parameter<int>("preproc.max_pts", 500);
    declare_parameter<int>("preproc.nb_neighbors", 20);
    declare_parameter<double>("preproc.std_dev", 2.0);

    declare_parameter<double>("glob_reg.dist_thresh", 0.02);
    declare_parameter<int>("glob_reg.min_inliers", 100);
    declare_parameter<double>("glob_reg.angle_thresh_degree", 30.0);

    declare_parameter<double>("loc_reg.icp_dist", 0.04);
    declare_parameter<int>("loc_reg.yaw_step", 30);

    auto calib_yaml_name = get_parameter("calib_yaml").as_string();
    calib_yaml_ = config_dir + "/" + calib_yaml_name;
    world_frame_ = get_parameter("world_frame").as_string();

    auto template_dir_ = get_parameter("template.dir").as_string();
    auto template_cad_name_ = get_parameter("template.cad_name").as_string();
    template_params_.n_points = get_parameter("template.n_points").as_int();
    template_params_.angle_deg = get_parameter("template.angle_deg").as_double();
    template_params_.cad_path = config_dir + "/" + template_cad_name_;
    template_params_.out_dir = config_dir + template_dir_;

    preproc_params_.max_pts = get_parameter("preproc.max_pts").as_int();
    preproc_params_.nb_neighbors = get_parameter("preproc.nb_neighbors").as_int();
    preproc_params_.std_dev = get_parameter("preproc.std_dev").as_double();

    double angle_thresh_deg =
      get_parameter("glob_reg.angle_thresh_degree").as_double();
    glob_reg_params_.angle_thresh = std::cos(angle_thresh_deg * M_PI / 180.0);
    glob_reg_params_.dist_thresh = get_parameter("glob_reg.dist_thresh").as_double();
    glob_reg_params_.min_inliers = get_parameter("glob_reg.min_inliers").as_int();

    loc_reg_params_.icp_dist = get_parameter("loc_reg.icp_dist").as_double();
    loc_reg_params_.yaw_step = get_parameter("loc_reg.yaw_step").as_int();

    if (calib_yaml_.empty()) {
      throw std::runtime_error(
              "Parameter 'calib_yaml' is empty. "
              "Expected path to calibration YAML.");
    }

    if (template_params_.out_dir.empty()) {
      throw std::runtime_error(
              "Parameter 'template_dir' is empty. "
              "Expected directory containing templates.");
    }

    if (template_cad_name_.empty()) {
      throw std::runtime_error(
              "Parameter 'template_cad_name' is empty. "
              "Expected CAD filename (e.g. ConcreteBlock.ply).");
    }

    // ------------------------------------------------------------
    // Validate files
    // ------------------------------------------------------------
    if (!std::filesystem::exists(calib_yaml_)) {
      throw std::runtime_error(
              "Calibration YAML not found: " + calib_yaml_);
    }
    if (!std::filesystem::exists(template_params_.out_dir)) {
      RCLCPP_INFO(get_logger(), "Generating templates from %s", template_params_.cad_path.c_str());
      generate_templates(template_params_);
    }

    // ------------------------------------------------------------
    // Load data
    // ------------------------------------------------------------
    T_P_C_ = load_T_4x4(calib_yaml_);
    K_ = load_camera_matrix(calib_yaml_);
    templates_ = load_templates(template_params_.out_dir);

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


    RCLCPP_INFO(
      get_logger(),
      "Block registration action ready");
    RCLCPP_INFO(
      get_logger(),
      "  calib_yaml:   %s", calib_yaml_.c_str());
    RCLCPP_INFO(
      get_logger(),
      "  template_dir: %s", template_params_.out_dir.c_str());
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

    TicToc tt_total;
    TicToc tt_stage;

    auto publish_feedback =
      [&](const std::string & stage, float progress)
      {
        RegisterBlockAction::Feedback fb;
        fb.stage = stage;
        fb.progress = progress;
        fb.elapsed_ms = tt_total.total();
        goal_handle->publish_feedback(
          std::make_shared<RegisterBlockAction::Feedback>(fb));

        tt_stage.tic();
      };

    LOG(get_logger(), "execute() start");

    publish_feedback("tf_lookup", 0.1f);
    geometry_msgs::msg::TransformStamped tf_cloud;
    if (!lookupCloudTransform(*goal, tf_cloud)) {
      result->success = false;
      goal_handle->abort(result);
      return;
    }
    if (check_cancel(goal_handle, result)) {return;}
    RCLCPP_INFO(
      get_logger(),
      "[tf_lookup] stage took %.1f ms",
      tt_stage.toc());


    publish_feedback("mask_conversion", 0.2f);
    cv::Mat mask = convertMask(*goal);
    publishDebugMask(*goal, mask);
    if (check_cancel(goal_handle, result)) {return;}
    RCLCPP_INFO(
      get_logger(),
      "[mask_conversion] stage took %.1f ms",
      tt_stage.toc());

    publish_feedback("cloud_conversion", 0.3f);
    auto scene = convertCloud(*goal);
    if (scene->points_.empty()) {
      result->success = false;
      goal_handle->abort(result);
      return;
    }
    if (check_cancel(goal_handle, result)) {return;}
    RCLCPP_INFO(
      get_logger(),
      "[cloud_conversion] stage took %.1f ms",
      tt_stage.toc());

    publish_feedback("cutout", 0.4f);
    geometry::PointCloud cutout;
    if (!computeCutout(*scene, mask, cutout)) {
      result->success = false;
      goal_handle->abort(result);
      return;
    }
    if (check_cancel(goal_handle, result)) {return;}
    RCLCPP_INFO(
      get_logger(),
      "[cutout] stage took %.1f ms",
      tt_stage.toc());

    publish_feedback("preprocess", 0.5f);
    preprocessCutout(cutout, tf_cloud);
    RCLCPP_INFO(
      get_logger(),
      "[preprocess] stage took %.1f ms",
      tt_stage.toc());

    publish_feedback("global_registration", 0.7f);
    GlobalRegistrationResult glob;
    if (!runGlobalRegistration(cutout, glob)) {
      result->success = false;
      goal_handle->abort(result);
      return;
    }

    // debug purpose:
    auto transform_ = globalResultToTransform(glob);
    result->pose = to_ros_pose(transform_);
    result->success = true;
    auto transformation = transform_;
    int template_index = 0;

    RCLCPP_INFO(
      get_logger(),
      "[global_registration] stage took %.1f ms",
      tt_stage.toc());

    // publish_feedback("local_registration", 0.9f);
    // LocalRegistrationResult reg;
    // if (!runLocalRegistration(cutout, glob, reg)) {
    //   result->success = false;
    //   goal_handle->abort(result);
    //   return;
    // }
    // RCLCPP_INFO(
    //   get_logger(),
    //   "[local_registration] stage took %.1f ms",
    //   tt_stage.toc());
    // auto transformation = reg.icp.transformations_;
    // int template_index = reg.template_index;

    publish_feedback("visualization", 0.95f);

    publishDebugVisualization(*goal, cutout, template_index, transformation);
    RCLCPP_INFO(
      get_logger(),
      "[visualization] stage took %.1f ms",
      tt_stage.toc());

    // result->pose = to_ros_pose(reg.icp.transformation_);
    // result->fitness = reg.icp.fitness_;
    // result->rmse = reg.icp.inlier_rmse_;
    // result->success = true;

    goal_handle->succeed(result);
    publish_feedback("done", 1.0f);

    RCLCPP_INFO(
      get_logger(),
      "EXEC DONE in %.1f ms | pts=%zu",
      tt_total.total(),
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
    open3d::geometry::PointCloud & cutout,
    const geometry_msgs::msg::TransformStamped & tf)
  {
    std::shared_ptr<open3d::geometry::PointCloud> pcd;
    std::vector<size_t> ind;

    std::tie(pcd, ind) =
      cutout.RemoveStatisticalOutliers(
      preproc_params_.nb_neighbors,
      preproc_params_.std_dev);

    cutout = *pcd;

    if (cutout.points_.size() > preproc_params_.max_pts) {
      std::vector<size_t> idx(cutout.points_.size());
      std::iota(idx.begin(), idx.end(), 0);
      static thread_local std::mt19937 rng{42};
      std::shuffle(idx.begin(), idx.end(), rng);
      idx.resize(preproc_params_.max_pts);
      cutout = *cutout.SelectByIndex(idx);
    }

    cutout.Transform(transformToEigen(tf));
    cutout.EstimateNormals(
      open3d::geometry::KDTreeSearchParamHybrid(0.02, 30));
  }


  bool runGlobalRegistration(
    const geometry::PointCloud & cutout,
    GlobalRegistrationResult & out)
  {
    out = compute_global_registration(
      cutout,
      glob_reg_params_.Z_WORLD,
      glob_reg_params_.angle_thresh,
      glob_reg_params_.MAX_PLANES,
      glob_reg_params_.dist_thresh,
      glob_reg_params_.min_inliers);

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
      loc_reg_params_.icp_dist,
      loc_reg_params_.yaw_step);

    return out.success;
  }

  void publishDebugVisualization(
    const RegisterBlockAction::Goal & goal,
    const geometry::PointCloud & cutout,
    const int template_index,
    const Eigen::Matrix4d homogeneous_transformation)
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
      *templates_[template_index].pcd);
    tpl->Transform(homogeneous_transformation);
    tpl->PaintUniformColor({0, 1, 0});

    debug_template_pub_->publish(
      open3d_to_pointcloud2_colored(
        *tpl, world_frame_, stamp));

    // TF frame
    geometry_msgs::msg::TransformStamped tf;
    tf.header.stamp = stamp;
    tf.header.frame_id = world_frame_;
    tf.child_frame_id = "block_debug";

    Eigen::Matrix4d T = homogeneous_transformation;
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


  std::string calib_yaml_;
  std::string world_frame_;
  TemplateGenerationParams template_params_;
  PreprocessingParams preproc_params_;
  GlobalRegistrationParams glob_reg_params_;
  LocalRegistrationParams loc_reg_params_;

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
