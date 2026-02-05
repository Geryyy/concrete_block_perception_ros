#include <rclcpp/rclcpp.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>

#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <visualization_msgs/msg/marker.hpp>

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

#include "concrete_block_perception/srv/register_block.hpp"
#include "concrete_block_perception/io_utils.hpp"

using RegisterBlock = concrete_block_perception::srv::RegisterBlock;
using namespace pcd_block;
using namespace open3d;

#define LOG(logger, ...) RCLCPP_INFO(logger, __VA_ARGS__)

class BlockRegistrationNode : public rclcpp::Node
{
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
    declare_parameter<double>("dist_thresh", 0.02);
    declare_parameter<int>("min_inliers", 100);
    declare_parameter<double>("icp_dist", 0.04);
    declare_parameter<double>("angle_thresh_degree", 30.0);
    declare_parameter<int>("yaw_step", 30);

    world_frame_ =
      declare_parameter<std::string>("world_frame", "world");

    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ =
      std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
    tf_broadcaster_ =
      std::make_shared<tf2_ros::TransformBroadcaster>(this);

    calib_yaml = get_parameter("calib_yaml").as_string();
    template_dir = get_parameter("template_dir").as_string();

    if (calib_yaml.empty()) {
      calib_yaml = default_calib_yaml;
    }
    if (template_dir.empty()) {
      template_dir = default_template_dir;
    }

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
    service_cb_group_ =
      create_callback_group(
      rclcpp::CallbackGroupType::MutuallyExclusive);

    service_ = create_service<RegisterBlock>(
      "register_block_pose",
      std::bind(
        &BlockRegistrationNode::handle_request,
        this,
        std::placeholders::_1,
        std::placeholders::_2),
      rmw_qos_profile_services_default,
      service_cb_group_);


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
      "Block registration service ready");
    RCLCPP_INFO(
      get_logger(),
      "  calib_yaml:   %s", calib_yaml.c_str());
    RCLCPP_INFO(
      get_logger(),
      "  template_dir: %s", template_dir.c_str());
  }

private:
  // --------------------------------------------------------
  // Service callback
  // --------------------------------------------------------
  void handle_request(
    const std::shared_ptr<RegisterBlock::Request> req,
    std::shared_ptr<RegisterBlock::Response> res)
  {
    const auto t0 = this->now();
    const auto tid =
      std::hash<std::thread::id>{}(std::this_thread::get_id());

    LOG(
      get_logger(),
      "handle_request() start [tid=%lu]", tid);

    // ------------------------------------------------------------
    // get tf transformation: lidar -> world
    // ------------------------------------------------------------

    const std::string cloud_frame = req->cloud.header.frame_id;
    const rclcpp::Time cloud_time(req->cloud.header.stamp);

    geometry_msgs::msg::TransformStamped tf_cloud_to_world;

    try {
      tf_cloud_to_world =
        tf_buffer_->lookupTransform(
        world_frame_,   // target
        cloud_frame,    // source
        cloud_time,
        rclcpp::Duration::from_seconds(0.1));
    } catch (const tf2::TransformException & ex) {
      RCLCPP_WARN(
        get_logger(),
        "TF lookup failed (%s -> %s): %s",
        cloud_frame.c_str(),
        world_frame_.c_str(),
        ex.what());
      res->success = false;
      return;
    }


    try {
      // ------------------------------------------------------------
      // Mask
      // ------------------------------------------------------------
      cv::Mat mask =
        cv_bridge::toCvCopy(
        req->mask, "mono8")->image;

      // ------------------------------------------------------------
      // Debug: publish segmentation mask (scaled for visualization)
      // ------------------------------------------------------------
      if (publish_debug_mask_ && debug_mask_pub_) {
        cv::Mat mask_vis;
        mask.convertTo(mask_vis, CV_8UC1, 255.0);

        sensor_msgs::msg::Image::SharedPtr mask_msg =
          cv_bridge::CvImage(
          req->mask.header,
          "mono8",
          mask_vis).toImageMsg();

        debug_mask_pub_->publish(*mask_msg);

        LOG(
          get_logger(),
          "Published debug mask (%dx%d)",
          mask_vis.cols,
          mask_vis.rows);
      }


      // ------------------------------------------------------------
      // PointCloud2 → Open3D
      // ------------------------------------------------------------
      LOG(
        get_logger(),
        "Incoming cloud: width=%u height=%u data=%zu bytes",
        req->cloud.width,
        req->cloud.height,
        req->cloud.data.size());

      auto scene =
        pointcloud2_to_open3d(req->cloud);

      LOG(
        get_logger(),
        "Converted cloud: %zu points",
        scene->points_.size());

      if (scene->points_.empty()) {
        res->success = false;
        LOG(get_logger(), "Empty scene cloud after conversion");
        return;
      }

      // ------------------------------------------------------------
      // Mask-based cutout
      // ------------------------------------------------------------
      LOG(get_logger(), "Mask-based pcl cutout");

      auto pts_sel = select_points_by_mask(
        scene->points_,
        mask,
        K_,
        T_P_C_);

      if (pts_sel.empty()) {
        res->success = false;
        LOG(get_logger(), "No points after masking");
        return;
      }

      geometry::PointCloud cutout;
      cutout.points_ = pts_sel;
      cutout.EstimateNormals();


      // ------------------------------------------------------------
      // Pre-process pointcloud
      // ------------------------------------------------------------

      // Convert TF → Eigen
      Eigen::Matrix4d T_world_cloud =
        transformToEigen(tf_cloud_to_world);

      // Optional filtering before transform
      cutout.RemoveStatisticalOutliers(20, 2.0);

      // Apply transform
      cutout.Transform(T_world_cloud);
      cutout.RemoveStatisticalOutliers(20, 2.0);
      cutout.EstimateNormals();

      // ------------------------------------------------------------
      // Global registration
      // ------------------------------------------------------------
      LOG(get_logger(), "Global registration");

      GlobalRegistrationResult globreg =
        compute_global_registration(
        cutout,
        Z_WORLD,
        angle_thresh,
        MAX_PLANES,
        dist_thresh,
        min_inliers);

      if (globreg.success == false) {
        res->success = false;
        return;
      }

      // ------------------------------------------------------------
      // Local registration
      // ------------------------------------------------------------
      LOG(get_logger(), "Local registration");

      LocalRegistrationResult result =
        compute_local_registration(
        cutout,
        templates_,
        globreg,
        icp_dist,
        yaw_step);

      if (!result.success) {
        res->success = false;
        LOG(get_logger(), "Local registration failed");
        return;
      }

      // ------------------------------------------------------------
      // debug visualize cutout and fitted template with frame
      // ------------------------------------------------------------
      if (publish_debug_cutout_) {

        const auto stamp =
          rclcpp::Time(req->cloud.header.stamp);
        const auto frame_id = world_frame_;

        // ------------------------------------------------------------
        // 1) CUTOUT CLOUD (RED)
        // ------------------------------------------------------------
        geometry::PointCloud cutout_vis = cutout;
        cutout_vis.PaintUniformColor({1.0, 0.0, 0.0});

        auto cutout_msg =
          open3d_to_pointcloud2_colored(
          cutout_vis, frame_id, stamp);

        debug_cutout_pub_->publish(cutout_msg);

        // ------------------------------------------------------------
        // 2) FITTED TEMPLATE (GREEN)
        // ------------------------------------------------------------
        auto template_vis =
          std::make_shared<geometry::PointCloud>(
          *templates_[result.template_index].pcd);

        template_vis->Transform(result.icp.transformation_);
        template_vis->PaintUniformColor({0.0, 1.0, 0.0});

        auto template_msg =
          open3d_to_pointcloud2_colored(
          *template_vis, frame_id, stamp);

        debug_template_pub_->publish(template_msg);

        // ------------------------------------------------------------
        // Publish estimated block pose as TF
        // ------------------------------------------------------------
        geometry_msgs::msg::TransformStamped tf_msg;

        tf_msg.header.stamp = stamp;
        tf_msg.header.frame_id = world_frame_; // e.g. "world"
        tf_msg.child_frame_id = "block_debug";

        Eigen::Matrix4d T = result.icp.transformation_;
        Eigen::Quaterniond q(T.block<3, 3>(0, 0));

        tf_msg.transform.translation.x = T(0, 3);
        tf_msg.transform.translation.y = T(1, 3);
        tf_msg.transform.translation.z = T(2, 3);

        tf_msg.transform.rotation.x = q.x();
        tf_msg.transform.rotation.y = q.y();
        tf_msg.transform.rotation.z = q.z();
        tf_msg.transform.rotation.w = q.w();

        tf_broadcaster_->sendTransform(tf_msg);

        LOG(
          get_logger(),
          "Published debug vis: cutout=%zu template=%zu",
          cutout.points_.size(),
          template_vis->points_.size());
      }


      LOG(
        get_logger(),
        "Cutout stats: n=%zu (min required=%d)",
        cutout.points_.size(),
        min_inliers);


      // ------------------------------------------------------------
      // Response
      // ------------------------------------------------------------
      res->pose = to_ros_pose(
        result.icp.transformation_);
      res->fitness = result.icp.fitness_;
      res->rmse = result.icp.inlier_rmse_;
      res->success = true;

    } catch (const std::exception & e) {
      RCLCPP_ERROR(
        get_logger(),
        "Registration failed: %s", e.what());
      res->success = false;
    }

    LOG(
      get_logger(),
      "handle_request() finish (%.3f s)",
      (this->now() - t0).seconds());
  }

  static Eigen::Matrix4d
  transformToEigen(const geometry_msgs::msg::TransformStamped & tf)
  {
    Eigen::Quaterniond q(
      tf.transform.rotation.w,
      tf.transform.rotation.x,
      tf.transform.rotation.y,
      tf.transform.rotation.z);

    Eigen::Vector3d t(
      tf.transform.translation.x,
      tf.transform.translation.y,
      tf.transform.translation.z);

    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.block<3, 3>(0, 0) = q.toRotationMatrix();
    T.block<3, 1>(0, 3) = t;
    return T;
  }


  // --------------------------------------------------------
  // Members
  // --------------------------------------------------------
  rclcpp::Service<RegisterBlock>::SharedPtr service_;
  rclcpp::CallbackGroup::SharedPtr service_cb_group_;

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

  try {
    auto node =
      std::make_shared<BlockRegistrationNode>();
    rclcpp::spin(node);
  } catch (const std::exception & e) {
    RCLCPP_FATAL(
      rclcpp::get_logger("block_registration_node"),
      "Fatal error during startup: %s",
      e.what());
  }

  rclcpp::shutdown();
  return 0;
}
