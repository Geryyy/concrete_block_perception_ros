#include <rclcpp/rclcpp.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>

#include <vision_msgs/msg/detection2_d_array.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <Eigen/Dense>
#include <cv_bridge/cv_bridge.h>

#include "concrete_block_perception/msg/block.hpp"
#include "concrete_block_perception/msg/block_array.hpp"

#include "concrete_block_perception/img_utils.hpp"
#include "concrete_block_perception/io_utils.hpp"
#include "concrete_block_perception/visu_utils.hpp"

#include "pcd_block_estimation/mask_projection.hpp"
#include "pcd_block_estimation/utils.hpp"
#include "pcd_block_estimation/yaml_utils.hpp"

#include "concrete_block_perception/tracking/multi_object_tracker.hpp"
#include "concrete_block_perception/tracking/tracker_config.hpp"
#include "concrete_block_perception/tracking/measurement.hpp"

using Block = concrete_block_perception::msg::Block;
using BlockArray = concrete_block_perception::msg::BlockArray;
using namespace pcd_block;

class WorldModelNode : public rclcpp::Node
{
public:
  WorldModelNode()
  : Node("block_world_model_node")
  {
    // ----------------------------
    // Parameters
    // ----------------------------
    declare_parameter<std::string>("calib_yaml", "");
// ----------------------------
// World model
// ----------------------------
    declare_parameter<std::string>("world_model.world_frame", "world");
    declare_parameter<int>("world_model.min_points", 30);

// ----------------------------
// Tracker – lifecycle / gating
// ----------------------------
    declare_parameter<int>("tracker.min_hits", 5);
    declare_parameter<int>("tracker.max_misses", 10);
    declare_parameter<double>("tracker.chi2_gate", 7.815);

    declare_parameter<double>("tracker.deduplication_radius", 0.5);
    declare_parameter<double>("tracker.iou_thresh", 0.3);
    declare_parameter<double>("tracker.birth_suppression_radius", 0.5);

// ----------------------------
// Tracker – motion model
// ----------------------------
    declare_parameter<double>("tracker.velocity_damping", 0.0);

// ----------------------------
// Tracker – process noise
// ----------------------------
    declare_parameter<double>("tracker.process_noise.px", 1e-6);
    declare_parameter<double>("tracker.process_noise.py", 1e-6);
    declare_parameter<double>("tracker.process_noise.pz", 1e-6);
    declare_parameter<double>("tracker.process_noise.yaw", 1e-6);
    declare_parameter<double>("tracker.process_noise.vx", 5e-4);
    declare_parameter<double>("tracker.process_noise.vy", 5e-4);
    declare_parameter<double>("tracker.process_noise.vz", 5e-4);

// ----------------------------
// Tracker – measurement noise
// ----------------------------
    declare_parameter<double>("tracker.measurement_noise.px", 0.1);
    declare_parameter<double>("tracker.measurement_noise.py", 0.1);
    declare_parameter<double>("tracker.measurement_noise.pz", 0.1);
    declare_parameter<double>("tracker.measurement_noise.yaw", 0.15);

    // ----------------------------
    // Read parameters
    // ----------------------------
    calib_yaml_ = get_parameter("calib_yaml").as_string();
    // ----------------------------
// Read world model params
// ----------------------------
    world_frame_ =
      get_parameter("world_model.world_frame").as_string();

    min_points_ =
      get_parameter("world_model.min_points").as_int();

    // ----------------------------
    // Read tracker lifecycle params
    // ----------------------------
    tracker_cfg_.min_hits =
      get_parameter("tracker.min_hits").as_int();

    tracker_cfg_.max_misses =
      get_parameter("tracker.max_misses").as_int();

    tracker_cfg_.chi2_gate =
      get_parameter("tracker.chi2_gate").as_double();

    tracker_cfg_.deduplication_radius =
      get_parameter("tracker.deduplication_radius").as_double();

    tracker_cfg_.iou_thresh =
      get_parameter("tracker.iou_thresh").as_double();

    tracker_cfg_.birth_suppression_radius =
      get_parameter("tracker.birth_suppression_radius").as_double();

    // ----------------------------
    // Read tracker motion model
    // ----------------------------
    tracker_cfg_.velocity_damping =
      get_parameter("tracker.velocity_damping").as_double();

    tracker_cfg_.Q.setZero();

    tracker_cfg_.Q.diagonal() <<
      get_parameter("tracker.process_noise.px").as_double(),
      get_parameter("tracker.process_noise.py").as_double(),
      get_parameter("tracker.process_noise.pz").as_double(),
      get_parameter("tracker.process_noise.yaw").as_double(),
      get_parameter("tracker.process_noise.vx").as_double(),
      get_parameter("tracker.process_noise.vy").as_double(),
      get_parameter("tracker.process_noise.vz").as_double();


    tracker_cfg_.R.setZero();

    tracker_cfg_.R.diagonal() <<
      std::pow(get_parameter("tracker.measurement_noise.px").as_double(), 2),
      std::pow(get_parameter("tracker.measurement_noise.py").as_double(), 2),
      std::pow(get_parameter("tracker.measurement_noise.pz").as_double(), 2),
      std::pow(get_parameter("tracker.measurement_noise.yaw").as_double(), 2);


    tracker_ =
      std::make_unique<cbp::tracking::MultiObjectTracker>(tracker_cfg_, this->get_logger());

    RCLCPP_INFO(
      get_logger(),
      "Tracker: min_hits=%d max_misses=%d chi2=%.2f dedup=%.2f iou=%.2f birth=%.2f vel_damp=%.2f",
      tracker_cfg_.min_hits,
      tracker_cfg_.max_misses,
      tracker_cfg_.chi2_gate,
      tracker_cfg_.deduplication_radius,
      tracker_cfg_.iou_thresh,
      tracker_cfg_.birth_suppression_radius,
      tracker_cfg_.velocity_damping);

    // ----------------------------
    // Publishers
    // ----------------------------
    world_pub_ =
      create_publisher<BlockArray>("block_world_model", 10);

    marker_pub_ =
      create_publisher<visualization_msgs::msg::MarkerArray>(
      "block_markers", 10);

    // ----------------------------
    // Subscribers (sync)
    // ----------------------------
    det_sub_.subscribe(this, "detections");
    cloud_sub_.subscribe(this, "points");
    image_sub_.subscribe(this, "image");

    sync_ = std::make_shared<Synchronizer>(
      SyncPolicy(10), det_sub_, cloud_sub_, image_sub_);

    sync_->registerCallback(
      std::bind(
        &WorldModelNode::callback,
        this,
        std::placeholders::_1,
        std::placeholders::_2,
        std::placeholders::_3));

    // ----------------------------
    // TF
    // ----------------------------
    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(get_clock());
    tf_listener_ =
      std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    // ----------------------------
    // Calibration
    // ----------------------------
    T_P_C_ = load_T_4x4(calib_yaml_);
    K_ = load_camera_matrix(calib_yaml_);

    RCLCPP_INFO(get_logger(), "World model node started");
  }

private:
  // ------------------------------------------------------------
  // Callback
  // ------------------------------------------------------------
  void callback(
    const vision_msgs::msg::Detection2DArray::ConstSharedPtr & detections,
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr & cloud,
    const sensor_msgs::msg::Image::ConstSharedPtr & mask_msg)
  {
    std::vector<cbp::tracking::Measurement> measurements;

    const auto stamp = cloud->header.stamp;
    const auto scene = pointcloud2_to_open3d(*cloud);
    const cv::Mat full_mask =
      cv_bridge::toCvCopy(mask_msg, "mono8")->image;

    const int img_w = full_mask.cols;
    const int img_h = full_mask.rows;

    for (const auto & det : detections->detections) {

      const cv::Mat det_mask = extract_mask_roi(full_mask, det);
      if (det_mask.empty()) {
        continue;
      }

      const auto pts = select_points_by_mask(
        scene->points_, det_mask, K_, T_P_C_);

      if (static_cast<int>(pts.size()) < min_points_) {
        continue;
      }

      const Eigen::Vector3d c = compute_center(pts);

      geometry_msgs::msg::Pose pose_lidar;
      pose_lidar.position.x = c.x();
      pose_lidar.position.y = c.y();
      pose_lidar.position.z = c.z();
      pose_lidar.orientation.w = 1.0;

      geometry_msgs::msg::PoseStamped pose_world;
      try {
        pose_world = transform_pose_to_world(
          pose_lidar, cloud->header.frame_id, stamp);
      } catch (const tf2::TransformException &) {
        continue;
      }

      cbp::tracking::Measurement m;
      m.position = Eigen::Vector3d(
        pose_world.pose.position.x,
        pose_world.pose.position.y,
        pose_world.pose.position.z);

      m.yaw = 0.0;
      m.R = tracker_cfg_.R;
      m.confidence =
        det.results.empty() ? 0.0 : det.results[0].hypothesis.score;
      m.stamp = stamp;

      // --------------------------------------------------
      // Bounding box (image space, CLAMPED)
      // --------------------------------------------------
      const auto & bb = det.bbox;

      int x = static_cast<int>(bb.center.position.x - 0.5 * bb.size_x);
      int y = static_cast<int>(bb.center.position.y - 0.5 * bb.size_y);
      int w = static_cast<int>(bb.size_x);
      int h = static_cast<int>(bb.size_y);

      // Clamp to image bounds
      x = std::clamp(x, 0, img_w - 1);
      y = std::clamp(y, 0, img_h - 1);
      w = std::min(w, img_w - x);
      h = std::min(h, img_h - y);

      // Skip degenerate boxes
      if (w <= 0 || h <= 0) {
        continue;
      }

      m.bbox = cv::Rect(x, y, w, h);

      // --------------------------------------------------
      // Mask (full image, CLONED for safety)
      // --------------------------------------------------
      m.mask = det_mask.clone();

      // --------------------------------------------------
      // Debug output (safe)
      // --------------------------------------------------
      RCLCPP_DEBUG_STREAM(get_logger(), m.to_string());

      measurements.push_back(std::move(m));
    }

    tracker_->step(measurements, stamp);

    publish();
    publish_markers();
  }


  geometry_msgs::msg::PoseStamped
  transform_pose_to_world(
    const geometry_msgs::msg::Pose & pose,
    const std::string & frame,
    const rclcpp::Time & stamp)
  {
    geometry_msgs::msg::PoseStamped in, out;
    in.header.frame_id = frame;
    in.header.stamp = stamp;
    in.pose = pose;

    return tf_buffer_->transform(
      in, world_frame_, tf2::durationFromSec(0.1));
  }

  // ------------------------------------------------------------
  // Publishing
  // ------------------------------------------------------------
  void publish()
  {
    BlockArray msg;
    msg.header.frame_id = world_frame_;
    msg.header.stamp = now();

    for (const auto & track : tracker_->tracks()) {

      if (track.hits < tracker_cfg_.min_hits) {
        continue;
      }

      Block b;
      b.id = "block_" + std::to_string(track.id);

      b.pose.position.x = track.kf.x()(0);
      b.pose.position.y = track.kf.x()(1);
      b.pose.position.z = track.kf.x()(2);
      b.pose.orientation.w = 1.0;

      b.confidence = track.confidence;
      b.last_seen = track.last_update;

      b.pose_status = track.pose_status;
      b.task_status = track.task_status;

      msg.blocks.push_back(b);
    }

    world_pub_->publish(msg);
  }

  void publish_markers()
  {
    visualization_msgs::msg::MarkerArray arr;

    for (const auto & track : tracker_->tracks()) {

      if (track.hits < tracker_cfg_.min_hits) {
        continue;
      }

      Block b;
      b.id = "block_" + std::to_string(track.id);

      b.pose.position.x = track.kf.x()(0);
      b.pose.position.y = track.kf.x()(1);
      b.pose.position.z = track.kf.x()(2);
      b.pose.orientation.w = 1.0;

      b.confidence = track.confidence;
      int id = track.id;
      b.id = "block_" + std::to_string(track.id);
      b.pose_status = track.pose_status;
      b.task_status = track.task_status;

      arr.markers.push_back(
        make_block_marker(b, world_frame_, id, now()));
      arr.markers.push_back(
        make_text_marker(b, world_frame_, id, now()));

    }

    marker_pub_->publish(arr);
  }

private:
  using SyncPolicy =
    message_filters::sync_policies::ApproximateTime<
    vision_msgs::msg::Detection2DArray,
    sensor_msgs::msg::PointCloud2,
    sensor_msgs::msg::Image>;

  using Synchronizer =
    message_filters::Synchronizer<SyncPolicy>;

  message_filters::Subscriber<vision_msgs::msg::Detection2DArray> det_sub_;
  message_filters::Subscriber<sensor_msgs::msg::PointCloud2> cloud_sub_;
  message_filters::Subscriber<sensor_msgs::msg::Image> image_sub_;
  std::shared_ptr<Synchronizer> sync_;

  rclcpp::Publisher<BlockArray>::SharedPtr world_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;

  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  std::unique_ptr<cbp::tracking::MultiObjectTracker> tracker_;
  cbp::tracking::TrackerConfig tracker_cfg_;

  std::string calib_yaml_;
  std::string world_frame_;
  int min_points_{30};

  Eigen::Matrix3d K_;
  Eigen::Matrix4d T_P_C_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<WorldModelNode>());
  rclcpp::shutdown();
  return 0;
}
