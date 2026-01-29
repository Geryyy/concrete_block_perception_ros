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

#include <unordered_map>
#include <string>

#include <Eigen/Dense>
#include <cv_bridge/cv_bridge.h>
#include "concrete_block_perception/msg/block.hpp"
#include "concrete_block_perception/msg/block_array.hpp"

#include "concrete_block_perception/io_utils.hpp"
#include "concrete_block_perception/img_utils.hpp"
#include "pcd_block_estimation/mask_projection.hpp"
#include "pcd_block_estimation/utils.hpp"
#include "pcd_block_estimation/yaml_utils.hpp"

using Block = concrete_block_perception::msg::Block;
using BlockArray = concrete_block_perception::msg::BlockArray;
using namespace pcd_block;


static std::string poseStatusToString(int status)
{
  switch (status) {
    case Block::POSE_COARSE:
      return "coarse";
    case Block::POSE_PRECISE:
      return "precise";
    case Block::POSE_UNKNOWN:
    default:
      return "unknown";
  }
}


static std::string taskStatusToString(int status)
{
  switch (status) {
    case Block::TASK_FREE:
      return "free";
    case Block::TASK_PLACED:
      return "placed";
    case Block::TASK_MOVE:
      return "move";
    case Block::TASK_REMOVED:
      return "removed";
    case Block::TASK_UNKNOWN:
    default:
      return "unknown";
  }
}


class WorldModelNode : public rclcpp::Node
{
public:
  WorldModelNode()
  : Node("block_world_model_node")
  {
    // ----------------------------
    // Parameters
    // ----------------------------
    const std::string default_calib_yaml =
      ament_index_cpp::get_package_share_directory(
      "concrete_block_perception") +
      "/config/calib_zed2i_to_seyond.yaml";


    declare_parameter<std::string>("calib_yaml", default_calib_yaml);
    declare_parameter<std::string>("world_frame", "world");
    declare_parameter<double>("assoc_dist", 0.15);
    declare_parameter<int>("min_points", 30);

    calib_yaml_ = get_parameter("calib_yaml").as_string();
    world_frame_ = get_parameter("world_frame").as_string();
    assoc_dist_ = get_parameter("assoc_dist").as_double();
    min_points_ = get_parameter("min_points").as_int();

    // ----------------------------
    // Publisher
    // ----------------------------
    world_pub_ =
      create_publisher<BlockArray>("block_world_model", 10);

    marker_pub_ =
      create_publisher<visualization_msgs::msg::MarkerArray>(
      "block_markers", 10);

    // ----------------------------
    // Subscribers (synchronized)
    // ----------------------------
    det_sub_.subscribe(this, "detections");
    cloud_sub_.subscribe(this, "points");
    image_sub_.subscribe(this, "image");

    sync_ = std::make_shared<Synchronizer>(
      SyncPolicy(10),
      det_sub_,
      cloud_sub_,
      image_sub_
    );

    sync_->registerCallback(
      std::bind(
        &WorldModelNode::callback,
        this,
        std::placeholders::_1,
        std::placeholders::_2,
        std::placeholders::_3
      )
    );

    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ =
      std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    if (calib_yaml_.empty()) {
      throw std::runtime_error("Missing parameters");
    }

    if (world_frame_.empty()) {
      throw std::runtime_error("Parameter 'world_frame' must not be empty");
    }


    T_P_C_ = load_T_4x4(calib_yaml_);
    K_ = load_camera_matrix(calib_yaml_);

    RCLCPP_INFO(get_logger(), "World model node started");
  }

private:
  // ------------------------------------------------------------
  // Internal block state
  // ------------------------------------------------------------
  struct BlockState
  {
    Block msg;
  };

  // ------------------------------------------------------------
  // Callback
  // ------------------------------------------------------------
  void callback(
    const vision_msgs::msg::Detection2DArray::ConstSharedPtr & detections,
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr & cloud,
    const sensor_msgs::msg::Image::ConstSharedPtr & mask_msg)
  {
    auto scene = pointcloud2_to_open3d(*cloud);

    cv::Mat full_mask =
      cv_bridge::toCvCopy(mask_msg, "mono8")->image;

    for (const auto & det : detections->detections) {

      // 🔹 Extract per-detection mask
      cv::Mat det_mask = extract_mask_roi(full_mask, det);

      if (det_mask.empty()) {
        continue;
      }

      auto pts = select_points_by_mask(
        scene->points_,
        det_mask,
        K_,
        T_P_C_
      );

      if (static_cast<int>(pts.size()) < min_points_) {
        continue;
      }

      Eigen::Vector3d c = compute_center(pts);

      geometry_msgs::msg::Pose pose;
      pose.position.x = c.x();
      pose.position.y = c.y();
      pose.position.z = c.z();
      pose.orientation.w = 1.0;

      const std::string id = associate_or_create(pose);

      auto & block = blocks_[id].msg;
      block.pose = pose;
      block.confidence =
        det.results.empty() ?
        0.0f :
        det.results[0].hypothesis.score;
      block.last_seen = now();
    }

    publish();
  }

  geometry_msgs::msg::PoseStamped
  transform_pose_to_world(
    const geometry_msgs::msg::Pose & pose_lidar,
    const std::string & lidar_frame,
    const rclcpp::Time & stamp)
  {
    geometry_msgs::msg::PoseStamped in, out;

    in.header.stamp = stamp;
    in.header.frame_id = lidar_frame;
    in.pose = pose_lidar;

    out = tf_buffer_->transform(
      in,
      world_frame_,
      tf2::durationFromSec(0.1)
    );

    return out;
  }


  // ------------------------------------------------------------
  // Data association (v1)
  // ------------------------------------------------------------
  std::string associate_or_create(
    const geometry_msgs::msg::Pose & pose)
  {
    for (auto & [id, state] : blocks_) {
      if (distance(state.msg.pose, pose) < assoc_dist_) {
        return id;
      }
    }

    // New block
    std::string id = "block_" + std::to_string(next_id_++);
    BlockState state;
    state.msg.id = id;
    blocks_[id] = state;
    return id;
  }

  double distance(
    const geometry_msgs::msg::Pose & a,
    const geometry_msgs::msg::Pose & b)
  {
    const double dx = a.position.x - b.position.x;
    const double dy = a.position.y - b.position.y;
    const double dz = a.position.z - b.position.z;
    return std::sqrt(dx * dx + dy * dy + dz * dz);
  }

  // ------------------------------------------------------------
  // Helpers
  // ------------------------------------------------------------

  visualization_msgs::msg::Marker
  make_block_marker(
    const Block & block,
    const std::string & frame,
    int id)
  {
    visualization_msgs::msg::Marker m;
    m.header.frame_id = frame;
    m.header.stamp = rclcpp::Clock().now();

    m.ns = "blocks";
    m.id = id;
    m.type = visualization_msgs::msg::Marker::CUBE;
    m.action = visualization_msgs::msg::Marker::ADD;

    m.pose = block.pose;

    // Rough concrete block size (adjust!)
    m.scale.x = 0.9;
    m.scale.y = 0.6;
    m.scale.z = 0.6;

    // Color by pose status
    if (block.pose_status == Block::POSE_COARSE) {
      m.color = make_color(1.0, 0.5, 0.0, 0.8); // orange
    } else {
      m.color = make_color(0.0, 1.0, 0.0, 0.8); // green
    }

    return m;
  }

  visualization_msgs::msg::Marker
  make_text_marker(const Block & block, std::string frame, int id)
  {
    visualization_msgs::msg::Marker m;
    m.header.frame_id = frame;
    m.header.stamp = now();

    m.ns = "block_labels";
    m.id = id;
    m.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;

    m.pose = block.pose;
    m.pose.position.z += 0.3;

    m.scale.z = 0.12;
    m.color = make_color(1, 1, 1, 1);

    m.text =
      block.id +
      "\npose=" + poseStatusToString(block.pose_status) +
      "\ntask=" + taskStatusToString(block.task_status);

    return m;
  }

  std_msgs::msg::ColorRGBA make_color(
    float r, float g, float b, float a)
  {
    std_msgs::msg::ColorRGBA c;
    c.r = r; c.g = g; c.b = b; c.a = a;
    return c;
  }

  // ------------------------------------------------------------
  // Publish world model
  // ------------------------------------------------------------
  void publish()
  {
    BlockArray msg;
    msg.header.stamp = now();
    msg.header.frame_id = world_frame_;

    for (const auto & [_, state] : blocks_) {
      msg.blocks.push_back(state.msg);
    }

    world_pub_->publish(msg);
  }

  void publish_markers()
  {
    visualization_msgs::msg::MarkerArray arr;

    int id = 0;

    for (const auto & [_, state] : blocks_) {

      // -----------------------
      // Geometry marker
      // -----------------------
      arr.markers.push_back(
        make_block_marker(
          state.msg,
          world_frame_,
          id
        )
      );

      // -----------------------
      // Text marker
      // -----------------------
      arr.markers.push_back(
        make_text_marker(
          state.msg,
          world_frame_,
          id
        )
      );

      ++id;
    }

    marker_pub_->publish(arr);
  }
  // ------------------------------------------------------------
  // Members
  // ------------------------------------------------------------
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

  std::unordered_map<std::string, BlockState> blocks_;
  int next_id_ = 0;

  std::string calib_yaml_;
  std::string world_frame_;
  double assoc_dist_;
  int min_points_;

  Eigen::Matrix3d K_;       // camera intrinsics
  Eigen::Matrix4d T_P_C_;   // pointcloud → camera
};


int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<WorldModelNode>());
  rclcpp::shutdown();
  return 0;
}
