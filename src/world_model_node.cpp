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
#include <fstream>
#include <rcpputils/filesystem_helper.hpp>

#include <Eigen/Dense>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgcodecs.hpp>
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

    RCLCPP_DEBUG(
      get_logger(),
      "T_P_C_:\n"
      "[ %.4f %.4f %.4f %.4f ]\n"
      "[ %.4f %.4f %.4f %.4f ]\n"
      "[ %.4f %.4f %.4f %.4f ]\n"
      "[ %.4f %.4f %.4f %.4f ]",
      T_P_C_(0, 0), T_P_C_(0, 1), T_P_C_(0, 2), T_P_C_(0, 3),
      T_P_C_(1, 0), T_P_C_(1, 1), T_P_C_(1, 2), T_P_C_(1, 3),
      T_P_C_(2, 0), T_P_C_(2, 1), T_P_C_(2, 2), T_P_C_(2, 3),
      T_P_C_(3, 0), T_P_C_(3, 1), T_P_C_(3, 2), T_P_C_(3, 3)
    );

    RCLCPP_DEBUG(
      get_logger(),
      "K:\n"
      "[ %.2f %.2f %.2f ]\n"
      "[ %.2f %.2f %.2f ]\n"
      "[ %.2f %.2f %.2f ]",
      K_(0, 0), K_(0, 1), K_(0, 2),
      K_(1, 0), K_(1, 1), K_(1, 2),
      K_(2, 0), K_(2, 1), K_(2, 2)
    );

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
    RCLCPP_DEBUG(
      this->get_logger(), "callback() with %ld detections",
      detections->detections.size());

    lidar_frame_ = cloud->header.frame_id;
    auto time_stamp = cloud->header.stamp;
    auto scene = pointcloud2_to_open3d(*cloud);

    cv::Mat full_mask =
      cv_bridge::toCvCopy(mask_msg, "mono8")->image;

    for (const auto & det : detections->detections) {

      // Extract per-detection mask
      cv::Mat det_mask = extract_mask_roi(full_mask, det);

      if (det_mask.empty()) {
        RCLCPP_WARN(this->get_logger(), "detection mask empty");
        continue;
      }

      int count = cv::countNonZero(full_mask);
      RCLCPP_DEBUG(this->get_logger(), "full mask > 1: %d", count);
      count = cv::countNonZero(det_mask);
      RCLCPP_DEBUG(this->get_logger(), "detection mask > 1: %d", count);
      RCLCPP_DEBUG(this->get_logger(), "scene->points.size(): %ld", scene->points_.size());


      //////////////////////////////////////////////////////////////////////////////////////////////
      // DEBUG DUMP
      //////////////////////////////////////////////////////////////////////////////////////////////

// ------------------------------------------------------------------
// DEBUG DUMP (one-shot or per detection)
// ------------------------------------------------------------------
      // static bool dumped_once = false;
      // if (!dumped_once) {
      //   dumped_once = true;

      //   const std::string dump_dir = "/tmp/world_model_debug";
      //   rcpputils::fs::create_directories(dump_dir);

      //   const auto stamp =
      //     std::to_string(cloud->header.stamp.sec) + "_" +
      //     std::to_string(cloud->header.stamp.nanosec);

      //   // --- Save full mask ---
      //   {
      //     std::string path = dump_dir + "/full_mask_" + stamp + ".png";
      //     cv::imwrite(path, full_mask);
      //     RCLCPP_DEBUG(get_logger(), "Saved full mask: %s", path.c_str());
      //   }

      //   // --- Save detection mask ---
      //   {
      //     std::string path = dump_dir + "/det_mask_" + stamp + ".png";
      //     cv::imwrite(path, det_mask);
      //     RCLCPP_DEBUG(get_logger(), "Saved detection mask: %s", path.c_str());
      //   }

      //   // --- Save full point cloud ---
      //   {
      //     std::string path = dump_dir + "/scene_" + stamp + ".ply";
      //     open3d::io::WritePointCloud(path, *scene);
      //     RCLCPP_DEBUG(get_logger(), "Saved pointcloud: %s", path.c_str());
      //   }

      //   // --- Save intrinsics K ---
      //   {
      //     std::string path = dump_dir + "/K_" + stamp + ".txt";
      //     std::ofstream f(path);
      //     f << K_ << std::endl;
      //     RCLCPP_DEBUG(get_logger(), "Saved K: %s", path.c_str());
      //   }

      //   // --- Save extrinsics T_P_C ---
      //   {
      //     std::string path = dump_dir + "/T_P_C_" + stamp + ".txt";
      //     std::ofstream f(path);
      //     f << T_P_C_ << std::endl;
      //     RCLCPP_DEBUG(get_logger(), "Saved T_P_C: %s", path.c_str());
      //   }

      //   RCLCPP_WARN(
      //     get_logger(),
      //     "World-model debug dump written to %s",
      //     dump_dir.c_str()
      //   );
      // }

      //////////////////////////////////////////////////////////////////////////////////////////////
      // DEBUG DUMP END
      //////////////////////////////////////////////////////////////////////////////////////////////


      auto pts = select_points_by_mask(
        scene->points_,
        det_mask,
        K_,
        T_P_C_
      );

      if (static_cast<int>(pts.size()) < min_points_) {
        RCLCPP_WARN(
          this->get_logger(), "points in pointcloud (%ld) less than min_points (%d)",
          pts.size(), min_points_);
        continue;
      }

      Eigen::Vector3d c = compute_center(pts);

      geometry_msgs::msg::Pose pose_lidar_frame;
      pose_lidar_frame.position.x = c.x();
      pose_lidar_frame.position.y = c.y();
      pose_lidar_frame.position.z = c.z();
      pose_lidar_frame.orientation.w = 1.0;

      auto pose_stamped = transform_pose_to_world(pose_lidar_frame, lidar_frame_, time_stamp);
      auto pose = pose_stamped.pose;

      RCLCPP_DEBUG(
        this->get_logger(), "Associate block at x: %3.2f, y: %3.2f, z: %3.2f", pose.position.x,
        pose.position.y, pose.position.z);

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
    publish_markers();
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
      // if (state.msg.task_status == state.msg.TASK_MOVE) {
      //   // do not update block if
      //   continue;
      // }

      if (distance(state.msg.pose, pose) < assoc_dist_) {
        RCLCPP_DEBUG(this->get_logger(), "Found block with '%s'", id.c_str());
        return id;
      }
    }


    // New block
    std::string id = "block_" + std::to_string(next_id_++);
    BlockState state;

    RCLCPP_INFO(
      this->get_logger(),
      "Create new block '%s' at pose [x=%.3f y=%.3f z=%.3f | "
      "qx=%.3f qy=%.3f qz=%.3f qw=%.3f]",
      id.c_str(),
      pose.position.x,
      pose.position.y,
      pose.position.z,
      pose.orientation.x,
      pose.orientation.y,
      pose.orientation.z,
      pose.orientation.w
    );

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
  std::string lidar_frame_;
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
