#include "concrete_block_perception/dataset_export/scene_dataset_exporter_node.hpp"

#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <unordered_set>

#include <Eigen/Geometry>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>

namespace concrete_block_perception
{
namespace
{

// Message-stamp nanoseconds. Deliberately not rclcpp::Time/Duration: the
// export gate must depend only on the data's own timestamps, not on the
// node's clock type or whether use_sim_time/--rate matches wall time.
int64_t stampNanoseconds(const builtin_interfaces::msg::Time & stamp)
{
  return static_cast<int64_t>(stamp.sec) * 1000000000LL + static_cast<int64_t>(stamp.nanosec);
}

std::string yamlEscape(const std::string & value)
{
  std::string out;
  out.reserve(value.size());
  for (const char c : value) {
    if (c == '\\' || c == '"') {
      out.push_back('\\');
    }
    out.push_back(c);
  }
  return out;
}

std::string sanitizeForPath(const std::string & value)
{
  std::string out;
  out.reserve(value.size());
  for (const char c : value) {
    out.push_back(std::isalnum(static_cast<unsigned char>(c)) ? c : '_');
  }
  return out.empty() ? std::string("bag") : out;
}

std::string frameToken(const std::string & frame)
{
  std::string out;
  out.reserve(frame.size());
  for (const char c : frame) {
    out.push_back((c == '/' || c == ' ') ? '_' : c);
  }
  return out;
}

// URDF/tf convention: R = Rz(yaw) * Ry(pitch) * Rx(roll). Returns [roll, pitch, yaw].
std::array<double, 3> rpyFromRotation(const Eigen::Matrix3d & rot)
{
  const double pitch = std::asin(std::clamp(-rot(2, 0), -1.0, 1.0));
  const double roll = std::atan2(rot(2, 1), rot(2, 2));
  const double yaw = std::atan2(rot(1, 0), rot(0, 0));
  return {roll, pitch, yaw};
}

void writeTfEntry(
  std::ostream & out, const std::string & parent, const std::string & child,
  const std::string & lookup, bool available, const std::string & reason,
  const geometry_msgs::msg::Transform & tf)
{
  out << "  - name: \"T_" << frameToken(parent) << "_" << frameToken(child) << "\"\n";
  out << "    parent: \"" << yamlEscape(parent) << "\"\n";
  out << "    child: \"" << yamlEscape(child) << "\"\n";
  out << "    lookup: \"" << lookup << "\"\n";
  out << "    available: " << (available ? "true" : "false") << "\n";
  if (!available) {
    out << "    reason: \"" << yamlEscape(reason) << "\"\n";
    return;
  }
  if (!reason.empty()) {
    out << "    note: \"" << yamlEscape(reason) << "\"\n";
  }
  const auto & t = tf.translation;
  const auto & q = tf.rotation;
  const Eigen::Quaterniond quat = Eigen::Quaterniond(q.w, q.x, q.y, q.z).normalized();
  const Eigen::Matrix3d rot = quat.toRotationMatrix();
  const auto rpy = rpyFromRotation(rot);
  const std::array<double, 3> trans{t.x, t.y, t.z};
  std::ostringstream body;
  body << std::setprecision(9);
  body << "    xyz: [" << t.x << ", " << t.y << ", " << t.z << "]\n";
  body << "    rpy: [" << rpy[0] << ", " << rpy[1] << ", " << rpy[2] << "]\n";
  body << "    quaternion_xyzw: [" << q.x << ", " << q.y << ", " << q.z << ", " << q.w << "]\n";
  body << "    matrix:\n";
  for (int r = 0; r < 3; ++r) {
    body << "      - [" << rot(r, 0) << ", " << rot(r, 1) << ", " << rot(r, 2) << ", "
         << trans[static_cast<size_t>(r)] << "]\n";
  }
  body << "      - [0.0, 0.0, 0.0, 1.0]\n";
  out << body.str();
}

bool writeCloudPcd(
  const std::filesystem::path & path, const sensor_msgs::msg::PointCloud2 & cloud,
  const rclcpp::Logger & logger)
{
  try {
    sensor_msgs::PointCloud2ConstIterator<float> x(cloud, "x");
    sensor_msgs::PointCloud2ConstIterator<float> y(cloud, "y");
    sensor_msgs::PointCloud2ConstIterator<float> z(cloud, "z");
    std::vector<std::array<float, 3>> points;
    points.reserve(static_cast<size_t>(cloud.width) * static_cast<size_t>(cloud.height));
    for (; x != x.end(); ++x, ++y, ++z) {
      if (std::isfinite(*x) && std::isfinite(*y) && std::isfinite(*z)) {
        points.push_back({*x, *y, *z});
      }
    }
    std::ofstream out(path);
    if (!out.is_open()) {
      RCLCPP_WARN(logger, "Dataset export: failed to open %s", path.c_str());
      return false;
    }
    out << "# .PCD v0.7 - Point Cloud Data file format\n"
        << "VERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n"
        << "WIDTH " << points.size() << "\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\n"
        << "POINTS " << points.size() << "\nDATA ascii\n";
    out << std::setprecision(9);
    for (const auto & point : points) {
      out << point[0] << ' ' << point[1] << ' ' << point[2] << '\n';
    }
    return true;
  } catch (const std::exception & ex) {
    RCLCPP_WARN(logger, "Dataset export: cloud export failed: %s", ex.what());
    return false;
  }
}

bool writeImagePng(
  const std::filesystem::path & path, const sensor_msgs::msg::Image & image,
  const rclcpp::Logger & logger)
{
  try {
    auto cv_image = cv_bridge::toCvCopy(image);
    cv::Mat output = cv_image->image;
    if (cv_image->encoding == "rgb8") {
      cv::cvtColor(output, output, cv::COLOR_RGB2BGR);
    } else if (cv_image->encoding == "rgba8") {
      cv::cvtColor(output, output, cv::COLOR_RGBA2BGRA);
    }
    if (!cv::imwrite(path.string(), output)) {
      RCLCPP_WARN(logger, "Dataset export: failed to write %s", path.c_str());
      return false;
    }
    return true;
  } catch (const std::exception & ex) {
    RCLCPP_WARN(logger, "Dataset export: RGB export failed: %s", ex.what());
    return false;
  }
}

}  // namespace

SceneDatasetExporterNode::SceneDatasetExporterNode(const rclcpp::NodeOptions & options)
: Node("scene_dataset_exporter_node", options)
{
  export_interval_s_ = declare_parameter<double>("export_interval_s", 1.0);
  if (!std::isfinite(export_interval_s_) || export_interval_s_ < 0.0) {
    throw std::invalid_argument("export_interval_s must be a finite value >= 0");
  }
  world_frame_ = declare_parameter<std::string>("world_frame", "world");
  output_dir_ = declare_parameter<std::string>("output_dir", "dataset");
  source_bag_ = declare_parameter<std::string>("source_bag", "");
  extra_tf_frames_ = declare_parameter<std::vector<std::string>>(
    "tf_frames", std::vector<std::string>{
      "K0_mounting_base", "elastic/K9", "elastic/K10_left_rail", "elastic/K11",
      "elastic/K12_right_rail", "elastic/K8_tool_center_point"});
  const double sync_slop_s = declare_parameter<double>("sync_slop_s", 0.06);

  try {
    std::filesystem::create_directories(output_dir_);
  } catch (const std::exception & ex) {
    throw std::invalid_argument(
            std::string("cannot create output_dir '") + output_dir_.string() + "': " + ex.what());
  }

  tf_buffer_ = std::make_shared<tf2_ros::Buffer>(get_clock());
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

  camera_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
    "camera_info", rclcpp::SensorDataQoS(),
    std::bind(&SceneDatasetExporterNode::cameraInfoCallback, this, std::placeholders::_1));

  image_sub_.subscribe(this, "image");
  cloud_sub_.subscribe(this, "points");
  sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
    SyncPolicy(10), image_sub_, cloud_sub_);
  sync_->setMaxIntervalDuration(rclcpp::Duration::from_seconds(sync_slop_s));
  sync_->registerCallback(
    std::bind(
      &SceneDatasetExporterNode::syncedCallback, this, std::placeholders::_1,
      std::placeholders::_2));

  RCLCPP_INFO(
    get_logger(),
    "Dataset export: every %.3f s of message time, source_bag='%s', output_dir='%s'",
    export_interval_s_, source_bag_.c_str(), output_dir_.c_str());
}

void SceneDatasetExporterNode::cameraInfoCallback(
  sensor_msgs::msg::CameraInfo::ConstSharedPtr msg)
{
  last_camera_info_ = std::move(msg);
}

void SceneDatasetExporterNode::syncedCallback(
  const sensor_msgs::msg::Image::ConstSharedPtr & image,
  const sensor_msgs::msg::PointCloud2::ConstSharedPtr & cloud)
{
  const int64_t cloud_stamp_ns = stampNanoseconds(cloud->header.stamp);
  if (has_last_export_stamp_) {
    const double elapsed_s =
      static_cast<double>(cloud_stamp_ns - last_export_stamp_ns_) * 1e-9;
    // A negative elapsed time (bag looped, or a second `ros2 bag play` started)
    // is treated as a new sequence: export rather than silently stall forever.
    if (elapsed_s >= 0.0 && elapsed_s < export_interval_s_) {
      return;
    }
  }
  if (!last_camera_info_) {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 5000,
      "Dataset export: skipping sample, no CameraInfo received yet.");
    return;
  }

  // Resolve TF before creating anything: right when bag playback's /clock
  // starts, tf2_ros::Buffer can detect a backward time jump and clear itself,
  // discarding the one-shot /tf_static content it already had -- observed to
  // silently drop world<->sensor transforms for the first ~1 s of playback
  // even with a startup delay. A sample missing those is geometrically
  // useless for reprojection, so skip it rather than write incomplete data;
  // the interval gate is not consumed, so the next resolvable moment is used.
  const auto tf_snapshot = resolveTfSnapshot(
    image->header.frame_id, cloud->header.frame_id, cloud->header.stamp);
  if (!tf_snapshot.core_complete) {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 5000,
      "Dataset export: skipping sample at %d.%09u, core TF (world<->sensor) not resolved yet.",
      cloud->header.stamp.sec, cloud->header.stamp.nanosec);
    return;
  }

  std::ostringstream name;
  name << sanitizeForPath(source_bag_) << "__" << cloud->header.stamp.sec << "_"
       << std::setw(9) << std::setfill('0') << cloud->header.stamp.nanosec;
  const auto dir = output_dir_ / name.str();
  try {
    std::filesystem::create_directories(dir);
  } catch (const std::exception & ex) {
    RCLCPP_WARN(get_logger(), "Dataset export: cannot create %s: %s", dir.c_str(), ex.what());
    return;
  }

  const bool cloud_written = writeCloudPcd(dir / "cloud.pcd", *cloud, get_logger());
  const bool image_written = writeImagePng(dir / "rgb.png", *image, get_logger());
  std::ofstream(dir / "tf.yaml") << tf_snapshot.yaml;
  writeCameraInfoYaml(dir / "camera_info.yaml");

  std::ofstream metadata(dir / "metadata.yaml");
  if (metadata.is_open()) {
    metadata << "schema_version: 1\n"
             << "capture_kind: bag_periodic_sample\n"
             << "source_bag: \"" << yamlEscape(source_bag_) << "\"\n"
             << "export_interval_s: " << export_interval_s_ << "\n"
             << "reference_frame: \"" << yamlEscape(world_frame_) << "\"\n"
             << "cloud:\n"
             << "  frame_id: \"" << yamlEscape(cloud->header.frame_id) << "\"\n"
             << "  stamp: {sec: " << cloud->header.stamp.sec
             << ", nanosec: " << cloud->header.stamp.nanosec << "}\n"
             << "  file: " << (cloud_written ? "cloud.pcd" : "null") << "\n"
             << "rgb:\n"
             << "  frame_id: \"" << yamlEscape(image->header.frame_id) << "\"\n"
             << "  stamp: {sec: " << image->header.stamp.sec
             << ", nanosec: " << image->header.stamp.nanosec << "}\n"
             << "  file: " << (image_written ? "rgb.png" : "null") << "\n"
             << "camera_info:\n"
             << "  file: camera_info.yaml\n";
  }

  has_last_export_stamp_ = true;
  last_export_stamp_ns_ = cloud_stamp_ns;
  ++exported_count_;
  RCLCPP_INFO(get_logger(), "Dataset export: sample %zu saved: %s", exported_count_, dir.c_str());
}

SceneDatasetExporterNode::TfSnapshot SceneDatasetExporterNode::resolveTfSnapshot(
  const std::string & image_frame, const std::string & cloud_frame,
  const builtin_interfaces::msg::Time & stamp)
{
  std::ostringstream out;
  out << "# TF snapshot for this bag-sampled scene.\n";
  out << "# Convention: T_parent_child maps points from child coords into parent coords\n";
  out << "#   (p_parent = T_parent_child * p_child).\n";
  out << "# lookup: \"stamp\" = at the cloud stamp, \"latest\" = newest available (see note),\n";
  out << "#   \"identity\" = parent == child.\n";
  out << "# rgb.png is in image_frame, cloud.pcd is in cloud_frame.\n";
  out << "reference_frame: \"" << yamlEscape(world_frame_) << "\"\n";
  out << "stamp:\n  sec: " << stamp.sec << "\n  nanosec: " << stamp.nanosec << "\n";
  out << "image_frame: \"" << yamlEscape(image_frame) << "\"\n";
  out << "cloud_frame: \"" << yamlEscape(cloud_frame) << "\"\n";
  out << "transforms:\n";

  std::vector<std::pair<std::string, std::string>> pairs;
  std::unordered_set<std::string> seen;
  const auto add_pair = [&pairs, &seen](const std::string & parent, const std::string & child) {
      if (parent.empty() || child.empty()) {return;}
      if (seen.insert(parent + '\n' + child).second) {pairs.emplace_back(parent, child);}
    };
  add_pair(world_frame_, image_frame);
  add_pair(world_frame_, cloud_frame);
  add_pair(cloud_frame, image_frame);
  const std::size_t core_pair_count = pairs.size();
  for (const auto & frame : extra_tf_frames_) {
    add_pair(world_frame_, frame);
  }

  std::size_t resolved = 0;
  bool core_complete = true;
  for (std::size_t index = 0; index < pairs.size(); ++index) {
    const auto & pair = pairs[index];
    geometry_msgs::msg::TransformStamped tf;
    std::string lookup;
    std::string reason;
    bool ok = false;
    if (pair.first == pair.second) {
      tf.header.frame_id = pair.first;
      tf.child_frame_id = pair.second;
      tf.transform.rotation.w = 1.0;
      lookup = "identity";
      ok = true;
    } else {
      try {
        tf = tf_buffer_->lookupTransform(pair.first, pair.second, stamp, tf2::durationFromSec(0.1));
        lookup = "stamp";
        ok = true;
      } catch (const tf2::TransformException & ex_stamp) {
        try {
          tf = tf_buffer_->lookupTransform(pair.first, pair.second, tf2::TimePointZero);
          lookup = "latest";
          reason = std::string("stamp lookup failed, used latest available: ") + ex_stamp.what();
          ok = true;
        } catch (const tf2::TransformException & ex_latest) {
          reason = ex_latest.what();
        }
      }
    }
    if (ok) {
      ++resolved;
    } else if (index < core_pair_count) {
      core_complete = false;
    }
    writeTfEntry(out, pair.first, pair.second, lookup, ok, reason, tf.transform);
  }
  return {out.str(), core_complete};
}

void SceneDatasetExporterNode::writeCameraInfoYaml(const std::filesystem::path & path)
{
  std::ofstream out(path);
  if (!out.is_open() || !last_camera_info_) {
    return;
  }
  const auto & info = *last_camera_info_;
  out << std::setprecision(12);
  out << "frame_id: \"" << yamlEscape(info.header.frame_id) << "\"\n"
      << "stamp: {sec: " << info.header.stamp.sec << ", nanosec: " << info.header.stamp.nanosec
      << "}\n"
      << "height: " << info.height << "\n"
      << "width: " << info.width << "\n"
      << "distortion_model: \"" << yamlEscape(info.distortion_model) << "\"\n"
      << "d: [";
  for (std::size_t i = 0; i < info.d.size(); ++i) {out << (i ? ", " : "") << info.d[i];}
  out << "]\nk: [";
  for (std::size_t i = 0; i < info.k.size(); ++i) {out << (i ? ", " : "") << info.k[i];}
  out << "]\nr: [";
  for (std::size_t i = 0; i < info.r.size(); ++i) {out << (i ? ", " : "") << info.r[i];}
  out << "]\np: [";
  for (std::size_t i = 0; i < info.p.size(); ++i) {out << (i ? ", " : "") << info.p[i];}
  out << "]\n";
}

}  // namespace concrete_block_perception

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<concrete_block_perception::SceneDatasetExporterNode>());
  rclcpp::shutdown();
  return 0;
}
