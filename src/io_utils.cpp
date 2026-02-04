#include "concrete_block_perception/io_utils.hpp"
#include <Eigen/Geometry>
#include <sensor_msgs/msg/point_field.hpp>
#include <rclcpp/rclcpp.hpp>

std::shared_ptr<open3d::geometry::PointCloud>
pointcloud2_to_open3d(
  const sensor_msgs::msg::PointCloud2 & msg)
{
  auto cloud = std::make_shared<open3d::geometry::PointCloud>();

  sensor_msgs::PointCloud2ConstIterator<float> iter_x(msg, "x");
  sensor_msgs::PointCloud2ConstIterator<float> iter_y(msg, "y");
  sensor_msgs::PointCloud2ConstIterator<float> iter_z(msg, "z");

  for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z) {
    if (!std::isfinite(*iter_x) ||
      !std::isfinite(*iter_y) ||
      !std::isfinite(*iter_z))
    {
      continue;
    }

    cloud->points_.emplace_back(*iter_x, *iter_y, *iter_z);
  }

  return cloud;
}

sensor_msgs::msg::PointCloud2
open3d_to_pointcloud2(
  const open3d::geometry::PointCloud & cloud,
  const std::string & frame_id,
  const rclcpp::Time & stamp)
{
  sensor_msgs::msg::PointCloud2 msg;

  msg.header.frame_id = frame_id;
  msg.header.stamp = stamp;

  msg.height = 1;
  msg.width = static_cast<uint32_t>(cloud.points_.size());

  msg.is_dense = false;
  msg.is_bigendian = false;

  // ------------------------------------------------------------
  // Fields: x y z
  // ------------------------------------------------------------
  msg.fields.resize(3);

  msg.fields[0].name = "x";
  msg.fields[0].offset = 0;
  msg.fields[0].datatype = sensor_msgs::msg::PointField::FLOAT32;
  msg.fields[0].count = 1;

  msg.fields[1].name = "y";
  msg.fields[1].offset = 4;
  msg.fields[1].datatype = sensor_msgs::msg::PointField::FLOAT32;
  msg.fields[1].count = 1;

  msg.fields[2].name = "z";
  msg.fields[2].offset = 8;
  msg.fields[2].datatype = sensor_msgs::msg::PointField::FLOAT32;
  msg.fields[2].count = 1;

  msg.point_step = 12;  // 3 * float32
  msg.row_step = msg.point_step * msg.width;

  msg.data.resize(msg.row_step);

  // ------------------------------------------------------------
  // Copy data
  // ------------------------------------------------------------
  uint8_t * ptr = msg.data.data();

  for (const auto & p : cloud.points_) {
    float * f = reinterpret_cast<float *>(ptr);

    f[0] = static_cast<float>(p.x());
    f[1] = static_cast<float>(p.y());
    f[2] = static_cast<float>(p.z());

    ptr += msg.point_step;
  }

  return msg;
}


geometry_msgs::msg::Pose
to_ros_pose(const Eigen::Matrix4d & T)
{
  geometry_msgs::msg::Pose pose;

  // Translation
  pose.position.x = T(0, 3);
  pose.position.y = T(1, 3);
  pose.position.z = T(2, 3);

  // Rotation
  Eigen::Matrix3d R = T.block<3, 3>(0, 0);
  Eigen::Quaterniond q(R);

  q.normalize();    // important for numerical stability

  pose.orientation.x = q.x();
  pose.orientation.y = q.y();
  pose.orientation.z = q.z();
  pose.orientation.w = q.w();

  return pose;
}
