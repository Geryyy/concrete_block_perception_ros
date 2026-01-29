#include "concrete_block_perception/io_utils.hpp"
#include <Eigen/Geometry>

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
