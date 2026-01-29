#pragma once

#include <Eigen/Dense>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <open3d/Open3D.h>

std::shared_ptr<open3d::geometry::PointCloud>
pointcloud2_to_open3d(
  const sensor_msgs::msg::PointCloud2 & msg);


geometry_msgs::msg::Pose
to_ros_pose(const Eigen::Matrix4d & T);
