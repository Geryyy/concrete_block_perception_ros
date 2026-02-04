#pragma once

#include <Eigen/Dense>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <open3d/Open3D.h>
#include <rclcpp/rclcpp.hpp>

std::shared_ptr<open3d::geometry::PointCloud>
pointcloud2_to_open3d(
  const sensor_msgs::msg::PointCloud2 & msg);


sensor_msgs::msg::PointCloud2
open3d_to_pointcloud2(
  const open3d::geometry::PointCloud & cloud,
  const std::string & frame_id,
  const rclcpp::Time & stamp);

geometry_msgs::msg::Pose
to_ros_pose(const Eigen::Matrix4d & T);
