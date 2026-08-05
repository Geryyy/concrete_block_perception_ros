cmake_minimum_required(VERSION 3.8)
set(CMAKE_POLICY_DEFAULT_CMP0072 NEW)

project(concrete_block_perception)

# ------------------------------------------------------------
# Compiler warnings
# ------------------------------------------------------------
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ------------------------------------------------------------
# ROS dependencies
# ------------------------------------------------------------
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(Eigen3 REQUIRED)
cmake_policy(SET CMP0072 NEW)
find_package(Open3D CONFIG REQUIRED)
find_package(message_filters REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(OpenCV REQUIRED)
find_package(ros2_yolos_cpp REQUIRED)
find_package(concrete_block_perception_interfaces REQUIRED)

# ------------------------------------------------------------
# External library
# ------------------------------------------------------------
add_subdirectory(external/pcd_block_estimation)

# ============================================================
# BLOCK DETECTION TRACKING NODE
# ============================================================
add_executable(block_detection_tracking_node
  src/nodes/block_detection_tracking_node.cpp
  src/utils/io_utils.cpp
  src/utils/img_utils.cpp
  src/utils/detection_association.cpp
  src/detection/block_detection_tracker.cpp
  src/detection/detection_config.cpp
)

target_include_directories(block_detection_tracking_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(block_detection_tracking_node
  pcd_block_estimation
)

ament_target_dependencies(block_detection_tracking_node
  concrete_block_perception_interfaces
  rclcpp
  sensor_msgs
  geometry_msgs
  std_msgs
  vision_msgs
  cv_bridge
)

# ============================================================
# BLOCK REGISTRATION NODE (UPDATED)
# ============================================================
add_executable(block_registration_node
  src/nodes/block_registration_node.cpp
  src/registration/block_registration_pipeline.cpp
  src/registration/registration_config.cpp
  src/registration/ros_debug_helpers.cpp
  src/utils/io_utils.cpp
)

target_include_directories(block_registration_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(block_registration_node
  pcd_block_estimation
  Open3D::Open3D
  ${OpenCV_LIBS}
)

ament_target_dependencies(block_registration_node
  concrete_block_perception_interfaces
  rclcpp
  rclcpp_action
  sensor_msgs
  geometry_msgs
  vision_msgs
  visualization_msgs
  cv_bridge
  tf2
  tf2_ros
  tf2_geometry_msgs
)

# ============================================================
# SCENE DATASET EXPORTER NODE (bag -> unlabeled snapshot dataset)
# ============================================================
add_executable(scene_dataset_exporter_node
  src/nodes/scene_dataset_exporter_node.cpp
)

target_include_directories(scene_dataset_exporter_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(scene_dataset_exporter_node
  ${OpenCV_LIBS}
  Eigen3::Eigen
)

ament_target_dependencies(scene_dataset_exporter_node
  rclcpp
  sensor_msgs
  geometry_msgs
  cv_bridge
  message_filters
  tf2
  tf2_ros
)

# ------------------------------------------------------------
# Install
# ------------------------------------------------------------
install(TARGETS
  block_detection_tracking_node
  block_registration_node
  scene_dataset_exporter_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  config
  rviz
  DESTINATION share/${PROJECT_NAME}
)

# ------------------------------------------------------------
# Testing
# ------------------------------------------------------------
if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_cpplint_FOUND TRUE)
  set(ament_cmake_copyright_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()

endif()

ament_package()
