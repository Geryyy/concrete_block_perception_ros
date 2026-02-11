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
find_package(Open3D CONFIG REQUIRED)
find_package(message_filters REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(OpenCV REQUIRED)

# ------------------------------------------------------------
# External library
# ------------------------------------------------------------
add_subdirectory(external/pcd_block_estimation)

# ------------------------------------------------------------
# Interfaces
# ------------------------------------------------------------
rosidl_generate_interfaces(${PROJECT_NAME}
  "action/RegisterBlock.action"
  "srv/SegmentAtPoint.srv"
  "srv/RegisterBlock.srv"
  "msg/Block.msg"
  "msg/BlockArray.msg"
  "msg/TrackedDetection.msg"
  "msg/TrackedDetectionArray.msg"
  DEPENDENCIES
    sensor_msgs
    geometry_msgs
    builtin_interfaces
    std_msgs
    visualization_msgs
    vision_msgs
)

rosidl_get_typesupport_target(
  cpp_typesupport_target
  ${PROJECT_NAME}
  "rosidl_typesupport_cpp"
)

ament_export_dependencies(rosidl_default_runtime)

# ============================================================
# BLOCK DETECTION TRACKING NODE
# ============================================================
add_executable(block_detection_tracking_node
  src/nodes/block_detection_tracking_node.cpp
  src/utils/io_utils.cpp
  src/utils/img_utils.cpp
  src/utils/block_utils.cpp
  src/utils/visu_utils.cpp
  src/utils/detection_association.cpp
)

target_include_directories(block_detection_tracking_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(block_detection_tracking_node
  pcd_block_estimation
  ${cpp_typesupport_target}
)

ament_target_dependencies(block_detection_tracking_node
  rclcpp
  sensor_msgs
  geometry_msgs
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
  ${cpp_typesupport_target}
  Open3D::Open3D
  ${OpenCV_LIBS}
)

ament_target_dependencies(block_registration_node
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
# WORLD MODEL NODE
# ============================================================
add_executable(world_model_node
  src/nodes/world_model_node.cpp
  src/utils/io_utils.cpp
  src/utils/img_utils.cpp
  src/utils/block_utils.cpp
  src/utils/visu_utils.cpp
  # src/tracking/damped_cv_kalman_filter.cpp
  # src/tracking/gating.cpp
  # src/tracking/hungarian.cpp
  # src/tracking/multi_object_tracker.cpp
)

target_include_directories(world_model_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(world_model_node
  pcd_block_estimation
  ${cpp_typesupport_target}
  Open3D::Open3D
  ${OpenCV_LIBS}
)

ament_target_dependencies(world_model_node
  rclcpp
  std_msgs
  sensor_msgs
  geometry_msgs
  vision_msgs
  cv_bridge
  message_filters
  visualization_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
)

# ------------------------------------------------------------
# Install
# ------------------------------------------------------------
install(TARGETS
  block_detection_tracking_node
  block_registration_node
  world_model_node
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/detection_node.py
  scripts/segmentation_node.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}
)

# ------------------------------------------------------------
# Testing
# ------------------------------------------------------------
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_cpplint_FOUND TRUE)
  set(ament_cmake_copyright_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
