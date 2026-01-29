cmake_minimum_required(VERSION 3.8)


# --- Fix OpenGL policy (CMP0072) ---
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
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Open3D CONFIG REQUIRED)

find_package(rosidl_default_generators REQUIRED)

# ------------------------------------------------------------
# Build external library (vendored)
# ------------------------------------------------------------
# add_subdirectory(
#   ${CMAKE_CURRENT_SOURCE_DIR}/external/pcd_block_estimation
#   ${CMAKE_BINARY_DIR}/pcd_block_estimation
# )

add_subdirectory(external/pcd_block_estimation)


# ------------------------------------------------------------
# Generate services
# ------------------------------------------------------------
rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/SegmentAtPoint.srv"
  "srv/RegisterBlock.srv"
  DEPENDENCIES sensor_msgs geometry_msgs
)

rosidl_get_typesupport_target(
  cpp_typesupport_target
  ${PROJECT_NAME}
  "rosidl_typesupport_cpp"
)


ament_export_dependencies(rosidl_default_runtime)

# ------------------------------------------------------------
# Block registration node
# ------------------------------------------------------------
add_executable(block_registration_node
  src/block_registration_node.cpp
  src/io_utils.cpp
)

target_include_directories(block_registration_node
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(block_registration_node
  pcd_block_estimation
  ${cpp_typesupport_target}
)


ament_target_dependencies(block_registration_node
  rclcpp
  sensor_msgs
  geometry_msgs
  vision_msgs
  cv_bridge
)

# Make sure generated service headers are visible
# rosidl_target_interfaces(block_registration_node
#   ${PROJECT_NAME} "rosidl_typesupport_cpp")

# ------------------------------------------------------------
# Install targets
# ------------------------------------------------------------
install(TARGETS
  block_registration_node
  DESTINATION lib/${PROJECT_NAME}
)

install(
  PROGRAMS
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
