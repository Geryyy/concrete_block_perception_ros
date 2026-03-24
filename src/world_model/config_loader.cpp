#include "concrete_block_perception/world_model/config_loader.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <unordered_set>

#include "concrete_block_perception/msg/block.hpp"
#include <yaml-cpp/yaml.h>

namespace cbp::world_model
{

namespace
{

using concrete_block_perception::msg::Block;

int parsePoseStatus(const YAML::Node & node, int fallback)
{
  if (!node || node.IsNull()) {
    return fallback;
  }
  if (node.IsScalar()) {
    const std::string value = node.as<std::string>("");
    if (value == "POSE_UNKNOWN") {
      return Block::POSE_UNKNOWN;
    }
    if (value == "POSE_COARSE") {
      return Block::POSE_COARSE;
    }
    if (value == "POSE_PRECISE") {
      return Block::POSE_PRECISE;
    }
    try {
      return std::stoi(value);
    } catch (...) {
      return fallback;
    }
  }
  return fallback;
}

int parseTaskStatus(const YAML::Node & node, int fallback)
{
  if (!node || node.IsNull()) {
    return fallback;
  }
  if (node.IsScalar()) {
    const std::string value = node.as<std::string>("");
    if (value == "TASK_UNKNOWN") {
      return Block::TASK_UNKNOWN;
    }
    if (value == "TASK_FREE") {
      return Block::TASK_FREE;
    }
    if (value == "TASK_MOVE") {
      return Block::TASK_MOVE;
    }
    if (value == "TASK_PLACED") {
      return Block::TASK_PLACED;
    }
    if (value == "TASK_REMOVED") {
      return Block::TASK_REMOVED;
    }
    try {
      return std::stoi(value);
    } catch (...) {
      return fallback;
    }
  }
  return fallback;
}

bool isKnownPoseStatus(int value)
{
  return value == Block::POSE_UNKNOWN ||
         value == Block::POSE_COARSE ||
         value == Block::POSE_PRECISE;
}

bool isKnownTaskStatus(int value)
{
  return value == Block::TASK_UNKNOWN ||
         value == Block::TASK_FREE ||
         value == Block::TASK_MOVE ||
         value == Block::TASK_PLACED ||
         value == Block::TASK_REMOVED;
}

std::vector<InitialBlockConfig> parseInitialBlocksYaml(
  rclcpp::Logger logger,
  const std::string & world_frame,
  const std::string & yaml_payload)
{
  std::vector<InitialBlockConfig> out;
  if (yaml_payload.empty()) {
    return out;
  }

  YAML::Node root;
  try {
    root = YAML::Load(yaml_payload);
  } catch (const std::exception & exc) {
    RCLCPP_ERROR(logger, "Failed to parse world_model.initial_blocks YAML: %s", exc.what());
    return out;
  }

  if (!root || !root.IsSequence()) {
    RCLCPP_ERROR(logger, "world_model.initial_blocks must be a YAML sequence.");
    return out;
  }

  std::unordered_set<std::string> seen_ids;
  for (std::size_t idx = 0; idx < root.size(); ++idx) {
    const YAML::Node node = root[idx];
    if (!node.IsMap()) {
      RCLCPP_WARN(logger, "Skipping initial block %zu: expected mapping.", idx + 1);
      continue;
    }

    InitialBlockConfig block;
    block.id = node["id"].as<std::string>("");
    if (block.id.empty()) {
      RCLCPP_WARN(logger, "Skipping initial block %zu: id must not be empty.", idx + 1);
      continue;
    }
    if (!seen_ids.insert(block.id).second) {
      RCLCPP_WARN(logger, "Skipping initial block '%s': duplicate id.", block.id.c_str());
      continue;
    }

    block.frame_id = node["frame_id"].as<std::string>(world_frame);
    if (block.frame_id.empty()) {
      block.frame_id = world_frame;
    }
    if (block.frame_id != world_frame) {
      RCLCPP_WARN(
        logger,
        "Skipping initial block '%s': frame_id '%s' must match world_frame '%s'.",
        block.id.c_str(),
        block.frame_id.c_str(),
        world_frame.c_str());
      continue;
    }

    const YAML::Node position = node["position"];
    if (!position || !position.IsSequence() || position.size() != 3) {
      RCLCPP_WARN(
        logger,
        "Skipping initial block '%s': position must be a 3-element sequence.",
        block.id.c_str());
      continue;
    }
    try {
      for (std::size_t axis = 0; axis < 3; ++axis) {
        block.position[axis] = position[axis].as<double>();
      }
    } catch (...) {
      RCLCPP_WARN(
        logger,
        "Skipping initial block '%s': position values must be numeric.",
        block.id.c_str());
      continue;
    }
    if (!std::isfinite(block.position[0]) ||
        !std::isfinite(block.position[1]) ||
        !std::isfinite(block.position[2])) {
      RCLCPP_WARN(
        logger,
        "Skipping initial block '%s': position values must be finite.",
        block.id.c_str());
      continue;
    }

    try {
      block.yaw_deg = node["yaw_deg"].as<double>(0.0);
      block.confidence = node["confidence"].as<double>(1.0);
    } catch (...) {
      RCLCPP_WARN(
        logger,
        "Skipping initial block '%s': yaw_deg/confidence must be numeric.",
        block.id.c_str());
      continue;
    }
    if (!std::isfinite(block.yaw_deg) || !std::isfinite(block.confidence)) {
      RCLCPP_WARN(
        logger,
        "Skipping initial block '%s': yaw_deg/confidence must be finite.",
        block.id.c_str());
      continue;
    }

    block.pose_status = parsePoseStatus(node["pose_status"], Block::POSE_COARSE);
    block.task_status = parseTaskStatus(node["task_status"], Block::TASK_PLACED);
    if (!isKnownPoseStatus(block.pose_status)) {
      RCLCPP_WARN(
        logger,
        "Skipping initial block '%s': unsupported pose_status.",
        block.id.c_str());
      continue;
    }
    if (!isKnownTaskStatus(block.task_status)) {
      RCLCPP_WARN(
        logger,
        "Skipping initial block '%s': unsupported task_status.",
        block.id.c_str());
      continue;
    }

    out.push_back(block);
  }

  return out;
}

}  // namespace

WorldModelConfig loadWorldModelConfig(rclcpp::Node & node)
{
  WorldModelConfig cfg;

  cfg.pipeline_mode_str = node.declare_parameter<std::string>("pipeline_mode", "full");
  cfg.perception_mode_str = node.declare_parameter<std::string>("perception_mode", "FULL");
  cfg.min_fitness = node.declare_parameter<double>("min_fitness", 0.3);
  cfg.max_rmse = node.declare_parameter<double>("max_rmse", 0.05);
  cfg.object_class = node.declare_parameter<std::string>("object_class", "concrete_block");
  cfg.world_frame = node.declare_parameter<std::string>("world_frame", "world");
  cfg.max_sync_delta_s = node.declare_parameter<double>("sync.max_delta_s", 0.06);
  cfg.object_timeout_s = node.declare_parameter<double>("world_model.object_timeout_s", 10.0);
  cfg.association_max_distance_m =
    node.declare_parameter<double>("world_model.association_max_distance_m", 0.45);
  cfg.association_max_age_s = node.declare_parameter<double>("world_model.association_max_age_s", 20.0);
  cfg.min_update_confidence = node.declare_parameter<double>("world_model.min_update_confidence", 0.25);
  cfg.refine_target_max_distance_m =
    node.declare_parameter<double>("world_model.refine_target_max_distance_m", 1.2);
  cfg.scene_discovery_coarse_fallback_enabled =
    node.declare_parameter<bool>("world_model.scene_discovery_coarse_fallback.enable", true);
  cfg.scene_discovery_coarse_fallback_min_points =
    node.declare_parameter<int>("world_model.scene_discovery_coarse_fallback.min_points", 120);
  cfg.coarse_surface_square_ratio_thresh = node.declare_parameter<double>(
    "world_model.scene_discovery_coarse_fallback.surface_shape.square_ratio_thresh", 1.35);
  cfg.coarse_front_center_offset_square_m = node.declare_parameter<double>(
    "world_model.scene_discovery_coarse_fallback.center_offset.square_m", 0.45);
  cfg.coarse_front_center_offset_rect_m = node.declare_parameter<double>(
    "world_model.scene_discovery_coarse_fallback.center_offset.rect_m", 0.30);
  cfg.debug_detection_overlay_enabled = node.declare_parameter<bool>("debug.publish_detection_overlay", true);
  cfg.debug_refine_grasped_roi_input_enabled =
    node.declare_parameter<bool>("debug.publish_refine_grasped_roi_input", true);
  cfg.perf_log_timing_enabled = node.declare_parameter<bool>("perf.log_timing", true);
  cfg.perf_log_every_n_frames = node.declare_parameter<int>("perf.log_every_n_frames", 20);
  cfg.marker_refresh_period_s = node.declare_parameter<double>("world_model.marker_refresh_period_s", 0.5);

  cfg.refine_grasped_use_fk_roi = node.declare_parameter<bool>("refine_grasped.use_fk_roi", true);
  cfg.refine_grasped_tcp_frame =
    node.declare_parameter<std::string>("refine_grasped.tcp_frame", "elastic/K8_tool_center_point");
  cfg.refine_grasped_camera_frame =
    node.declare_parameter<std::string>("refine_grasped.camera_frame", "");
  cfg.refine_grasped_camera_info_topic = node.declare_parameter<std::string>(
    "refine_grasped.camera_info_topic", "/zed2i/warped/left/camera_info");
  cfg.refine_grasped_min_depth_m = node.declare_parameter<double>("refine_grasped.min_depth_m", 0.5);
  cfg.refine_grasped_max_depth_m = node.declare_parameter<double>("refine_grasped.max_depth_m", 30.0);
  cfg.refine_grasped_segmentation_timeout_s =
    node.declare_parameter<double>("refine_grasped.segmentation_timeout_s", 3.0);
  cfg.refine_grasped_use_black_bg =
    node.declare_parameter<bool>("refine_grasped.segmentation_input.use_black_background", false);
  cfg.refine_grasped_blur_kernel_size =
    node.declare_parameter<int>("refine_grasped.segmentation_input.blur_kernel_size", 31);
  cfg.refine_grasped_pose_fusion.enabled =
    node.declare_parameter<bool>("refine_grasped.pose_fusion.enable", true);
  cfg.refine_grasped_pose_fusion.mode = node.declare_parameter<std::string>(
    "refine_grasped.pose_fusion.mode",
    "position_from_registration_orientation_from_fk");
  cfg.refine_grasped_pose_fusion.max_translation_jump_m = node.declare_parameter<double>(
    "refine_grasped.pose_fusion.max_translation_jump_m", 0.35);
  cfg.refine_grasped_pose_fusion.max_z_delta_m = node.declare_parameter<double>(
    "refine_grasped.pose_fusion.max_z_delta_m", 0.25);
  cfg.refine_grasped_pose_fusion.debug_log =
    node.declare_parameter<bool>("refine_grasped.pose_fusion.debug_log", true);
  cfg.refine_grasped_tcp_to_block_xyz =
    node.declare_parameter<std::vector<double>>("refine_grasped.tcp_to_block.xyz", {0.0, 0.0, 0.0});
  cfg.refine_grasped_tcp_to_block_rpy =
    node.declare_parameter<std::vector<double>>("refine_grasped.tcp_to_block.rpy", {0.0, 0.0, 0.0});
  cfg.refine_grasped_roi_size_m =
    node.declare_parameter<std::vector<double>>("refine_grasped.roi_size_m", {0.60, 0.40});

  cfg.refine_block_use_pose_roi = node.declare_parameter<bool>("refine_block.use_pose_roi", false);
  cfg.refine_block_roi_size_m =
    node.declare_parameter<std::vector<double>>("refine_block.roi_size_m", {1.20, 1.00});
  cfg.refine_block_min_depth_m = node.declare_parameter<double>("refine_block.min_depth_m", 0.5);
  cfg.refine_block_max_depth_m = node.declare_parameter<double>("refine_block.max_depth_m", 30.0);
  cfg.refine_block_segmentation_timeout_s =
    node.declare_parameter<double>("refine_block.segmentation_timeout_s", 3.0);
  cfg.refine_block_use_black_bg =
    node.declare_parameter<bool>("refine_block.segmentation_input.use_black_background", false);
  cfg.refine_block_blur_kernel_size =
    node.declare_parameter<int>("refine_block.segmentation_input.blur_kernel_size", 31);
  cfg.initial_blocks_yaml =
    node.declare_parameter<std::string>("world_model.initial_blocks", "");

  // Keep for launch-file compatibility; no longer used in one-shot flow.
  (void)node.declare_parameter<std::string>("calib_yaml", "");
  cfg.initial_blocks = parseInitialBlocksYaml(node.get_logger(), cfg.world_frame, cfg.initial_blocks_yaml);
  return cfg;
}

void normalizeWorldModelConfig(rclcpp::Logger logger, WorldModelConfig & cfg)
{
  auto clamp_min = [logger](double & value, double min_value, const char * name) {
      if (value < min_value) {
        RCLCPP_WARN(logger, "Invalid %s=%.3f, clamping to %.3f", name, value, min_value);
        value = min_value;
      }
    };
  auto normalize_blur = [logger](int & kernel, const char * name) {
      if (kernel < 1) {
        RCLCPP_WARN(logger, "Invalid %s=%d, clamping to 1", name, kernel);
        kernel = 1;
      }
      if ((kernel % 2) == 0) {
        RCLCPP_WARN(logger, "Invalid %s=%d (must be odd), incrementing to %d", name, kernel, kernel + 1);
        kernel += 1;
      }
    };
  auto clamp_min_i = [logger](int & value, int min_value, const char * name) {
      if (value < min_value) {
        RCLCPP_WARN(logger, "Invalid %s=%d, clamping to %d", name, value, min_value);
        value = min_value;
      }
    };

  clamp_min(cfg.min_fitness, 0.0, "min_fitness");
  clamp_min(cfg.max_rmse, 0.0, "max_rmse");
  clamp_min_i(
    cfg.scene_discovery_coarse_fallback_min_points, 1,
    "scene_discovery_coarse_fallback.min_points");
  clamp_min(
    cfg.coarse_surface_square_ratio_thresh, 1.0,
    "scene_discovery_coarse_fallback.surface_shape.square_ratio_thresh");
  clamp_min(
    cfg.coarse_front_center_offset_square_m, 0.0,
    "scene_discovery_coarse_fallback.center_offset.square_m");
  clamp_min(
    cfg.coarse_front_center_offset_rect_m, 0.0,
    "scene_discovery_coarse_fallback.center_offset.rect_m");

  if (cfg.refine_grasped_min_depth_m > cfg.refine_grasped_max_depth_m) {
    RCLCPP_WARN(
      logger,
      "Invalid refine_grasped depth range [%.3f, %.3f], swapping bounds",
      cfg.refine_grasped_min_depth_m, cfg.refine_grasped_max_depth_m);
    std::swap(cfg.refine_grasped_min_depth_m, cfg.refine_grasped_max_depth_m);
  }
  if (cfg.refine_block_min_depth_m > cfg.refine_block_max_depth_m) {
    RCLCPP_WARN(
      logger,
      "Invalid refine_block depth range [%.3f, %.3f], swapping bounds",
      cfg.refine_block_min_depth_m, cfg.refine_block_max_depth_m);
    std::swap(cfg.refine_block_min_depth_m, cfg.refine_block_max_depth_m);
  }
  normalize_blur(
    cfg.refine_grasped_blur_kernel_size, "refine_grasped.segmentation_input.blur_kernel_size");
  normalize_blur(cfg.refine_block_blur_kernel_size, "refine_block.segmentation_input.blur_kernel_size");
  if (!cfg.initial_blocks.empty()) {
    RCLCPP_INFO(logger, "Configured %zu seeded world-model blocks for startup.", cfg.initial_blocks.size());
  }
}

double vectorComponent(
  rclcpp::Logger logger,
  const std::vector<double> & values,
  size_t index,
  double fallback,
  const char * param_name)
{
  if (index < values.size()) {
    return values[index];
  }
  RCLCPP_WARN(
    logger,
    "Parameter '%s' expected at least %zu entries, got %zu. Using fallback %.3f for index %zu.",
    param_name,
    index + 1,
    values.size(),
    fallback,
    index);
  return fallback;
}

}  // namespace cbp::world_model
