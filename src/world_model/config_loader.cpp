#include "concrete_block_perception/world_model/config_loader.hpp"

#include <algorithm>

namespace cbp::world_model
{

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

  // Keep for launch-file compatibility; no longer used in one-shot flow.
  (void)node.declare_parameter<std::string>("calib_yaml", "");
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

