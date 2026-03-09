# Vision Pipeline Modes and Task Configuration

This document describes:
- which one-shot perception modes are available,
- how they map to your block-stacking tasks,
- how to configure the pipeline via launch arguments and YAML.

## 1) One-Shot Modes

The world model is driven via:
- `/world_model_node/run_pose_estimation` (`concrete_block_perception/srv/RunPoseEstimation`)

Supported modes:

1. `SCENE_DISCOVERY`
- Purpose: detect and register all visible blocks in current view.
- Typical use: block inventory update after moving crane to a fixed scan viewpoint.

2. `REFINE_BLOCK`
- Purpose: refine one known world-model block pose.
- Input: `target_block_id` should be set (for example `wm_block_1`).
- Typical use: refine already placed reference block before relative assembly step.
- Optional robust variant: `refine_block.use_pose_roi=true`
  - uses persistent world-model target pose as ROI seed,
  - runs ROI-focused segmentation + registration,
  - falls back to default full-image `REFINE_BLOCK` if ROI preconditions fail.

3. `REFINE_GRASPED`
- Purpose: refine currently grasped block pose.
- Target resolution:
  - if `target_block_id` is empty, world model uses newest block with `TASK_MOVE`.
- Current robust path:
  - FK + ROI in world model for segmentation focus,
  - registration translation + FK orientation fusion.

## 2) Launch Test-Mode Controls (Rosbag)

File:
- [rosbag_block_world_model_test_modes.launch.py](/workspaces/ros2_baustelle_ws/src/concrete_block_stack/concrete_block_perception/launch/rosbag_block_world_model_test_modes.launch.py)

Main arguments:

- `run_scene_discovery` (default `true`)
- `run_refine_block` (default `false`)
- `run_set_task_move` (default `false`)
- `run_refine_grasped` (default `false`)
- `refine_block_id` (default `wm_block_1`)
- `task_move_block_id` (default `wm_block_1`)

Timed call sequence in this launch:

1. `SCENE_DISCOVERY` at 8 s
2. `REFINE_BLOCK` at 11 s (optional)
3. `SetBlockTaskStatus(TASK_MOVE)` at 16 s (optional)
4. `REFINE_GRASPED` at 18 s (optional)

## 3) Mapping to Your Tasks

### A. Detect all blocks in scene (scan viewpoints)

Use `SCENE_DISCOVERY` at each chosen crane viewpoint.

Current recommended approach:
- BT moves crane to fixed scan pose(s),
- trigger one-shot `SCENE_DISCOVERY` per pose,
- world model associates detections to existing blocks or creates new ones.

### B. Refine reference block before assembly

Use `REFINE_BLOCK` with `target_block_id=<wm_block_x>`.

### C. Refine grasped block during pick/place

Use:
1. set grasped block to `TASK_MOVE` (or pass explicit target id),
2. call `REFINE_GRASPED`.

## 4) World Model Configuration (FK+ROI + Fusion)

File:
- [world_model.yaml](/workspaces/ros2_baustelle_ws/src/concrete_block_stack/concrete_block_perception/config/world_model.yaml)

Relevant keys:

- `refine_grasped.use_fk_roi`
- `refine_grasped.tcp_frame`
- `refine_grasped.camera_frame`
- `refine_grasped.camera_info_topic`
- `refine_grasped.tcp_to_block.xyz/rpy`
- `refine_grasped.roi_size_m`
- `refine_grasped.min_depth_m`, `max_depth_m`
- `refine_grasped.segmentation_timeout_s`
- `refine_grasped.segmentation_input.use_black_background`
- `refine_grasped.segmentation_input.blur_kernel_size`

Pose fusion for grasped mode:

- `refine_grasped.pose_fusion.enable`
- `refine_grasped.pose_fusion.mode=position_from_registration_orientation_from_fk`
- `refine_grasped.pose_fusion.max_translation_jump_m`
- `refine_grasped.pose_fusion.max_z_delta_m`
- `refine_grasped.pose_fusion.debug_log`

## 5) Registration Configuration (Robustness)

File:
- [block_registration.yaml](/workspaces/ros2_baustelle_ws/src/concrete_block_stack/concrete_block_perception/config/block_registration.yaml)

Local ICP controls:

- `loc_reg.icp_dist`
- `loc_reg.relax_num_faces_match`
- `loc_reg.use_fk_translation_seed`
- `loc_reg.icp_dist_multipliers`
- `loc_reg.enable_point_to_point_fallback`
- `loc_reg.fk_seed.tcp_frame`
- `loc_reg.fk_seed.tcp_to_block_xyz`

Note on FK seed scope:
- FK translation seed is applied only for `REFINE_GRASPED` action calls (tagged by world model), not for scene discovery/refine-block service path.

## 6) Failure-Debug Dump Package

When enabled, failed registration writes artifacts for offline analysis.

Config:
- `dump.enable: true`
- `dump.failure_package: true`
- `dump.dir: dump`

Outputs per failure:
- `*_fail_mask.png`
- `*_fail_cloud.ply`
- `*_fail_cutout_world.ply`
- `*_fail_meta.yaml`

Default location in installed workspace:
- `/workspaces/ros2_baustelle_ws/install/concrete_block_perception/share/concrete_block_perception/config/dump/`
