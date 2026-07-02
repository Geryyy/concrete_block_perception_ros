# concrete_block_perception

ROS 2 Humble package for 6-DOF pose estimation of concrete blocks from a synchronized ZED2i camera + Seyond LiDAR sensor pair. Outputs a persistent world model of detected blocks, used by the BT-based pick-and-place pipeline.

## Architecture

```
ZED2i image ──────────────────────────────────────────────────────────────┐
                                                                           │
  compressed ──► image_transport/republish ──► /zed2i/.../image_raw       │
  /seyond_points/compressed ──► cloudini_topic_converter ──► /seyond_points│
                                                                           ▼
ros2_yolos_cpp (YOLO segmentor service) ◄── block_detection_tracking_node
                                                │
                                                │ (sync: image + points)
                                                ▼
                                        world_model_node  ◄──► block_registration_node
                                           (orchestrator)          (ICP pipeline)
                                                │
                                                ├──► /cbp/block_world_model
                                                └──► /cbp/block_world_model_markers
```

Three executables:

| Executable | Config | Role |
|---|---|---|
| `block_detection_tracking_node` | `block_detection_tracking.yaml` | Tracks detections frame-to-frame via `TrackDetections` service |
| `block_registration_node` | `block_registration.yaml` | Action server; runs ICP registration pipeline (cutout → preprocess → global reg → local ICP) |
| `world_model_node` | `world_model.yaml` | Orchestrates the full pipeline; owns the persistent block world model; exposes all BT-facing services |

## Dependencies & interactions

| Direction | Package / lib | Via |
|---|---|---|
| **out** | [concrete_block_perception_interfaces](../concrete_block_perception_interfaces/) | `TrackedDetectionArray`, `RegisterBlock` action, `TrackDetections` / `SegmentAtPoint` / `ExtractMaskCutout` srv |
| **lib** | `ros2_yolos_cpp` | YOLO / ONNX segmentor service |
| **lib** | `sam2_segmentation` | optional SAM2 mask source |
| **consumed by** | [concrete_block_world_model](../concrete_block_world_model/) | `world_model_node` subscribes to the detections and calls the `RegisterBlock` action |

> The `world_model_node` in the architecture diagram is the orchestrator and is **built by [concrete_block_world_model](../concrete_block_world_model/)**, not this package — here we build only `block_detection_tracking_node` and `block_registration_node`. This package emits the *transient* perception interfaces; the *persistent* `BlockArray` is owned by the world model. Note this package does **not** depend on `concrete_block_world_model_interfaces`.

## Input / Output topics

| Topic | Type | Direction | Description |
|---|---|---|---|
| `/zed2i/warped/left/image_rect_color/image_raw` | `sensor_msgs/Image` | in | Rectified left camera image (remapped to `image`) |
| `/seyond_points` | `sensor_msgs/PointCloud2` | in | LiDAR point cloud (remapped to `points`) |
| `/cbp/block_world_model` | `BlockArray` | out | Persistent block world model (all known blocks + poses + status) |
| `/cbp/block_world_model_markers` | `MarkerArray` | out | RViz visualization: block cubes + axes arrows |
| `/cbp/debug/detection_overlay` | `sensor_msgs/Image` | debug | YOLO mask overlay on camera image |
| `/cbp/debug/registration_mask_cutout` | `sensor_msgs/PointCloud2` | debug | Raw mask-selected cloud before registration preprocessing |
| `/cbp/debug/registration_cleaned_cutout` | `sensor_msgs/PointCloud2` | debug | Mask cutout after preprocessing; input to plane-based global registration |
| `/cbp/debug/registration_plane_cloud` | `sensor_msgs/PointCloud2` | debug | Plane inliers selected by global registration and used as the local ICP scene |
| `/cbp/debug/registration_template` | `sensor_msgs/PointCloud2` | debug | Matched template cloud at ICP result pose |
| `/cbp/debug/registration_mask` | `sensor_msgs/Image` | debug | Binary segmentation mask passed to ICP |
| `/cbp/debug/refine_grasped_roi_input` | `sensor_msgs/Image` | debug | Cropped ROI image sent to YOLO for REFINE_GRASPED |

## Services (world_model_node)

All services are under the node namespace (`~/` = `/world_model_node/`).

| Service | Type | Description |
|---|---|---|
| `~/run_pose_estimation` | `RunPoseEstimation` | Trigger a one-shot pose estimation (see modes below). Blocks until result or timeout. Returns updated world snapshot. |
| `~/set_mode` | `SetPerceptionMode` | Set the continuous pipeline mode (see runtime modes below). |
| `~/get_coarse_blocks` | `GetCoarseBlocks` | Query the current world model snapshot without triggering any processing. |
| `~/get_planning_scene` | `GetPlanningScene` | Return blocks + static obstacles as a planning scene (for motion planning). |
| `~/set_block_task_status` | `SetBlockTaskStatus` | Update a block's task status (FREE → MOVE → PLACED / REMOVED). Enforces valid transitions. |
| `~/upsert_block` | `UpsertBlock` | Insert or overwrite a block entry in the world model by ID, bypassing the sensor pipeline. Used to seed known poses from the assembly plan. |

### Block task status lifecycle

```
TASK_FREE ──► TASK_MOVE ──► TASK_PLACED
    │              │              │
    └──────────────┴──► TASK_REMOVED
```

Call `set_block_task_status` with `TASK_MOVE` before `REFINE_GRASPED` so the orchestrator can auto-resolve which block is being transported.

## Perception modes

### Runtime modes (`~/set_mode`)

Set the continuous pipeline behavior between one-shot calls:

| Mode string | Pipeline | Registration on demand | Typical use |
|---|---|---|---|
| `IDLE` | off | no | Standby; no sensor processing |
| `SCENE_SCAN` | track | no | Continuous coarse tracking; no ICP |
| `PRE_GRASP` | full | yes | Before pickup: full YOLO + ICP loop |
| `GRASP_EXECUTE` | track | no | During grasp motion; tracking only |
| `TRANSPORT` | segment | no | Block in transit; segmentation only |
| `PRE_ASSEMBLY` | full | yes | Before placement: full YOLO + ICP loop |
| `ASSEMBLY_EXECUTE` | track | no | During placement motion; tracking only |

Legacy low-level aliases (`SEGMENT`, `TRACK`, `REGISTER`, `FULL`) are still accepted.

### One-shot modes (`~/run_pose_estimation`)

Explicit triggered estimation; call from BT or CLI:

| `mode` string | Description |
|---|---|
| `SCENE_DISCOVERY` | Full YOLO scan of the entire image → register all detected blocks → add new entries to world model. Entry point for initial scene mapping. |
| `REFINE_BLOCK` | Re-register a specific known block (`target_block_id`). Uses the block's stored world model pose as ROI seed (if `refine_block.use_pose_roi: true`). Falls back to full-image YOLO otherwise. |
| `REFINE_GRASPED` | Register the block currently held in the gripper. Uses FK → predicted block center → image ROI → YOLO → ICP. Pose fusion: position from ICP, orientation from FK TCP transform. Auto-resolves the grasped block by finding the newest `TASK_MOVE` entry; pass `target_block_id` explicitly as override. |

## Registration pipeline (`block_registration_node`)

For each detected block mask:

1. **Cutout** — project 3-D LiDAR points through the calibrated camera model; keep only points inside the 2-D segmentation mask.
2. **Preprocess** — statistical outlier removal → optional DBSCAN dominant-cluster filter → downsample to `max_pts` → transform to world frame.
3. **Global registration** — RANSAC plane extraction on the cutout cloud; two planes expected (top + front face). PCA classifies square vs. rectangular front face to build an initial 6-DOF pose estimate. Bypassed when an FK pose seed is available (REFINE_GRASPED).
4. **Local ICP** — Open3D point-to-plane ICP refines the global estimate. Tries templates at multiple yaw hypotheses and `icp_dist` multipliers. Optional point-to-point fallback.

### FK pose seed (REFINE_GRASPED)

When `loc_reg.use_fk_translation_seed: true` and the mode string contains `#REFINE_GRASPED`, the registration node looks up TF for `tcp_frame` and computes:

```
T_world_block = T_world_tcp × T_tcp_block
```

where `T_tcp_block` is set via `tcp_to_block_xyz` (translation) and `tcp_to_block_rpy` (rotation in radians, roll–pitch–yaw).

Standard robot tool frames have **Z pointing down** toward the gripped object. Apply a 180° roll to flip to block-frame Z-up:
```yaml
tcp_to_block_rpy: [3.14159265, 0.0, 0.0]
```

This seed bypasses the global plane extraction entirely and initializes ICP directly from FK, making grasped-block registration robust to gripper occlusion.

## Block coordinate frame convention

The template CAD model uses:

| Axis | Meaning | Block dimension |
|---|---|---|
| +X | Normal to square end-face (depth direction) | 0.9 m |
| +Y | Normal to long rectangular face | 0.6 m |
| +Z | Up (top face normal) | 0.6 m |

The `block_dimensions_m` parameter maps directly to RViz cube marker `scale.x/y/z`:
```yaml
block_dimensions_m: [0.9, 0.6, 0.6]   # [X, Y, Z]
```

## Pick-and-place workflow

Typical BT sequence for one pick-and-place cycle:

```
SCENE_DISCOVERY               → populate world model
GetNextAssemblyTask           → get target_block_id + reference_block_id
REFINE_BLOCK(target)          → precise pickup pose
[ grasp block ]
SetBlockTaskStatus(TASK_MOVE) → mark block as transported
REFINE_GRASPED                → measure grasped block pose
REFINE_BLOCK(reference)       → measure reference block pose (already placed)
[ compute relative transform → execute placement ]
SetBlockTaskStatus(TASK_PLACED)
REFINE_BLOCK(target)          → optional post-placement verification
```

The relative measurement approach (REFINE_GRASPED vs. REFINE_BLOCK on an already-placed reference) cancels absolute FK errors from crane elasticities.

## Configuration files

| File | Node | Key parameters |
|---|---|---|
| `concrete_block_world_model/config/world_model.yaml` | world_model_node | block dimensions, association thresholds, registration gates, REFINE_GRASPED FK/ROI config, static scene objects |
| `config/block_registration.yaml` | block_registration_node | calibration YAML selection, ICP dist, global reg thresholds, FK seed TCP frame + offset, plane clipping |
| `config/block_detection_tracking.yaml` | block_detection_tracking_node | YOLO confidence threshold, tracking params |
| `config/calib_zed2i_to_seyond_new_sensor_head.yaml` | block_registration_node | Live-crane ZED2i/Seyond extrinsic and camera intrinsics K |
| `config/calib_zed2i_to_seyond_old_sensor_head.yaml` | block_registration_node | Old sensor-head calibration for historical rosbags |
| `config/calib_zed2i_to_seyond.yaml` | block_registration_node | Legacy alias kept for compatibility |

All parameters are loaded at startup via ROS 2 parameter files. Changes take effect on next launch (symlink install; no rebuild needed for YAML-only changes).

## Launch files

| File | Purpose |
|---|---|
| `perception.launch.py` | **Primary** — starts processing components plus `concrete_block_world_model/world_model_node`. Args: `pipeline_mode`, `use_gpu`, `use_sim_time`, `start_processing_stack`, `start_world_model`, `calib_yaml`. |
| `rosbag_block_world_model_test_modes.launch.py` | Development/testing — replays a bag and optionally triggers SCENE_DISCOVERY / REFINE_BLOCK / REFINE_GRASPED via launch args. See `README_PERCEPTION_MODES.md` for all args. |
| `block_registration.launch.py` | Registration node only (standalone ICP testing). |
| `detection_tracking.launch.py` | Detection + tracking node only. |
| `world_node.launch.py` | world_model_node only (no processing stack). |
| `commissioning.launch.py` | Hardware commissioning helpers. |

### Minimal test run with a rosbag

```bash
ros2 launch concrete_block_perception rosbag_block_world_model_test_modes.launch.py \
  bag:=/path/to/your/bag \
  calib_yaml:=calib_zed2i_to_seyond_old_sensor_head.yaml \
  run_scene_discovery:=true \
  run_set_task_move:=true \
  task_move_block_id:=wm_block_1 \
  run_refine_grasped:=true
```

## Calibration

Sensor fusion requires a calibrated extrinsic between the ZED2i camera and the Seyond LiDAR. The calibration files contain:
- `T_P_C`: 4×4 transform from camera frame to point-cloud frame (`T_seyond_zed2i_left_optical`)
- `K`: 3×3 camera intrinsic matrix

The live crane defaults to `calib_zed2i_to_seyond_new_sensor_head.yaml`:

```bash
ros2 launch concrete_block_perception perception.launch.py
```

Historical rosbags recorded with the old sensor head should override the calibration:

```bash
ros2 launch concrete_block_perception rosbag_block_world_model_test_modes.launch.py \
  bag:=/path/to/old_sensor_head_bag \
  calib_yaml:=calib_zed2i_to_seyond_old_sensor_head.yaml
```

The new-sensor-head file is derived from the current `sensor_description`
`blackfly_rotated` to `seyond` transform. Re-run the extrinsic calibration
procedure and update `calib_zed2i_to_seyond_new_sensor_head.yaml` if the
relative sensor rig is remounted.
