# Perception Modes Overview

This file summarizes the perception/world-model modes and how they are used by:
`launch/rosbag_block_world_model_test_modes.launch.py`.

## 1) Startup/runtime modes (`perception_mode`)
These are handled by `world_model_node` and mapped to pipeline behavior:

- `IDLE`: no processing
- `SCENE_SCAN`: tracking + coarse world publish
- `PRE_GRASP`: full pipeline, registration on demand
- `GRASP_EXECUTE`: tracking only
- `TRANSPORT`: segmentation only
- `PRE_ASSEMBLY`: full pipeline, registration on demand
- `ASSEMBLY_EXECUTE`: tracking only

Legacy-compatible aliases are also accepted:
- `SEGMENT`, `TRACK`, `REGISTER`, `FULL`

## 2) Low-level pipeline modes (`pipeline_mode`)
Available low-level pipeline settings:

- `idle`
- `segment`
- `track`
- `register`
- `full`

## 3) One-shot service modes (`RunPoseEstimation.mode`)
For explicit one-shot estimation requests:

- `SCENE_DISCOVERY`
- `REFINE_BLOCK`
- `REFINE_GRASPED`

## 4) What the current test launch does
In `launch/rosbag_block_world_model_test_modes.launch.py`:

- Starts world model with `perception_mode: IDLE`
- Triggers one-shot `SCENE_DISCOVERY` after a timer (enabled by default)
- Supports optional one-shot calls for `REFINE_BLOCK` and `REFINE_GRASPED` via launch args

Default behavior still tests only scene discovery mode (`run_refine_block:=false`, `run_refine_grasped:=false`).

## 5) Simplified launch control
The test launch was simplified so you no longer need to comment/uncomment actions in code.

Useful args:
- `perception_mode:=IDLE|SCENE_SCAN|PRE_GRASP|...`
- `run_scene_discovery:=true|false`
- `run_refine_block:=true|false`
- `run_set_task_move:=true|false`
- `task_move_block_id:=<block_id>`
- `run_refine_grasped:=true|false`
- `refine_block_id:=<block_id>`

Example:
```bash
ros2 launch concrete_block_perception rosbag_block_world_model_test_modes.launch.py \
  bag:=/path/to/bag \
  run_scene_discovery:=true \
  run_refine_block:=true \
  refine_block_id:=wm_block_1 \
  run_set_task_move:=true \
  task_move_block_id:=wm_block_1 \
  run_refine_grasped:=true
```

`run_set_task_move:=true` calls `~/set_block_task_status` at 13s and sets the
selected block to `TASK_MOVE`, which enables `REFINE_GRASPED` without passing
an explicit target id.

## Source of truth in code
- `src/utils/world_model_utils.cpp`
- `src/nodes/world_model_node.cpp`
- `srv/RunPoseEstimation.srv`
- `srv/SetPerceptionMode.srv`
- `srv/SetBlockTaskStatus.srv`
