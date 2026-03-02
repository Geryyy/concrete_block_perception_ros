# BT-Driven Perception API Design (On-Demand Pose Estimation)

## Goal
Refactor perception from continuous full-pipeline processing to **phase-driven, on-demand processing** controlled by the behavior tree (BT):
- Run cheap perception for scene awareness.
- Run expensive registration only when needed for grasp/assembly.
- Keep world model as a cache with explicit freshness.

This preserves current components (`segment -> track -> register -> world`) but changes **when** each stage runs.

---

## Current State (Summary)
Current world node executes an asynchronous chain each frame:
1. Segment (`/yolos_segmentor_service/segment`)
2. Track (`/block_detection_tracking_node/track`)
3. Register (`register_block` action)
4. Publish world model (`/cbp/block_world_model`)

This is robust but expensive, especially when registration dominates runtime.

---

## Target Architecture
Introduce a BT-facing `perception_orchestrator` node that controls existing services/actions.

### Responsibilities
- Hold current **perception mode**.
- Serve BT requests for:
  - coarse scene update
  - on-demand pose refinement of one target
- Publish state and freshness metadata for downstream planning/control.

### Keep Existing Nodes
- `yolos_segmentor_service` (segmentation)
- `block_detection_tracking_node` (tracking)
- `block_registration_node` (precise pose)
- `world_model_node` (can be simplified into orchestrator over time)

---

## Proposed Modes
Use explicit one-shot modes. The node stays idle between requests.

1. `SCENE_DISCOVERY`
- Run detect + segment + register for all visible blocks once.
- Update persistent world model.

2. `REFINE_BLOCK`
- Run detect + segment + register for one given `block_id`.
- Replace block pose in persistent world model.

3. `REFINE_GRASPED`
- Run detect + segment + register for the grasped block once.
- Target block id comes from request, or from world-model `TASK_MOVE`.

---

## Proposed ROS Interfaces
These are intentionally minimal and BT-friendly.

### 1) One-shot Pose Request
File: `srv/RunPoseEstimation.srv`

```srv
string mode            # SCENE_DISCOVERY | REFINE_BLOCK | REFINE_GRASPED
string target_block_id # required for REFINE_BLOCK, optional for REFINE_GRASPED
bool enable_debug
float32 timeout_s
---
bool success
concrete_block_perception/msg/BlockArray blocks
string message
```

### 2) Optional Cache Query
File: `srv/GetBlockState.srv`

```srv
string block_id
---
bool found
concrete_block_perception/msg/Block block
float32 age_s
string quality   # COARSE | PRECISE
```

---

## Rosbag Testability Requirements
To evaluate modes reliably with rosbags, the API must support **time-aware requests** and deterministic replay windows.

### Required behavior
1. All nodes run with `use_sim_time=true`.
2. All responses include or preserve source timestamps from bag data.
3. If `query_stamp` is set in a request, server returns data valid for that stamp (or nearest within tolerance), not just latest.
4. BT test runner can replay the same bag segment repeatedly.

### Replay window control
For testing, use bag + time window:
- `bag_uri`
- `start_offset_s`
- `duration_s`
- optional `rate`

Implementation note:
- Keep this in a **test harness node/launch** (not core runtime API).
- The harness starts `ros2 bag play` with the requested window and triggers BT/perception calls at known `/clock` times.

---

## Optional Test Harness API (Recommended)
Create a test-only service in a separate package (for example `concrete_block_perception_test_tools`):

File: `srv/RunPerceptionScenario.srv`

```srv
string bag_uri
float32 start_offset_s
float32 duration_s
float32 rate
string mode                # SCENE_SCAN, PRE_GRASP, ...
string target_block_id
string reference_block_id
---
bool success
string run_id
string message
```

This avoids polluting production interfaces while making regression tests reproducible.

---

## Message-Level Metadata (Recommended)
Extend or wrap block outputs with:
- `quality`: `COARSE` or `PRECISE`
- `age_s` or `last_update_stamp`
- `source`: `SEGMENT`, `TRACK`, `REGISTER`

If `Block.msg` should stay unchanged, publish side-channel status topic:
- `/cbp/block_state_meta` keyed by block id.

---

## BT Blackboard Contract (Recommended)
Use fixed keys to decouple BT/action nodes from perception internals.

- `target_block_id: string`
- `reference_block_id: string`
- `target_block_pose_coarse: PoseStamped`
- `target_block_pose_precise: PoseStamped`
- `reference_block_pose_precise: PoseStamped`
- `perception_mode: string`
- `perception_ok: bool`

---

## Suggested BT Sequence Integration

### Pickup
1. `RunPoseEstimation(SCENE_DISCOVERY)`
2. choose target from world model
3. `RunPoseEstimation(REFINE_BLOCK, target_id)`
5. execute grasp

### Transport
1. no perception requests (node remains idle)

### Assembly
1. `RunPoseEstimation(REFINE_BLOCK, reference_id)`
2. `RunPoseEstimation(REFINE_GRASPED, grasped_id or empty)`
4. execute assembly

---

## Performance Strategy
Prioritize this order:

1. **Registration on demand only**
- biggest practical gain.

2. **ROI refinement**
- crop image/cloud around seed pose before segmentation/registration.

3. **No background processing**
- node is idle unless a one-shot request is active.

---

## Incremental Migration Plan

### Phase 1 (Current)
- Add `RunPoseEstimation` service with three one-shot modes.
- Keep node idle between calls.
- Persist updated precise poses in world model.

### Phase 2
- Add ROI-seeded refinement path from FK pose.

### Phase 3
- Optional component composition/intra-process optimization.

---

## Backward Compatibility
- Keep current launch files and topics (`/cbp/block_world_model`, `/cbp/tracked_detections`, debug overlays).
- New interfaces are additive.
- Existing `pipeline_mode` can map to default orchestrator mode for non-BT usage.

---

## Acceptance Criteria
1. During `TRANSPORT`, registration count is zero.
2. During pickup/assembly, refine action can be triggered and returns precise pose.
3. End-to-end task time improves without reducing successful grasp/assembly rate.
4. Debug overlays remain available when mode enables them.
5. Rosbag tests are reproducible for fixed `(bag_uri, start_offset_s, duration_s, mode)`.
