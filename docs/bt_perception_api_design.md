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
Use explicit modes matching task phases.

1. `SCENE_SCAN`
- Detect/segment all visible blocks.
- Compute coarse 3D pose (centroid + yaw heuristic).
- No full ICP per frame.

2. `PRE_GRASP`
- Focus on selected target block.
- Optional refine on demand.

3. `GRASP_EXECUTE`
- Freeze or very low-rate updates.

4. `TRANSPORT`
- Perception mostly off.

5. `PRE_ASSEMBLY`
- Refine grasped block and reference block pose.

6. `ASSEMBLY_EXECUTE`
- ROI-only updates if needed.

---

## Proposed ROS Interfaces
These are intentionally minimal and BT-friendly.

### 1) Set Mode
File: `srv/SetPerceptionMode.srv`

```srv
string mode            # SCENE_SCAN, PRE_GRASP, GRASP_EXECUTE, TRANSPORT, PRE_ASSEMBLY, ASSEMBLY_EXECUTE
string target_block_id # optional, empty if none
bool enable_debug
---
bool success
string message
```

### 2) Coarse Scene Update
File: `srv/GetCoarseBlocks.srv`

```srv
bool force_refresh
float32 timeout_s
builtin_interfaces/Time query_stamp  # optional; zero = latest
---
bool success
concrete_block_perception/msg/BlockArray blocks
string message
```

Semantics:
- Returns coarse pose blocks quickly.
- Pose quality marked as coarse in metadata (see below).

### 3) Precise Pose On Demand
File: `action/RefineBlockPose.action`

```action
string block_id
geometry_msgs/PoseStamped seed_pose   # optional; frame must match cloud/image frame
float32 timeout_s
bool use_roi
float32 roi_size_m                     # used when use_roi=true
builtin_interfaces/Time query_stamp    # optional; zero = latest
---
bool success
concrete_block_perception/msg/Block refined_block
string message
---
string stage                           # SEGMENT, TRACK, REGISTER
float32 progress
```

Semantics:
- Runs expensive registration for one target.
- Updates world cache only for that target on success.

### 4) Optional Cache Query
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
1. `SetPerceptionMode(SCENE_SCAN)`
2. `GetCoarseBlocks` -> choose target
3. `SetPerceptionMode(PRE_GRASP, target_id)`
4. `RefineBlockPose(target_id)`
5. execute grasp

### Transport
1. `SetPerceptionMode(TRANSPORT)`
2. no updates

### Assembly
1. `SetPerceptionMode(PRE_ASSEMBLY, target_id)`
2. `RefineBlockPose(reference_id)`
3. `RefineBlockPose(target_id)`
4. execute assembly

---

## Performance Strategy
Prioritize this order:

1. **Registration on demand only**
- biggest practical gain.

2. **Coarse pose path without ICP**
- centroid from segmented cloud + simple yaw estimate.

3. **ROI refinement**
- crop image/cloud around seed pose before segmentation/registration.

4. **Tracking-only updates between refines**
- no repeated registration.

---

## Incremental Migration Plan

### Phase 1 (Low risk)
- Add `SetPerceptionMode` + `GetCoarseBlocks`.
- Keep current world node; gate registration by mode.
- Keep existing topics unchanged.

### Phase 2
- Add `RefineBlockPose` action.
- Move precise registration to explicit BT requests.

### Phase 3
- Introduce ROI refine path.
- Optionally split orchestration into dedicated node.

### Phase 4
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
