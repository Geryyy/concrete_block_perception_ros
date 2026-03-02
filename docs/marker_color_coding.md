# RViz Block Marker Color Coding

This is the state-to-color mapping used by `world_model_node` for block markers.

## Mapping
- `TASK_REMOVED` -> **red** (`r=0.9, g=0.1, b=0.1`)
- `TASK_PLACED` -> **cyan** (`r=0.1, g=0.9, b=0.9`)
- `TASK_MOVE` -> **blue** (`r=0.2, g=0.4, b=1.0`)
- `POSE_PRECISE` -> **green** (`r=0.1, g=0.8, b=0.2`)
- `POSE_COARSE` -> **yellow** (`r=1.0, g=0.8, b=0.1`)
- otherwise (`POSE_UNKNOWN` / `TASK_UNKNOWN`) -> **gray** (`r=0.5, g=0.5, b=0.5`)

Marker alpha is fixed at `a=0.6`.

## Priority
The logic is evaluated in this order:
1. `TASK_REMOVED`
2. `TASK_PLACED`
3. `TASK_MOVE`
4. `POSE_PRECISE`
5. `POSE_COARSE`
6. fallback gray

So task-state colors override pose-state colors when both are set.
