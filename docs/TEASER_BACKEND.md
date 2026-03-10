# TEASER++ Backend Integration

This package now supports two registration backends with identical ROS interfaces:

- `legacy` -> `block_registration_node`
- `teaser` -> `concrete_block_registration_teaser/block_registration_teaser_node`

Both expose:
- action: `register_block`
- service: `register_block_pose`

## Launch selection (default non-breaking)

`launch/perception.launch.py` adds:

- `registration_backend:=legacy|teaser` (default: `legacy`)

Example:

```bash
ros2 launch concrete_block_perception perception.launch.py registration_backend:=legacy
ros2 launch concrete_block_perception perception.launch.py registration_backend:=teaser
```

## Config

TEASER parameters are in:

- `config/block_registration.yaml` under `teaser_reg`

Legacy parameters remain in `preproc`, `glob_reg`, `loc_reg`.

## Dependency

TEASER++ is vendored at:

- `external/teaser-plusplus`

If `CBP_ENABLE_TEASER=ON` and submodule is missing, CMake fails with a clear error.

## A/B benchmark scripts (paper artifact)

1. Run benchmark per backend:

```bash
ros2 run concrete_block_perception run_pose_estimation_benchmark.py \
  --backend-label legacy --mode SCENE_DISCOVERY --iterations 20 --output /tmp/legacy.json

ros2 run concrete_block_perception run_pose_estimation_benchmark.py \
  --backend-label teaser --mode SCENE_DISCOVERY --iterations 20 --output /tmp/teaser.json
```

2. Compare and export summary:

```bash
ros2 run concrete_block_perception compare_backend_results.py \
  --legacy /tmp/legacy.json \
  --teaser /tmp/teaser.json \
  --out-json /tmp/backend_compare.json \
  --out-csv /tmp/backend_compare.csv
```
