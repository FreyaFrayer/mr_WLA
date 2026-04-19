# panda_ik_window

MoveIt2 (MoveItPy) + Franka Panda IK sampling benchmark (window policy evaluation).

This package:

1. **Generates a dataset** (depends on `seed`):
   - Samples `num_points = n` reachable Cartesian target poses `p1..pN` (with `p0` as the start state).
   - For each point `p_i`, samples exactly `num_solutions = m` IK solutions (stored in `p{i}.json`).

2. **Evaluates window policies** (depends only on `window_size = ws`):
   - For a single dataset, evaluates **only the requested** `ws` (a single value or a list).
   - Use `window_size:=all` to sweep `ws = 1, 2, ..., n` (backward-compatible default).
   - `ws = 1` is equivalent to **greedy**.
   - `ws = n` is equivalent to **global optimal**.
   - The window logic is optimized so that when `remain <= ws`, it solves the remaining points **once** (full DP) and finishes.

`summary.json` records results **only for the evaluated `ws`**.

## Build

```bash
cd ~/ws_moveit2
colcon build --packages-select panda_ik_window
source install/setup.bash
```

## Run (single experiment)

```bash
ros2 launch panda_ik_window ik_benchmark.launch.py num_points:=8 seed:=7 window_size:=3
```

Default target sampling workspace:
- outer box: `x in [-0.75, 0.75]`, `y in [-0.55, 0.55]`, `z in [0.05, 0.85]`
- inner XY exclusion: `sqrt(x^2 + y^2) >= 0.25`, so targets too close to the robot body are skipped

You can override it from launch if needed:

```bash
ros2 launch panda_ik_window ik_benchmark.launch.py \
  ws_x_min:=-0.75 ws_x_max:=0.75 \
  ws_y_min:=-0.55 ws_y_max:=0.55 \
  ws_z_min:=0.05 ws_z_max:=0.85 \
  ws_xy_inner_radius:=0.25
```

## Verify Self-Collision On `dataset_ws3_top50_sort50`

This package also provides a checker for:
- each `q_cur` sample
- all next-point candidates in `q_cand_fut[:,0,:,:]` (masked by `cand_mask[:,0,:]`)
- synchronized trapezoid-velocity interpolation between `q_cur -> q_cand_next`
- self-collision query at every sampled state

Run with MoveIt parameters preloaded by launch:

```bash
ros2 launch panda_ik_window ik_self_collision_check.launch.py \
  dataset:=/Users/<you>/Desktop/dataset_ws3_top50_sort50 \
  sample_dt:=0.02 \
  min_samples:=5
```

Useful limits for smoke-test:

```bash
ros2 launch panda_ik_window ik_self_collision_check.launch.py \
  dataset:=/Users/<you>/Desktop/dataset_ws3_top50_sort50 \
  max_samples:=100 \
  max_candidates:=50
```

Outputs are written to dataset dir by default:
- `self_collision_trapezoid.npz`
- `self_collision_trapezoid_summary.json`

## Visualize One Collision Pair In RViz

After generating `collision_pairs.json`, you can pick one pair and replay
`q_cur -> q_cand_next` with synchronized trapezoid timing in a loop.

```bash
ros2 launch panda_ik_window ik_collision_pair_playback_rviz.launch.py \
  dataset:=/Users/<you>/Desktop/dataset_ws3_top50_sort50 \
  pair_index:=0 \
  sample_dt:=0.02 \
  min_samples:=5 \
  speed_scale:=1.0
```

Optional: directly specify `(i, k)` (bypass `collision_pairs.json`):

```bash
ros2 launch panda_ik_window ik_collision_pair_playback_rviz.launch.py \
  dataset:=/Users/<you>/Desktop/dataset_ws3_top50_sort50 \
  sample_i:=123 \
  candidate_k:=7
```

By default, `p0` start state is now a seeded random joint state (controlled by `seed`).
You can still force a named SRDF state with `named_start:=ready`, or provide explicit joints via `--p0`.
Use `p0_down:=true` to force the `p0` tip orientation to vertical-down (world `-Z`) by IK while keeping the same tip position.

Select path pattern (defined in `path_pattern_definition.md`):

```bash
ros2 launch panda_ik_window ik_benchmark.launch.py num_points:=8 seed:=7 window_size:=3 path_pattern:=trend
```

`path_pattern` choices:
- `trend`: direction-consistent trend
- `switching`: multi-directional switching
- `random`: unconstrained random (may be trend/switching/unstructured, default)
- `trend_plus`: trend-dominant with periodic switch (every 3~5 trend points) and danger-zone turn-back to safe zone

For `trend` / `trend_plus`, you can additionally limit the point-to-point step length:

```bash
ros2 launch panda_ik_window ik_benchmark.launch.py \
  num_points:=8 seed:=7 window_size:=3 path_pattern:=trend \
  trend_max_step:=0.25
```

In `trend` / `trend_plus`, the sampler now ranks a small pool of valid FK candidates and prefers the continuation that is more direction-consistent, uses a steadier step length, and stays farther from workspace edges. It also reuses a stable reference orientation instead of carrying fully random FK orientations point-to-point.

When the last accepted trend point is already near the workspace edge, the sampler only accepts candidates that move back toward the safer middle region, which reduces "edge to more-edge" dead ends.

Evaluate multiple window sizes in one run:

```bash
ros2 launch panda_ik_window ik_benchmark.launch.py num_points:=8 seed:=7 window_size:="1,3,8"
```

Reuse an existing candidate set (`targets.json` + `p*.json`) for repeatable evaluation:

```bash
# 1) Generate candidates once
ros2 launch panda_ik_window ik_benchmark.launch.py \
  num_points:=8 seed:=7 window_size:="1,3,8" data_root:=data_window/shared_seed7

# 2) Re-evaluate on exactly the same candidates
ros2 launch panda_ik_window ik_benchmark.launch.py \
  num_points:=8 seed:=7 window_size:="1,3,8" \
  reuse_candidates_dir:=data_window/shared_seed7/<timestamp>
```

Sweep all (ws=1..n), old behavior:

```bash
ros2 launch panda_ik_window ik_benchmark.launch.py num_points:=8 seed:=7 window_size:=all
```

Data will be saved under:

```
<data_root>/<timestamp>/
```

(`data_root` defaults to `./data_three`.)

## Batch run

See `batch_ik_window.py` at the package root. It runs one dataset per seed and writes a CSV summary.

Typical usage:

```bash
python3 batch_ik_window.py --num-points 8 --seeds 7,8,9
```

## Key outputs

- `targets.json`: start `p0` joint positions + sampled Cartesian target poses `p1..pN` (`x,y,z,qx,qy,qz,qw`)
- `p1.json`..`pN.json`: IK solutions for each target pose
- `summary.json`: unified report with window results (only evaluated ws), plus `origin` (direct planner point-to-point time through `p0->p1..pN`, no IK candidate selection). `origin.joint_positions_by_point` records per-point joint angles (`p0..pN`) for each axis.
- `summary.json.trapezoid_solutions_true_plan`: replay the trapezoid-selected joint targets with point-to-point planner calls and record per-`ws` segment/total time.
