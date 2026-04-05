# panda_ik_window

MoveIt2 (MoveItPy) + Franka Panda IK sampling benchmark (window policy evaluation).

This package:

1. **Generates a dataset** (depends on `seed`):
   - Samples `num_points = n` reachable Cartesian targets `p1..pN` (with `p0` as the start state).
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

By default, `p0` start state is now a seeded random joint state (controlled by `seed`).
You can still force a named SRDF state with `named_start:=ready`, or provide explicit joints via `--p0`.

Select path pattern (defined in `path_pattern_definition.md`):

```bash
ros2 launch panda_ik_window ik_benchmark.launch.py num_points:=8 seed:=7 window_size:=3 path_pattern:=trend
```

`path_pattern` choices:
- `trend`: direction-consistent trend
- `switching`: multi-directional switching
- `random`: unconstrained random (may be trend/switching/unstructured, default)

Evaluate multiple window sizes in one run:

```bash
ros2 launch panda_ik_window ik_benchmark.launch.py num_points:=8 seed:=7 window_size:="1,3,8"
```

Reuse an existing candidate set (`targets.json` + `p*.json`) for fair time-model comparison:

```bash
# 1) Generate candidates once (example: totg run)
ros2 launch panda_ik_window ik_benchmark.launch.py \
  num_points:=8 seed:=7 window_size:="1,3,8" time_model:=totg data_root:=data_window/shared_seed7

# 2) Re-evaluate with another time model on exactly the same candidates
ros2 launch panda_ik_window ik_benchmark.launch.py \
  num_points:=8 seed:=7 window_size:="1,3,8" time_model:=trapezoid \
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

- `targets.json`: start `p0` joint positions + sampled Cartesian target points `p1..pN`
- `p1.json`..`pN.json`: IK solutions for each target point
- `summary.json`: unified report with window results (only evaluated ws)
