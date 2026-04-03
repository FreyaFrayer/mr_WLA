# panda_ik_waypoint

`panda_ik_waypoint` is a lightweight RViz2 visualization package for `panda_ik_window` outputs.

It provides:
- path target point markers (spheres + labels)
- path line strip in selected visiting order
- sequential point-to-point arm playback using selected `window_size` result

## Launch

```bash
ros2 launch panda_ik_waypoint ik_waypoint_rviz.launch.py \
  result_dir:=data_window/np3/seed10/20260402_135424 \
  window_size:=3
```

`result_dir` can also be a parent folder (the node will auto-pick a latest run containing `summary.json`).

## Key Parameters

- `result_dir` (string): run directory or parent directory
- `window_size` (int): which ws result to visualize (`-1` means max ws)
- `publish_rate_hz` (float): joint-state publish rate
- `speed_scale` (float): playback speed multiplier
- `loop` (bool): replay loop on/off
