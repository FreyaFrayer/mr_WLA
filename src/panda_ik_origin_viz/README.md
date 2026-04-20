# panda_ik_origin_viz

用于 `panda_ik_window` 输出结果的离线可视化：把同一个 `summary.json` 里的
`origin`、`window.results_by_ws[ws]` 和 `trapezoid_solutions_true_plan.results_by_ws[ws]`
三条关节轨迹在 `RViz2` 中同时播放，默认对比 `origin`、`ws=1` 和 `ws=1 planner`。

## 功能

- 同时显示三台 Panda，并排对比：
  - 下方：`ORIGIN`
  - 中间：`WS=1`
  - 上方：`WS=1 PLANNER`
- 从 `summary.json` 读取三条轨迹的逐段关节目标和段时间
- 以 quintic time-scaling 插值播放，不依赖 MoveIt 在线执行
- 在 RViz 中标出 `p0..pN` 路径点，并在标题里直接显示三条轨迹的总时间
- 支持循环、同步循环、暂停、seek 等播放器 topic

## 输入数据

默认读取同一个 run 目录下的 `summary.json`：

- `origin.segments[*]`
- `window.results_by_ws["1"].final_path.segments`
- `trapezoid_solutions_true_plan.results_by_ws["1"].segments`

如果你想看别的窗口大小，可通过 launch 参数传 `window_size:=3`。第三条 planner
轨迹默认跟随同一个 `window_size`；如果想单独切换，可再传 `planner_window_size:=5`。

注意：`summary.json` 必须真的包含对应 `ws` 的 `window` 和 `trapezoid_solutions_true_plan`
结果；如果这次 benchmark 没评估该窗口，播放器会直接报错。

## 安装

将本包放到工作空间 `src/` 下并编译：

```bash
cd ~/ws
colcon build --symlink-install --packages-select panda_ik_origin_viz
source install/setup.bash
```

## 运行

```bash
ros2 launch panda_ik_origin_viz ik_origin_compare_rviz.launch.py \
  result_dir:=/absolute/path/to/run_dir \
  window_size:=1
```

`result_dir` 也可以给父目录，程序会自动递归找到其中最新的一个 `summary.json` 所在目录。

## 常用参数

- `window_size`：默认 `1`
- `planner_window_size`：默认 `0`，表示跟随 `window_size`
- `fixed_frame`：默认 `world`
- `publish_rate_hz`：默认 `50.0`
- `speed_scale`：默认 `1.0`
- `loop`：默认 `true`
- `sync_loop`：默认 `true`
- `hold_time_s`：默认 `0.5`

## RViz 配置

本包自带 `rviz/ik_origin_compare.rviz`，launch 会自动加载，包含：

- `/origin/robot_description` + `TF Prefix: origin_`
- `/ws/robot_description` + `TF Prefix: ws_`
- `/ws_planner/robot_description` + `TF Prefix: ws_planner_`
- `MarkerArray: /ik_origin/markers`

## 说明

- 该包默认把 `origin`、`ws=1` 和 `ws=1 planner` 放在同一个 `summary.json` 下比较。
- 如果你后续要比较别的 `ws`，只需要改 `window_size`；如果 planner 想单独看别的 `ws`，
  再额外设置 `planner_window_size` 即可。
