# panda_ik_benchmark_viz

用于 **panda_ik_benchmark** 的离线可视化：把一次 benchmark 的输出结果（`summary.json` 等）在 **RViz2** 中同时播放 **greedy** 和 **global** 两种方法的关节运动，并把路径点高亮标注出来。

## 功能

- 同时显示两台 Panda（同一 URDF），左右并排对比：
  - 左侧：`GREEDY`
  - 右侧：`GLOBAL`
- 播放两条关节轨迹（从 `summary.json` 读取段时间 `time_s / segment_time_s`）
- 在 RViz 内高亮标注访问顺序的路径点（球 + 文本 + 连线）
- 不依赖 MoveIt 执行（仅做可视化播放），插值采用 quintic time-scaling（段端点零速度/加速度）

## 依赖

- ROS 2（任意支持 `robot_state_publisher` / `rviz2` 的发行版）
- `robot_state_publisher` 会发布 `robot_description` 话题（transient local），用于 RViz2 的 RobotModel 显示
- Panda URDF/Xacro：
  - 优先尝试 `moveit_resources_panda_description`
  - 其次尝试 `franka_description`

若你的系统里 Panda URDF 不在上述包里，请编辑 `launch/ik_benchmark_compare_rviz.launch.py` 中 `_pick_panda_description_file()`。

## 安装

将本包放到工作空间 `src/` 下并编译：

```bash
cd ~/ws/src
# 把本包目录 panda_ik_benchmark_viz 拷贝到这里
cd ..
colcon build --symlink-install
source install/setup.bash
```

## 运行

如果你的输出在 `data/20260121_083922/`：

```bash
ros2 launch panda_ik_benchmark_viz ik_benchmark_compare_rviz.launch.py result_dir:=/absolute/path/to/data/20260121_083922
```

也可以直接给父目录 `data/`，程序会自动选择里面最新的一个子目录（按时间戳目录名优先）：

```bash
ros2 launch panda_ik_benchmark_viz ik_benchmark_compare_rviz.launch.py result_dir:=/absolute/path/to/data
```

### 常用参数

- `fixed_frame`：默认 `world`
- Node `ik_benchmark_compare_player` 的参数可在 launch 中继续传入，例如：

```bash
ros2 launch panda_ik_benchmark_viz ik_benchmark_compare_rviz.launch.py \
  result_dir:=/absolute/path/to/data \
  fixed_frame:=world
```

在 `ik_benchmark_compare_player.py` 里还支持：
- `publish_rate_hz`（默认 50）
- `hold_time_s`（每个路径点停留，默认 0.5s）
- `loop`（循环播放，默认 true）
- `speed_scale`（播放速度倍数，默认 1.0）
- `greedy_offset_xyz / global_offset_xyz`（两台机器人摆放位置）

## RViz 配置

本包自带 `rviz/ik_benchmark_compare.rviz`，launch 会自动加载，包含：
- 两个 RobotModel：
  - /greedy/robot_description + TF Prefix: greedy_
  - /global/robot_description + TF Prefix: global_
- MarkerArray: /ik_benchmark/markers

若你想手动配置 RViz，也可以照此添加显示项。


## Notes for ROS 2 Jazzy

- This package uses `FindExecutable` from `launch.substitutions` (required on Jazzy).

## 叠加对比模式（同一基座 + 染色“滤镜” + 同步循环）

除了左右并排对比，本包还提供一个 **叠加对比** 的 launch + RViz 配置：

- 两台机械臂 **基座完全重合**（从同一起始位姿开始），沿各自方法的解算路径运动
- greedy 机械臂：**深蓝色**（半透明）
- global 机械臂：**浅黄色**（更透明一些）
- 循环播放时：每一轮 **同时开始**，一般不会同时结束；先结束的一台会保持末端姿态，
  等另一台结束后两者再同时从起点重新开始（`sync_loop=true`）

运行：

```bash
ros2 launch panda_ik_benchmark_viz ik_benchmark_compare_overlay_tint_rviz.launch.py \
  result_dir:=/absolute/path/to/data/20260121_083922
```

说明：由于 Panda 的视觉网格是 `.dae`（通常带有**嵌入材质**），RViz2 RobotModel 很可能会忽略
URDF 的 `<material><color .../>`。因此叠加对比模式使用 **每个 link 的 mesh MarkerArray** 来强制着色：

- `/greedy/mesh_markers`
- `/global/mesh_markers`

并且设置 `mesh_use_embedded_materials=false`，从而保证“滤镜”颜色一定生效。

你也可以在 launch 里调整：
- 两台机器人的 tint 颜色/透明度（见 `launch/ik_benchmark_compare_overlay_tint_rviz.launch.py` 里的 `greedy_tint/global_tint`）
- 播放器参数（`hold_time_s`、`speed_scale`、`publish_rate_hz` 等）

## 叠加对比模式（带暂停/进度条面板）

为了方便停下来逐帧/细粒度对比，本仓库还附带一个 **RViz Panel 插件**（Qt 按钮 + 时间滑条）：

- **Pause / Play** 按钮
- 时间进度条（可拖动进行 seek）

它通过 topic 控制播放器：

- `/ik_benchmark/playback/pause` (std_msgs/Bool)
- `/ik_benchmark/playback/seek` (std_msgs/Float32, 0..1)

并订阅播放器状态：

- `/ik_benchmark/playback/progress`
- `/ik_benchmark/playback/time_s`
- `/ik_benchmark/playback/cycle_s`
- `/ik_benchmark/playback/paused`

> 注意：面板插件在单独的包里：`panda_ik_benchmark_viz_rviz_plugins`（需要一起放进 workspace 编译）。

运行（带控制面板的 RViz 配置）：

```bash
ros2 launch panda_ik_benchmark_viz ik_benchmark_compare_overlay_tint_control_rviz.launch.py \
  result_dir:=/absolute/path/to/data/20260121_083922
```
