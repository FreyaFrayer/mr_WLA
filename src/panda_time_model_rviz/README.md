# panda_time_model_rviz

用于在 RViz2 中对比同一个 `seed` 目录下两个结果目录的选解播放结果。

## 功能

- 双 Panda 机械臂并排显示（左/右）
- 自动读取 `seed_dir` 下两个模型目录的**最新** run（按时间戳目录名优先）
- 读取同一 `window_size` 的 `final_path.segments` 并播放
- 可视化访问顺序（球 + 标签 + 折线）

## 目录约定

示例：

```text
data_window/batch_time_model_data/np3/seed11/
  model_a/<timestamp>/summary.json
  model_b/<timestamp>/summary.json
```

## 启动

```bash
ros2 launch panda_time_model_rviz time_model_compare_rviz.launch.py \
  seed_dir:=data_window/batch_time_model_data/np3/seed11
```

## 常用参数

- `window_size`：要播放的窗口大小，默认 `-1`（两侧共同可用的最大 ws）
- `model_a_dir`：默认 `model_a`
- `model_b_dir`：默认 `model_b`
- `sync_loop`：默认 `true`，两侧按同一周期同步重启
- `speed_scale`：播放倍率
- `hold_time_s`：每个路径点附加停留时间

例如固定查看 `ws=3`：

```bash
ros2 launch panda_time_model_rviz time_model_compare_rviz.launch.py \
  seed_dir:=data_window/batch_time_model_data/np3/seed11 \
  window_size:=3
```
