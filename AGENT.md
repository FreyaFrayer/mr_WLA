# AGENT.md

本文件面向 AI 编程助手（如 Cursor / Claude Code / ChatGPT Codex 类代理）。
目标是帮助代理在 **不破坏实验语义、输出格式和运行方式** 的前提下，理解并修改这个仓库。

---

## 0. 实验环境

- 环境采取：`ROS (Jazzy) on Ubuntu 24.04`

---

## 1. 项目是什么

这是一个基于 **ROS2 + MoveIt2 (MoveItPy) + Franka Panda** 的离线基准项目，核心任务有两部分：

1. **生成 IK 数据集**
   - 从起始状态 `p0` 出发，采样 `p1..pN` 个可达笛卡尔点。
   - 对每个点采样固定数量的 IK 候选解（通常 100 或 200 个）。
   - 每个点对应一个 `p{i}.json`。

2. **评估窗口策略（window policy）**
   - 对给定 `window_size = ws`，评估“只看未来若干步”的路径选择策略。
   - `ws = 1` 等价于 greedy。
   - `ws = n` 等价于全局最优。
   - 当剩余点数 `remain <= ws` 时，代码会对尾部 **一次性做完整 DP**，然后直接结束。

这个仓库不是在线控制器。它更像是：
- 离线数据生成工具
- 轨迹选择策略评测器
- 基于梯形时间模型的窗口策略评测平台

---

## 2. 仓库当前的真实结构

### 2.1 关键目录

```text
.
├── AGENT.md
├── path_pattern_definition.md
├── script/
│   ├── batch_ik_window.py
│   └── batch_run_window.py
└── src/
    ├── moveit_resources/
    └── panda_ik_window/
        ├── README.md
        ├── package.xml
        ├── setup.py
        ├── launch/
        │   └── ik_benchmark.launch.py
        ├── config/
        │   └── moveit_cpp_offline.yaml
        └── panda_ik_window/
            ├── types.py
            ├── ik/
            │   ├── sampler_space.py
            │   └── robust_sampler.py
            ├── planning/
            │   ├── search.py
            │   └── time_metric.py
            ├── scripts/
            │   └── run_benchmark.py
            └── utils/
                ├── reporting.py
                ├── robot.py
                ├── targets.py
                └── timestamp.py
```

### 2.2 可以忽略的脏内容

这些内容来自压缩包或本地环境，不属于核心代码：

- `__MACOSX/`
- `.DS_Store`
- `__pycache__/`

AI 修改代码时，不要围绕这些文件做任何逻辑性改动。

---

## 3. 从入口看完整执行流

### 3.1 主入口

主入口是：

```bash
ros2 launch panda_ik_window ik_benchmark.launch.py ...
```

launch 文件：

```text
src/panda_ik_window/launch/ik_benchmark.launch.py
```

它最终会调用 console script：

```text
panda_ik_window.scripts.run_benchmark:main
```

### 3.2 主流程在这里

```text
src/panda_ik_window/panda_ik_window/scripts/run_benchmark.py
```

这个文件负责：

1. 解析 CLI 参数。
2. 初始化 MoveItPy / Panda 机器人上下文。
3. 构造起始状态 `p0`。
4. 顺序采样 `p1..pN`。
5. 对每个点先做 **快速可行性预检查**，再做 **多轮 IK 采样并去重**。
6. 保存 `targets.json` 与各点的 `p{i}.json`。
7. 构造 `SegmentTimeModel`。
8. 对一个或多个 `ws` 执行窗口评估。
9. 输出 `summary.json` 和 `summary.txt`。

### 3.3 评测核心逻辑在哪

```text
src/panda_ik_window/panda_ik_window/planning/search.py
```

重点函数：

- `greedy_path(...)`
- `window_path_receding_horizon(...)`
- `global_optimal_path(...)`
- `_optimal_path_indices_dp(...)`

其中 `window_path_receding_horizon(...)` 是最重要的函数。

---

## 4. 必须保护的语义约束

这是最重要的部分。AI 可以重构实现，但不要改变这些语义。

### 4.1 `ws` 的定义不能改

当前语义是：

- `ws = 1`：只看当前一步，等价 greedy。
- `ws > 1`：在固定窗口内做 DP，但 **只提交第一个决策**。
- 当 `remain <= ws`：对剩余点 **一次性解完整 DP**，并提交整段尾部路径。

这不是普通的“每一步都做长度固定为 ws 的滑窗并只提交一步”实现。尾部逻辑是特意优化过的。

### 4.2 每个点当前要求“严格找到固定数量的 IK 解”

在 `run_benchmark.py` 里，当前逻辑是严格的：

- 若某点 `best_found <= 0`，直接报错。
- 若某点 `best_found < requested`，也直接报错。
- 也就是说，当前数据集生成阶段不接受 shortfall。

虽然 `robust_sampler.py` 和 `reporting.py` 兼容 `accepted_with_shortfall` / `shortfall` 字段，但 **当前 benchmark 主流程默认不允许 shortfall 被接受**。

### 4.3 `summary.json` 是外部契约

下游批处理脚本会解析：

- `window.results_by_ws`
- `window.total_time_s_by_ws`
- `window.selection_timing_by_ws`
- `window.results[*].segment_times_s`

因此：

- 尽量 **追加字段**，不要重命名已有字段。
- 不要随意改变 `summary.json` 的层级。
- 若必须改格式，请同步更新批处理脚本，并在 PR / 变更说明中明确写出。

### 4.4 `p{i}.json` 的 payload 结构要稳定

采样模块返回的是：

- `meta`
- `solutions`
- `solutions_obj`

其中：

- `solutions_obj` 是运行时对象，方便搜索算法使用。
- `save_ik_json(...)` 落盘时只会写 `meta` 和 `solutions`。

如果改 `sample_ik_solutions(...)` 或 `sample_ik_solutions_multi_pass(...)`，不要破坏这个约定。

### 4.5 GPU 加速只针对 trapezoid DP

`search.py` 中的 GPU 加速条件是：

- `torch` 可导入
- CUDA 可用
- `time_model.info.effective == "trapezoid"`

### 4.6 默认起始状态是“按 seed 确定的随机关节状态”

如果：

- 用户没有传 `--p0`
- `--named-start=random`

那么起始状态 `p0` 不是固定的 `ready`，而是 **由 seed 决定的随机状态**。

不要误改成固定起点，否则会改变实验分布。

---

## 5. 关键模块说明

### 5.1 `run_benchmark.py`

位置：

```text
src/panda_ik_window/panda_ik_window/scripts/run_benchmark.py
```

作用：主程序。它把“采点、采 IK、评估策略、写结果”串起来。

你在这里最可能做的修改：

- 新增 CLI 参数
- 调整 summary 字段
- 更改实验流程
- 增加额外统计项

注意：

- 这里有参数解析、执行逻辑、summary 组装三部分。
- 如果新增参数，通常需要 **同时修改 launch 文件**。

### 5.2 `planning/search.py`

作用：路径选择。

主要逻辑：

- `greedy_path`: 逐点选当前最短时间。
- `window_path_receding_horizon`: 核心滑窗策略。
- `global_optimal_path`: 整条路径做 DP。

修改建议：

- 如果你只是在试新策略，优先新增函数，不要直接破坏原有 `window_path_receding_horizon(...)`。
- 若改 DP 后端，要保持 NumPy 与 Torch 两条路径结果一致。

### 5.3 `planning/time_metric.py`

作用：段时间估计。

当前支持：

- `trapezoid`

其中：

- `trapezoid` 是解析型、快、可向量化。

若你修改时间模型：

- 保持 `SegmentTimeModel.segment_time_s(...)` 与 `segment_time_matrix_s(...)` 语义一致。
- 若调整 summary 结构，记得同步更新 `meta.time_model`。

### 5.4 `ik/sampler_space.py`

作用：单点 IK 采样。

当前思想：

- 对末端姿态做 yaw 空间扰动
- 用 nullspace exploration 扩展解
- 去重
- 输出固定格式 payload

这是“基础采样器”。

### 5.5 `ik/robust_sampler.py`

作用：多轮采样与合并。

当前做法：

- 以不同 seed 做多轮 pass
- 合并唯一解
- 返回合并后的 `meta / solutions / solutions_obj`

若你提高采样成功率，优先改这里，而不是把 `run_benchmark.py` 写得更复杂。

### 5.6 `utils/targets.py`

作用：目标点采样与路径模式约束。

当前 `path_pattern`：

- `trend`
- `switching`
- `random`

如果要改路径模式判据：

- 同时更新这里的实现
- 以及根目录的 `path_pattern_definition.md`

不要只改文档，不改代码；也不要只改代码，不改文档。

### 5.7 `utils/robot.py`

作用：MoveItPy 机器人上下文封装。

包括：

- 加载 robot model
- 推断 tip link
- 读取关节限位
- 生成 named / custom robot state

这里依赖 MoveItPy 绑定。没有正确的 ROS2 / MoveIt 环境时，导入和运行都可能失败。

### 5.8 `utils/reporting.py`

作用：写 `targets.json`、`robot_info.txt`、`summary.json` 等。

如果只是想“给 summary 多加一个字段”，优先看这里和 `run_benchmark.py` 的 summary 组装逻辑。

---

## 6. 批处理脚本说明

### 6.1 `script/batch_ik_window.py`

这是更完整的批处理入口。作用是：

- 按 `window_size` 和 seed 循环运行 benchmark
- 等待 `summary.json`
- 从 summary 中抽取数据
- 汇总为 CSV

### 6.2 `script/batch_run_window.py`

与上面的脚本功能相近，但它看起来更像一个较早版本。当前文件里有一些参数被注释掉了，例如：

- `time_model`
- `device`
- `dp_block_size`
- `extra_launch_args`

因此，若要做批处理增强，优先基于 `batch_ik_window.py`，不要盲目同时维护两个脚本的复杂逻辑，除非你明确要统一它们。

---

## 7. 运行方式

### 7.1 构建

```bash
cd ~/ws_moveit2
colcon build --packages-select panda_ik_window
source install/setup.bash
```

### 7.2 单次运行

```bash
ros2 launch panda_ik_window ik_benchmark.launch.py \
  num_points:=8 \
  seed:=7 \
  window_size:=3
```

### 7.3 指定路径模式

```bash
ros2 launch panda_ik_window ik_benchmark.launch.py \
  num_points:=8 \
  seed:=7 \
  window_size:=3 \
  path_pattern:=trend
```

### 7.4 一次评估多个 `ws`

```bash
ros2 launch panda_ik_window ik_benchmark.launch.py \
  num_points:=8 \
  seed:=7 \
  window_size:="1,3,8"
```

### 7.5 Sweep 全部窗口

```bash
ros2 launch panda_ik_window ik_benchmark.launch.py \
  num_points:=8 \
  seed:=7 \
  window_size:=all
```

### 7.6 批处理

```bash
python3 script/batch_ik_window.py --num-points 8 --seeds 7,8,9
```

---

## 8. AI 修改代码时的推荐策略

### 8.1 想新增实验统计

优先改：

- `run_benchmark.py` 中 summary 组装部分
- `utils/reporting.py`
- 必要时同步修改 `batch_ik_window.py`

建议：

- 追加字段，不重命名旧字段
- 把统计放进 `window.results_by_ws[ws]` 或 `meta` 下
- 保留旧字段以兼容已有脚本

### 8.2 想新增一种路径策略

优先改：

- `planning/search.py`

建议：

- 新增一个函数，例如 `beam_path(...)`、`window_path_soft_commit(...)`
- 不要直接把 `window_path_receding_horizon(...)` 改成另一种完全不同的策略
- 若需要 CLI 暴露新策略，再去改 `run_benchmark.py` 和 launch

### 8.3 想提升速度

先判断瓶颈在哪：

1. **IK 采样慢**：看 `sampler_space.py` / `robust_sampler.py`
2. **DP 慢**：看 `planning/search.py`

当前最容易带来收益的方向通常是：

- 减少不必要的 Python 循环
- 提前缓存 joint arrays
- 提高 GPU 路径的复用程度
- 减少 summary 中冗余数据拷贝

### 8.4 想改 CLI 参数

通常需要同时修改三个位置：

1. `run_benchmark.py` 参数解析
2. `launch/ik_benchmark.launch.py` 的 `DeclareLaunchArgument`
3. 文档（至少 `README.md`）

---

## 9. 不能忽视的仓库约束与坑

### 9.1 launch 默认值与 Python 默认值不完全一致

这里有一个实际存在的不一致：

- `run_benchmark.py` 里 `--data-root` 默认是 `data_three`
- `launch/ik_benchmark.launch.py` 里 `data_root` 默认是 `data_window`
- README / 批处理脚本更接近 `data_window`

因此：

- 从 launch 跑时，默认输出在 `data_window`
- 直接运行 Python 脚本时，默认可能在 `data_three`

如果要修这个问题，请统一三处并同步文档。

### 9.2 这个项目依赖 ROS2 launch 注入参数

直接执行：

```bash
python src/panda_ik_window/panda_ik_window/scripts/run_benchmark.py
```

通常并不可靠，因为 MoveItPy 需要：

- `robot_description`
- `robot_description_semantic`
- kinematics / planning 参数

这些通常由 launch 注入。

### 9.3 `selection_timing` 不是全段都有

当前 `window.selection_timing` 只记录 **完整窗口决策**。

当进入尾部 `remain <= ws` 时，虽然也会做一次 DP，但这部分不会逐段写入 `selection_timing`。这是当前设计，不要误判为漏统计。

### 9.4 `global_optimal_path(...)` 已存在，但主流程不一定直接调用

主流程主要是通过 `window_size` 统一表达：

- `ws = 1` 对应 greedy
- `ws = n` 对应 global

因此如果看到已有 `global_optimal_path(...)` 没被主流程显式调用，不代表它无用。

### 9.5 不要把文档里的理论阈值和代码里的阈值混为一谈

`path_pattern_definition.md` 更偏论文/说明文本。

`utils/targets.py` 里的阈值是实际执行阈值。两者可能不是逐字完全一致。修改模式分类时，要明确你是在：

- 改理论说明
- 还是改代码判据
- 还是两者一起改

---

## 10. 建议保留的编码风格

当前项目的 Python 风格有这些特点：

- 大量使用类型标注
- 小而清晰的 dataclass
- JSON 输出使用 UTF-8、可读缩进
- 逻辑上偏“显式而非魔法”
- 对异常情况更倾向于直接报错，而不是静默吞掉

AI 修改时请尽量保持：

- 函数签名清晰
- 新字段命名稳定
- 错误信息能指导实验者调参
- 不要引入重量级依赖，除非明确必要

---

## 11. 如果你是 AI 助手，优先这样理解这个仓库

### 一句话理解

这是一个 **先为每个笛卡尔点生成一组 IK 候选，再用时间代价模型在候选图上选路径** 的实验仓库。

### 更具体一点

- 节点是每个点的 IK 候选解。
- 边代价是相邻两个关节状态之间的运动时间。
- `window` / `greedy` / `global` 的差别，不在 IK 本身，而在“如何在层状图上选路径”。

所以：

- IK 采样模块负责“候选集质量”
- 时间模型负责“边代价是否合理”
- 搜索模块负责“路径选择是否合理”

多数修改都应该先想清楚自己是在改哪一层。

---

## 12. 修改后最少要做的检查

### 12.1 纯 Python 级检查

```bash
python -m compileall src/panda_ik_window/panda_ik_window script
```

这一步不能证明 ROS2/MoveIt 运行没问题，但至少能排除明显语法错误。

### 12.2 构建检查

```bash
colcon build --packages-select panda_ik_window
```

### 12.3 最小运行检查

```bash
source install/setup.bash
ros2 launch panda_ik_window ik_benchmark.launch.py num_points:=3 seed:=7 window_size:=1
```

至少确认：

- 能正常采样 `p1..pN`
- 会生成 `targets.json`
- 会生成 `p{i}.json`
- 会生成 `summary.json`

### 12.4 若动了 summary 格式

必须额外检查：

```bash
python3 script/batch_ik_window.py --num-points 3 --seeds 7
```

看批处理脚本还能不能解析最新输出。

---

## 13. 给 AI 的实际操作建议

### 适合直接做的事

- 增加统计项
- 增加日志信息
- 抽取重复代码
- 为现有策略增加新参数
- 给 summary 增加兼容字段
- 修复默认值不一致

### 需要谨慎的事

- 改 `summary.json` 结构
- 改 `window` 语义
- 改 IK 去重规则
- 改路径模式判别阈值

### 不建议默认去做的事

- 删除兼容字段
- 把严格报错改成静默降级
- 仅凭“代码看起来重复”就合并 batch 脚本而不验证

---

## 14. 当前代码中值得优先关注的几个问题

这些不是必须立刻修，但它们是比较真实的维护点：

1. `data_root` 默认值在不同入口不一致。
2. 顶层有两个 batch 脚本，功能部分重叠。
3. 仓库里带有 `__MACOSX`、`.DS_Store`、`__pycache__` 等非源码内容。
4. 文档中的 path pattern 定义与代码阈值可能并非完全同步。

---

## 15. 结论

如果你要修改这个仓库，请优先守住三件事：

1. **不要改坏实验语义**：特别是 `ws`、严格 `num_solutions`、尾部一次性 DP。
2. **不要改坏输出契约**：特别是 `summary.json` 和 `p{i}.json`。
3. **不要忽略运行上下文**：这是 ROS2 + MoveIt2 项目，不是纯 Python 脚本仓库。

在这个前提下，增量式改动、追加字段、局部重构，通常都是安全的。
