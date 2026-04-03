# panda_ik_solutions_viz

A small ROS 2 (Jazzy) package to visualize **many Panda IK solutions at once** in RViz.

The trick is to publish a single `moveit_msgs/DisplayTrajectory` message whose trajectory points are the IK solutions.
In the MoveIt RViz plugin, enable the **Planned Path -> Show Trail** option to see many semi-transparent "ghost" robots at once.

## Dependencies

- Ubuntu 24.04
- ROS 2 Jazzy
- MoveIt 2
- `moveit_resources_panda_moveit_config`

## Usage

1) Put your `p1.json` into this package:

```bash
cp /path/to/your/p1.json <your_ws>/src/panda_ik_solutions_viz/config/p1.json
```

2) Build:

```bash
cd <your_ws>
colcon build --packages-select panda_ik_solutions_viz
source install/setup.bash
```

3) Launch:

```bash
ros2 launch panda_ik_solutions_viz visualize_ik_solutions.launch.py
```

Or specify a custom JSON path:

```bash
ros2 launch panda_ik_solutions_viz visualize_ik_solutions.launch.py ik_json:=/absolute/path/to/p1.json
```

4) In RViz (MotionPlanning display):
- Expand **Planned Path**
- Check **Show Trail**
- Set **Trail Step Size** to `1` to show every solution
- Lower **Robot Alpha** (e.g. `0.15`) to better see overlapping solutions

## JSON formats supported

The node tries to accept several common formats:

- `{ "solutions": [ { "joint_state": { "name": [...], "position": [...] } }, ... ] }`
- `[ { "joint_names": [...], "joint_positions": [...] }, ... ]`
- `[ [q1, q2, q3, q4, q5, q6, q7], ... ]` (assumes Panda joint order)
- `[ { "panda_joint1": 0.1, ... }, ... ]`

If your values are in degrees, set `input_in_degrees:=true`.
