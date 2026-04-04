# panda_try_rviz

Interactive Panda RViz package with:

- 7 sliders to control `panda_joint1..7`
- one button to save current EE position
- saved EE positions shown as yellow points
- self-collision status + contact markers shown in RViz

## Run

```bash
ros2 launch panda_try_rviz panda_try_rviz.launch.py
```

## UI/Topics

- RViz Panel: `Panda Try Control`
- Joint command input: `/panda_try/command_joint_states`
- Joint state output: `/panda_try/joint_states`
- Save EE button topic: `/panda_try/save_ee_point`
- Self-collision bool: `/panda_try/self_collision`
- Markers: `/panda_try/markers`

## Notes

- Collision check uses MoveIt service: `/check_state_validity`.
- Launch starts `move_group` automatically.
- Collision contact points are rendered as red spheres, so collision locations are visible on the arm.
