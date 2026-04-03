#!/usr/bin/env python3

import json
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

from builtin_interfaces.msg import Duration
from moveit_msgs.msg import DisplayTrajectory, RobotState, RobotTrajectory
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


DEFAULT_ARM_JOINTS = [f"panda_joint{i}" for i in range(1, 8)]
DEFAULT_GRIPPER_JOINTS = ["panda_finger_joint1", "panda_finger_joint2"]


def _find_solutions_list(data: Any) -> List[Any]:
    """Try hard to extract a list of solutions from arbitrary JSON structures."""
    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        # Common keys
        for key in [
            "solutions",
            "ik_solutions",
            "ik",
            "results",
            "result",
            "data",
            "samples",
        ]:
            if key in data and isinstance(data[key], list):
                return data[key]

        # Nested common patterns
        # e.g. {"response": {"solutions": [...]}}
        for key in ["response", "responses", "payload", "msg", "message"]:
            if key in data:
                inner = data[key]
                sols = _find_solutions_list(inner)
                if sols:
                    return sols

    raise ValueError(
        "Unable to find a list of IK solutions in JSON. "
        "Expected a list, or a dict containing one under keys like 'solutions'."
    )


def _as_joint_mapping(sol: Any, default_joint_order: Sequence[str]) -> Optional[Dict[str, float]]:
    """Convert one solution record into a mapping joint_name -> position."""
    # 1) ROS JointState-like
    if isinstance(sol, dict):
        # {"joint_state": {"name": [...], "position": [...]}}
        if "joint_state" in sol and isinstance(sol["joint_state"], dict):
            js = sol["joint_state"]
            names = js.get("name") or js.get("names")
            pos = js.get("position") or js.get("positions")
            if isinstance(names, list) and isinstance(pos, list) and len(names) == len(pos):
                return {str(n): float(p) for n, p in zip(names, pos)}

        # {"name": [...], "position": [...]} or {"names": ..., "positions": ...}
        names = sol.get("name") or sol.get("names") or sol.get("joint_names")
        pos = sol.get("position") or sol.get("positions") or sol.get("joint_positions")
        if isinstance(names, list) and isinstance(pos, list) and len(names) == len(pos):
            return {str(n): float(p) for n, p in zip(names, pos)}

        # {"joints": {"panda_joint1": 0.1, ...}}
        if "joints" in sol and isinstance(sol["joints"], dict):
            return {str(k): float(v) for k, v in sol["joints"].items()}

        # Sometimes it is already {joint_name: value, ...}
        if sol and all(isinstance(v, (int, float)) for v in sol.values()):
            return {str(k): float(v) for k, v in sol.items()}

        # Sometimes it is {"solution": {...}}
        if "solution" in sol:
            return _as_joint_mapping(sol["solution"], default_joint_order)

    # 2) Raw array of positions in a known order
    if isinstance(sol, (list, tuple)):
        if len(sol) >= len(default_joint_order) and all(isinstance(v, (int, float)) for v in sol):
            return {jn: float(sol[i]) for i, jn in enumerate(default_joint_order)}

    return None


def _duration_from_seconds(t: float) -> Duration:
    sec = int(math.floor(t))
    nsec = int((t - sec) * 1e9)
    msg = Duration()
    msg.sec = sec
    msg.nanosec = nsec
    return msg


class IKSolutionDisplayPublisher(Node):
    def __init__(self) -> None:
        super().__init__("ik_solution_display_publisher")

        pkg_share = os.environ.get("AMENT_PREFIX_PATH", "")
        _ = pkg_share  # silence lint; just to show it's intentionally unused

        self.declare_parameter("ik_json_path", "")
        self.declare_parameter("display_topic", "/display_planned_path")
        self.declare_parameter("publish_period_s", 1.0)
        self.declare_parameter("input_in_degrees", False)
        self.declare_parameter("arm_joint_names", DEFAULT_ARM_JOINTS)
        self.declare_parameter("include_gripper", True)
        self.declare_parameter("gripper_opening", 0.04)
        self.declare_parameter("time_step_s", 0.1)

        self._ik_json_path: str = self.get_parameter("ik_json_path").get_parameter_value().string_value
        self._topic: str = self.get_parameter("display_topic").get_parameter_value().string_value
        self._publish_period: float = (
            self.get_parameter("publish_period_s").get_parameter_value().double_value
        )
        self._input_in_degrees: bool = (
            self.get_parameter("input_in_degrees").get_parameter_value().bool_value
        )
        self._arm_joint_names: List[str] = [
            str(s) for s in self.get_parameter("arm_joint_names").value
        ]
        self._include_gripper: bool = (
            self.get_parameter("include_gripper").get_parameter_value().bool_value
        )
        self._gripper_opening: float = (
            self.get_parameter("gripper_opening").get_parameter_value().double_value
        )
        self._time_step_s: float = self.get_parameter("time_step_s").get_parameter_value().double_value

        if not self._arm_joint_names:
            self._arm_joint_names = list(DEFAULT_ARM_JOINTS)

        qos = QoSProfile(depth=1)
        qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        qos.reliability = ReliabilityPolicy.RELIABLE

        self._pub = self.create_publisher(DisplayTrajectory, self._topic, qos)

        if not self._ik_json_path:
            self.get_logger().error(
                "Parameter 'ik_json_path' is empty. Please provide the path to p1.json."
            )
            self._msg: Optional[DisplayTrajectory] = None
        else:
            self._msg = self._load_and_build_message(self._ik_json_path)

        self._timer = self.create_timer(max(0.05, self._publish_period), self._on_timer)

    def _load_and_build_message(self, path: str) -> Optional[DisplayTrajectory]:
        path = os.path.expanduser(path)
        if not os.path.isabs(path):
            # Allow relative to current working directory
            path = os.path.abspath(path)

        if not os.path.exists(path):
            self.get_logger().error(f"IK json file does not exist: {path}")
            return None

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            self.get_logger().error(f"Failed to read json '{path}': {e}")
            return None

        try:
            raw_solutions = _find_solutions_list(data)
        except Exception as e:
            self.get_logger().error(str(e))
            return None

        joint_names: List[str] = list(self._arm_joint_names)
        if self._include_gripper:
            joint_names.extend(DEFAULT_GRIPPER_JOINTS)

        default_order_for_raw_arrays = list(self._arm_joint_names)

        solutions: List[List[float]] = []
        dropped = 0

        for sol in raw_solutions:
            mapping = _as_joint_mapping(sol, default_order_for_raw_arrays)
            if mapping is None:
                dropped += 1
                continue

            positions: List[float] = []
            for jn in joint_names:
                if jn in mapping:
                    v = float(mapping[jn])
                else:
                    if jn in DEFAULT_GRIPPER_JOINTS:
                        v = float(self._gripper_opening)
                    else:
                        v = 0.0
                positions.append(v)

            if self._input_in_degrees:
                # Convert only arm joints
                for i, jn in enumerate(joint_names):
                    if jn.startswith("panda_joint"):
                        positions[i] = math.radians(positions[i])

            solutions.append(positions)

        if not solutions:
            self.get_logger().error(
                f"No usable IK solutions were parsed from {path}. Dropped {dropped} entries."
            )
            return None

        # Basic sanity warning for likely degrees-vs-radians mistakes
        max_abs = max(abs(v) for sol in solutions for v in sol[: len(self._arm_joint_names)])
        if not self._input_in_degrees and max_abs > (2.0 * math.pi + 0.5):
            self.get_logger().warning(
                "Some joint values are > 2π rad. If your JSON uses degrees, set input_in_degrees:=true."
            )

        self.get_logger().info(
            f"Loaded {len(solutions)} IK solutions (dropped {dropped}) from: {path}"
        )

        # Build DisplayTrajectory
        msg = DisplayTrajectory()
        msg.model_id = "moveit_resources_panda"

        start_state = RobotState()
        start_js = JointState()
        start_js.name = list(joint_names)
        start_js.position = list(solutions[0])
        start_state.joint_state = start_js
        msg.trajectory_start = start_state

        joint_traj = JointTrajectory()
        joint_traj.joint_names = list(joint_names)

        for i, positions in enumerate(solutions):
            pt = JointTrajectoryPoint()
            pt.positions = list(positions)
            pt.time_from_start = _duration_from_seconds(i * max(0.001, self._time_step_s))
            joint_traj.points.append(pt)

        robot_traj = RobotTrajectory()
        robot_traj.joint_trajectory = joint_traj
        msg.trajectory.append(robot_traj)

        return msg

    def _on_timer(self) -> None:
        if self._msg is None:
            return
        self._msg.trajectory_start.joint_state.header.stamp = self.get_clock().now().to_msg()
        # (JointTrajectory has no header in message definition in ROS2; it does, but not used here)
        self._pub.publish(self._msg)


def main() -> None:
    rclpy.init()
    node = IKSolutionDisplayPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
