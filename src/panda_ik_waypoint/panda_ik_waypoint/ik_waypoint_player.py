#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""panda_ik_waypoint.ik_waypoint_player

Visualize one Panda IK waypoint solution sequence in RViz2.

Features:
- Reads benchmark output directory (summary.json, targets.json, p*.json).
- Chooses one window-size result (ws) and replays its final_path segments.
- Publishes /waypoint/joint_states for robot_state_publisher.
- Publishes /ik_waypoint/markers (points, labels, line strip).
- Publishes static TF world -> waypoint_/panda_link0 (configurable).
"""

from __future__ import annotations

import bisect
import glob
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import rclpy
from geometry_msgs.msg import Point, TransformStamped
from rclpy.node import Node
from sensor_msgs.msg import JointState
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray


def _read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_result_dir(path: str) -> str:
    """Accept run dir or parent dir; return a directory containing summary.json."""
    path = os.path.abspath(os.path.expanduser(path))
    if os.path.isfile(os.path.join(path, "summary.json")):
        return path

    candidates = glob.glob(os.path.join(path, "**", "summary.json"), recursive=True)
    if not candidates:
        raise FileNotFoundError(
            f"Could not find summary.json in '{path}' or its subdirectories."
        )

    def _key(summary_path: str) -> Tuple[int, str, float]:
        run_dir = os.path.basename(os.path.dirname(summary_path))
        is_timestamp = 1 if run_dir.isdigit() else 0
        return (is_timestamp, run_dir, os.path.getmtime(summary_path))

    best = max(candidates, key=_key)
    return os.path.dirname(best)


def _find_joint_names(result_dir: str) -> List[str]:
    """Try reading joint_names from p*.json; fallback to panda_joint1..7."""
    for p in sorted(glob.glob(os.path.join(result_dir, "p*.json"))):
        base = os.path.basename(p)
        if base == "p0.json":
            continue
        try:
            data = _read_json(p)
            sols = data.get("solutions", [])
            if sols and isinstance(sols[0], dict):
                names = sols[0].get("joint_names", [])
                if isinstance(names, list) and names:
                    return [str(n) for n in names]
        except Exception:
            continue
    return [f"panda_joint{i}" for i in range(1, 8)]


def _load_targets(result_dir: str, summary: dict) -> List[dict]:
    """Load target points, preferring targets.json."""
    targets_path = os.path.join(result_dir, "targets.json")
    if os.path.isfile(targets_path):
        try:
            data = _read_json(targets_path)
            if isinstance(data, dict):
                ts = data.get("targets", [])
                if isinstance(ts, list):
                    return ts
            if isinstance(data, list):
                return data
        except Exception:
            pass

    ts = summary.get("targets", [])
    if isinstance(ts, list):
        return ts
    return []


def _pick_ws_data(summary: dict, ws_param: int) -> Tuple[int, dict]:
    """Pick one ws result from summary['window']['results_by_ws'].

    ws_param:
      -1 => choose max available ws
      >=0 => choose exact ws
    """
    window = summary.get("window", {})
    results_by_ws = window.get("results_by_ws", {})

    if isinstance(results_by_ws, dict) and results_by_ws:
        ws_keys: List[int] = []
        for k in results_by_ws.keys():
            try:
                ws_keys.append(int(k))
            except Exception:
                continue
        if not ws_keys:
            raise RuntimeError("results_by_ws exists but has no valid numeric keys.")

        if ws_param < 0:
            ws = max(ws_keys)
        else:
            ws = int(ws_param)
            if ws not in ws_keys:
                raise KeyError(
                    f"Requested window_size={ws} not found. Available: {sorted(ws_keys)}"
                )

        return ws, results_by_ws[str(ws)]

    # Legacy fallback: window.results = [ ... ]
    results = window.get("results", [])
    if isinstance(results, list) and results:
        if ws_param < 0:
            picked = max(results, key=lambda r: int(r.get("window_size", 0)))
            ws = int(picked.get("window_size", 0))
            return ws, picked

        for r in results:
            try:
                if int(r.get("window_size", -1)) == ws_param:
                    return ws_param, r
            except Exception:
                continue

        avail = sorted(
            int(r.get("window_size", 0))
            for r in results
            if isinstance(r, dict) and "window_size" in r
        )
        raise KeyError(
            f"Requested window_size={ws_param} not found in window.results. Available: {avail}"
        )

    raise KeyError("Could not find window results in summary.json")


def _quintic_time_scaling(s: float) -> float:
    s = max(0.0, min(1.0, s))
    return 10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5


@dataclass
class Segment:
    q0: List[float]
    q1: List[float]
    duration: float
    to_name: str


@dataclass
class PiecewiseTrajectory:
    segments: List[Segment]
    ends_s: List[float]
    total_s: float

    @staticmethod
    def build(segments: Sequence[Segment]) -> "PiecewiseTrajectory":
        ends_s: List[float] = []
        t = 0.0
        for seg in segments:
            t += max(1e-6, float(seg.duration))
            ends_s.append(t)
        return PiecewiseTrajectory(segments=list(segments), ends_s=ends_s, total_s=t)

    def evaluate(self, t_s: float) -> List[float]:
        if not self.segments:
            return []

        if self.total_s <= 1e-9:
            return list(self.segments[-1].q1)

        t = min(max(0.0, t_s), self.total_s)
        idx = bisect.bisect_left(self.ends_s, t)
        if idx >= len(self.segments):
            return list(self.segments[-1].q1)

        seg = self.segments[idx]
        seg_end = self.ends_s[idx]
        seg_start = 0.0 if idx == 0 else self.ends_s[idx - 1]
        seg_dur = max(1e-6, seg_end - seg_start)
        u = _quintic_time_scaling((t - seg_start) / seg_dur)
        return [a + (b - a) * u for a, b in zip(seg.q0, seg.q1)]


def _sort_point_name_key(name: str) -> Tuple[int, int, str]:
    if isinstance(name, str) and name.startswith("p"):
        try:
            return (0, int(name[1:]), name)
        except Exception:
            pass
    return (1, 0, str(name))


class IKWaypointPlayer(Node):
    def __init__(self) -> None:
        super().__init__("ik_waypoint_player")

        # Inputs
        self.declare_parameter("result_dir", "data_window")
        self.declare_parameter("window_size", -1)  # -1 => max ws

        # Frame / robot layout
        self.declare_parameter("fixed_frame", "world")
        self.declare_parameter("root_link", "panda_link0")
        self.declare_parameter("robot_prefix", "waypoint_/")
        self.declare_parameter("base_offset_xyz", [0.0, 0.0, 0.0])

        # Playback
        self.declare_parameter("publish_rate_hz", 50.0)
        self.declare_parameter("loop", True)
        self.declare_parameter("speed_scale", 1.0)
        self.declare_parameter("hold_time_s", 0.0)

        # Gripper (optional)
        self.declare_parameter("publish_gripper", True)
        self.declare_parameter("gripper_joint_names", ["panda_finger_joint1", "panda_finger_joint2"])
        self.declare_parameter("gripper_joint_positions", [0.04, 0.04])

        # Marker style
        self.declare_parameter("marker_topic", "/ik_waypoint/markers")
        self.declare_parameter("point_scale", 0.045)
        self.declare_parameter("line_width", 0.012)
        self.declare_parameter("label_scale_z", 0.05)
        self.declare_parameter("label_dz", 0.07)
        self.declare_parameter("point_rgba", [0.20, 0.70, 1.00, 1.00])
        self.declare_parameter("line_rgba", [1.00, 0.45, 0.10, 0.90])
        self.declare_parameter("label_rgba", [1.00, 1.00, 1.00, 1.00])

        # Resolve and load inputs
        result_dir_in = str(self.get_parameter("result_dir").value)
        self.result_dir = _resolve_result_dir(result_dir_in)

        summary_path = os.path.join(self.result_dir, "summary.json")
        self.summary = _read_json(summary_path)

        ws_param = int(self.get_parameter("window_size").value)
        self.ws, ws_data = _pick_ws_data(self.summary, ws_param)

        start = self.summary.get("start", {})
        start_q = [float(x) for x in start.get("joint_positions", [])]
        if not start_q:
            raise RuntimeError("summary.json missing start.joint_positions")

        self.fixed_frame = str(self.get_parameter("fixed_frame").value)
        self.root_link = str(self.get_parameter("root_link").value)
        self.robot_prefix = str(self.get_parameter("robot_prefix").value)
        self.base_offset = [float(v) for v in list(self.get_parameter("base_offset_xyz").value)]

        rate_hz = max(1.0, float(self.get_parameter("publish_rate_hz").value))
        self.dt = 1.0 / rate_hz
        self.loop = bool(self.get_parameter("loop").value)
        self.speed_scale = max(1e-6, float(self.get_parameter("speed_scale").value))
        self.hold_time_s = max(0.0, float(self.get_parameter("hold_time_s").value))

        self.publish_gripper = bool(self.get_parameter("publish_gripper").value)
        self.gripper_joint_names = [str(x) for x in list(self.get_parameter("gripper_joint_names").value)]
        self.gripper_joint_positions = [
            float(x) for x in list(self.get_parameter("gripper_joint_positions").value)
        ]
        if len(self.gripper_joint_positions) != len(self.gripper_joint_names):
            self.gripper_joint_positions = [0.04] * len(self.gripper_joint_names)

        self.point_scale = float(self.get_parameter("point_scale").value)
        self.line_width = float(self.get_parameter("line_width").value)
        self.label_scale_z = float(self.get_parameter("label_scale_z").value)
        self.label_dz = float(self.get_parameter("label_dz").value)

        self.point_rgba = self._rgba_param("point_rgba", (0.2, 0.7, 1.0, 1.0))
        self.line_rgba = self._rgba_param("line_rgba", (1.0, 0.45, 0.1, 0.9))
        self.label_rgba = self._rgba_param("label_rgba", (1.0, 1.0, 1.0, 1.0))

        self.joint_names = _find_joint_names(self.result_dir)
        if len(self.joint_names) != len(start_q):
            self.get_logger().warn(
                "Joint-name count does not match start joint vector length; using panda_joint1..N."
            )
            self.joint_names = [f"panda_joint{i}" for i in range(1, len(start_q) + 1)]

        seg_dicts = (
            ws_data.get("final_path", {}).get("segments", [])
            if isinstance(ws_data, dict)
            else []
        )
        if not isinstance(seg_dicts, list) or not seg_dicts:
            raise RuntimeError("No final_path.segments found for selected ws")

        # Build playback segments from chosen path.
        q_prev = list(start_q)
        segments: List[Segment] = []
        self.path_order: List[str] = []
        for seg in seg_dicts:
            q_next = seg.get("joint_positions", None)
            if not isinstance(q_next, list) or len(q_next) != len(q_prev):
                continue
            to_name = str(seg.get("to", ""))
            dur = float(seg.get("time_s", 0.0))
            if dur <= 0.0:
                dur = 1.0
            q1 = [float(x) for x in q_next]
            segments.append(Segment(q0=list(q_prev), q1=q1, duration=dur, to_name=to_name))
            q_prev = q1
            self.path_order.append(to_name)

            if self.hold_time_s > 0.0:
                segments.append(
                    Segment(
                        q0=list(q_prev),
                        q1=list(q_prev),
                        duration=self.hold_time_s,
                        to_name=to_name,
                    )
                )

        if not segments:
            raise RuntimeError("Unable to build playback segments from summary")

        self.traj = PiecewiseTrajectory.build(segments)

        # Target points for markers
        targets_list = _load_targets(self.result_dir, self.summary)
        self.targets_by_name: Dict[str, dict] = {}
        for t in targets_list:
            if not isinstance(t, dict):
                continue
            if all(k in t for k in ("name", "x", "y", "z")):
                self.targets_by_name[str(t["name"])] = t

        if not self.targets_by_name:
            self.get_logger().warn("No target points found in targets.json/summary.json")

        # Publishers
        self.js_pub = self.create_publisher(JointState, "/waypoint/joint_states", 10)
        marker_topic = str(self.get_parameter("marker_topic").value)
        self.marker_pub = self.create_publisher(MarkerArray, marker_topic, 10)

        # Static TF for base placement
        self.static_tf = StaticTransformBroadcaster(self)
        self._publish_static_tf()

        # Runtime state
        self._elapsed_s = 0.0
        self._last_tick = self.get_clock().now()

        # Timers
        self._play_timer = self.create_timer(self.dt, self._on_timer)
        self._marker_timer = self.create_timer(1.0, self._on_marker_timer)

        # Initial publish
        self._publish_joint_state(self.traj.evaluate(0.0))
        self._marker_msg = self._make_marker_array()
        self.marker_pub.publish(self._marker_msg)

        total_summary = ws_data.get("final_path", {}).get("total_time_s", self.traj.total_s)
        self.get_logger().info(
            f"Loaded run: {self.result_dir} | ws={self.ws} | segments={len(seg_dicts)} | "
            f"path_time={float(total_summary):.4f}s | playback_time={self.traj.total_s:.4f}s"
        )

    def _rgba_param(self, name: str, default: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
        v = self.get_parameter(name).value
        try:
            if isinstance(v, (list, tuple)) and len(v) == 4:
                r, g, b, a = [float(x) for x in v]
                return (r, g, b, a)
        except Exception:
            pass
        return default

    def _publish_static_tf(self) -> None:
        ts = TransformStamped()
        ts.header.stamp = self.get_clock().now().to_msg()
        ts.header.frame_id = self.fixed_frame
        ts.child_frame_id = f"{self.robot_prefix}{self.root_link}"
        ts.transform.translation.x = float(self.base_offset[0])
        ts.transform.translation.y = float(self.base_offset[1])
        ts.transform.translation.z = float(self.base_offset[2])
        ts.transform.rotation.w = 1.0
        self.static_tf.sendTransform([ts])

    def _publish_joint_state(self, q: Sequence[float]) -> None:
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()

        if self.publish_gripper:
            msg.name = list(self.joint_names) + list(self.gripper_joint_names)
            msg.position = [float(x) for x in q] + list(self.gripper_joint_positions)
        else:
            msg.name = list(self.joint_names)
            msg.position = [float(x) for x in q]

        self.js_pub.publish(msg)

    def _ordered_target_names(self) -> List[str]:
        names_in_path = [n for n in self.path_order if n in self.targets_by_name]
        if names_in_path:
            return names_in_path
        return sorted(self.targets_by_name.keys(), key=_sort_point_name_key)

    def _make_marker_array(self) -> MarkerArray:
        ma = MarkerArray()
        now = self.get_clock().now().to_msg()
        frame = f"{self.robot_prefix}{self.root_link}"

        # Clear stale markers from previous runs/layouts.
        clear = Marker()
        clear.header.stamp = now
        clear.header.frame_id = frame
        clear.action = Marker.DELETEALL
        ma.markers.append(clear)

        ordered_names = self._ordered_target_names()
        line_points: List[Tuple[float, float, float]] = []

        for idx, pname in enumerate(ordered_names, start=1):
            t = self.targets_by_name.get(pname)
            if not t:
                continue

            xyz = (float(t["x"]), float(t["y"]), float(t["z"]))
            line_points.append(xyz)

            sphere = Marker()
            sphere.header.stamp = now
            sphere.header.frame_id = frame
            sphere.ns = "ik_waypoint_points"
            sphere.id = 1000 + idx
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD
            sphere.pose.position.x = xyz[0]
            sphere.pose.position.y = xyz[1]
            sphere.pose.position.z = xyz[2]
            sphere.pose.orientation.w = 1.0
            sphere.scale.x = self.point_scale
            sphere.scale.y = self.point_scale
            sphere.scale.z = self.point_scale
            sphere.color.r, sphere.color.g, sphere.color.b, sphere.color.a = self.point_rgba
            ma.markers.append(sphere)

            label = Marker()
            label.header.stamp = now
            label.header.frame_id = frame
            label.ns = "ik_waypoint_labels"
            label.id = 2000 + idx
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.position.x = xyz[0]
            label.pose.position.y = xyz[1]
            label.pose.position.z = xyz[2] + self.label_dz
            label.pose.orientation.w = 1.0
            label.scale.z = self.label_scale_z
            label.color.r, label.color.g, label.color.b, label.color.a = self.label_rgba
            label.text = f"{idx}:{pname}"
            ma.markers.append(label)

        if len(line_points) >= 2:
            line = Marker()
            line.header.stamp = now
            line.header.frame_id = frame
            line.ns = "ik_waypoint_line"
            line.id = 3000
            line.type = Marker.LINE_STRIP
            line.action = Marker.ADD
            line.pose.orientation.w = 1.0
            line.scale.x = self.line_width
            line.color.r, line.color.g, line.color.b, line.color.a = self.line_rgba

            for xyz in line_points:
                p = Point()
                p.x = xyz[0]
                p.y = xyz[1]
                p.z = xyz[2]
                line.points.append(p)
            ma.markers.append(line)

        title = Marker()
        title.header.stamp = now
        title.header.frame_id = self.fixed_frame
        title.ns = "ik_waypoint_title"
        title.id = 4000
        title.type = Marker.TEXT_VIEW_FACING
        title.action = Marker.ADD
        title.pose.position.x = float(self.base_offset[0])
        title.pose.position.y = float(self.base_offset[1])
        title.pose.position.z = float(self.base_offset[2]) + 0.9
        title.pose.orientation.w = 1.0
        title.scale.z = 0.08
        title.color.r = 1.0
        title.color.g = 1.0
        title.color.b = 1.0
        title.color.a = 1.0
        title.text = f"WAYPOINT (ws={self.ws})"
        ma.markers.append(title)

        return ma

    def _on_marker_timer(self) -> None:
        self._marker_msg = self._make_marker_array()
        self.marker_pub.publish(self._marker_msg)

    def _on_timer(self) -> None:
        now = self.get_clock().now()
        dt_s = (now - self._last_tick).nanoseconds * 1e-9
        self._last_tick = now

        if dt_s < 0.0 or not math.isfinite(dt_s):
            dt_s = self.dt

        self._elapsed_s += dt_s * self.speed_scale

        if self.traj.total_s <= 1e-9:
            q = self.traj.evaluate(0.0)
            self._publish_joint_state(q)
            return

        if self.loop:
            t = self._elapsed_s % self.traj.total_s
        else:
            t = min(self._elapsed_s, self.traj.total_s)

        q = self.traj.evaluate(t)
        self._publish_joint_state(q)


def main(args: Optional[Sequence[str]] = None) -> None:
    rclpy.init(args=args)
    node = IKWaypointPlayer()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
