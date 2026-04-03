#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""panda_ik_benchmark_viz.ik_benchmark_compare_player

Play back Panda IK benchmark results (greedy vs global) in RViz2.

It reads the benchmark output directory (containing summary.json, targets.json, p*.json),
reconstructs the greedy and global waypoint sequences, and publishes:

- /greedy/joint_states   (sensor_msgs/JointState)
- /global/joint_states   (sensor_msgs/JointState)
- /ik_benchmark/markers  (visualization_msgs/MarkerArray)  # points/labels/lines

Additionally, it publishes static TF transforms to place the two robots:
  fixed_frame -> <greedy_prefix><root_link>
  fixed_frame -> <global_prefix><root_link>

This node does NOT require MoveIt. It interpolates joint positions between waypoints using
smooth quintic time-scaling (zero vel/acc at segment endpoints), using segment durations
from summary.json (e.g., TOTG timing results).

Looping modes:
- Independent looping (default): each trajectory loops with its own period.
- Synchronized looping (sync_loop=true): both trajectories restart together each cycle.
  The cycle duration is max(T_greedy, T_global). The shorter one holds its last pose
  until the cycle ends, then both restart simultaneously.

Playback control topics (for an RViz panel):
- pause_topic: std_msgs/Bool    (True => pause, False => play)
- seek_topic:  std_msgs/Float32 normalized [0..1] within the current shared cycle
State topics:
- progress_topic: std_msgs/Float32  0..1
- time_topic:     std_msgs/Float32  seconds within current cycle
- cycle_topic:    std_msgs/Float32  cycle duration seconds
- paused_topic:   std_msgs/Bool
Waypoint timing (for tick marks on the timeline):
- waypoints_topic: std_msgs/String  JSON with per-method waypoint arrival times
"""

from __future__ import annotations

import glob
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float32, String
from tf2_ros import Buffer, TransformListener
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray


def _quintic_time_scaling(s: float) -> float:
    """Smooth step from 0..1 with zero 1st/2nd derivatives at endpoints."""
    s = max(0.0, min(1.0, s))
    return 10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5


def _resolve_result_dir(path: str) -> str:
    """Accept either a run dir (contains summary.json) or a data dir (contains run subdirs)."""
    path = os.path.expanduser(path)
    if os.path.isfile(os.path.join(path, "summary.json")):
        return path

    candidates = glob.glob(os.path.join(path, "*", "summary.json"))
    if not candidates:
        raise FileNotFoundError(
            f"Could not find summary.json in '{path}' or any subdirectory like '{path}/*/summary.json'."
        )

    def _key(p: str) -> Tuple[int, str, float]:
        run_dir = os.path.basename(os.path.dirname(p))
        is_ts = 1 if run_dir.isdigit() else 0
        return (is_ts, run_dir, os.path.getmtime(p))

    best = max(candidates, key=_key)
    return os.path.dirname(best)


def _read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_targets(result_dir: str, summary: dict) -> List[dict]:
    """Load targets from targets.json (preferred) or summary['targets'] fallback.

    Expected target format:
      {"name":"p1","x":..., "y":..., "z":...}
    """
    tj = os.path.join(result_dir, "targets.json")
    if os.path.isfile(tj):
        try:
            data = _read_json(tj)
            if isinstance(data, list):
                return data
            if isinstance(data, dict) and "targets" in data and isinstance(data["targets"], list):
                return data["targets"]
        except Exception:
            pass
    # Fallback
    t = summary.get("targets", [])
    if isinstance(t, list):
        return t
    return []


def _find_joint_names(result_dir: str) -> List[str]:
    """Try to read joint_names from p*.json; fallback to panda_joint1..7."""
    for p in sorted(glob.glob(os.path.join(result_dir, "p*.json"))):
        base = os.path.basename(p)
        if base in ("p0.json",):
            continue
        try:
            data = _read_json(p)
            sols = data.get("solutions", [])
            if sols and "joint_names" in sols[0]:
                return list(sols[0]["joint_names"])
        except Exception:
            continue

    return [f"panda_joint{i}" for i in range(1, 8)]


@dataclass
class Segment:
    q0: List[float]
    q1: List[float]
    duration: float


@dataclass
class Trajectory:
    joint_names: List[str]
    samples: List[List[float]]  # joint positions per sample
    dt: float
    total_time: float


def _build_segments_from_summary(summary: dict, method: str) -> Tuple[List[Segment], List[str]]:
    """Return (segments, visit_order_names) for method in {'greedy','global'}."""
    start_q = summary["start"]["joint_positions"]
    segments: List[Segment] = []
    order: List[str] = []

    if method == "greedy":
        greedy = summary.get("greedy", {})
        steps = greedy.get("steps", [])
        q_prev = list(start_q)
        for step in steps:
            sol = step.get("solution", {})
            q_next = sol.get("joint_positions", None)
            if q_next is None:
                continue
            dur = float(step.get("segment_time_s", 0.0))
            if dur <= 0.0 and "segment_times_s" in greedy:
                idx = len(segments)
                if idx < len(greedy["segment_times_s"]):
                    dur = float(greedy["segment_times_s"][idx])
            if dur <= 0.0:
                dur = 1.0

            segments.append(Segment(q0=q_prev, q1=list(q_next), duration=dur))
            q_prev = list(q_next)
            if "point" in step:
                order.append(str(step["point"]))

    elif method == "global":
        glob_res = summary.get("global", {})
        final_path = glob_res.get("final_path", {})
        segs = final_path.get("segments", [])
        q_prev = list(start_q)
        for seg in segs:
            q_next = seg.get("joint_positions", None)
            if q_next is None:
                continue
            dur = float(seg.get("time_s", 0.0))
            if dur <= 0.0:
                dur = 1.0
            segments.append(Segment(q0=q_prev, q1=list(q_next), duration=dur))
            q_prev = list(q_next)
            if "to" in seg:
                order.append(str(seg["to"]))

    else:
        raise ValueError(f"Unknown method: {method}")

    return segments, order


def _segments_to_trajectory(segments: Sequence[Segment], joint_names: List[str], dt: float, hold_time_s: float) -> Trajectory:
    samples: List[List[float]] = []
    t_total = 0.0
    q_last: Optional[List[float]] = None

    for si, seg in enumerate(segments):
        dur = max(1e-6, float(seg.duration))
        n = max(2, int(math.ceil(dur / dt)) + 1)

        for i in range(n):
            if si > 0 and i == 0:
                continue
            t = min(dur, i * dt)
            u = _quintic_time_scaling(t / dur)
            q = [a + (b - a) * u for a, b in zip(seg.q0, seg.q1)]
            samples.append(q)
            q_last = q

        t_total += dur

        if hold_time_s > 0.0 and q_last is not None:
            n_hold = int(math.ceil(hold_time_s / dt))
            for _ in range(n_hold):
                samples.append(list(q_last))
            t_total += n_hold * dt

    if not samples and segments:
        samples = [list(segments[0].q0), list(segments[-1].q1)]
        t_total = dt * (len(samples) - 1)

    return Trajectory(joint_names=joint_names, samples=samples, dt=dt, total_time=t_total)


def _sort_point_name_key(name: str) -> Tuple[int, str]:
    # Prefer p<number> ordering; otherwise lexical.
    if isinstance(name, str) and name.startswith("p"):
        try:
            return (0, f"{int(name[1:]):08d}")
        except Exception:
            pass
    return (1, str(name))


class IKBenchmarkComparePlayer(Node):
    def __init__(self) -> None:
        super().__init__("ik_benchmark_compare_player")

        # -------------------------
        # Parameters
        # -------------------------
        self.declare_parameter("result_dir", "")
        self.declare_parameter("fixed_frame", "world")
        self.declare_parameter("root_link", "panda_link0")
        self.declare_parameter("ee_link", "panda_hand")

        # Prefix must end with '/' if you want frames like 'greedy_/panda_link0'
        self.declare_parameter("greedy_prefix", "greedy_/" )
        self.declare_parameter("global_prefix", "global_/" )

        # Base placement (world offsets)
        self.declare_parameter("greedy_offset_xyz", [0.0, -0.6, 0.0])
        self.declare_parameter("global_offset_xyz", [0.0, 0.6, 0.0])

        # Playback
        self.declare_parameter("publish_rate_hz", 50.0)
        self.declare_parameter("hold_time_s", 0.5)
        self.declare_parameter("loop", True)
        self.declare_parameter("speed_scale", 1.0)
        self.declare_parameter("sync_loop", False)

        # Gripper joints (so TF exists for panda_leftfinger/rightfinger, and meshes can be shown)
        self.declare_parameter("publish_gripper", True)
        self.declare_parameter("gripper_joint_names", ["panda_finger_joint1", "panda_finger_joint2"])
        self.declare_parameter("gripper_joint_positions", [0.04, 0.04])  # open by default (meters)

        # Playback control topics (for RViz panel)
        self.declare_parameter("pause_topic", "/ik_benchmark/playback/pause")
        self.declare_parameter("seek_topic", "/ik_benchmark/playback/seek")
        self.declare_parameter("progress_topic", "/ik_benchmark/playback/progress")
        self.declare_parameter("time_topic", "/ik_benchmark/playback/time_s")
        self.declare_parameter("cycle_topic", "/ik_benchmark/playback/cycle_s")
        self.declare_parameter("paused_topic", "/ik_benchmark/playback/paused")
        self.declare_parameter("waypoints_topic", "/ik_benchmark/playback/waypoints_json")
        # Republish waypoint timing periodically so late-joining UIs (or non-latched subscribers)
        # still receive tick marks.
        # Set <= 0 to disable periodic republish.
        self.declare_parameter("waypoints_republish_period_s", 2.0)

        # Marker mode
        # - per_method: separate colored spheres+labels for global/greedy (old behavior)
        # - shared: one white set p0..pN + colored line strips for global/greedy orders + one title per robot
        self.declare_parameter("marker_mode", "per_method")

        # Marker style (used by both modes)
        self.declare_parameter("greedy_rgba", [1.0, 0.95, 0.30, 0.55])  # light yellow (default)
        self.declare_parameter("global_rgba", [0.05, 0.15, 0.80, 0.80])  # deep blue (default)
        self.declare_parameter("shared_point_rgba", [1.0, 1.0, 1.0, 1.0])  # white
        self.declare_parameter("label_scale_z", 0.06)
        self.declare_parameter("label_dz", 0.08)

        # -------------------------
        # Resolve result dir
        # -------------------------
        result_dir_in = self.get_parameter("result_dir").get_parameter_value().string_value
        if not result_dir_in:
            result_dir_in = "data"

        self.result_dir = _resolve_result_dir(result_dir_in)
        self.get_logger().info(f"Using result_dir: {self.result_dir}")

        self.fixed_frame = self.get_parameter("fixed_frame").get_parameter_value().string_value
        self.root_link = self.get_parameter("root_link").get_parameter_value().string_value
        self.ee_link = self.get_parameter("ee_link").get_parameter_value().string_value

        self.greedy_prefix = self.get_parameter("greedy_prefix").get_parameter_value().string_value
        self.global_prefix = self.get_parameter("global_prefix").get_parameter_value().string_value

        self.greedy_offset = list(self.get_parameter("greedy_offset_xyz").get_parameter_value().double_array_value)
        self.global_offset = list(self.get_parameter("global_offset_xyz").get_parameter_value().double_array_value)

        publish_rate = float(self.get_parameter("publish_rate_hz").value)
        publish_rate = max(1.0, publish_rate)
        self.dt = 1.0 / publish_rate

        self.hold_time_s = float(self.get_parameter("hold_time_s").value)
        self.loop = bool(self.get_parameter("loop").value)
        self.speed_scale = float(self.get_parameter("speed_scale").value)
        self.sync_loop = bool(self.get_parameter("sync_loop").value)

        if self.speed_scale <= 0.0:
            self.speed_scale = 1.0

        # Gripper
        self.publish_gripper = bool(self.get_parameter("publish_gripper").value)
        self.gripper_joint_names = list(self.get_parameter("gripper_joint_names").value)
        self.gripper_joint_positions = [float(x) for x in list(self.get_parameter("gripper_joint_positions").value)]
        if len(self.gripper_joint_positions) != len(self.gripper_joint_names):
            # fallback
            self.gripper_joint_positions = [0.04] * len(self.gripper_joint_names)

        # Colors
        self.greedy_rgba = self._rgba_param("greedy_rgba", (1.0, 0.95, 0.30, 0.55))
        self.global_rgba = self._rgba_param("global_rgba", (0.05, 0.15, 0.80, 0.80))
        self.shared_point_rgba = self._rgba_param("shared_point_rgba", (1.0, 1.0, 1.0, 1.0))
        self.label_scale_z = float(self.get_parameter("label_scale_z").value)
        self.label_dz = float(self.get_parameter("label_dz").value)

        # Control topics
        self.pause_topic = str(self.get_parameter("pause_topic").value)
        self.seek_topic = str(self.get_parameter("seek_topic").value)
        self.progress_topic = str(self.get_parameter("progress_topic").value)
        self.time_topic = str(self.get_parameter("time_topic").value)
        self.cycle_topic = str(self.get_parameter("cycle_topic").value)
        self.paused_topic = str(self.get_parameter("paused_topic").value)
        self.waypoints_topic = str(self.get_parameter("waypoints_topic").value)
        self.waypoints_republish_period_s = float(self.get_parameter("waypoints_republish_period_s").value)

        self.marker_mode = str(self.get_parameter("marker_mode").value).strip().lower()

        # -------------------------
        # Load result data
        # -------------------------
        summary = _read_json(os.path.join(self.result_dir, "summary.json"))
        self.summary = summary

        self.targets_list = _load_targets(self.result_dir, summary)
        # Normalize target dict
        self.targets_by_name: Dict[str, dict] = {}
        for t in self.targets_list:
            if not isinstance(t, dict):
                continue
            name = str(t.get("name", ""))
            if not name:
                continue
            if all(k in t for k in ("x", "y", "z")):
                self.targets_by_name[name] = t

        joint_names = _find_joint_names(self.result_dir)

        greedy_segments, greedy_order = _build_segments_from_summary(summary, "greedy")
        global_segments, global_order = _build_segments_from_summary(summary, "global")

        self.greedy_segments = greedy_segments
        self.global_segments = global_segments

        self.greedy_order = greedy_order
        self.global_order = global_order

        self.greedy_traj = _segments_to_trajectory(greedy_segments, joint_names, self.dt, self.hold_time_s)
        self.global_traj = _segments_to_trajectory(global_segments, joint_names, self.dt, self.hold_time_s)

        self.get_logger().info(
            f"Greedy traj: {len(self.greedy_traj.samples)} samples, total_time={self.greedy_traj.total_time:.3f}s"
        )
        self.get_logger().info(
            f"Global traj: {len(self.global_traj.samples)} samples, total_time={self.global_traj.total_time:.3f}s"
        )

        # Shared UI cycle time (also used for synchronized looping)
        self._cycle_time = float(max(self.greedy_traj.total_time, self.global_traj.total_time))
        if self.loop and self.sync_loop:
            self.get_logger().info(f"Sync loop enabled. cycle_time={self._cycle_time:.3f}s")

        # -------------------------
        # Publishers
        # -------------------------
        self.greedy_js_pub = self.create_publisher(JointState, "/greedy/joint_states", 10)
        self.global_js_pub = self.create_publisher(JointState, "/global/joint_states", 10)
        self.marker_pub = self.create_publisher(MarkerArray, "/ik_benchmark/markers", 10)

        # Playback state publishers (for RViz panel)
        self.progress_pub = self.create_publisher(Float32, self.progress_topic, 10)
        self.time_pub = self.create_publisher(Float32, self.time_topic, 10)
        self.cycle_pub = self.create_publisher(Float32, self.cycle_topic, 10)
        self.paused_pub = self.create_publisher(Bool, self.paused_topic, 10)

        # Waypoint timing publisher (latched / transient local)
        qos = QoSProfile(depth=1)
        qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        qos.reliability = ReliabilityPolicy.RELIABLE
        self.waypoints_pub = self.create_publisher(String, self.waypoints_topic, qos)

        # Playback control subscribers (for RViz panel)
        self.pause_sub = self.create_subscription(Bool, self.pause_topic, self._on_pause_cmd, 10)
        self.seek_sub = self.create_subscription(Float32, self.seek_topic, self._on_seek_cmd, 10)

        # Static TF
        self.static_tf = StaticTransformBroadcaster(self)
        self._publish_static_transforms()

        # TF listener (for p0 marker)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Markers
        self._p0_xyz_fixed: Optional[Tuple[float, float, float]] = None
        self._marker_msg = self._make_marker_array()
        self._try_p0_timer = self.create_timer(0.2, self._try_compute_p0)

        # Publish waypoint timing once (latched)
        self._publish_waypoints_json()
        if self.waypoints_republish_period_s > 0.0:
            self._waypoints_timer = self.create_timer(self.waypoints_republish_period_s, self._publish_waypoints_json)

        # Playback state
        self._paused = False
        self._t_abs = 0.0  # absolute time accumulator in seconds
        self._last_tick = self.get_clock().now()

        # Timers
        self._play_timer = self.create_timer(self.dt, self._on_timer)
        self._marker_timer = self.create_timer(1.0, self._on_marker_timer)

        # Publish initial UI state immediately
        self._publish_playback_state()

    # -------------------------
    # Helpers
    # -------------------------
    def _rgba_param(self, name: str, default: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
        v = self.get_parameter(name).value
        try:
            if isinstance(v, (list, tuple)) and len(v) == 4:
                r, g, b, a = [float(x) for x in v]
                return (r, g, b, a)
        except Exception:
            pass
        return default

    def _publish_static_transforms(self) -> None:
        msgs: List[TransformStamped] = []

        for prefix, offset in (
            (self.greedy_prefix, self.greedy_offset),
            (self.global_prefix, self.global_offset),
        ):
            ts = TransformStamped()
            ts.header.stamp = self.get_clock().now().to_msg()
            ts.header.frame_id = self.fixed_frame
            ts.child_frame_id = f"{prefix}{self.root_link}"
            ts.transform.translation.x = float(offset[0])
            ts.transform.translation.y = float(offset[1])
            ts.transform.translation.z = float(offset[2])
            ts.transform.rotation.w = 1.0
            msgs.append(ts)

        self.static_tf.sendTransform(msgs)

    def _offset_for_shared_markers(self) -> Tuple[float, float, float]:
        """In shared marker mode, assume both robots share the same base offset.
        We use the greedy offset (and warn if they differ significantly)."""
        dx = abs(self.greedy_offset[0] - self.global_offset[0])
        dy = abs(self.greedy_offset[1] - self.global_offset[1])
        dz = abs(self.greedy_offset[2] - self.global_offset[2])
        if dx + dy + dz > 1e-6:
            self.get_logger().warn(
                "marker_mode=shared but greedy_offset_xyz != global_offset_xyz; markers will follow greedy offset."
            )
        return (float(self.greedy_offset[0]), float(self.greedy_offset[1]), float(self.greedy_offset[2]))

    def _make_marker_array(self) -> MarkerArray:
        if self.marker_mode == "shared":
            return self._make_marker_array_shared()
        return self._make_marker_array_per_method()

    # -------------------------
    # Markers: per-method (legacy)
    # -------------------------
    def _make_marker_array_per_method(self) -> MarkerArray:
        ma = MarkerArray()
        now = self.get_clock().now().to_msg()

        def add_text(frame_id: str, mid: int, xyz: Tuple[float, float, float], text: str, rgba, dz: float) -> None:
            m = Marker()
            m.header.stamp = now
            m.header.frame_id = frame_id
            m.ns = "ik_benchmark_labels"
            m.id = mid
            m.type = Marker.TEXT_VIEW_FACING
            m.action = Marker.ADD
            m.pose.position.x = float(xyz[0])
            m.pose.position.y = float(xyz[1])
            m.pose.position.z = float(xyz[2]) + float(dz)
            m.pose.orientation.w = 1.0
            m.scale.z = self.label_scale_z
            m.color.r, m.color.g, m.color.b, m.color.a = rgba
            m.text = text
            ma.markers.append(m)

        def add_sphere(frame_id: str, mid: int, xyz: Tuple[float, float, float], rgba) -> None:
            m = Marker()
            m.header.stamp = now
            m.header.frame_id = frame_id
            m.ns = "ik_benchmark_points"
            m.id = mid
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = float(xyz[0])
            m.pose.position.y = float(xyz[1])
            m.pose.position.z = float(xyz[2])
            m.pose.orientation.w = 1.0
            m.scale.x = 0.05
            m.scale.y = 0.05
            m.scale.z = 0.05
            m.color.r, m.color.g, m.color.b, m.color.a = rgba
            ma.markers.append(m)

        def add_line_strip(frame_id: str, mid: int, xyz_list: List[Tuple[float, float, float]], rgba) -> None:
            if len(xyz_list) < 2:
                return
            from geometry_msgs.msg import Point

            m = Marker()
            m.header.stamp = now
            m.header.frame_id = frame_id
            m.ns = "ik_benchmark_lines"
            m.id = mid
            m.type = Marker.LINE_STRIP
            m.action = Marker.ADD
            m.pose.orientation.w = 1.0
            m.scale.x = 0.01
            r, g, b, a = rgba
            m.color.r, m.color.g, m.color.b, m.color.a = (r, g, b, min(0.8, a))

            for xyz in xyz_list:
                p = Point()
                p.x, p.y, p.z = float(xyz[0]), float(xyz[1]), float(xyz[2])
                m.points.append(p)
            ma.markers.append(m)

        # Titles above each robot (in fixed frame)
        gx, gy, gz = float(self.greedy_offset[0]), float(self.greedy_offset[1]), float(self.greedy_offset[2])
        hx, hy, hz = float(self.global_offset[0]), float(self.global_offset[1]), float(self.global_offset[2])

        add_text(self.fixed_frame, 1, (gx, gy, gz + 0.9), "GREEDY", self.gSreedy_rgba, dz=0.0)
        add_text(self.fixed_frame, 2, (hx, hy, hz + 0.9), "OURS", self.global_rgba, dz=0.0)

        # Per-method point markers (in each robot base frame)
        def add_method_points(method_name: str, prefix: str, order: List[str], base_id: int, rgba) -> None:
            frame = f"{prefix}{self.root_link}"
            pts: List[Tuple[float, float, float]] = []
            for i, pname in enumerate(order, start=1):
                t = self.targets_by_name.get(pname)
                if not t:
                    continue
                xyz = (float(t["x"]), float(t["y"]), float(t["z"]))
                pts.append(xyz)

                sid = base_id + i
                lid = base_id + 1000 + i
                label = f"{method_name} #{i}: {pname}"
                add_sphere(frame, sid, xyz, rgba)
                add_text(frame, lid, xyz, label, rgba, dz=self.label_dz)

            add_line_strip(frame, base_id + 5000, pts, rgba)

        add_method_points("greedy", self.greedy_prefix, self.greedy_order, base_id=10000, rgba=self.greedy_rgba)
        add_method_points("global", self.global_prefix, self.global_order, base_id=20000, rgba=self.global_rgba)

        return ma

    # -------------------------
    # Markers: shared (clean overlay mode)
    # -------------------------
    def _make_marker_array_shared(self) -> MarkerArray:
        ma = MarkerArray()
        now = self.get_clock().now().to_msg()

        def add_text(frame_id: str, ns: str, mid: int, xyz: Tuple[float, float, float], text: str, rgba, dz: float) -> None:
            m = Marker()
            m.header.stamp = now
            m.header.frame_id = frame_id
            m.ns = ns
            m.id = mid
            m.type = Marker.TEXT_VIEW_FACING
            m.action = Marker.ADD
            m.pose.position.x = float(xyz[0])
            m.pose.position.y = float(xyz[1])
            m.pose.position.z = float(xyz[2]) + float(dz)
            m.pose.orientation.w = 1.0
            m.scale.z = self.label_scale_z
            m.color.r, m.color.g, m.color.b, m.color.a = rgba
            m.text = text
            ma.markers.append(m)

        def add_sphere(frame_id: str, ns: str, mid: int, xyz: Tuple[float, float, float], rgba, scale: float = 0.05) -> None:
            m = Marker()
            m.header.stamp = now
            m.header.frame_id = frame_id
            m.ns = ns
            m.id = mid
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = float(xyz[0])
            m.pose.position.y = float(xyz[1])
            m.pose.position.z = float(xyz[2])
            m.pose.orientation.w = 1.0
            m.scale.x = scale
            m.scale.y = scale
            m.scale.z = scale
            m.color.r, m.color.g, m.color.b, m.color.a = rgba
            ma.markers.append(m)

        def add_line_strip(frame_id: str, ns: str, mid: int, xyz_list: List[Tuple[float, float, float]], rgba) -> None:
            if len(xyz_list) < 2:
                return
            from geometry_msgs.msg import Point

            m = Marker()
            m.header.stamp = now
            m.header.frame_id = frame_id
            m.ns = ns
            m.id = mid
            m.type = Marker.LINE_STRIP
            m.action = Marker.ADD
            m.pose.orientation.w = 1.0
            m.scale.x = 0.012
            r, g, b, a = rgba
            m.color.r, m.color.g, m.color.b, m.color.a = (r, g, b, min(0.9, a))

            for xyz in xyz_list:
                p = Point()
                p.x, p.y, p.z = float(xyz[0]), float(xyz[1]), float(xyz[2])
                m.points.append(p)
            ma.markers.append(m)

        # Titles: one for each robot (placed above base; if overlapped, shift X slightly so they are readable)
        ox, oy, oz = self._offset_for_shared_markers()
        shift = 0.15
        add_text(self.fixed_frame, "ik_benchmark_titles", 1, (ox - shift, oy, oz + 0.9), "OURS", self.global_rgba, dz=0.0)
        add_text(self.fixed_frame, "ik_benchmark_titles", 2, (ox + shift, oy, oz + 0.9), "GREEDY", self.greedy_rgba, dz=0.0)

        # Shared path points (white): p1..pN
        # Use fixed_frame positions by applying the shared base offset.
        targets_sorted = sorted(self.targets_by_name.values(), key=lambda t: _sort_point_name_key(str(t.get("name", ""))))
        base_id = 100
        label_id = 1100

        # p0 (start EE) if available
        if self._p0_xyz_fixed is not None:
            add_sphere(self.fixed_frame, "ik_benchmark_p0", base_id, self._p0_xyz_fixed, self.shared_point_rgba, scale=0.055)
            add_text(self.fixed_frame, "ik_benchmark_p0_label", label_id, self._p0_xyz_fixed, "p0", self.shared_point_rgba, dz=self.label_dz)
            base_id += 1
            label_id += 1

        for t in targets_sorted:
            name = str(t.get("name", ""))
            if not name:
                continue
            xyz = (float(t["x"]) + ox, float(t["y"]) + oy, float(t["z"]) + oz)
            add_sphere(self.fixed_frame, "ik_benchmark_points", base_id, xyz, self.shared_point_rgba, scale=0.05)
            add_text(self.fixed_frame, "ik_benchmark_point_labels", label_id, xyz, name, self.shared_point_rgba, dz=self.label_dz)
            base_id += 1
            label_id += 1

        # Colored line strips (orders)
        def order_to_xyz(order: List[str]) -> List[Tuple[float, float, float]]:
            out: List[Tuple[float, float, float]] = []

            if self._p0_xyz_fixed is not None:
                out.append(self._p0_xyz_fixed)

            for pname in order:
                t = self.targets_by_name.get(pname)
                if not t:
                    continue
                out.append((float(t["x"]) + ox, float(t["y"]) + oy, float(t["z"]) + oz))
            return out

        add_line_strip(self.fixed_frame, "ik_benchmark_lines_global", 5000, order_to_xyz(self.global_order), self.global_rgba)
        add_line_strip(self.fixed_frame, "ik_benchmark_lines_greedy", 5001, order_to_xyz(self.greedy_order), self.greedy_rgba)

        return ma

    # -------------------------
    # JointState publishing
    # -------------------------
    def _compose_joint_state(self, traj: Trajectory, idx: int) -> JointState:
        idx = max(0, min(idx, len(traj.samples) - 1))
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()

        names = list(traj.joint_names)
        pos = list(traj.samples[idx])

        # Ensure gripper joints are present so TF exists for finger links.
        if self.publish_gripper and self.gripper_joint_names:
            for jn, jp in zip(self.gripper_joint_names, self.gripper_joint_positions):
                if jn not in names:
                    names.append(str(jn))
                    pos.append(float(jp))

        msg.name = names
        msg.position = pos
        return msg

    def _publish_joint_state(self, pub, traj: Trajectory, idx: int) -> None:
        if not traj.samples:
            return
        pub.publish(self._compose_joint_state(traj, idx))

    # -------------------------
    # UI state + waypoint ticks
    # -------------------------
    def _ui_time_in_cycle(self) -> float:
        """Return the UI time in seconds within the shared cycle [0..cycle_time]."""
        cycle = float(self._cycle_time)
        if cycle <= 1e-6:
            return 0.0
        if self.loop:
            t_mod = float(self._t_abs % cycle)
            # If user seeks to 1.0 or we land exactly on a boundary, show "end of cycle".
            if self._t_abs > 1e-6 and t_mod < 1e-6:
                return float(cycle)
            return t_mod
        return float(min(self._t_abs, cycle))

    def _publish_playback_state(self) -> None:
        """Publish playback UI state for a pause/seek panel."""
        cycle = float(self._cycle_time)
        t_ui = self._ui_time_in_cycle()
        progress = 0.0 if cycle <= 1e-6 else float(t_ui / cycle)

        msg_p = Float32()
        msg_p.data = float(progress)
        self.progress_pub.publish(msg_p)

        msg_t = Float32()
        msg_t.data = float(t_ui)
        self.time_pub.publish(msg_t)

        msg_c = Float32()
        msg_c.data = float(cycle)
        self.cycle_pub.publish(msg_c)

        msg_paused = Bool()
        msg_paused.data = bool(self._paused)
        self.paused_pub.publish(msg_paused)

    def _publish_waypoints_json(self) -> None:
        """Publish per-method waypoint arrival times as JSON (latched)."""
        def build(method: str, segs: List[Segment], order: List[str]) -> List[dict]:
            out: List[dict] = [{"name": "p0", "t": 0.0}]
            t = 0.0
            for dur, pname in zip([s.duration for s in segs], order):
                t += float(dur)
                out.append({"name": str(pname), "t": float(t)})
                # Hold after arriving (matches playback)
                if self.hold_time_s > 0.0:
                    t += float(self.hold_time_s)
            return out

        payload = {
            "cycle_s": float(self._cycle_time),
            "global": build("global", self.global_segments, self.global_order),
            "greedy": build("greedy", self.greedy_segments, self.greedy_order),
        }

        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.waypoints_pub.publish(msg)

    # -------------------------
    # Playback control callbacks
    # -------------------------
    def _on_pause_cmd(self, msg: Bool) -> None:
        # True => pause; False => play
        self._paused = bool(msg.data)
        # Reset tick so that resuming doesn't cause a large dt jump.
        self._last_tick = self.get_clock().now()
        self._publish_playback_state()

    def _on_seek_cmd(self, msg: Float32) -> None:
        # Normalized seek [0..1] within the shared cycle.
        cycle = float(self._cycle_time)
        if cycle <= 1e-6:
            return
        p = max(0.0, min(1.0, float(msg.data)))
        self._t_abs = p * cycle
        self._last_tick = self.get_clock().now()
        # Publish immediately so the UI reflects the new position.
        self._publish_playback_state()

    # -------------------------
    # Timers
    # -------------------------
    def _on_timer(self) -> None:
        now = self.get_clock().now()
        dt_wall = (now - self._last_tick).nanoseconds * 1e-9
        if dt_wall < 0.0:
            dt_wall = 0.0
        self._last_tick = now

        # Advance time only if not paused.
        if not self._paused:
            self._t_abs += dt_wall * self.speed_scale

        tg_total = float(self.greedy_traj.total_time)
        tl_total = float(self.global_traj.total_time)

        if self.loop:
            if self.sync_loop:
                # Shared cycle: both start together; shorter holds at end.
                t_cycle = self._ui_time_in_cycle()
                t_g = min(t_cycle, tg_total)
                t_l = min(t_cycle, tl_total)
            else:
                # Independent looping.
                t = float(self._t_abs)
                t_g = (t % tg_total) if tg_total > 1e-6 else 0.0
                t_l = (t % tl_total) if tl_total > 1e-6 else 0.0
        else:
            t = float(self._t_abs)
            t_g = min(t, tg_total)
            t_l = min(t, tl_total)

        if self.greedy_traj.samples:
            idx_g = int(t_g / self.greedy_traj.dt)
            self._publish_joint_state(self.greedy_js_pub, self.greedy_traj, idx_g)

        if self.global_traj.samples:
            idx_l = int(t_l / self.global_traj.dt)
            self._publish_joint_state(self.global_js_pub, self.global_traj, idx_l)

        # Update UI state (progress, time, paused)
        self._publish_playback_state()

    def _on_marker_timer(self) -> None:
        now = self.get_clock().now().to_msg()
        for m in self._marker_msg.markers:
            m.header.stamp = now
        self.marker_pub.publish(self._marker_msg)

    def _try_compute_p0(self) -> None:
        """Try to compute p0 (start EE position).

        We want p0 to represent the *start* configuration, even if playback has already
        begun. To make this robust, we momentarily publish the start joint state (idx=0)
        and query TF for the EE frame.
        """
        if self._p0_xyz_fixed is not None:
            self._try_p0_timer.cancel()
            return

        ee_frame = f"{self.greedy_prefix}{self.ee_link}"

        # Save current playback state; temporarily force start pose.
        prev_paused = bool(self._paused)
        prev_t_abs = float(self._t_abs)
        self._paused = True
        self._t_abs = 0.0

        # Publish start pose once (both robots) so robot_state_publisher has the correct TF.
        try:
            if self.greedy_traj.samples:
                self._publish_joint_state(self.greedy_js_pub, self.greedy_traj, 0)
            if self.global_traj.samples:
                self._publish_joint_state(self.global_js_pub, self.global_traj, 0)

            try:
                from rclpy.duration import Duration
                if not self.tf_buffer.can_transform(self.fixed_frame, ee_frame, rclpy.time.Time(), timeout=Duration(seconds=0.2)):
                    return
            except Exception:
                # can_transform may not be available on some builds; fall back to lookup.
                pass

            tf = self.tf_buffer.lookup_transform(self.fixed_frame, ee_frame, rclpy.time.Time())
            tr = tf.transform.translation
            self._p0_xyz_fixed = (float(tr.x), float(tr.y), float(tr.z))
            self.get_logger().info(
                f"Computed p0 from TF (start pose): {ee_frame} in {self.fixed_frame} => {self._p0_xyz_fixed}"
            )
            self._marker_msg = self._make_marker_array()
            self._try_p0_timer.cancel()
        except Exception:
            # TF may not be ready yet.
            return
        finally:
            # Restore playback state (and reset tick to avoid time jump on resume).
            self._paused = prev_paused
            self._t_abs = prev_t_abs
            self._last_tick = self.get_clock().now()
def main(args: Optional[List[str]] = None) -> None:
    rclpy.init(args=args)
    node = IKBenchmarkComparePlayer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
