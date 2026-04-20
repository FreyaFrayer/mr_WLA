#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Compare one run's origin, window and planner-replay paths in RViz2."""

from __future__ import annotations

import glob
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import rclpy
from geometry_msgs.msg import Point, TransformStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float32, String
from tf2_ros import Buffer, TransformListener
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray


def _read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _summary_pick_key(summary_path: str) -> Tuple[int, str, float]:
    run_dir = os.path.basename(os.path.dirname(summary_path))
    is_timestamp = 1 if run_dir.isdigit() else 0
    return (is_timestamp, run_dir, os.path.getmtime(summary_path))


def _resolve_result_dir(path: str) -> str:
    path = os.path.abspath(os.path.expanduser(path))
    if os.path.isfile(os.path.join(path, "summary.json")):
        return path

    candidates = glob.glob(os.path.join(path, "**", "summary.json"), recursive=True)
    if not candidates:
        raise FileNotFoundError(f"Could not find summary.json in '{path}' or its subdirectories.")

    best = max(candidates, key=_summary_pick_key)
    return os.path.dirname(best)


def _load_targets(result_dir: str, summary: dict) -> List[dict]:
    targets_path = os.path.join(result_dir, "targets.json")
    if os.path.isfile(targets_path):
        try:
            data = _read_json(targets_path)
            if isinstance(data, dict):
                targets = data.get("targets", [])
                if isinstance(targets, list):
                    return targets
            if isinstance(data, list):
                return data
        except Exception:
            pass

    targets = summary.get("targets", [])
    if isinstance(targets, list):
        return targets
    return []


def _find_joint_names(summary: dict, result_dir: str, dof: int) -> List[str]:
    origin = summary.get("origin", {})
    if isinstance(origin, dict):
        names = origin.get("joint_names", [])
        if isinstance(names, list) and len(names) == dof:
            return [str(v) for v in names]

    for p in sorted(glob.glob(os.path.join(result_dir, "p*.json"))):
        if os.path.basename(p) == "p0.json":
            continue
        try:
            data = _read_json(p)
            sols = data.get("solutions", [])
            if sols and isinstance(sols[0], dict):
                names = sols[0].get("joint_names", [])
                if isinstance(names, list) and len(names) == dof:
                    return [str(v) for v in names]
        except Exception:
            continue

    return [f"panda_joint{i}" for i in range(1, dof + 1)]


def _sort_point_name_key(name: str) -> Tuple[int, int, str]:
    if isinstance(name, str) and name.startswith("p"):
        try:
            return (0, int(name[1:]), name)
        except Exception:
            pass
    return (1, 0, str(name))


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
class Trajectory:
    joint_names: List[str]
    samples: List[List[float]]
    dt: float
    total_time: float


def _pick_result_by_ws(summary: dict, section_key: str, window_size: int) -> dict:
    section = summary.get(section_key, {})
    if not isinstance(section, dict):
        raise RuntimeError(f"summary.json missing {section_key} section")

    by_ws = section.get("results_by_ws", {})
    key = str(int(window_size))
    if isinstance(by_ws, dict) and by_ws:
        if key in by_ws and isinstance(by_ws[key], dict):
            return by_ws[key]
        raise KeyError(
            f"window_size={window_size} not found in {section_key}.results_by_ws; "
            f"available={sorted(by_ws.keys())}"
        )

    results = section.get("results", [])
    if isinstance(results, list):
        for rec in results:
            if not isinstance(rec, dict):
                continue
            try:
                rec_ws = int(rec.get("window_size", -1))
            except Exception:
                continue
            if rec_ws == int(window_size):
                return rec

    raise KeyError(f"window_size={window_size} not found in summary.json section '{section_key}'")


def _pick_window_result(summary: dict, window_size: int) -> dict:
    return _pick_result_by_ws(summary, "window", window_size)


def _pick_ws_planner_result(summary: dict, window_size: int) -> dict:
    return _pick_result_by_ws(summary, "trapezoid_solutions_true_plan", window_size)


def _extract_start_q(summary: dict) -> List[float]:
    start = summary.get("start", {})
    if not isinstance(start, dict):
        raise RuntimeError("summary.json missing start section")

    joint_positions = start.get("joint_positions", [])
    if not isinstance(joint_positions, list) or not joint_positions:
        raise RuntimeError("summary.json missing start.joint_positions")

    return [float(v) for v in joint_positions]


def _build_segments_from_raw(
    raw_segments: Sequence[dict],
    start_q: Sequence[float],
    *,
    q0_keys: Sequence[str],
    q1_keys: Sequence[str],
    duration_keys: Sequence[str],
    to_keys: Sequence[str],
    source_name: str,
) -> Tuple[List[Segment], List[str]]:
    q_prev = [float(v) for v in start_q]
    segments: List[Segment] = []
    order: List[str] = []

    for seg in raw_segments:
        if not isinstance(seg, dict):
            continue

        q0_raw = None
        for key in q0_keys:
            value = seg.get(key, None)
            if isinstance(value, list):
                q0_raw = value
                break
        if q0_raw is None:
            q0_raw = q_prev

        q1_raw = None
        for key in q1_keys:
            value = seg.get(key, None)
            if isinstance(value, list):
                q1_raw = value
                break

        if not isinstance(q0_raw, list) or not isinstance(q1_raw, list):
            continue
        if len(q0_raw) != len(q_prev) or len(q1_raw) != len(q_prev):
            continue

        dur = 0.0
        for key in duration_keys:
            if key not in seg:
                continue
            try:
                dur = float(seg.get(key, 0.0))
                break
            except Exception:
                dur = 0.0
        if dur <= 0.0:
            dur = 1.0

        q0 = [float(v) for v in q0_raw]
        q1 = [float(v) for v in q1_raw]
        to_name = ""
        for key in to_keys:
            value = str(seg.get(key, "")).strip()
            if value:
                to_name = value
                break
        segments.append(Segment(q0=q0, q1=q1, duration=dur, to_name=to_name))
        q_prev = q1
        if to_name:
            order.append(to_name)

    if not segments:
        raise RuntimeError(f"{source_name} segments exist but none could be parsed")

    return segments, order


def _build_origin_segments(summary: dict, start_q: Sequence[float]) -> Tuple[List[Segment], List[str], float]:
    origin = summary.get("origin", {})
    if not isinstance(origin, dict):
        raise RuntimeError("summary.json missing origin section")

    raw_segments = origin.get("segments", [])
    if not isinstance(raw_segments, list) or not raw_segments:
        status = str(origin.get("status", "unknown"))
        note = str(origin.get("note", "")).strip()
        suffix = f": {note}" if note else ""
        raise RuntimeError(f"origin has no valid segments (status={status}){suffix}")

    segments, order = _build_segments_from_raw(
        raw_segments,
        start_q,
        q0_keys=("start_joint_positions",),
        q1_keys=("end_joint_positions",),
        duration_keys=("trajectory_time_s", "time_s"),
        to_keys=("to",),
        source_name="origin",
    )

    total_time = float(origin.get("total_time_s", sum(seg.duration for seg in segments)))
    return segments, order, total_time


def _build_window_segments(
    summary: dict,
    start_q: Sequence[float],
    window_size: int,
) -> Tuple[List[Segment], List[str], float]:
    ws_data = _pick_window_result(summary, window_size)

    raw_segments: List[dict] = []
    final_path = ws_data.get("final_path", {})
    if isinstance(final_path, dict):
        items = final_path.get("segments", [])
        if isinstance(items, list):
            raw_segments = [v for v in items if isinstance(v, dict)]

    if not raw_segments:
        items = ws_data.get("segments", [])
        if isinstance(items, list):
            raw_segments = [v for v in items if isinstance(v, dict)]

    if not raw_segments:
        raise RuntimeError(f"window_size={window_size} has no valid segments")

    segments, order = _build_segments_from_raw(
        raw_segments,
        start_q,
        q0_keys=("start_joint_positions",),
        q1_keys=("joint_positions", "end_joint_positions"),
        duration_keys=("time_s", "trajectory_time_s"),
        to_keys=("to", "point"),
        source_name=f"window_size={window_size}",
    )

    total_time = float(final_path.get("total_time_s", ws_data.get("total_time_s", sum(seg.duration for seg in segments))))
    return segments, order, total_time


def _build_ws_planner_segments(
    summary: dict,
    start_q: Sequence[float],
    window_size: int,
) -> Tuple[List[Segment], List[str], float]:
    ws_planner_data = _pick_ws_planner_result(summary, window_size)

    raw_segments = ws_planner_data.get("segments", [])
    if not isinstance(raw_segments, list) or not raw_segments:
        status = str(ws_planner_data.get("status", "unknown"))
        note = str(ws_planner_data.get("note", "")).strip()
        suffix = f": {note}" if note else ""
        raise RuntimeError(
            f"window_size={window_size} planner replay has no valid segments (status={status}){suffix}"
        )

    segments, order = _build_segments_from_raw(
        raw_segments,
        start_q,
        q0_keys=("start_joint_positions",),
        q1_keys=("end_joint_positions", "joint_positions"),
        duration_keys=("time_s", "trajectory_time_s"),
        to_keys=("to", "point"),
        source_name=f"window_size={window_size} planner replay",
    )

    total_time = float(ws_planner_data.get("total_time_s", sum(seg.duration for seg in segments)))
    return segments, order, total_time


def _segments_to_trajectory(
    *,
    segments: Sequence[Segment],
    joint_names: List[str],
    dt: float,
    hold_time_s: float,
    start_q: Sequence[float],
) -> Trajectory:
    if not segments:
        return Trajectory(
            joint_names=list(joint_names),
            samples=[list(start_q)],
            dt=float(dt),
            total_time=0.0,
        )

    samples: List[List[float]] = []
    total_time = 0.0
    q_last = list(start_q)

    for seg_idx, seg in enumerate(segments):
        dur = max(1e-6, float(seg.duration))
        n = max(2, int(math.ceil(dur / dt)) + 1)

        for i in range(n):
            if seg_idx > 0 and i == 0:
                continue
            t = min(dur, i * dt)
            u = _quintic_time_scaling(t / dur)
            q = [a + (b - a) * u for a, b in zip(seg.q0, seg.q1)]
            samples.append(q)
            q_last = q

        total_time += dur

        if hold_time_s > 0.0:
            n_hold = int(math.ceil(hold_time_s / dt))
            for _ in range(n_hold):
                samples.append(list(q_last))
            total_time += n_hold * dt

    if not samples:
        samples = [list(start_q)]

    return Trajectory(
        joint_names=list(joint_names),
        samples=samples,
        dt=float(dt),
        total_time=float(max(0.0, total_time)),
    )


class IKOriginComparePlayer(Node):
    def __init__(self) -> None:
        super().__init__("ik_origin_compare_player")

        self.declare_parameter("result_dir", "data_window")
        self.declare_parameter("window_size", 1)

        self.declare_parameter("fixed_frame", "world")
        self.declare_parameter("root_link", "panda_link0")
        self.declare_parameter("ee_link", "panda_hand")
        self.declare_parameter("origin_prefix", "origin_/")
        self.declare_parameter("ws_prefix", "ws_/")
        self.declare_parameter("ws_planner_prefix", "ws_planner_/")
        self.declare_parameter("origin_label", "")
        self.declare_parameter("ws_label", "")
        self.declare_parameter("ws_planner_label", "")
        self.declare_parameter("origin_offset_xyz", [0.0, -1.2, 0.0])
        self.declare_parameter("ws_offset_xyz", [0.0, 0.0, 0.0])
        self.declare_parameter("ws_planner_offset_xyz", [0.0, 1.2, 0.0])
        self.declare_parameter("planner_window_size", 0)

        self.declare_parameter("publish_rate_hz", 50.0)
        self.declare_parameter("hold_time_s", 0.5)
        self.declare_parameter("loop", True)
        self.declare_parameter("sync_loop", True)
        self.declare_parameter("speed_scale", 1.0)

        self.declare_parameter("publish_gripper", True)
        self.declare_parameter("gripper_joint_names", ["panda_finger_joint1", "panda_finger_joint2"])
        self.declare_parameter("gripper_joint_positions", [0.04, 0.04])

        self.declare_parameter("marker_topic", "/ik_origin/markers")
        self.declare_parameter("origin_rgba", [0.08, 0.27, 0.95, 0.95])
        self.declare_parameter("ws_rgba", [1.00, 0.55, 0.10, 0.95])
        self.declare_parameter("ws_planner_rgba", [0.12, 0.72, 0.25, 0.95])
        self.declare_parameter("point_rgba", [1.0, 1.0, 1.0, 1.0])
        self.declare_parameter("point_scale", 0.05)
        self.declare_parameter("line_width", 0.012)
        self.declare_parameter("label_scale_z", 0.05)
        self.declare_parameter("label_dz", 0.07)

        self.declare_parameter("pause_topic", "/ik_origin/playback/pause")
        self.declare_parameter("seek_topic", "/ik_origin/playback/seek")
        self.declare_parameter("progress_topic", "/ik_origin/playback/progress")
        self.declare_parameter("time_topic", "/ik_origin/playback/time_s")
        self.declare_parameter("cycle_topic", "/ik_origin/playback/cycle_s")
        self.declare_parameter("paused_topic", "/ik_origin/playback/paused")
        self.declare_parameter("waypoints_topic", "/ik_origin/playback/waypoints_json")
        self.declare_parameter("waypoints_republish_period_s", 2.0)

        result_dir_in = str(self.get_parameter("result_dir").value)
        self.result_dir = _resolve_result_dir(result_dir_in)
        self.summary = _read_json(os.path.join(self.result_dir, "summary.json"))
        self.window_size = int(self.get_parameter("window_size").value)
        start_q = _extract_start_q(self.summary)

        joint_names = _find_joint_names(self.summary, self.result_dir, len(start_q))
        if len(joint_names) != len(start_q):
            joint_names = [f"panda_joint{i}" for i in range(1, len(start_q) + 1)]

        self.fixed_frame = str(self.get_parameter("fixed_frame").value)
        self.root_link = str(self.get_parameter("root_link").value)
        self.ee_link = str(self.get_parameter("ee_link").value)
        self.origin_prefix = str(self.get_parameter("origin_prefix").value)
        self.ws_prefix = str(self.get_parameter("ws_prefix").value)
        self.ws_planner_prefix = str(self.get_parameter("ws_planner_prefix").value)
        self.origin_offset = [float(v) for v in list(self.get_parameter("origin_offset_xyz").value)]
        self.ws_offset = [float(v) for v in list(self.get_parameter("ws_offset_xyz").value)]
        self.ws_planner_offset = [float(v) for v in list(self.get_parameter("ws_planner_offset_xyz").value)]
        planner_window_size_in = int(self.get_parameter("planner_window_size").value)
        self.planner_window_size = planner_window_size_in if planner_window_size_in > 0 else self.window_size

        rate_hz = max(1.0, float(self.get_parameter("publish_rate_hz").value))
        self.dt = 1.0 / rate_hz
        self.hold_time_s = max(0.0, float(self.get_parameter("hold_time_s").value))
        self.loop = bool(self.get_parameter("loop").value)
        self.sync_loop = bool(self.get_parameter("sync_loop").value)
        self.speed_scale = max(1e-6, float(self.get_parameter("speed_scale").value))

        self.publish_gripper = bool(self.get_parameter("publish_gripper").value)
        self.gripper_joint_names = [str(v) for v in list(self.get_parameter("gripper_joint_names").value)]
        self.gripper_joint_positions = [float(v) for v in list(self.get_parameter("gripper_joint_positions").value)]
        if len(self.gripper_joint_positions) != len(self.gripper_joint_names):
            self.gripper_joint_positions = [0.04] * len(self.gripper_joint_names)

        self.marker_topic = str(self.get_parameter("marker_topic").value)
        self.origin_rgba = self._rgba_param("origin_rgba", (0.08, 0.27, 0.95, 0.95))
        self.ws_rgba = self._rgba_param("ws_rgba", (1.00, 0.55, 0.10, 0.95))
        self.ws_planner_rgba = self._rgba_param("ws_planner_rgba", (0.12, 0.72, 0.25, 0.95))
        self.point_rgba = self._rgba_param("point_rgba", (1.0, 1.0, 1.0, 1.0))
        self.point_scale = float(self.get_parameter("point_scale").value)
        self.line_width = float(self.get_parameter("line_width").value)
        self.label_scale_z = float(self.get_parameter("label_scale_z").value)
        self.label_dz = float(self.get_parameter("label_dz").value)

        self.pause_topic = str(self.get_parameter("pause_topic").value)
        self.seek_topic = str(self.get_parameter("seek_topic").value)
        self.progress_topic = str(self.get_parameter("progress_topic").value)
        self.time_topic = str(self.get_parameter("time_topic").value)
        self.cycle_topic = str(self.get_parameter("cycle_topic").value)
        self.paused_topic = str(self.get_parameter("paused_topic").value)
        self.waypoints_topic = str(self.get_parameter("waypoints_topic").value)
        self.waypoints_republish_period_s = float(self.get_parameter("waypoints_republish_period_s").value)

        self.origin_segments, self.origin_order, self.origin_nominal_total_s = _build_origin_segments(
            self.summary,
            start_q,
        )
        self.ws_segments, self.ws_order, self.ws_nominal_total_s = _build_window_segments(
            self.summary,
            start_q,
            self.window_size,
        )
        self.ws_planner_segments, self.ws_planner_order, self.ws_planner_nominal_total_s = _build_ws_planner_segments(
            self.summary,
            start_q,
            self.planner_window_size,
        )

        self.origin_traj = _segments_to_trajectory(
            segments=self.origin_segments,
            joint_names=joint_names,
            dt=self.dt,
            hold_time_s=self.hold_time_s,
            start_q=start_q,
        )
        self.ws_traj = _segments_to_trajectory(
            segments=self.ws_segments,
            joint_names=joint_names,
            dt=self.dt,
            hold_time_s=self.hold_time_s,
            start_q=start_q,
        )
        self.ws_planner_traj = _segments_to_trajectory(
            segments=self.ws_planner_segments,
            joint_names=joint_names,
            dt=self.dt,
            hold_time_s=self.hold_time_s,
            start_q=start_q,
        )

        origin_label_in = str(self.get_parameter("origin_label").value).strip()
        ws_label_in = str(self.get_parameter("ws_label").value).strip()
        ws_planner_label_in = str(self.get_parameter("ws_planner_label").value).strip()
        self.origin_label = origin_label_in or "ORIGIN"
        self.ws_label = ws_label_in or f"WS={self.window_size}"
        self.ws_planner_label = ws_planner_label_in or f"WS={self.planner_window_size} PLANNER"

        self.targets_by_name: Dict[str, dict] = {}
        for target in _load_targets(self.result_dir, self.summary):
            if not isinstance(target, dict):
                continue
            if not all(key in target for key in ("name", "x", "y", "z")):
                continue
            name = str(target.get("name", "")).strip()
            if name:
                self.targets_by_name[name] = target

        self.origin_js_pub = self.create_publisher(JointState, "/origin/joint_states", 10)
        self.ws_js_pub = self.create_publisher(JointState, "/ws/joint_states", 10)
        self.ws_planner_js_pub = self.create_publisher(JointState, "/ws_planner/joint_states", 10)
        self.marker_pub = self.create_publisher(MarkerArray, self.marker_topic, 10)

        self.progress_pub = self.create_publisher(Float32, self.progress_topic, 10)
        self.time_pub = self.create_publisher(Float32, self.time_topic, 10)
        self.cycle_pub = self.create_publisher(Float32, self.cycle_topic, 10)
        self.paused_pub = self.create_publisher(Bool, self.paused_topic, 10)

        qos = QoSProfile(depth=1)
        qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        qos.reliability = ReliabilityPolicy.RELIABLE
        self.waypoints_pub = self.create_publisher(String, self.waypoints_topic, qos)

        self.pause_sub = self.create_subscription(Bool, self.pause_topic, self._on_pause_cmd, 10)
        self.seek_sub = self.create_subscription(Float32, self.seek_topic, self._on_seek_cmd, 10)

        self.static_tf = StaticTransformBroadcaster(self)
        self._publish_static_tf()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self._t_abs = 0.0
        self._last_tick = self.get_clock().now()
        self._paused = False
        self._cycle_time = float(
            max(
                self.origin_traj.total_time,
                self.ws_traj.total_time,
                self.ws_planner_traj.total_time,
            )
        )

        self._p0_xyz_origin_fixed: Optional[Tuple[float, float, float]] = None
        self._p0_xyz_ws_fixed: Optional[Tuple[float, float, float]] = None
        self._p0_xyz_ws_planner_fixed: Optional[Tuple[float, float, float]] = None
        self._marker_msg = self._make_marker_array()

        self._play_timer = self.create_timer(self.dt, self._on_timer)
        self._marker_timer = self.create_timer(1.0, self._on_marker_timer)
        self._try_p0_timer = self.create_timer(0.2, self._try_compute_p0)

        self._publish_waypoints_json()
        if self.waypoints_republish_period_s > 0.0:
            self._waypoints_timer = self.create_timer(self.waypoints_republish_period_s, self._publish_waypoints_json)

        self._publish_joint_state(self.origin_js_pub, self.origin_traj, 0)
        self._publish_joint_state(self.ws_js_pub, self.ws_traj, 0)
        self._publish_joint_state(self.ws_planner_js_pub, self.ws_planner_traj, 0)
        self.marker_pub.publish(self._marker_msg)
        self._publish_playback_state()

        self.get_logger().info(f"Using result_dir: {self.result_dir}")
        self.get_logger().info(
            f"{self.origin_label} total={self.origin_nominal_total_s:.6f}s, "
            f"{self.ws_label} total={self.ws_nominal_total_s:.6f}s, "
            f"{self.ws_planner_label} total={self.ws_planner_nominal_total_s:.6f}s, "
            f"sync_loop={self.sync_loop}"
        )

    def _rgba_param(self, name: str, default: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
        value = self.get_parameter(name).value
        try:
            if isinstance(value, (list, tuple)) and len(value) == 4:
                r, g, b, a = [float(v) for v in value]
                return (r, g, b, a)
        except Exception:
            pass
        return default

    def _publish_static_tf(self) -> None:
        msgs: List[TransformStamped] = []
        for prefix, offset in (
            (self.origin_prefix, self.origin_offset),
            (self.ws_prefix, self.ws_offset),
            (self.ws_planner_prefix, self.ws_planner_offset),
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

    def _compose_joint_state(self, traj: Trajectory, idx: int) -> JointState:
        idx = max(0, min(idx, len(traj.samples) - 1))
        q = traj.samples[idx]

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        names = list(traj.joint_names)
        pos = [float(v) for v in q]

        if self.publish_gripper and self.gripper_joint_names:
            for name, pos_value in zip(self.gripper_joint_names, self.gripper_joint_positions):
                if name not in names:
                    names.append(str(name))
                    pos.append(float(pos_value))

        msg.name = names
        msg.position = pos
        return msg

    def _publish_joint_state(self, pub, traj: Trajectory, idx: int) -> None:
        if not traj.samples:
            return
        pub.publish(self._compose_joint_state(traj, idx))

    def _title_text(self, label: str, total_time_s: float) -> str:
        return f"{label} {total_time_s:.3f}s"

    def _offset_xyz(
        self,
        target: dict,
        offset: Sequence[float],
    ) -> Tuple[float, float, float]:
        return (
            float(target["x"]) + float(offset[0]),
            float(target["y"]) + float(offset[1]),
            float(target["z"]) + float(offset[2]),
        )

    def _make_marker_array(self) -> MarkerArray:
        ma = MarkerArray()
        now = self.get_clock().now().to_msg()

        def add_text(ns: str, mid: int, xyz: Tuple[float, float, float], text: str, rgba, dz: float) -> None:
            marker = Marker()
            marker.header.stamp = now
            marker.header.frame_id = self.fixed_frame
            marker.ns = ns
            marker.id = mid
            marker.type = Marker.TEXT_VIEW_FACING
            marker.action = Marker.ADD
            marker.pose.position.x = float(xyz[0])
            marker.pose.position.y = float(xyz[1])
            marker.pose.position.z = float(xyz[2]) + float(dz)
            marker.pose.orientation.w = 1.0
            marker.scale.z = self.label_scale_z
            marker.color.r, marker.color.g, marker.color.b, marker.color.a = rgba
            marker.text = text
            ma.markers.append(marker)

        def add_sphere(ns: str, mid: int, xyz: Tuple[float, float, float], rgba, scale: float) -> None:
            marker = Marker()
            marker.header.stamp = now
            marker.header.frame_id = self.fixed_frame
            marker.ns = ns
            marker.id = mid
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = float(xyz[0])
            marker.pose.position.y = float(xyz[1])
            marker.pose.position.z = float(xyz[2])
            marker.pose.orientation.w = 1.0
            marker.scale.x = scale
            marker.scale.y = scale
            marker.scale.z = scale
            marker.color.r, marker.color.g, marker.color.b, marker.color.a = rgba
            ma.markers.append(marker)

        def add_line_strip(ns: str, mid: int, xyz_list: List[Tuple[float, float, float]], rgba) -> None:
            if len(xyz_list) < 2:
                return
            marker = Marker()
            marker.header.stamp = now
            marker.header.frame_id = self.fixed_frame
            marker.ns = ns
            marker.id = mid
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.pose.orientation.w = 1.0
            marker.scale.x = self.line_width
            marker.color.r, marker.color.g, marker.color.b, marker.color.a = rgba
            for xyz in xyz_list:
                point = Point()
                point.x = float(xyz[0])
                point.y = float(xyz[1])
                point.z = float(xyz[2])
                marker.points.append(point)
            ma.markers.append(marker)

        add_text(
            "ik_origin_titles",
            1,
            (float(self.origin_offset[0]), float(self.origin_offset[1]), float(self.origin_offset[2]) + 0.9),
            self._title_text(self.origin_label, self.origin_nominal_total_s),
            self.origin_rgba,
            0.0,
        )
        add_text(
            "ik_origin_titles",
            2,
            (float(self.ws_offset[0]), float(self.ws_offset[1]), float(self.ws_offset[2]) + 0.9),
            self._title_text(self.ws_label, self.ws_nominal_total_s),
            self.ws_rgba,
            0.0,
        )
        add_text(
            "ik_origin_titles",
            3,
            (
                float(self.ws_planner_offset[0]),
                float(self.ws_planner_offset[1]),
                float(self.ws_planner_offset[2]) + 0.9,
            ),
            self._title_text(self.ws_planner_label, self.ws_planner_nominal_total_s),
            self.ws_planner_rgba,
            0.0,
        )

        def add_path_markers(
            *,
            prefix: str,
            base_id: int,
            order: List[str],
            offset: Sequence[float],
            rgba,
            p0_xyz: Optional[Tuple[float, float, float]],
        ) -> None:
            points_for_line: List[Tuple[float, float, float]] = []
            if p0_xyz is not None:
                add_sphere(f"ik_origin_{prefix}_p0", base_id, p0_xyz, self.point_rgba, self.point_scale * 1.1)
                add_text(f"ik_origin_{prefix}_labels", base_id + 1000, p0_xyz, "p0", self.point_rgba, self.label_dz)
                points_for_line.append(p0_xyz)

            for idx, point_name in enumerate(order, start=1):
                target = self.targets_by_name.get(point_name)
                if not target:
                    continue
                xyz = self._offset_xyz(target, offset)
                add_sphere(f"ik_origin_{prefix}_points", base_id + idx, xyz, self.point_rgba, self.point_scale)
                add_text(
                    f"ik_origin_{prefix}_labels",
                    base_id + 1000 + idx,
                    xyz,
                    point_name,
                    self.point_rgba,
                    self.label_dz,
                )
                points_for_line.append(xyz)

            add_line_strip(f"ik_origin_{prefix}_line", base_id + 5000, points_for_line, rgba)

        add_path_markers(
            prefix="origin",
            base_id=10000,
            order=self.origin_order,
            offset=self.origin_offset,
            rgba=self.origin_rgba,
            p0_xyz=self._p0_xyz_origin_fixed,
        )
        add_path_markers(
            prefix="ws",
            base_id=20000,
            order=self.ws_order,
            offset=self.ws_offset,
            rgba=self.ws_rgba,
            p0_xyz=self._p0_xyz_ws_fixed,
        )
        add_path_markers(
            prefix="ws_planner",
            base_id=30000,
            order=self.ws_planner_order,
            offset=self.ws_planner_offset,
            rgba=self.ws_planner_rgba,
            p0_xyz=self._p0_xyz_ws_planner_fixed,
        )

        return ma

    def _ui_time_in_cycle(self) -> float:
        cycle = float(self._cycle_time)
        if cycle <= 1e-6:
            return 0.0
        if self.loop:
            t_mod = float(self._t_abs % cycle)
            if self._t_abs > 1e-6 and t_mod < 1e-6:
                return float(cycle)
            return t_mod
        return float(min(self._t_abs, cycle))

    def _publish_playback_state(self) -> None:
        cycle = float(self._cycle_time)
        t_ui = self._ui_time_in_cycle()
        progress = 0.0 if cycle <= 1e-6 else float(t_ui / cycle)

        msg_progress = Float32()
        msg_progress.data = float(progress)
        self.progress_pub.publish(msg_progress)

        msg_time = Float32()
        msg_time.data = float(t_ui)
        self.time_pub.publish(msg_time)

        msg_cycle = Float32()
        msg_cycle.data = float(cycle)
        self.cycle_pub.publish(msg_cycle)

        msg_paused = Bool()
        msg_paused.data = bool(self._paused)
        self.paused_pub.publish(msg_paused)

    def _publish_waypoints_json(self) -> None:
        def build(segments: List[Segment], order: List[str]) -> List[dict]:
            out: List[dict] = [{"name": "p0", "t": 0.0}]
            t = 0.0
            for seg, point_name in zip(segments, order):
                t += float(seg.duration)
                out.append({"name": str(point_name), "t": float(t)})
                if self.hold_time_s > 0.0:
                    t += float(self.hold_time_s)
            return out

        payload = {
            "cycle_s": float(self._cycle_time),
            "origin": build(self.origin_segments, self.origin_order),
            "ws": build(self.ws_segments, self.ws_order),
            "ws_planner": build(self.ws_planner_segments, self.ws_planner_order),
            "window_size": int(self.window_size),
            "planner_window_size": int(self.planner_window_size),
        }

        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.waypoints_pub.publish(msg)

    def _on_pause_cmd(self, msg: Bool) -> None:
        self._paused = bool(msg.data)
        self._last_tick = self.get_clock().now()
        self._publish_playback_state()

    def _on_seek_cmd(self, msg: Float32) -> None:
        cycle = float(self._cycle_time)
        if cycle <= 1e-6:
            return
        progress = max(0.0, min(1.0, float(msg.data)))
        self._t_abs = progress * cycle
        self._last_tick = self.get_clock().now()
        self._publish_playback_state()

    def _on_timer(self) -> None:
        now = self.get_clock().now()
        dt_wall = (now - self._last_tick).nanoseconds * 1e-9
        if dt_wall < 0.0:
            dt_wall = 0.0
        self._last_tick = now

        if not self._paused:
            self._t_abs += dt_wall * self.speed_scale

        origin_total = float(self.origin_traj.total_time)
        ws_total = float(self.ws_traj.total_time)
        ws_planner_total = float(self.ws_planner_traj.total_time)

        if self.loop:
            if self.sync_loop:
                t_cycle = self._ui_time_in_cycle()
                t_origin = min(t_cycle, origin_total)
                t_ws = min(t_cycle, ws_total)
                t_ws_planner = min(t_cycle, ws_planner_total)
            else:
                t = float(self._t_abs)
                t_origin = (t % origin_total) if origin_total > 1e-6 else 0.0
                t_ws = (t % ws_total) if ws_total > 1e-6 else 0.0
                t_ws_planner = (t % ws_planner_total) if ws_planner_total > 1e-6 else 0.0
        else:
            t = float(self._t_abs)
            t_origin = min(t, origin_total)
            t_ws = min(t, ws_total)
            t_ws_planner = min(t, ws_planner_total)

        if self.origin_traj.samples:
            self._publish_joint_state(self.origin_js_pub, self.origin_traj, int(t_origin / self.origin_traj.dt))
        if self.ws_traj.samples:
            self._publish_joint_state(self.ws_js_pub, self.ws_traj, int(t_ws / self.ws_traj.dt))
        if self.ws_planner_traj.samples:
            self._publish_joint_state(
                self.ws_planner_js_pub,
                self.ws_planner_traj,
                int(t_ws_planner / self.ws_planner_traj.dt),
            )

        self._publish_playback_state()

    def _on_marker_timer(self) -> None:
        now = self.get_clock().now().to_msg()
        for marker in self._marker_msg.markers:
            marker.header.stamp = now
        self.marker_pub.publish(self._marker_msg)

    def _try_compute_p0(self) -> None:
        if (
            self._p0_xyz_origin_fixed is not None
            and self._p0_xyz_ws_fixed is not None
            and self._p0_xyz_ws_planner_fixed is not None
        ):
            self._try_p0_timer.cancel()
            return

        prev_paused = bool(self._paused)
        prev_t_abs = float(self._t_abs)
        self._paused = True
        self._t_abs = 0.0

        try:
            self._publish_joint_state(self.origin_js_pub, self.origin_traj, 0)
            self._publish_joint_state(self.ws_js_pub, self.ws_traj, 0)
            self._publish_joint_state(self.ws_planner_js_pub, self.ws_planner_traj, 0)

            updated = False
            if self._p0_xyz_origin_fixed is None:
                try:
                    tf = self.tf_buffer.lookup_transform(
                        self.fixed_frame,
                        f"{self.origin_prefix}{self.ee_link}",
                        rclpy.time.Time(),
                    )
                    tr = tf.transform.translation
                    self._p0_xyz_origin_fixed = (float(tr.x), float(tr.y), float(tr.z))
                    updated = True
                except Exception:
                    pass

            if self._p0_xyz_ws_fixed is None:
                try:
                    tf = self.tf_buffer.lookup_transform(
                        self.fixed_frame,
                        f"{self.ws_prefix}{self.ee_link}",
                        rclpy.time.Time(),
                    )
                    tr = tf.transform.translation
                    self._p0_xyz_ws_fixed = (float(tr.x), float(tr.y), float(tr.z))
                    updated = True
                except Exception:
                    pass

            if self._p0_xyz_ws_planner_fixed is None:
                try:
                    tf = self.tf_buffer.lookup_transform(
                        self.fixed_frame,
                        f"{self.ws_planner_prefix}{self.ee_link}",
                        rclpy.time.Time(),
                    )
                    tr = tf.transform.translation
                    self._p0_xyz_ws_planner_fixed = (float(tr.x), float(tr.y), float(tr.z))
                    updated = True
                except Exception:
                    pass

            if updated:
                self._marker_msg = self._make_marker_array()
                self.marker_pub.publish(self._marker_msg)

            if (
                self._p0_xyz_origin_fixed is not None
                and self._p0_xyz_ws_fixed is not None
                and self._p0_xyz_ws_planner_fixed is not None
            ):
                self._try_p0_timer.cancel()
        finally:
            self._paused = prev_paused
            self._t_abs = prev_t_abs
            self._last_tick = self.get_clock().now()


def main(args: Optional[List[str]] = None) -> None:
    rclpy.init(args=args)
    node = IKOriginComparePlayer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
