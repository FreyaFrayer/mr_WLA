#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Compare two time-model selection results under one seed folder in RViz2.

Input layout example:
  seed11/
    model_totg/<timestamp>/summary.json
    model_trapezoid/<timestamp>/summary.json

This node resolves the latest run under each model directory, extracts one shared
window-size final_path, publishes two JointState streams, and overlays path markers.
"""

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
from sensor_msgs.msg import JointState
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray


def _read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _summary_pick_key(summary_path: str) -> Tuple[int, str, float]:
    run_dir = os.path.basename(os.path.dirname(summary_path))
    is_timestamp = 1 if run_dir.isdigit() else 0
    return (is_timestamp, run_dir, os.path.getmtime(summary_path))


def _resolve_model_result_dir(seed_dir: str, model_dir_name: str) -> str:
    seed_dir = os.path.abspath(os.path.expanduser(seed_dir))
    model_root = os.path.join(seed_dir, model_dir_name)

    if os.path.isfile(os.path.join(model_root, "summary.json")):
        return model_root

    if not os.path.isdir(model_root):
        raise FileNotFoundError(
            f"Model directory not found: {model_root}. "
            f"Expected structure: <seed_dir>/{model_dir_name}/<timestamp>/summary.json"
        )

    candidates = glob.glob(os.path.join(model_root, "*", "summary.json"))
    if not candidates:
        candidates = glob.glob(os.path.join(model_root, "**", "summary.json"), recursive=True)

    if not candidates:
        raise FileNotFoundError(f"No summary.json found under model directory: {model_root}")

    best = max(candidates, key=_summary_pick_key)
    return os.path.dirname(best)


def _extract_available_ws(summary: dict) -> List[int]:
    window = summary.get("window", {})
    if not isinstance(window, dict):
        return []

    out: List[int] = []

    by_ws = window.get("results_by_ws", {})
    if isinstance(by_ws, dict):
        for k in by_ws.keys():
            try:
                ws = int(k)
                if ws not in out:
                    out.append(ws)
            except Exception:
                continue

    results = window.get("results", [])
    if isinstance(results, list):
        for rec in results:
            if not isinstance(rec, dict):
                continue
            try:
                ws = int(rec.get("window_size", -1))
            except Exception:
                continue
            if ws > 0 and ws not in out:
                out.append(ws)

    out.sort()
    return out


def _pick_ws_data(summary: dict, ws: int) -> dict:
    window = summary.get("window", {})
    if not isinstance(window, dict):
        raise RuntimeError("summary.json missing window section")

    by_ws = window.get("results_by_ws", {})
    if isinstance(by_ws, dict) and by_ws:
        key = str(int(ws))
        if key in by_ws and isinstance(by_ws[key], dict):
            return by_ws[key]
        raise KeyError(
            f"window_size={ws} not found in window.results_by_ws; available={sorted(by_ws.keys())}"
        )

    results = window.get("results", [])
    if isinstance(results, list):
        for rec in results:
            if not isinstance(rec, dict):
                continue
            try:
                rec_ws = int(rec.get("window_size", -1))
            except Exception:
                continue
            if rec_ws == int(ws):
                return rec

        avail = sorted(
            int(rec.get("window_size", -1))
            for rec in results
            if isinstance(rec, dict) and "window_size" in rec
        )
        raise KeyError(f"window_size={ws} not found in window.results; available={avail}")

    raise RuntimeError("No valid window results found in summary.json")


def _find_joint_names(result_dir: str, fallback_count: int) -> List[str]:
    for p in sorted(glob.glob(os.path.join(result_dir, "p*.json"))):
        if os.path.basename(p) == "p0.json":
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

    n = max(1, int(fallback_count))
    return [f"panda_joint{i}" for i in range(1, n + 1)]


def _load_targets(result_dir: str, summary: dict) -> List[dict]:
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


def _model_label_from_summary(summary: dict, model_dir_name: str) -> str:
    meta = summary.get("meta", {})
    if isinstance(meta, dict):
        tm = meta.get("time_model", {})
        if isinstance(tm, dict):
            eff = str(tm.get("effective", "")).strip()
            if eff:
                return eff.upper()

    fallback = str(model_dir_name)
    if fallback.startswith("model_"):
        fallback = fallback[len("model_") :]
    return fallback.upper()


def _quintic_time_scaling(s: float) -> float:
    s = max(0.0, min(1.0, s))
    return 10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5


def _sort_point_name_key(name: str) -> Tuple[int, int, str]:
    if isinstance(name, str) and name.startswith("p"):
        try:
            return (0, int(name[1:]), name)
        except Exception:
            pass
    return (1, 0, str(name))


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


def _build_segments(summary: dict, ws: int) -> Tuple[List[float], List[Segment], List[str]]:
    start = summary.get("start", {})
    if not isinstance(start, dict):
        raise RuntimeError("summary.json missing start section")

    start_q_raw = start.get("joint_positions", [])
    if not isinstance(start_q_raw, list) or not start_q_raw:
        raise RuntimeError("summary.json missing start.joint_positions")

    start_q = [float(v) for v in start_q_raw]

    ws_data = _pick_ws_data(summary, ws)
    if not isinstance(ws_data, dict):
        raise RuntimeError(f"Invalid ws data for window_size={ws}")

    seg_dicts: List[dict] = []

    final_path = ws_data.get("final_path", {})
    if isinstance(final_path, dict):
        raw = final_path.get("segments", [])
        if isinstance(raw, list):
            seg_dicts = [x for x in raw if isinstance(x, dict)]

    if not seg_dicts:
        raw = ws_data.get("segments", [])
        if isinstance(raw, list):
            seg_dicts = [x for x in raw if isinstance(x, dict)]

    if not seg_dicts:
        raise RuntimeError(f"No segments found for window_size={ws}")

    q_prev = list(start_q)
    segments: List[Segment] = []
    order: List[str] = []

    for seg in seg_dicts:
        q_next_raw = seg.get("joint_positions", None)
        if not isinstance(q_next_raw, list) or len(q_next_raw) != len(q_prev):
            continue

        dur = float(seg.get("time_s", 0.0))
        if dur <= 0.0:
            dur = 1.0

        q_next = [float(v) for v in q_next_raw]
        to_name = str(seg.get("to", ""))

        segments.append(
            Segment(
                q0=list(q_prev),
                q1=q_next,
                duration=dur,
                to_name=to_name,
            )
        )

        q_prev = q_next
        if to_name:
            order.append(to_name)

    if not segments:
        raise RuntimeError(f"Failed to build valid segments for window_size={ws}")

    return start_q, segments, order


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

    samples: List[List[float]] = [list(start_q)]
    t_total = 0.0
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

        t_total += dur

        if hold_time_s > 0.0:
            n_hold = int(math.ceil(hold_time_s / dt))
            for _ in range(n_hold):
                samples.append(list(q_last))
            t_total += n_hold * dt

    return Trajectory(
        joint_names=list(joint_names),
        samples=samples,
        dt=float(dt),
        total_time=float(max(0.0, t_total)),
    )


class TimeModelComparePlayer(Node):
    def __init__(self) -> None:
        super().__init__("time_model_compare_player")

        self.declare_parameter("seed_dir", "data_window/batch_time_model_data/np3/seed11")
        self.declare_parameter("model_a_dir", "model_totg")
        self.declare_parameter("model_b_dir", "model_trapezoid")
        self.declare_parameter("window_size", -1)

        self.declare_parameter("fixed_frame", "world")
        self.declare_parameter("root_link", "panda_link0")
        self.declare_parameter("model_a_prefix", "model_a_/")
        self.declare_parameter("model_b_prefix", "model_b_/")
        self.declare_parameter("model_a_label", "")
        self.declare_parameter("model_b_label", "")
        self.declare_parameter("model_a_offset_xyz", [0.0, -0.6, 0.0])
        self.declare_parameter("model_b_offset_xyz", [0.0, 0.6, 0.0])

        self.declare_parameter("publish_rate_hz", 50.0)
        self.declare_parameter("hold_time_s", 0.4)
        self.declare_parameter("loop", True)
        self.declare_parameter("sync_loop", True)
        self.declare_parameter("speed_scale", 1.0)

        self.declare_parameter("publish_gripper", True)
        self.declare_parameter("gripper_joint_names", ["panda_finger_joint1", "panda_finger_joint2"])
        self.declare_parameter("gripper_joint_positions", [0.04, 0.04])

        self.declare_parameter("marker_topic", "/time_model_compare/markers")
        self.declare_parameter("model_a_rgba", [0.08, 0.27, 0.95, 0.95])
        self.declare_parameter("model_b_rgba", [1.00, 0.55, 0.10, 0.95])
        self.declare_parameter("point_scale", 0.05)
        self.declare_parameter("line_width", 0.012)
        self.declare_parameter("label_scale_z", 0.05)
        self.declare_parameter("label_dz", 0.07)

        self.seed_dir = str(self.get_parameter("seed_dir").value)
        self.model_a_dir = str(self.get_parameter("model_a_dir").value)
        self.model_b_dir = str(self.get_parameter("model_b_dir").value)

        self.fixed_frame = str(self.get_parameter("fixed_frame").value)
        self.root_link = str(self.get_parameter("root_link").value)
        self.model_a_prefix = str(self.get_parameter("model_a_prefix").value)
        self.model_b_prefix = str(self.get_parameter("model_b_prefix").value)

        self.model_a_offset = [float(v) for v in list(self.get_parameter("model_a_offset_xyz").value)]
        self.model_b_offset = [float(v) for v in list(self.get_parameter("model_b_offset_xyz").value)]

        rate_hz = max(1.0, float(self.get_parameter("publish_rate_hz").value))
        self.dt = 1.0 / rate_hz
        self.hold_time_s = max(0.0, float(self.get_parameter("hold_time_s").value))
        self.loop = bool(self.get_parameter("loop").value)
        self.sync_loop = bool(self.get_parameter("sync_loop").value)
        self.speed_scale = max(1e-6, float(self.get_parameter("speed_scale").value))

        self.publish_gripper = bool(self.get_parameter("publish_gripper").value)
        self.gripper_joint_names = [str(x) for x in list(self.get_parameter("gripper_joint_names").value)]
        self.gripper_joint_positions = [float(x) for x in list(self.get_parameter("gripper_joint_positions").value)]
        if len(self.gripper_joint_positions) != len(self.gripper_joint_names):
            self.gripper_joint_positions = [0.04] * len(self.gripper_joint_names)

        self.marker_topic = str(self.get_parameter("marker_topic").value)
        self.model_a_rgba = self._rgba_param("model_a_rgba", (0.08, 0.27, 0.95, 0.95))
        self.model_b_rgba = self._rgba_param("model_b_rgba", (1.00, 0.55, 0.10, 0.95))
        self.point_scale = float(self.get_parameter("point_scale").value)
        self.line_width = float(self.get_parameter("line_width").value)
        self.label_scale_z = float(self.get_parameter("label_scale_z").value)
        self.label_dz = float(self.get_parameter("label_dz").value)

        self.model_a_result_dir = _resolve_model_result_dir(self.seed_dir, self.model_a_dir)
        self.model_b_result_dir = _resolve_model_result_dir(self.seed_dir, self.model_b_dir)

        self.summary_a = _read_json(os.path.join(self.model_a_result_dir, "summary.json"))
        self.summary_b = _read_json(os.path.join(self.model_b_result_dir, "summary.json"))

        ws_param = int(self.get_parameter("window_size").value)
        avail_a = _extract_available_ws(self.summary_a)
        avail_b = _extract_available_ws(self.summary_b)

        if not avail_a:
            raise RuntimeError(f"No available window_size in {self.model_a_result_dir}/summary.json")
        if not avail_b:
            raise RuntimeError(f"No available window_size in {self.model_b_result_dir}/summary.json")

        if ws_param < 0:
            common = sorted(set(avail_a).intersection(avail_b))
            if not common:
                raise RuntimeError(
                    f"No common window_size between models: {self.model_a_dir}={avail_a}, {self.model_b_dir}={avail_b}"
                )
            self.ws = int(common[-1])
        else:
            self.ws = int(ws_param)
            if self.ws not in avail_a or self.ws not in avail_b:
                raise RuntimeError(
                    f"Requested window_size={self.ws} not available for both models: "
                    f"{self.model_a_dir}={avail_a}, {self.model_b_dir}={avail_b}"
                )

        start_a, segs_a, order_a = _build_segments(self.summary_a, self.ws)
        start_b, segs_b, order_b = _build_segments(self.summary_b, self.ws)

        names_a = _find_joint_names(self.model_a_result_dir, len(start_a))
        if len(names_a) != len(start_a):
            names_a = [f"panda_joint{i}" for i in range(1, len(start_a) + 1)]

        names_b = _find_joint_names(self.model_b_result_dir, len(start_b))
        if len(names_b) != len(start_b):
            names_b = [f"panda_joint{i}" for i in range(1, len(start_b) + 1)]

        self.model_a_traj = _segments_to_trajectory(
            segments=segs_a,
            joint_names=names_a,
            dt=self.dt,
            hold_time_s=self.hold_time_s,
            start_q=start_a,
        )
        self.model_b_traj = _segments_to_trajectory(
            segments=segs_b,
            joint_names=names_b,
            dt=self.dt,
            hold_time_s=self.hold_time_s,
            start_q=start_b,
        )

        self.model_a_order = list(order_a)
        self.model_b_order = list(order_b)

        model_a_label_in = str(self.get_parameter("model_a_label").value).strip()
        model_b_label_in = str(self.get_parameter("model_b_label").value).strip()
        self.model_a_label = model_a_label_in or _model_label_from_summary(self.summary_a, self.model_a_dir)
        self.model_b_label = model_b_label_in or _model_label_from_summary(self.summary_b, self.model_b_dir)

        self.targets_by_name: Dict[str, dict] = {}
        for t in _load_targets(self.model_a_result_dir, self.summary_a) + _load_targets(
            self.model_b_result_dir, self.summary_b
        ):
            if not isinstance(t, dict):
                continue
            if not all(k in t for k in ("name", "x", "y", "z")):
                continue
            name = str(t.get("name", ""))
            if not name:
                continue
            if name not in self.targets_by_name:
                self.targets_by_name[name] = t

        self.js_pub_a = self.create_publisher(JointState, "/model_a/joint_states", 10)
        self.js_pub_b = self.create_publisher(JointState, "/model_b/joint_states", 10)
        self.marker_pub = self.create_publisher(MarkerArray, self.marker_topic, 10)

        self.static_tf = StaticTransformBroadcaster(self)
        self._publish_static_tf()

        self._t_abs = 0.0
        self._last_tick = self.get_clock().now()
        self._cycle_time = float(max(self.model_a_traj.total_time, self.model_b_traj.total_time))

        self._marker_msg = self._make_marker_array()

        self._play_timer = self.create_timer(self.dt, self._on_timer)
        self._marker_timer = self.create_timer(1.0, self._on_marker_timer)

        self._publish_joint_state(self.js_pub_a, self.model_a_traj, 0)
        self._publish_joint_state(self.js_pub_b, self.model_b_traj, 0)
        self.marker_pub.publish(self._marker_msg)

        self.get_logger().info(
            f"seed_dir={os.path.abspath(os.path.expanduser(self.seed_dir))}, ws={self.ws}, "
            f"{self.model_a_label}={self.model_a_result_dir}, {self.model_b_label}={self.model_b_result_dir}"
        )
        self.get_logger().info(
            f"traj_time: {self.model_a_label}={self.model_a_traj.total_time:.4f}s, "
            f"{self.model_b_label}={self.model_b_traj.total_time:.4f}s, sync_loop={self.sync_loop}"
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
            (self.model_a_prefix, self.model_a_offset),
            (self.model_b_prefix, self.model_b_offset),
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

    def _ordered_names_for(self, order: Sequence[str]) -> List[str]:
        names = [n for n in order if n in self.targets_by_name]
        if names:
            return names
        return sorted(self.targets_by_name.keys(), key=_sort_point_name_key)

    def _make_marker_array(self) -> MarkerArray:
        ma = MarkerArray()
        now = self.get_clock().now().to_msg()

        clear = Marker()
        clear.header.stamp = now
        clear.header.frame_id = self.fixed_frame
        clear.action = Marker.DELETEALL
        ma.markers.append(clear)

        self._add_title_marker(
            ma=ma,
            now=now,
            marker_id=10,
            text=f"{self.model_a_label} (ws={self.ws})",
            xyz=(self.model_a_offset[0], self.model_a_offset[1], self.model_a_offset[2] + 0.9),
            rgba=self.model_a_rgba,
        )
        self._add_title_marker(
            ma=ma,
            now=now,
            marker_id=11,
            text=f"{self.model_b_label} (ws={self.ws})",
            xyz=(self.model_b_offset[0], self.model_b_offset[1], self.model_b_offset[2] + 0.9),
            rgba=self.model_b_rgba,
        )

        self._add_model_path_markers(
            ma=ma,
            now=now,
            prefix=self.model_a_prefix,
            ns_prefix="model_a",
            order=self._ordered_names_for(self.model_a_order),
            rgba=self.model_a_rgba,
            id_base=1000,
        )
        self._add_model_path_markers(
            ma=ma,
            now=now,
            prefix=self.model_b_prefix,
            ns_prefix="model_b",
            order=self._ordered_names_for(self.model_b_order),
            rgba=self.model_b_rgba,
            id_base=2000,
        )

        return ma

    def _add_title_marker(
        self,
        *,
        ma: MarkerArray,
        now,
        marker_id: int,
        text: str,
        xyz: Tuple[float, float, float],
        rgba: Tuple[float, float, float, float],
    ) -> None:
        m = Marker()
        m.header.stamp = now
        m.header.frame_id = self.fixed_frame
        m.ns = "time_model_title"
        m.id = int(marker_id)
        m.type = Marker.TEXT_VIEW_FACING
        m.action = Marker.ADD
        m.pose.position.x = float(xyz[0])
        m.pose.position.y = float(xyz[1])
        m.pose.position.z = float(xyz[2])
        m.pose.orientation.w = 1.0
        m.scale.z = 0.075
        m.color.r, m.color.g, m.color.b, m.color.a = rgba
        m.text = text
        ma.markers.append(m)

    def _add_model_path_markers(
        self,
        *,
        ma: MarkerArray,
        now,
        prefix: str,
        ns_prefix: str,
        order: Sequence[str],
        rgba: Tuple[float, float, float, float],
        id_base: int,
    ) -> None:
        frame = f"{prefix}{self.root_link}"
        line_points: List[Tuple[float, float, float]] = []

        for idx, name in enumerate(order, start=1):
            target = self.targets_by_name.get(name)
            if not target:
                continue

            xyz = (float(target["x"]), float(target["y"]), float(target["z"]))
            line_points.append(xyz)

            sphere = Marker()
            sphere.header.stamp = now
            sphere.header.frame_id = frame
            sphere.ns = f"{ns_prefix}_points"
            sphere.id = int(id_base + idx)
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD
            sphere.pose.position.x = xyz[0]
            sphere.pose.position.y = xyz[1]
            sphere.pose.position.z = xyz[2]
            sphere.pose.orientation.w = 1.0
            sphere.scale.x = self.point_scale
            sphere.scale.y = self.point_scale
            sphere.scale.z = self.point_scale
            sphere.color.r, sphere.color.g, sphere.color.b, sphere.color.a = rgba
            ma.markers.append(sphere)

            label = Marker()
            label.header.stamp = now
            label.header.frame_id = frame
            label.ns = f"{ns_prefix}_labels"
            label.id = int(id_base + 500 + idx)
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.position.x = xyz[0]
            label.pose.position.y = xyz[1]
            label.pose.position.z = xyz[2] + self.label_dz
            label.pose.orientation.w = 1.0
            label.scale.z = self.label_scale_z
            label.color.r, label.color.g, label.color.b, label.color.a = rgba
            label.text = f"{idx}:{name}"
            ma.markers.append(label)

        if len(line_points) >= 2:
            line = Marker()
            line.header.stamp = now
            line.header.frame_id = frame
            line.ns = f"{ns_prefix}_line"
            line.id = int(id_base + 9000)
            line.type = Marker.LINE_STRIP
            line.action = Marker.ADD
            line.pose.orientation.w = 1.0
            line.scale.x = self.line_width
            line.color.r, line.color.g, line.color.b, line.color.a = rgba

            for xyz in line_points:
                p = Point()
                p.x, p.y, p.z = xyz
                line.points.append(p)

            ma.markers.append(line)

    def _time_for_traj(self, total_s: float) -> float:
        if total_s <= 1e-9:
            return 0.0

        if self.loop:
            if self.sync_loop:
                if self._cycle_time <= 1e-9:
                    return 0.0
                t_cycle = self._t_abs % self._cycle_time
                return min(float(t_cycle), float(total_s))
            return float(self._t_abs % total_s)

        return min(float(self._t_abs), float(total_s))

    def _on_timer(self) -> None:
        now = self.get_clock().now()
        dt_s = (now - self._last_tick).nanoseconds * 1e-9
        self._last_tick = now

        if dt_s < 0.0 or not math.isfinite(dt_s):
            dt_s = self.dt

        self._t_abs += dt_s * self.speed_scale

        t_a = self._time_for_traj(self.model_a_traj.total_time)
        t_b = self._time_for_traj(self.model_b_traj.total_time)

        idx_a = int(t_a / self.model_a_traj.dt) if self.model_a_traj.samples else 0
        idx_b = int(t_b / self.model_b_traj.dt) if self.model_b_traj.samples else 0

        self._publish_joint_state(self.js_pub_a, self.model_a_traj, idx_a)
        self._publish_joint_state(self.js_pub_b, self.model_b_traj, idx_b)

    def _on_marker_timer(self) -> None:
        now = self.get_clock().now().to_msg()
        for m in self._marker_msg.markers:
            m.header.stamp = now
        self.marker_pub.publish(self._marker_msg)


def main(args: Optional[Sequence[str]] = None) -> None:
    rclpy.init(args=args)
    node = TimeModelComparePlayer()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
