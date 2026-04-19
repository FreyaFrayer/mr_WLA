#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Play one collision pair (q_cur -> q_cand_next) in RViz with trapezoid timing.

This node:
- Loads dataset_ws*.npz and collision_pairs.json
- Picks one pair (i, k)
- Builds synchronized trapezoid joint trajectory samples
- Publishes /collision_pair/joint_states in a loop for RViz observation
- Publishes static TF: fixed_frame -> <robot_prefix><root_link>
"""

from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from sensor_msgs.msg import JointState
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster

from panda_ik_window.collision.dataset_loader import resolve_dataset_npz
from panda_ik_window.collision.trapezoid_sampler import sample_synchronized_trapezoid_segment


def _read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_prefix(prefix: str) -> str:
    p = str(prefix).strip()
    if not p:
        return ""
    return p if p.endswith("/") else (p + "/")


class CollisionPairPlayer(Node):
    def __init__(self) -> None:
        super().__init__("collision_pair_player")

        # Inputs
        self.declare_parameter("dataset", "dataset_ws3_top50_sort50")
        self.declare_parameter("preferred_ws", 3)
        self.declare_parameter("collision_pairs_json", "")
        self.declare_parameter("pair_index", 0)
        self.declare_parameter("sample_i", -1)
        self.declare_parameter("candidate_k", -1)

        # Trajectory / playback
        self.declare_parameter("sample_dt", 0.02)
        self.declare_parameter("min_samples", 5)
        self.declare_parameter("publish_rate_hz", 60.0)
        self.declare_parameter("speed_scale", 1.0)
        self.declare_parameter("loop", True)
        self.declare_parameter("hold_start_s", 0.25)
        self.declare_parameter("hold_end_s", 0.50)

        # Frames / robot layout
        self.declare_parameter("fixed_frame", "world")
        self.declare_parameter("root_link", "panda_link0")
        self.declare_parameter("robot_prefix", "collision_pair_/")
        self.declare_parameter("base_offset_xyz", [0.0, 0.0, 0.0])

        # Joint names
        self.declare_parameter("joint_names", [])

        # Optional gripper
        self.declare_parameter("publish_gripper", True)
        self.declare_parameter("gripper_joint_names", ["panda_finger_joint1", "panda_finger_joint2"])
        self.declare_parameter("gripper_joint_positions", [0.04, 0.04])

        dataset = str(self.get_parameter("dataset").value)
        preferred_ws = int(self.get_parameter("preferred_ws").value)
        pair_index = int(self.get_parameter("pair_index").value)
        sample_i = int(self.get_parameter("sample_i").value)
        candidate_k = int(self.get_parameter("candidate_k").value)

        self.sample_dt = float(self.get_parameter("sample_dt").value)
        self.min_samples = int(self.get_parameter("min_samples").value)

        publish_rate_hz = max(float(self.get_parameter("publish_rate_hz").value), 1.0)
        self.speed_scale = max(float(self.get_parameter("speed_scale").value), 1e-6)
        self.loop = bool(self.get_parameter("loop").value)
        self.hold_start_s = max(float(self.get_parameter("hold_start_s").value), 0.0)
        self.hold_end_s = max(float(self.get_parameter("hold_end_s").value), 0.0)

        self.fixed_frame = str(self.get_parameter("fixed_frame").value)
        self.root_link = str(self.get_parameter("root_link").value)
        self.robot_prefix = _normalize_prefix(str(self.get_parameter("robot_prefix").value))
        self.base_offset = list(self.get_parameter("base_offset_xyz").value)
        if len(self.base_offset) != 3:
            self.base_offset = [0.0, 0.0, 0.0]

        # Resolve dataset and pair
        ds = resolve_dataset_npz(dataset, preferred_ws=preferred_ws)
        self.get_logger().info(f"Using dataset npz: {ds.npz_path}")

        pair_meta = None
        if sample_i >= 0 and candidate_k >= 0:
            i = int(sample_i)
            k = int(candidate_k)
            self.get_logger().info(f"Pair selected from params: i={i}, k={k}")
        else:
            cp_json = str(self.get_parameter("collision_pairs_json").value).strip()
            if not cp_json:
                cp_json = os.path.join(ds.dataset_dir, "collision_pairs.json")

            if not os.path.isfile(cp_json):
                raise FileNotFoundError(
                    f"collision_pairs.json not found: {cp_json}. "
                    "Set --ros-args -p collision_pairs_json:=<path> or provide sample_i/candidate_k."
                )

            data = _read_json(cp_json)
            pairs = data.get("pairs", []) if isinstance(data, dict) else []
            if not isinstance(pairs, list) or not pairs:
                raise RuntimeError(f"No pairs in collision_pairs.json: {cp_json}")

            if pair_index < 0 or pair_index >= len(pairs):
                raise IndexError(f"pair_index={pair_index} out of range [0, {len(pairs) - 1}]")

            pair_meta = pairs[pair_index]
            i = int(pair_meta.get("i", -1))
            k = int(pair_meta.get("k", -1))
            if i < 0 or k < 0:
                raise RuntimeError(f"Invalid pair at index {pair_index}: {pair_meta}")

            self.get_logger().info(
                f"Pair selected from collision_pairs.json: pair_index={pair_index}, i={i}, k={k}"
            )

        with np.load(ds.npz_path, allow_pickle=False, mmap_mode="r") as arr:
            q_cur = arr["q_cur"]
            q_cand_fut = arr["q_cand_fut"]
            cand_mask = arr["cand_mask"]

            n = int(q_cur.shape[0])
            kmax = int(q_cand_fut.shape[2])
            dof = int(q_cur.shape[1])

            if i < 0 or i >= n:
                raise IndexError(f"sample i={i} out of range [0, {n - 1}]")
            if k < 0 or k >= kmax:
                raise IndexError(f"candidate k={k} out of range [0, {kmax - 1}]")

            if not bool(cand_mask[i, 0, k]):
                self.get_logger().warn(
                    f"Selected pair (i={i},k={k}) is masked invalid in cand_mask[:,0,:]. Will still play raw q."
                )

            self.q_start = np.asarray(q_cur[i], dtype=float)
            self.q_end = np.asarray(q_cand_fut[i, 0, k], dtype=float)

        # Joint names
        names_param = list(self.get_parameter("joint_names").value)
        if names_param:
            self.arm_joint_names = [str(x) for x in names_param]
        else:
            self.arm_joint_names = [f"panda_joint{j}" for j in range(1, dof + 1)]

        if len(self.arm_joint_names) != len(self.q_start):
            raise RuntimeError(
                f"joint_names length mismatch: names={len(self.arm_joint_names)} dof={len(self.q_start)}"
            )

        # Trapezoid trajectory generation
        vmax = np.array([1.0] * len(self.q_start), dtype=float)
        amax = np.array([1.0] * len(self.q_start), dtype=float)

        self.times_s, self.q_samples, self.duration_s = sample_synchronized_trapezoid_segment(
            self.q_start,
            self.q_end,
            max_vel_rad_s=vmax,
            max_acc_rad_s2=amax,
            sample_dt=self.sample_dt,
            min_samples=self.min_samples,
        )

        self.cycle_s = float(self.hold_start_s + self.duration_s + self.hold_end_s)

        # Optional gripper
        self.publish_gripper = bool(self.get_parameter("publish_gripper").value)
        self.gripper_joint_names = [str(x) for x in list(self.get_parameter("gripper_joint_names").value)]
        self.gripper_joint_positions = [float(x) for x in list(self.get_parameter("gripper_joint_positions").value)]

        if self.publish_gripper and len(self.gripper_joint_names) != len(self.gripper_joint_positions):
            raise RuntimeError(
                "gripper_joint_names and gripper_joint_positions length mismatch: "
                f"{len(self.gripper_joint_names)} vs {len(self.gripper_joint_positions)}"
            )

        # Publishers
        self.js_pub = self.create_publisher(JointState, "/collision_pair/joint_states", 10)

        # Static TF
        self._static_tf = StaticTransformBroadcaster(self)
        self._publish_static_tf()

        # Playback clock
        self._sim_t = 0.0
        self._last_wall_ns: Optional[int] = None

        self.timer = self.create_timer(1.0 / publish_rate_hz, self._on_timer)

        self.get_logger().info(
            "Playback ready: i={} k={} duration_s={:.3f} cycle_s={:.3f} samples={} loop={} speed_scale={:.3f}".format(
                i,
                k,
                float(self.duration_s),
                float(self.cycle_s),
                int(len(self.times_s)),
                bool(self.loop),
                float(self.speed_scale),
            )
        )

        if pair_meta is not None:
            self.get_logger().info(f"collision pair meta: {pair_meta}")

    def _publish_static_tf(self) -> None:
        tf = TransformStamped()
        tf.header.stamp = self.get_clock().now().to_msg()
        tf.header.frame_id = self.fixed_frame
        tf.child_frame_id = f"{self.robot_prefix}{self.root_link}"
        tf.transform.translation.x = float(self.base_offset[0])
        tf.transform.translation.y = float(self.base_offset[1])
        tf.transform.translation.z = float(self.base_offset[2])
        tf.transform.rotation.x = 0.0
        tf.transform.rotation.y = 0.0
        tf.transform.rotation.z = 0.0
        tf.transform.rotation.w = 1.0
        self._static_tf.sendTransform(tf)

    def _evaluate_q(self, sim_t: float) -> np.ndarray:
        if self.duration_s <= 1e-12 or len(self.times_s) == 0:
            return np.asarray(self.q_end, dtype=float)

        if sim_t <= self.hold_start_s:
            return np.asarray(self.q_start, dtype=float)

        if sim_t >= (self.hold_start_s + self.duration_s):
            return np.asarray(self.q_end, dtype=float)

        t_local = float(sim_t - self.hold_start_s)

        idx = int(np.searchsorted(self.times_s, t_local, side="right"))
        if idx <= 0:
            return np.asarray(self.q_samples[0], dtype=float)
        if idx >= len(self.times_s):
            return np.asarray(self.q_samples[-1], dtype=float)

        t0 = float(self.times_s[idx - 1])
        t1 = float(self.times_s[idx])
        q0 = np.asarray(self.q_samples[idx - 1], dtype=float)
        q1 = np.asarray(self.q_samples[idx], dtype=float)

        if t1 <= t0 + 1e-12:
            return q1

        alpha = (t_local - t0) / (t1 - t0)
        alpha = max(0.0, min(1.0, float(alpha)))
        return q0 + alpha * (q1 - q0)

    def _on_timer(self) -> None:
        now = self.get_clock().now()
        now_ns = int(now.nanoseconds)

        if self._last_wall_ns is None:
            self._last_wall_ns = now_ns
        else:
            dt_wall = max(0.0, float(now_ns - self._last_wall_ns) * 1e-9)
            self._last_wall_ns = now_ns
            self._sim_t += dt_wall * self.speed_scale

        if self.cycle_s > 1e-12:
            if self.loop:
                while self._sim_t >= self.cycle_s:
                    self._sim_t -= self.cycle_s
            else:
                self._sim_t = min(self._sim_t, self.cycle_s)

        q = self._evaluate_q(self._sim_t)

        msg = JointState()
        msg.header.stamp = now.to_msg()
        msg.name = list(self.arm_joint_names)
        msg.position = [float(v) for v in q.tolist()]

        if self.publish_gripper:
            msg.name.extend(self.gripper_joint_names)
            msg.position.extend(self.gripper_joint_positions)

        self.js_pub.publish(msg)


def main() -> None:
    rclpy.init()
    node = None
    try:
        node = CollisionPairPlayer()
        rclpy.spin(node)
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
