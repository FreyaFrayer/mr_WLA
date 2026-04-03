#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Publish a tinted mesh MarkerArray for a URDF.

Why this node exists
--------------------
RViz2's RobotModel display typically renders Panda visuals from Collada (.dae) meshes.
Those meshes often contain embedded materials, and RViz will prefer the embedded
materials over URDF <material><color .../> overrides. As a result, trying to "tint"
the robot by injecting URDF colors may have no visible effect.

This node solves it robustly by publishing visualization markers for each URDF visual:
  - For meshes: Marker.MESH_RESOURCE with mesh_use_embedded_materials = False
  - For primitive shapes: cube/cylinder/sphere markers

Because we force mesh_use_embedded_materials=False, the marker RGBA is always used,
regardless of what the .dae contains.

The markers are frame-locked to each link TF, so they move with joint_states/TF.
"""

from __future__ import annotations

import math
import os
import xml.etree.ElementTree as ET
from typing import List, Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

from geometry_msgs.msg import Quaternion
from visualization_msgs.msg import Marker, MarkerArray

from ament_index_python.packages import PackageNotFoundError, get_package_share_directory


def _parse_floats(s: Optional[str], n: int, default: float = 0.0) -> List[float]:
    if not s:
        return [default] * n
    parts = [p for p in s.replace(",", " ").split() if p]
    vals: List[float] = []
    for i in range(n):
        try:
            vals.append(float(parts[i]))
        except Exception:
            vals.append(default)
    return vals


def _rpy_to_quat(roll: float, pitch: float, yaw: float) -> Quaternion:
    """Convert roll-pitch-yaw to quaternion (x,y,z,w)."""
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)

    q = Quaternion()
    q.w = cr * cp * cy + sr * sp * sy
    q.x = sr * cp * cy - cr * sp * sy
    q.y = cr * sp * cy + sr * cp * sy
    q.z = cr * cp * sy - sr * sp * cy
    return q


def _get_child(elem: ET.Element, tag: str) -> Optional[ET.Element]:
    for c in list(elem):
        if c.tag == tag:
            return c
    return None


def _resolve_mesh_resource(uri: str) -> str:
    """Resolve a URDF mesh filename to something RViz can load.

    RViz usually supports package:// URIs via resource_retriever, but in some
    environments it can fail (or users forget to source the workspace where the
    description package is installed). To make visualization more robust, we try
    to resolve package://<pkg>/<relpath> into an absolute file:// URI.

    If resolution fails, we fall back to the original URI.
    """

    if not uri:
        return uri

    if uri.startswith("package://"):
        rest = uri[len("package://") :]
        parts = rest.split("/", 1)
        pkg = parts[0]
        rel = parts[1] if len(parts) == 2 else ""
        try:
            share = get_package_share_directory(pkg)
            abs_path = os.path.join(share, rel)
            if os.path.exists(abs_path):
                return f"file://{abs_path}"
        except PackageNotFoundError:
            return uri
        except Exception:
            return uri

    return uri


class URDFMeshTintMarkers(Node):
    def __init__(self) -> None:
        super().__init__("urdf_mesh_tint_markers")

        self.declare_parameter("robot_description", "")
        self.declare_parameter("frame_prefix", "")
        self.declare_parameter("rgba", [0.2, 0.8, 1.0, 0.7])
        self.declare_parameter("marker_ns", "robot_mesh")
        self.declare_parameter("marker_topic", "mesh_markers")
        self.declare_parameter("republish_period_s", 5.0)  # 0 => publish once

        urdf_xml = self.get_parameter("robot_description").get_parameter_value().string_value
        if not urdf_xml:
            raise RuntimeError("Parameter 'robot_description' is empty. Pass a URDF XML string.")

        self._prefix = self.get_parameter("frame_prefix").get_parameter_value().string_value
        rgba = self.get_parameter("rgba").get_parameter_value().double_array_value
        if len(rgba) != 4:
            rgba = [0.2, 0.8, 1.0, 0.7]
        self._rgba = [float(x) for x in rgba]

        self._marker_ns = self.get_parameter("marker_ns").get_parameter_value().string_value
        topic = self.get_parameter("marker_topic").get_parameter_value().string_value
        self._topic = topic

        qos = QoSProfile(depth=1)
        qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        qos.reliability = ReliabilityPolicy.RELIABLE
        self._pub = self.create_publisher(MarkerArray, topic, qos)

        self._markers = self._build_markers(urdf_xml)

        # Publish immediately (latched via transient local)
        self._publish()

        repub = float(self.get_parameter("republish_period_s").get_parameter_value().double_value)
        if repub > 0.0:
            self.create_timer(repub, self._publish)

        self.get_logger().info(
            f"Publishing {len(self._markers.markers)} tinted URDF visual markers on '{self._topic}' "
            f"with prefix='{self._prefix}' ns='{self._marker_ns}'."
        )

    def _publish(self) -> None:
        # Update stamps so RViz can resolve TF at 'now'.
        now = self.get_clock().now().to_msg()
        for m in self._markers.markers:
            m.header.stamp = now
        self._pub.publish(self._markers)

    def _frame(self, link_name: str) -> str:
        return f"{self._prefix}{link_name}" if self._prefix else link_name

    def _colorize(self, marker: Marker) -> None:
        marker.color.r = self._rgba[0]
        marker.color.g = self._rgba[1]
        marker.color.b = self._rgba[2]
        marker.color.a = self._rgba[3]

    def _build_markers(self, urdf_xml: str) -> MarkerArray:
        root = ET.fromstring(urdf_xml)
        out = MarkerArray()

        mid = 0

        for link in root.findall("link"):
            link_name = link.get("name")
            if not link_name:
                continue

            for vi, visual in enumerate(link.findall("visual")):
                geom = _get_child(visual, "geometry")
                if geom is None:
                    continue

                origin = _get_child(visual, "origin")
                xyz = _parse_floats(origin.get("xyz") if origin is not None else None, 3, 0.0)
                rpy = _parse_floats(origin.get("rpy") if origin is not None else None, 3, 0.0)

                pose_q = _rpy_to_quat(rpy[0], rpy[1], rpy[2])

                # --- Mesh ---
                mesh = _get_child(geom, "mesh")
                if mesh is not None:
                    filename = mesh.get("filename") or mesh.get("file")
                    if not filename:
                        continue
                    scale = _parse_floats(mesh.get("scale"), 3, 1.0)

                    m = Marker()
                    m.header.frame_id = self._frame(link_name)
                    m.ns = self._marker_ns
                    m.id = mid
                    mid += 1
                    m.type = Marker.MESH_RESOURCE
                    m.action = Marker.ADD
                    m.mesh_resource = _resolve_mesh_resource(filename)
                    m.mesh_use_embedded_materials = False  # <-- critical for "tint" to actually work
                    m.frame_locked = True
                    m.pose.position.x = xyz[0]
                    m.pose.position.y = xyz[1]
                    m.pose.position.z = xyz[2]
                    m.pose.orientation = pose_q
                    m.scale.x = scale[0]
                    m.scale.y = scale[1]
                    m.scale.z = scale[2]
                    self._colorize(m)
                    out.markers.append(m)
                    continue

                # --- Box ---
                box = _get_child(geom, "box")
                if box is not None:
                    size = _parse_floats(box.get("size"), 3, 0.1)
                    m = Marker()
                    m.header.frame_id = self._frame(link_name)
                    m.ns = self._marker_ns
                    m.id = mid
                    mid += 1
                    m.type = Marker.CUBE
                    m.action = Marker.ADD
                    m.frame_locked = True
                    m.pose.position.x = xyz[0]
                    m.pose.position.y = xyz[1]
                    m.pose.position.z = xyz[2]
                    m.pose.orientation = pose_q
                    m.scale.x = max(1e-6, size[0])
                    m.scale.y = max(1e-6, size[1])
                    m.scale.z = max(1e-6, size[2])
                    self._colorize(m)
                    out.markers.append(m)
                    continue

                # --- Cylinder ---
                cyl = _get_child(geom, "cylinder")
                if cyl is not None:
                    r = float(cyl.get("radius") or 0.05)
                    l = float(cyl.get("length") or 0.1)
                    m = Marker()
                    m.header.frame_id = self._frame(link_name)
                    m.ns = self._marker_ns
                    m.id = mid
                    mid += 1
                    m.type = Marker.CYLINDER
                    m.action = Marker.ADD
                    m.frame_locked = True
                    m.pose.position.x = xyz[0]
                    m.pose.position.y = xyz[1]
                    m.pose.position.z = xyz[2]
                    m.pose.orientation = pose_q
                    m.scale.x = max(1e-6, 2.0 * r)
                    m.scale.y = max(1e-6, 2.0 * r)
                    m.scale.z = max(1e-6, l)
                    self._colorize(m)
                    out.markers.append(m)
                    continue

                # --- Sphere ---
                sph = _get_child(geom, "sphere")
                if sph is not None:
                    r = float(sph.get("radius") or 0.05)
                    m = Marker()
                    m.header.frame_id = self._frame(link_name)
                    m.ns = self._marker_ns
                    m.id = mid
                    mid += 1
                    m.type = Marker.SPHERE
                    m.action = Marker.ADD
                    m.frame_locked = True
                    m.pose.position.x = xyz[0]
                    m.pose.position.y = xyz[1]
                    m.pose.position.z = xyz[2]
                    m.pose.orientation = pose_q
                    m.scale.x = max(1e-6, 2.0 * r)
                    m.scale.y = max(1e-6, 2.0 * r)
                    m.scale.z = max(1e-6, 2.0 * r)
                    self._colorize(m)
                    out.markers.append(m)
                    continue

        return out


def main(args=None) -> None:
    rclpy.init(args=args)
    node = URDFMeshTintMarkers()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
