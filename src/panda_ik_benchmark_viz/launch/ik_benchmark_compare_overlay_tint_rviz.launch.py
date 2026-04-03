#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Overlay + tint RViz visualization .

- Two Panda robots are shown *overlapped* (same base pose) for direct visual comparison.
- global robot: deep blue tint
- greedy robot: light yellow tint
- Path points are shown once (white p0..pN), plus two colored line strips indicating
  the visiting order for global/greedy.

If using the control version, the RViz config includes a custom RViz Panel:
  panda_ik_benchmark_viz_rviz_plugins/IKBenchmarkPlaybackPanel
"""

from __future__ import annotations

import os
import subprocess
import xml.etree.ElementTree as ET
from typing import List, Tuple

from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _pick_panda_description_file() -> str:
    """Try common locations for Panda URDF/XACRO."""
    try:
        pkg = get_package_share_directory("moveit_resources_panda_description")
        for rel in ("urdf/panda.urdf.xacro", "urdf/panda.urdf"):
            p = os.path.join(pkg, rel)
            if os.path.exists(p):
                return p
    except PackageNotFoundError:
        pass

    try:
        pkg = get_package_share_directory("franka_description")
        candidates = [
            "robots/panda/panda.urdf.xacro",
            "robots/panda_arm_hand.urdf.xacro",
            "robots/panda_arm.urdf.xacro",
            "robots/panda/panda.urdf",
        ]
        for rel in candidates:
            p = os.path.join(pkg, rel)
            if os.path.exists(p):
                return p
    except PackageNotFoundError:
        pass

    raise RuntimeError(
        "Could not locate a Panda URDF/XACRO. Install either moveit_resources_panda_description or franka_description, "
        "or edit this launch file to point to your Panda URDF."
    )


def _load_urdf_from_file(desc_file: str) -> str:
    """Return a URDF XML string. If given a .xacro, it will be expanded."""
    if desc_file.endswith(".xacro"):
        try:
            import xacro  # type: ignore

            doc = xacro.process_file(desc_file)
            return doc.toxml()
        except Exception:
            out = subprocess.check_output(["xacro", desc_file], text=True)
            return out

    with open(desc_file, "r", encoding="utf-8") as f:
        return f.read()


def _tint_urdf_visuals(urdf_xml: str, rgba: Tuple[float, float, float, float], tint_name: str) -> str:
    """Force all <visual> materials to a single RGBA (may be ignored by .dae embedded materials)."""
    r, g, b, a = [float(x) for x in rgba]
    rgba_str = f"{r:.4f} {g:.4f} {b:.4f} {a:.4f}"

    root = ET.fromstring(urdf_xml)

    for link in root.findall("link"):
        link_name = link.get("name", "link")
        visuals = link.findall("visual")
        for vi, visual in enumerate(visuals):
            mat = visual.find("material")
            if mat is None:
                mat = ET.SubElement(visual, "material")

            mat.set("name", f"{tint_name}_{link_name}_{vi}")
            for child in list(mat):
                mat.remove(child)
            color = ET.SubElement(mat, "color")
            color.set("rgba", rgba_str)

    return ET.tostring(root, encoding="unicode")


def _launch_setup(context, *args, **kwargs) -> List:
    result_dir = LaunchConfiguration("result_dir").perform(context)
    fixed_frame = LaunchConfiguration("fixed_frame").perform(context)

    desc_file = _pick_panda_description_file()
    base_urdf = _load_urdf_from_file(desc_file)

    # Colors:
    global_tint = (0.05, 0.15, 0.80, 0.80)  # deep blue
    greedy_tint = (1.00, 0.95, 0.30, 0.55)  # light yellow

    # Optional tinted URDF strings (RobotModel sometimes ignores this for .dae meshes)
    global_urdf = _tint_urdf_visuals(base_urdf, global_tint, tint_name="tint_global")
    greedy_urdf = _tint_urdf_visuals(base_urdf, greedy_tint, tint_name="tint_greedy")

    rsp_greedy = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        namespace="greedy",
        name="robot_state_publisher",
        output="screen",
        parameters=[
            {"robot_description": greedy_urdf},
            {"frame_prefix": "greedy_/"},
            {"publish_frequency": 50.0},
        ],
    )

    rsp_global = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        namespace="global",
        name="robot_state_publisher",
        output="screen",
        parameters=[
            {"robot_description": global_urdf},
            {"frame_prefix": "global_/"},
            {"publish_frequency": 50.0},
        ],
    )

    greedy_mesh = Node(
        package="panda_ik_benchmark_viz",
        executable="urdf_mesh_tint_markers",
        namespace="greedy",
        name="urdf_mesh_tint_markers",
        output="screen",
        parameters=[
            {"robot_description": base_urdf},
            {"frame_prefix": "greedy_/"},
            {"rgba": list(greedy_tint)},
            {"marker_ns": "greedy_robot"},
            {"marker_topic": "mesh_markers"},
            {"republish_period_s": 5.0},
        ],
    )

    global_mesh = Node(
        package="panda_ik_benchmark_viz",
        executable="urdf_mesh_tint_markers",
        namespace="global",
        name="urdf_mesh_tint_markers",
        output="screen",
        parameters=[
            {"robot_description": base_urdf},
            {"frame_prefix": "global_/"},
            {"rgba": list(global_tint)},
            {"marker_ns": "global_robot"},
            {"marker_topic": "mesh_markers"},
            {"republish_period_s": 5.0},
        ],
    )

    player = Node(
        package="panda_ik_benchmark_viz",
        executable="ik_benchmark_compare_player",
        name="ik_benchmark_compare_player",
        output="screen",
        parameters=[
            {"result_dir": result_dir},
            {"fixed_frame": fixed_frame},
            {"greedy_prefix": "greedy_/"},
            {"global_prefix": "global_/"},

            # Overlapped bases
            {"greedy_offset_xyz": [0.0, 0.0, 0.0]},
            {"global_offset_xyz": [0.0, 0.0, 0.0]},

            # Looping behavior
            {"loop": True},
            {"sync_loop": True},

            # Clean markers: one shared white set + two order lines
            {"marker_mode": "shared"},
            {"global_rgba": list(global_tint)},
            {"greedy_rgba": list(greedy_tint)},
            {"shared_point_rgba": [1.0, 1.0, 1.0, 1.0]},
            {"label_scale_z": 0.06},
            {"label_dz": 0.08},

            # Ensure gripper TF exists so the mesh marker display can show fingers
            {"publish_gripper": True},
            {"gripper_joint_names": ["panda_finger_joint1", "panda_finger_joint2"]},
            {"gripper_joint_positions": [0.04, 0.04]},
        ],
    )

    rviz_cfg = LaunchConfiguration("rviz_config").perform(context)
    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", rviz_cfg],
    )

    return [rsp_greedy, rsp_global, greedy_mesh, global_mesh, player, rviz]


def generate_launch_description():
    pkg_share = get_package_share_directory("panda_ik_benchmark_viz")
    default_rviz = os.path.join(pkg_share, "rviz", "ik_benchmark_compare_overlay_tint.rviz")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "result_dir",
                default_value="data",
                description="Benchmark output directory or its parent (e.g., data/ or data/20260121_083922).",
            ),
            DeclareLaunchArgument(
                "fixed_frame",
                default_value="world",
                description="RViz Fixed Frame and the parent frame for the two robot base offsets.",
            ),
            DeclareLaunchArgument(
                "rviz_config",
                default_value=default_rviz,
                description="RViz2 config file.",
            ),
            OpaqueFunction(function=_launch_setup),
        ]
    )
