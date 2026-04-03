#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from typing import List

from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import Command, LaunchConfiguration, FindExecutable
from launch_ros.actions import Node


def _pick_panda_description_file() -> str:
    """
    Try common locations for Panda URDF/XACRO.

    Priority:
      1) moveit_resources_panda_description (MoveIt2 resource package)
      2) franka_description (franka_ros)
    """
    # 1) MoveIt resources
    try:
        pkg = get_package_share_directory("moveit_resources_panda_description")
        for rel in ("urdf/panda.urdf.xacro", "urdf/panda.urdf"):
            p = os.path.join(pkg, rel)
            if os.path.exists(p):
                return p
    except PackageNotFoundError:
        pass

    # 2) Franka description
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


def _make_robot_description_param(desc_file: str):
    if desc_file.endswith(".xacro"):
        return {"robot_description": Command([FindExecutable(name="xacro"), " ", desc_file])}
    # URDF file
    with open(desc_file, "r", encoding="utf-8") as f:
        return {"robot_description": f.read()}


def _launch_setup(context, *args, **kwargs) -> List:
    result_dir = LaunchConfiguration("result_dir").perform(context)
    fixed_frame = LaunchConfiguration("fixed_frame").perform(context)

    desc_file = _pick_panda_description_file()
    robot_description = _make_robot_description_param(desc_file)

    # Two robot_state_publisher nodes with different namespaces + frame_prefix
    #
    # IMPORTANT:
    # RViz RobotModel's "TF Prefix" resolves link frames as: <prefix> + "/" + <link_name>
    # robot_state_publisher's "frame_prefix" simply prepends the string as-is.
    #
    # Therefore we include a trailing '/' in frame_prefix (e.g. "greedy_/"),
    # so the published TF frames match what RViz expects when TF Prefix is "greedy_".
    rsp_greedy = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        namespace="greedy",
        name="robot_state_publisher",
        output="screen",
        parameters=[
            robot_description,
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
            robot_description,
            {"frame_prefix": "global_/"},
            {"publish_frequency": 50.0},
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

    return [rsp_greedy, rsp_global, player, rviz]


def generate_launch_description():
    pkg_share = get_package_share_directory("panda_ik_benchmark_viz")
    default_rviz = os.path.join(pkg_share, "rviz", "ik_benchmark_compare.rviz")

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
