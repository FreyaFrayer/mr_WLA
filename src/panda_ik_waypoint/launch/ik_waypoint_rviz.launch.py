#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from typing import List

from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import Command, FindExecutable, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _pick_panda_description_file() -> str:
    """Try common Panda URDF/XACRO package locations."""
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
        "Could not locate Panda URDF/XACRO. Install moveit_resources_panda_description or franka_description."
    )


def _make_robot_description_param(desc_file: str):
    if desc_file.endswith(".xacro"):
        return {
            "robot_description": Command([FindExecutable(name="xacro"), " ", desc_file])
        }

    with open(desc_file, "r", encoding="utf-8") as f:
        return {"robot_description": f.read()}


def _launch_setup(context, *args, **kwargs) -> List:
    result_dir = LaunchConfiguration("result_dir").perform(context)
    fixed_frame = LaunchConfiguration("fixed_frame").perform(context)
    rviz_config = LaunchConfiguration("rviz_config").perform(context)

    desc_file = _pick_panda_description_file()
    robot_description = _make_robot_description_param(desc_file)

    rsp = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        namespace="waypoint",
        name="robot_state_publisher",
        output="screen",
        parameters=[
            robot_description,
            {"frame_prefix": "waypoint_/"},
            {"publish_frequency": 50.0},
        ],
    )

    player = Node(
        package="panda_ik_waypoint",
        executable="ik_waypoint_player",
        name="ik_waypoint_player",
        output="screen",
        parameters=[
            {"result_dir": result_dir},
            {
                "window_size": ParameterValue(
                    LaunchConfiguration("window_size"), value_type=int
                )
            },
            {"fixed_frame": fixed_frame},
            {"robot_prefix": "waypoint_/"},
            {
                "publish_rate_hz": ParameterValue(
                    LaunchConfiguration("publish_rate_hz"), value_type=float
                )
            },
            {
                "speed_scale": ParameterValue(
                    LaunchConfiguration("speed_scale"), value_type=float
                )
            },
            {
                "loop": ParameterValue(
                    LaunchConfiguration("loop"), value_type=bool
                )
            },
        ],
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", rviz_config],
    )

    return [rsp, player, rviz]


def generate_launch_description() -> LaunchDescription:
    pkg_share = get_package_share_directory("panda_ik_waypoint")
    default_rviz = os.path.join(pkg_share, "rviz", "ik_waypoint.rviz")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "result_dir",
                default_value="data_window",
                description="Run output directory or parent directory containing summary.json.",
            ),
            DeclareLaunchArgument(
                "window_size",
                default_value="-1",
                description="Window size to visualize (-1 means use max available ws).",
            ),
            DeclareLaunchArgument(
                "fixed_frame",
                default_value="world",
                description="RViz fixed frame and static TF parent frame.",
            ),
            DeclareLaunchArgument(
                "publish_rate_hz",
                default_value="50.0",
                description="JointState publish rate.",
            ),
            DeclareLaunchArgument(
                "speed_scale",
                default_value="1.0",
                description="Playback speed scale.",
            ),
            DeclareLaunchArgument(
                "loop",
                default_value="true",
                description="Whether playback loops.",
            ),
            DeclareLaunchArgument(
                "rviz_config",
                default_value=default_rviz,
                description="RViz config file.",
            ),
            OpaqueFunction(function=_launch_setup),
        ]
    )
