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
    try:
        pkg = get_package_share_directory("moveit_resources_panda_description")
        for rel in ("urdf/panda.urdf.xacro", "urdf/panda.urdf"):
            path = os.path.join(pkg, rel)
            if os.path.exists(path):
                return path
    except PackageNotFoundError:
        pass

    try:
        pkg = get_package_share_directory("franka_description")
        for rel in (
            "robots/panda/panda.urdf.xacro",
            "robots/panda_arm_hand.urdf.xacro",
            "robots/panda_arm.urdf.xacro",
            "robots/panda/panda.urdf",
        ):
            path = os.path.join(pkg, rel)
            if os.path.exists(path):
                return path
    except PackageNotFoundError:
        pass

    raise RuntimeError(
        "Could not locate Panda URDF/XACRO. Install moveit_resources_panda_description or franka_description."
    )


def _make_robot_description_param(desc_file: str):
    if desc_file.endswith(".xacro"):
        return {"robot_description": Command([FindExecutable(name="xacro"), " ", desc_file])}

    with open(desc_file, "r", encoding="utf-8") as f:
        return {"robot_description": f.read()}


def _launch_setup(context, *args, **kwargs) -> List:
    result_dir = LaunchConfiguration("result_dir").perform(context)
    fixed_frame = LaunchConfiguration("fixed_frame").perform(context)
    rviz_config = LaunchConfiguration("rviz_config").perform(context)

    desc_file = _pick_panda_description_file()
    robot_description = _make_robot_description_param(desc_file)

    rsp_origin = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        namespace="origin",
        name="robot_state_publisher",
        output="screen",
        parameters=[
            robot_description,
            {"frame_prefix": "origin_/"},
            {"publish_frequency": 50.0},
        ],
    )

    rsp_ws = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        namespace="ws",
        name="robot_state_publisher",
        output="screen",
        parameters=[
            robot_description,
            {"frame_prefix": "ws_/"},
            {"publish_frequency": 50.0},
        ],
    )

    rsp_ws_planner = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        namespace="ws_planner",
        name="robot_state_publisher",
        output="screen",
        parameters=[
            robot_description,
            {"frame_prefix": "ws_planner_/"},
            {"publish_frequency": 50.0},
        ],
    )

    player = Node(
        package="panda_ik_origin_viz",
        executable="ik_origin_compare_player",
        name="ik_origin_compare_player",
        output="screen",
        parameters=[
            {"result_dir": result_dir},
            {"fixed_frame": fixed_frame},
            {"origin_prefix": "origin_/"},
            {"ws_prefix": "ws_/"},
            {"ws_planner_prefix": "ws_planner_/"},
            {"origin_label": LaunchConfiguration("origin_label")},
            {"ws_label": LaunchConfiguration("ws_label")},
            {"ws_planner_label": LaunchConfiguration("ws_planner_label")},
            {
                "window_size": ParameterValue(
                    LaunchConfiguration("window_size"),
                    value_type=int,
                )
            },
            {
                "planner_window_size": ParameterValue(
                    LaunchConfiguration("planner_window_size"),
                    value_type=int,
                )
            },
            {
                "publish_rate_hz": ParameterValue(
                    LaunchConfiguration("publish_rate_hz"),
                    value_type=float,
                )
            },
            {
                "speed_scale": ParameterValue(
                    LaunchConfiguration("speed_scale"),
                    value_type=float,
                )
            },
            {
                "hold_time_s": ParameterValue(
                    LaunchConfiguration("hold_time_s"),
                    value_type=float,
                )
            },
            {
                "loop": ParameterValue(
                    LaunchConfiguration("loop"),
                    value_type=bool,
                )
            },
            {
                "sync_loop": ParameterValue(
                    LaunchConfiguration("sync_loop"),
                    value_type=bool,
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

    return [rsp_origin, rsp_ws, rsp_ws_planner, player, rviz]


def generate_launch_description() -> LaunchDescription:
    pkg_share = get_package_share_directory("panda_ik_origin_viz")
    default_rviz = os.path.join(pkg_share, "rviz", "ik_origin_compare.rviz")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "result_dir",
                default_value="data_window",
                description="Run directory or parent directory containing summary.json.",
            ),
            DeclareLaunchArgument(
                "window_size",
                default_value="1",
                description="window_size for the middle WS robot. Default: 1.",
            ),
            DeclareLaunchArgument(
                "planner_window_size",
                default_value="0",
                description="window_size for the WS planner replay robot; 0 means follow window_size.",
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
                "hold_time_s",
                default_value="0.5",
                description="Hold time at each waypoint during playback.",
            ),
            DeclareLaunchArgument(
                "loop",
                default_value="true",
                description="Whether playback loops.",
            ),
            DeclareLaunchArgument(
                "sync_loop",
                default_value="true",
                description="Whether origin, ws and planner replay restart together every cycle.",
            ),
            DeclareLaunchArgument(
                "origin_label",
                default_value="",
                description="Optional custom title for the origin robot.",
            ),
            DeclareLaunchArgument(
                "ws_label",
                default_value="",
                description="Optional custom title for the WS robot.",
            ),
            DeclareLaunchArgument(
                "ws_planner_label",
                default_value="",
                description="Optional custom title for the WS planner replay robot.",
            ),
            DeclareLaunchArgument(
                "rviz_config",
                default_value=default_rviz,
                description="RViz config file.",
            ),
            OpaqueFunction(function=_launch_setup),
        ]
    )
