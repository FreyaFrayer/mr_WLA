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
            p = os.path.join(pkg, rel)
            if os.path.exists(p):
                return p
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
        return {"robot_description": Command([FindExecutable(name="xacro"), " ", desc_file])}

    with open(desc_file, "r", encoding="utf-8") as f:
        return {"robot_description": f.read()}


def _launch_setup(context, *args, **kwargs) -> List:
    seed_dir = LaunchConfiguration("seed_dir").perform(context)
    model_a_dir = LaunchConfiguration("model_a_dir").perform(context)
    model_b_dir = LaunchConfiguration("model_b_dir").perform(context)
    fixed_frame = LaunchConfiguration("fixed_frame").perform(context)
    rviz_config = LaunchConfiguration("rviz_config").perform(context)

    desc_file = _pick_panda_description_file()
    robot_description = _make_robot_description_param(desc_file)

    rsp_a = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        namespace="model_a",
        name="robot_state_publisher",
        output="screen",
        parameters=[
            robot_description,
            {"frame_prefix": "model_a_/"},
            {"publish_frequency": 50.0},
        ],
    )

    rsp_b = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        namespace="model_b",
        name="robot_state_publisher",
        output="screen",
        parameters=[
            robot_description,
            {"frame_prefix": "model_b_/"},
            {"publish_frequency": 50.0},
        ],
    )

    player = Node(
        package="panda_time_model_rviz",
        executable="time_model_compare_player",
        name="time_model_compare_player",
        output="screen",
        parameters=[
            {"seed_dir": seed_dir},
            {"model_a_dir": model_a_dir},
            {"model_b_dir": model_b_dir},
            {"fixed_frame": fixed_frame},
            {"model_a_prefix": "model_a_/"},
            {"model_b_prefix": "model_b_/"},
            {
                "window_size": ParameterValue(
                    LaunchConfiguration("window_size"), value_type=int
                )
            },
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
            {
                "sync_loop": ParameterValue(
                    LaunchConfiguration("sync_loop"), value_type=bool
                )
            },
            {
                "hold_time_s": ParameterValue(
                    LaunchConfiguration("hold_time_s"), value_type=float
                )
            },
            {"model_a_label": LaunchConfiguration("model_a_label")},
            {"model_b_label": LaunchConfiguration("model_b_label")},
        ],
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", rviz_config],
    )

    return [rsp_a, rsp_b, player, rviz]


def generate_launch_description() -> LaunchDescription:
    pkg_share = get_package_share_directory("panda_time_model_rviz")
    default_rviz = os.path.join(pkg_share, "rviz", "time_model_compare.rviz")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "seed_dir",
                default_value="data_window/batch_time_model_data/np3/seed11",
                description="Seed root directory containing model_a/model_b subdirectories.",
            ),
            DeclareLaunchArgument(
                "model_a_dir",
                default_value="model_a",
                description="Subdirectory name under seed_dir for model A.",
            ),
            DeclareLaunchArgument(
                "model_b_dir",
                default_value="model_b",
                description="Subdirectory name under seed_dir for model B.",
            ),
            DeclareLaunchArgument(
                "window_size",
                default_value="-1",
                description="window_size to visualize. -1 means common max ws across two models.",
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
                "sync_loop",
                default_value="true",
                description="Whether two models restart together every cycle.",
            ),
            DeclareLaunchArgument(
                "hold_time_s",
                default_value="0.4",
                description="Hold time at each waypoint during playback.",
            ),
            DeclareLaunchArgument(
                "model_a_label",
                default_value="",
                description="Optional custom title for model A.",
            ),
            DeclareLaunchArgument(
                "model_b_label",
                default_value="",
                description="Optional custom title for model B.",
            ),
            DeclareLaunchArgument(
                "rviz_config",
                default_value=default_rviz,
                description="RViz config file.",
            ),
            OpaqueFunction(function=_launch_setup),
        ]
    )
