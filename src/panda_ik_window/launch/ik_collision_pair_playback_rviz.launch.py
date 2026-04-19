#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
from typing import List

from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import Command, FindExecutable, LaunchConfiguration
from launch_ros.actions import Node


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
        return {"robot_description": Command([FindExecutable(name="xacro"), " ", desc_file])}

    with open(desc_file, "r", encoding="utf-8") as f:
        return {"robot_description": f.read()}


def _launch_setup(context, *args, **kwargs) -> List:
    dataset = LaunchConfiguration("dataset").perform(context)
    preferred_ws = int(LaunchConfiguration("preferred_ws").perform(context))
    collision_pairs_json = LaunchConfiguration("collision_pairs_json").perform(context)
    pair_index = int(LaunchConfiguration("pair_index").perform(context))
    sample_i = int(LaunchConfiguration("sample_i").perform(context))
    candidate_k = int(LaunchConfiguration("candidate_k").perform(context))

    sample_dt = float(LaunchConfiguration("sample_dt").perform(context))
    min_samples = int(LaunchConfiguration("min_samples").perform(context))
    publish_rate_hz = float(LaunchConfiguration("publish_rate_hz").perform(context))
    speed_scale = float(LaunchConfiguration("speed_scale").perform(context))
    loop = str(LaunchConfiguration("loop").perform(context)).strip().lower() in ("1", "true", "yes", "on")
    hold_start_s = float(LaunchConfiguration("hold_start_s").perform(context))
    hold_end_s = float(LaunchConfiguration("hold_end_s").perform(context))

    fixed_frame = LaunchConfiguration("fixed_frame").perform(context)
    rviz_config = LaunchConfiguration("rviz_config").perform(context)

    desc_file = _pick_panda_description_file()
    robot_description = _make_robot_description_param(desc_file)

    rsp = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        namespace="collision_pair",
        name="robot_state_publisher",
        output="screen",
        parameters=[
            robot_description,
            {"frame_prefix": "collision_pair_/"},
            {"publish_frequency": 60.0},
        ],
        remappings=[("joint_states", "/collision_pair/joint_states")],
    )

    player = Node(
        package="panda_ik_window",
        executable="ik_collision_pair_player",
        name="ik_collision_pair_player",
        output="screen",
        parameters=[
            {"dataset": dataset},
            {"preferred_ws": preferred_ws},
            {"collision_pairs_json": collision_pairs_json},
            {"pair_index": pair_index},
            {"sample_i": sample_i},
            {"candidate_k": candidate_k},
            {"sample_dt": sample_dt},
            {"min_samples": min_samples},
            {"publish_rate_hz": publish_rate_hz},
            {"speed_scale": speed_scale},
            {"loop": loop},
            {"hold_start_s": hold_start_s},
            {"hold_end_s": hold_end_s},
            {"fixed_frame": fixed_frame},
            {"robot_prefix": "collision_pair_/"},
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
    pkg_share = get_package_share_directory("panda_ik_window")
    default_rviz = os.path.join(pkg_share, "rviz", "collision_pair_playback.rviz")

    return LaunchDescription(
        [
            DeclareLaunchArgument("dataset", default_value="dataset_ws3_top50_sort50"),
            DeclareLaunchArgument("preferred_ws", default_value="3"),
            DeclareLaunchArgument("collision_pairs_json", default_value=""),
            DeclareLaunchArgument("pair_index", default_value="0"),
            DeclareLaunchArgument("sample_i", default_value="-1"),
            DeclareLaunchArgument("candidate_k", default_value="-1"),
            DeclareLaunchArgument("sample_dt", default_value="0.02"),
            DeclareLaunchArgument("min_samples", default_value="5"),
            DeclareLaunchArgument("publish_rate_hz", default_value="60.0"),
            DeclareLaunchArgument("speed_scale", default_value="1.0"),
            DeclareLaunchArgument("loop", default_value="true"),
            DeclareLaunchArgument("hold_start_s", default_value="0.25"),
            DeclareLaunchArgument("hold_end_s", default_value="0.50"),
            DeclareLaunchArgument("fixed_frame", default_value="world"),
            DeclareLaunchArgument("rviz_config", default_value=default_rviz),
            OpaqueFunction(function=_launch_setup),
        ]
    )
