#!/usr/bin/env python3

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description() -> LaunchDescription:
    pkg_share = get_package_share_directory("panda_try_rviz")
    default_rviz = os.path.join(pkg_share, "rviz", "panda_try.rviz")

    moveit_config = (
        MoveItConfigsBuilder(
            robot_name="moveit_resources_panda",
            package_name="moveit_resources_panda_moveit_config",
        )
        .planning_pipelines(pipelines=["pilz_industrial_motion_planner"])
        .to_moveit_configs()
    )

    rviz_arg = DeclareLaunchArgument(
        "rviz_config",
        default_value=default_rviz,
        description="RViz config file",
    )

    static_tf_node = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="world_to_panda_try_base",
        arguments=["0", "0", "0", "0", "0", "0", "world", "panda_try_/panda_link0"],
        output="screen",
    )

    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        namespace="panda_try",
        name="robot_state_publisher",
        output="screen",
        parameters=[
            moveit_config.robot_description,
            {"frame_prefix": "panda_try_/"},
            {"publish_frequency": 50.0},
        ],
    )

    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[moveit_config.to_dict()],
    )

    backend_node = Node(
        package="panda_try_rviz",
        executable="panda_try_backend_node",
        name="panda_try_backend",
        output="screen",
        parameters=[
            {"group_name": "panda_arm"},
            {"marker_frame": "world"},
            {"ee_link_frame": "panda_try_/panda_hand"},
        ],
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", LaunchConfiguration("rviz_config")],
        parameters=[moveit_config.to_dict()],
    )

    return LaunchDescription(
        [
            rviz_arg,
            static_tf_node,
            robot_state_publisher_node,
            move_group_node,
            backend_node,
            rviz_node,
        ]
    )
