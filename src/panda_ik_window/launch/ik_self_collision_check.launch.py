from __future__ import annotations

import os
import yaml

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def _load_yaml(package_name: str, file_path: str):
    package_path = get_package_share_directory(package_name)
    absolute_file_path = os.path.join(package_path, file_path)
    with open(absolute_file_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def generate_launch_description() -> LaunchDescription:
    dataset = LaunchConfiguration("dataset")
    preferred_ws = LaunchConfiguration("preferred_ws")

    output_dir = LaunchConfiguration("output_dir")
    output_prefix = LaunchConfiguration("output_prefix")
    details_jsonl = LaunchConfiguration("details_jsonl")

    sample_dt = LaunchConfiguration("sample_dt")
    min_samples = LaunchConfiguration("min_samples")
    max_samples = LaunchConfiguration("max_samples")
    max_candidates = LaunchConfiguration("max_candidates")
    progress_every = LaunchConfiguration("progress_every")

    group = LaunchConfiguration("group")
    tip_link = LaunchConfiguration("tip_link")
    node_name = LaunchConfiguration("node_name")

    moveit_cpp_yaml = os.path.join(
        get_package_share_directory("panda_ik_window"),
        "config",
        "moveit_cpp_offline.yaml",
    )

    moveit_config = (
        MoveItConfigsBuilder(
            robot_name="moveit_resources_panda",
            package_name="moveit_resources_panda_moveit_config",
        )
        .planning_pipelines(pipelines=["pilz_industrial_motion_planner"])
        .moveit_cpp(file_path=moveit_cpp_yaml)
        .to_moveit_configs()
    )
    moveit_params = moveit_config.to_dict()
    moveit_params["robot_description_kinematics"] = _load_yaml(
        "moveit_resources_panda_moveit_config",
        "config/trac_ik_kinematics.yaml",
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("dataset", default_value="dataset_ws3_top50_sort50"),
            DeclareLaunchArgument("preferred_ws", default_value="3"),
            DeclareLaunchArgument("output_dir", default_value=""),
            DeclareLaunchArgument("output_prefix", default_value="self_collision_trapezoid"),
            DeclareLaunchArgument("details_jsonl", default_value=""),
            DeclareLaunchArgument("sample_dt", default_value="0.02"),
            DeclareLaunchArgument("min_samples", default_value="5"),
            DeclareLaunchArgument("max_samples", default_value="0"),
            DeclareLaunchArgument("max_candidates", default_value="0"),
            DeclareLaunchArgument("progress_every", default_value="200"),
            DeclareLaunchArgument("group", default_value="panda_arm"),
            DeclareLaunchArgument("tip_link", default_value=""),
            DeclareLaunchArgument("node_name", default_value="panda_ws3_self_collision_check"),
            Node(
                package="panda_ik_window",
                executable="ik_check_self_collision",
                output="screen",
                parameters=[moveit_params],
                arguments=[
                    "--dataset",
                    dataset,
                    "--preferred-ws",
                    preferred_ws,
                    "--output-dir",
                    output_dir,
                    "--output-prefix",
                    output_prefix,
                    "--details-jsonl",
                    details_jsonl,
                    "--sample-dt",
                    sample_dt,
                    "--min-samples",
                    min_samples,
                    "--max-samples",
                    max_samples,
                    "--max-candidates",
                    max_candidates,
                    "--progress-every",
                    progress_every,
                    "--group",
                    group,
                    "--tip-link",
                    tip_link,
                    "--node-name",
                    node_name,
                ],
            ),
        ]
    )
