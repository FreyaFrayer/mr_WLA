from __future__ import annotations

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

import os
from moveit_configs_utils import MoveItConfigsBuilder
from ament_index_python.packages import get_package_share_directory


def generate_launch_description() -> LaunchDescription:
    # You can override these from CLI, e.g.:
    #   ros2 launch panda_ik_window ik_benchmark.launch.py num_points:=8 seed:=7
    num_points = LaunchConfiguration("num_points")
    seed = LaunchConfiguration("seed")
    path_pattern = LaunchConfiguration("path_pattern")

    group = LaunchConfiguration("group")
    named_start = LaunchConfiguration("named_start")
    data_root = LaunchConfiguration("data_root")
    reuse_candidates_dir = LaunchConfiguration("reuse_candidates_dir")

    # IK sampling
    num_solutions = LaunchConfiguration("num_solutions")
    num_spaces = LaunchConfiguration("num_spaces")
    max_attempts = LaunchConfiguration("max_attempts")
    ik_timeout = LaunchConfiguration("ik_timeout")

    # Robust sampling / auto-resample
    resample_max = LaunchConfiguration("resample_max")
    topup_passes = LaunchConfiguration("topup_passes")
    precheck_attempts = LaunchConfiguration("precheck_attempts")
    precheck_num_spaces = LaunchConfiguration("precheck_num_spaces")

    # Workspace bounds
    ws_x_min = LaunchConfiguration("ws_x_min")
    ws_x_max = LaunchConfiguration("ws_x_max")
    ws_y_min = LaunchConfiguration("ws_y_min")
    ws_y_max = LaunchConfiguration("ws_y_max")
    ws_z_min = LaunchConfiguration("ws_z_min")
    ws_z_max = LaunchConfiguration("ws_z_max")
    min_sep = LaunchConfiguration("min_sep")

    # Segment time model
    time_model = LaunchConfiguration("time_model")
    totg_vel_scale = LaunchConfiguration("totg_vel_scale")
    totg_acc_scale = LaunchConfiguration("totg_acc_scale")
    totg_path_tolerance = LaunchConfiguration("totg_path_tolerance")
    totg_resample_dt = LaunchConfiguration("totg_resample_dt")
    totg_min_angle_change = LaunchConfiguration("totg_min_angle_change")

    # Window policy evaluation
    # Accepts: "3" or "1,3,8" or "[1,3,8]" or "all" (default).
    window_size = LaunchConfiguration("window_size")

    # DP acceleration (trapezoid DP only; TOTG is CPU)
    device = LaunchConfiguration("device")
    dp_block_size = LaunchConfiguration("dp_block_size")

    # Panda MoveIt config (from moveit_resources)
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

    return LaunchDescription(
        [
            DeclareLaunchArgument("num_points", default_value="8"),
            DeclareLaunchArgument("seed", default_value="7"),
            DeclareLaunchArgument("path_pattern", default_value="random"),
            DeclareLaunchArgument("group", default_value="panda_arm"),
            DeclareLaunchArgument("named_start", default_value="random"),
            DeclareLaunchArgument("data_root", default_value="data_window"),
            DeclareLaunchArgument("reuse_candidates_dir", default_value=""),
            DeclareLaunchArgument("num_solutions", default_value="200"),
            DeclareLaunchArgument("num_spaces", default_value="10"),
            DeclareLaunchArgument("max_attempts", default_value="400"),
            DeclareLaunchArgument("ik_timeout", default_value="0.1"),
            DeclareLaunchArgument("resample_max", default_value="200"),
            DeclareLaunchArgument("topup_passes", default_value="3"),
            DeclareLaunchArgument("precheck_attempts", default_value="400"),
            DeclareLaunchArgument("precheck_num_spaces", default_value="5"),
            DeclareLaunchArgument("ws_x_min", default_value="-0.75"),
            DeclareLaunchArgument("ws_x_max", default_value="0.75"),
            DeclareLaunchArgument("ws_y_min", default_value="-0.55"),
            DeclareLaunchArgument("ws_y_max", default_value="0.55"),
            DeclareLaunchArgument("ws_z_min", default_value="0.05"),
            DeclareLaunchArgument("ws_z_max", default_value="0.85"),
            DeclareLaunchArgument("min_sep", default_value="0.06"),
            DeclareLaunchArgument("time_model", default_value="trapezoid"),
            DeclareLaunchArgument("window_size", default_value="all"),
            DeclareLaunchArgument("device", default_value="cuda"),
            DeclareLaunchArgument("dp_block_size", default_value="256"),
            DeclareLaunchArgument("totg_vel_scale", default_value="1.0"),
            DeclareLaunchArgument("totg_acc_scale", default_value="1.0"),
            DeclareLaunchArgument("totg_path_tolerance", default_value="0.1"),
            DeclareLaunchArgument("totg_resample_dt", default_value="0.1"),
            DeclareLaunchArgument("totg_min_angle_change", default_value="0.001"),
            Node(
                package="panda_ik_window",
                executable="ik_window",
                output="screen",
                parameters=[moveit_config.to_dict()],
                arguments=[
                    "--num-points",
                    num_points,
                    "--seed",
                    seed,
                    "--path-pattern",
                    path_pattern,
                    "--group",
                    group,
                    "--named-start",
                    named_start,
                    "--data-root",
                    data_root,
                    "--reuse-candidates-dir",
                    reuse_candidates_dir,
                    "--window-size",
                    window_size,
                    "--device",
                    device,
                    "--dp-block-size",
                    dp_block_size,
                    "--num-solutions",
                    num_solutions,
                    "--num-spaces",
                    num_spaces,
                    "--max-attempts",
                    max_attempts,
                    "--ik-timeout",
                    ik_timeout,
                    "--resample-max",
                    resample_max,
                    "--topup-passes",
                    topup_passes,
                    "--precheck-attempts",
                    precheck_attempts,
                    "--precheck-num-spaces",
                    precheck_num_spaces,
                    "--ws-x",
                    ws_x_min,
                    ws_x_max,
                    "--ws-y",
                    ws_y_min,
                    ws_y_max,
                    "--ws-z",
                    ws_z_min,
                    ws_z_max,
                    "--min-sep",
                    min_sep,
                    "--time-model",
                    time_model,
                    "--totg-vel-scale",
                    totg_vel_scale,
                    "--totg-acc-scale",
                    totg_acc_scale,
                    "--totg-path-tolerance",
                    totg_path_tolerance,
                    "--totg-resample-dt",
                    totg_resample_dt,
                    "--totg-min-angle-change",
                    totg_min_angle_change,
                ],
            ),
        ]
    )
