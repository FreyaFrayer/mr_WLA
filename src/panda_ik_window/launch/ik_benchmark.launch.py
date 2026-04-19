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
    trend_max_step = LaunchConfiguration("trend_max_step")

    group = LaunchConfiguration("group")
    named_start = LaunchConfiguration("named_start")
    p0_down = LaunchConfiguration("p0_down")
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
    ws_xy_inner_radius = LaunchConfiguration("ws_xy_inner_radius")
    min_sep = LaunchConfiguration("min_sep")

    # Window policy evaluation
    # Accepts: "3" or "1,3,8" or "[1,3,8]" or "all" (default).
    window_size = LaunchConfiguration("window_size")

    # DP acceleration
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
            DeclareLaunchArgument("trend_max_step", default_value="0.25"),
            DeclareLaunchArgument("group", default_value="panda_arm"),
            DeclareLaunchArgument("named_start", default_value="random"),
            DeclareLaunchArgument("p0_down", default_value="false"),
            DeclareLaunchArgument("data_root", default_value="data_window"),
            DeclareLaunchArgument("reuse_candidates_dir", default_value=""),
            DeclareLaunchArgument("num_solutions", default_value="100"),
            DeclareLaunchArgument("num_spaces", default_value="10"),
            DeclareLaunchArgument("max_attempts", default_value="400"),
            DeclareLaunchArgument("ik_timeout", default_value="0.05"),
            DeclareLaunchArgument("resample_max", default_value="200"),
            DeclareLaunchArgument("topup_passes", default_value="3"),
            DeclareLaunchArgument("precheck_attempts", default_value="200"),
            DeclareLaunchArgument("precheck_num_spaces", default_value="5"),
            DeclareLaunchArgument("ws_x_min", default_value="-0.75"),
            DeclareLaunchArgument("ws_x_max", default_value="0.75"),
            DeclareLaunchArgument("ws_y_min", default_value="-0.55"),
            DeclareLaunchArgument("ws_y_max", default_value="0.55"),
            DeclareLaunchArgument("ws_z_min", default_value="0.05"),
            DeclareLaunchArgument("ws_z_max", default_value="0.85"),
            DeclareLaunchArgument("ws_xy_inner_radius", default_value="0.25"),
            DeclareLaunchArgument("min_sep", default_value="0.06"),
            DeclareLaunchArgument("window_size", default_value="all"),
            DeclareLaunchArgument("device", default_value="cuda"),
            DeclareLaunchArgument("dp_block_size", default_value="256"),
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
                    "--trend-max-step",
                    trend_max_step,
                    "--group",
                    group,
                    "--named-start",
                    named_start,
                    "--p0-down",
                    p0_down,
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
                    "--ws-xy-inner-radius",
                    ws_xy_inner_radius,
                    "--min-sep",
                    min_sep,
                ],
            ),
        ]
    )
