import glob
import os
from typing import Tuple


from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

from moveit_configs_utils import MoveItConfigsBuilder


def _pick_rviz_config() -> str:
    # Try to reuse the RViz config shipped with moveit_resources_panda_moveit_config.
    share = get_package_share_directory("moveit_resources_panda_moveit_config")
    candidates = glob.glob(os.path.join(share, "**", "*.rviz"), recursive=True)
    if not candidates:
        return ""

    def score(p: str) -> Tuple[int, str]:
        name = os.path.basename(p).lower()
        s = 0
        if "moveit" in name:
            s += 10
        if "demo" in name:
            s += 5
        if "panda" in name:
            s += 2
        return (s, p)

    # Highest score wins.
    candidates = sorted(candidates, key=lambda p: score(p), reverse=True)
    return candidates[0]


def generate_launch_description() -> LaunchDescription:
    pkg_share = get_package_share_directory("panda_ik_solutions_viz")

    ik_json_default = os.path.join(pkg_share, "config", "p1.json")
    ik_json_arg = DeclareLaunchArgument(
        "ik_json",
        default_value=ik_json_default,
        description="Path to p1.json containing IK solutions",
    )

    input_in_degrees_arg = DeclareLaunchArgument(
        "input_in_degrees",
        default_value="false",
        description="Set true if joint values in JSON are in degrees",
    )

    time_step_s_arg = DeclareLaunchArgument(
        "time_step_s",
        default_value="0.1",
        description="Time step between consecutive trajectory points (seconds)",
    )

    publish_period_s_arg = DeclareLaunchArgument(
        "publish_period_s",
        default_value="1.0",
        description="How often to (re)publish the DisplayTrajectory (seconds)",
    )

    # Offline MoveItCpp config (as requested)
    moveit_cpp_yaml = os.path.join(pkg_share, "config", "moveit_cpp_offline.yaml")
    moveit_config = (
        MoveItConfigsBuilder(
            robot_name="moveit_resources_panda",
            package_name="moveit_resources_panda_moveit_config",
        )
        .moveit_cpp(file_path=moveit_cpp_yaml)
        .to_moveit_configs()
    )

    # Joint state + robot_state_publisher (so RViz and move_group get a current state)
    joint_state_publisher_node = Node(
        package="joint_state_publisher",
        executable="joint_state_publisher",
        name="joint_state_publisher",
        output="log",
        parameters=[moveit_config.robot_description],
    )

    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="log",
        parameters=[moveit_config.robot_description],
    )

    # Move group (needed by the MotionPlanning RViz plugin for monitored planning scene, etc.)
    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[moveit_config.to_dict()],
    )

    rviz_config_file = _pick_rviz_config()
    rviz_args = ["-d", rviz_config_file] if rviz_config_file else []

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=rviz_args,
        parameters=[moveit_config.to_dict()],
    )

    # World -> panda_link0 (identity). Harmless even if RViz fixed frame is panda_link0.
    static_tf_node = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="world_to_panda_link0",
        arguments=["0", "0", "0", "0", "0", "0", "world", "panda_link0"],
        output="log",
    )

    ik_display_node = Node(
        package="panda_ik_solutions_viz",
        executable="display_ik_solutions",
        name="display_ik_solutions",
        output="screen",
        parameters=[
            {
                "ik_json_path": LaunchConfiguration("ik_json"),
                "display_topic": "/display_planned_path",
                "publish_period_s": ParameterValue(LaunchConfiguration("publish_period_s"), value_type=float),
                # Tip: set to true if your JSON stores degrees
                "input_in_degrees": ParameterValue(LaunchConfiguration("input_in_degrees"), value_type=bool),
                "time_step_s": ParameterValue(LaunchConfiguration("time_step_s"), value_type=float),
            }
        ],
    )

    info = LogInfo(
        msg=(
            "Using RViz config: " + (rviz_config_file if rviz_config_file else "<none> (default RViz)")
        )
    )


    return LaunchDescription(
        [
            ik_json_arg,
            input_in_degrees_arg,
            time_step_s_arg,
            publish_period_s_arg,
            info,
            static_tf_node,
            joint_state_publisher_node,
            robot_state_publisher_node,
            move_group_node,
            rviz_node,
            ik_display_node,
        ]
    )
