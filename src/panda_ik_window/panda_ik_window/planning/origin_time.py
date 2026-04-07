from __future__ import annotations

import math
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, List, Sequence

from ..types import TargetPoint
from ..utils.robot import RobotContext, make_robot_state_from_joints


@dataclass(frozen=True)
class OriginSegmentResult:
    seg_idx_1based: int
    from_label: str
    to_label: str
    trajectory_time_s: float
    planning_elapsed_s: float
    start_joint_positions: Sequence[float]
    end_joint_positions: Sequence[float]


@dataclass(frozen=True)
class OriginPathResult:
    status: str
    method: str
    total_time_s: float
    planning_elapsed_total_s: float
    segments: Sequence[OriginSegmentResult] = field(default_factory=tuple)
    planning_frame: str = ""
    planner_id: str = ""
    note: str = ""


def _resolve_planning_frame(robot_model: Any) -> str:
    for attr in ("model_frame", "get_model_frame", "getModelFrame"):
        try:
            value = getattr(robot_model, attr)
            frame = value() if callable(value) else value
            text = str(frame).strip()
            if text:
                return text
        except Exception:
            continue
    return "panda_link0"


def _extract_trajectory(plan_result: Any) -> Any | None:
    for attr in ("trajectory", "planned_trajectory"):
        try:
            value = getattr(plan_result, attr)
            if value is not None:
                return value
        except Exception:
            continue
    return None


def _extract_trajectory_msg(robot_trajectory: Any) -> Any | None:
    for method in ("get_robot_trajectory_msg", "getRobotTrajectoryMsg"):
        fn = getattr(robot_trajectory, method, None)
        if fn is None:
            continue
        try:
            msg = fn()
            if msg is not None:
                return msg
        except Exception:
            continue
    return None


def _duration_from_msg_s(traj_msg: Any) -> float | None:
    try:
        jt = getattr(traj_msg, "joint_trajectory")
        points = list(getattr(jt, "points", []))
        if len(points) == 0:
            return None

        tfs = getattr(points[-1], "time_from_start", None)
        if tfs is None:
            return None

        sec = float(getattr(tfs, "sec", 0.0))
        nanosec = float(getattr(tfs, "nanosec", 0.0))
        out = sec + nanosec * 1e-9
        if math.isfinite(out) and out >= 0.0:
            return float(out)
        return None
    except Exception:
        return None


def _extract_trajectory_duration_s(robot_trajectory: Any) -> float:
    for attr in ("duration", "get_duration"):
        try:
            value = getattr(robot_trajectory, attr)
            out = float(value() if callable(value) else value)
            if math.isfinite(out) and out >= 0.0:
                return float(out)
        except Exception:
            continue

    traj_msg = _extract_trajectory_msg(robot_trajectory)
    if traj_msg is None:
        raise RuntimeError("cannot access RobotTrajectory message to read duration")

    out_from_msg = _duration_from_msg_s(traj_msg)
    if out_from_msg is None:
        raise RuntimeError("trajectory message has no valid time_from_start")
    return float(out_from_msg)


def _extract_last_joint_positions(
    robot_trajectory: Any,
    ordered_joint_names: Sequence[str],
) -> List[float] | None:
    traj_msg = _extract_trajectory_msg(robot_trajectory)
    if traj_msg is None:
        return None

    try:
        jt = getattr(traj_msg, "joint_trajectory")
        joint_names = [str(v) for v in list(getattr(jt, "joint_names", []))]
        points = list(getattr(jt, "points", []))
        if len(points) == 0:
            return None

        positions = [float(v) for v in list(getattr(points[-1], "positions", []))]
        if len(positions) == 0:
            return None

        if len(joint_names) == len(positions) and len(joint_names) > 0:
            by_name = {name: float(pos) for name, pos in zip(joint_names, positions)}
            if all(name in by_name for name in ordered_joint_names):
                return [float(by_name[name]) for name in ordered_joint_names]

        if len(positions) == len(ordered_joint_names):
            return [float(v) for v in positions]

        return None
    except Exception:
        return None


def _set_start_state(planning_component: Any, start_state: Any) -> None:
    fn = getattr(planning_component, "set_start_state", None)
    if fn is None:
        raise RuntimeError("PlanningComponent.set_start_state not available")

    for kwargs in (
        {"robot_state": start_state},
        {"start_state": start_state},
        {},
    ):
        try:
            if len(kwargs) == 0:
                out = fn(start_state)
            else:
                out = fn(**kwargs)
            if out is False:
                raise RuntimeError("set_start_state returned False")
            return
        except TypeError:
            continue
        except Exception as e:
            raise RuntimeError(f"set_start_state failed: {e}") from e

    raise RuntimeError("set_start_state signature is unsupported")


def _set_goal_pose(planning_component: Any, pose_stamped: Any, tip_link: str) -> None:
    fn = getattr(planning_component, "set_goal_state", None)
    if fn is None:
        raise RuntimeError("PlanningComponent.set_goal_state not available")

    candidates = [
        {"pose_stamped_msg": pose_stamped, "pose_link": str(tip_link)},
        {"pose_stamped_msg": pose_stamped, "link_name": str(tip_link)},
        {"pose_stamped": pose_stamped, "pose_link": str(tip_link)},
        {"pose_stamped": pose_stamped, "link_name": str(tip_link)},
        {"pose_stamped_msg": pose_stamped},
        {"pose_stamped": pose_stamped},
    ]

    for kwargs in candidates:
        try:
            out = fn(**kwargs)
            if out is False:
                raise RuntimeError("set_goal_state returned False")
            return
        except TypeError:
            continue
        except Exception as e:
            raise RuntimeError(f"set_goal_state failed: {e}") from e

    # Positional fallback
    try:
        out = fn(pose_stamped, str(tip_link))
        if out is False:
            raise RuntimeError("set_goal_state returned False")
        return
    except Exception as e:
        raise RuntimeError(f"set_goal_state signature is unsupported: {e}") from e


def _make_pose_stamped(*, point: TargetPoint, quat_xyzw: Sequence[float], frame_id: str) -> Any:
    from geometry_msgs.msg import PoseStamped

    pose_stamped = PoseStamped()
    pose_stamped.header.frame_id = str(frame_id)
    pose_stamped.pose.position.x = float(point.x)
    pose_stamped.pose.position.y = float(point.y)
    pose_stamped.pose.position.z = float(point.z)
    pose_stamped.pose.orientation.x = float(quat_xyzw[0])
    pose_stamped.pose.orientation.y = float(quat_xyzw[1])
    pose_stamped.pose.orientation.z = float(quat_xyzw[2])
    pose_stamped.pose.orientation.w = float(quat_xyzw[3])
    return pose_stamped


def compute_origin_path_time(
    *,
    ctx: RobotContext,
    start_q: Sequence[float],
    targets: Sequence[TargetPoint],
    tip_quat_xyzw: Sequence[float],
) -> OriginPathResult:
    """Compute origin path time by planner point-to-point calls without IK-choice DP.

    For each segment p{i-1} -> p{i}:
    - set explicit start state from previous segment end
    - plan to Cartesian pose target at p{i} (fixed orientation)
    - accumulate planned trajectory duration
    """

    method = "planner_point_to_point"
    try:
        planning_component = ctx.moveit_py.get_planning_component(str(ctx.group))
    except Exception as e:
        return OriginPathResult(
            status="failed",
            method=method,
            total_time_s=0.0,
            planning_elapsed_total_s=0.0,
            segments=tuple(),
            planning_frame="",
            planner_id="",
            note=f"cannot create planning component for group={ctx.group!r}: {e}",
        )

    planning_frame = _resolve_planning_frame(ctx.robot_model)

    quat = [float(v) for v in tip_quat_xyzw]
    if len(quat) != 4:
        return OriginPathResult(
            status="failed",
            method=method,
            total_time_s=0.0,
            planning_elapsed_total_s=0.0,
            segments=tuple(),
            planning_frame=str(planning_frame),
            planner_id="",
            note=f"tip_quat_xyzw length must be 4, got {len(quat)}",
        )

    if len(targets) == 0:
        return OriginPathResult(
            status="ok",
            method=method,
            total_time_s=0.0,
            planning_elapsed_total_s=0.0,
            segments=tuple(),
            planning_frame=str(planning_frame),
            planner_id="PTP",
            note="no target points",
        )

    current_q = [float(v) for v in start_q]
    total_time_s = 0.0
    planning_elapsed_total_s = 0.0
    segments: List[OriginSegmentResult] = []

    for i, target in enumerate(targets, start=1):
        from_label = f"p{i-1}"
        to_label = f"p{i}"

        start_state = make_robot_state_from_joints(ctx, current_q)
        goal_pose = _make_pose_stamped(point=target, quat_xyzw=quat, frame_id=planning_frame)

        t0 = perf_counter()
        try:
            _set_start_state(planning_component, start_state)
            _set_goal_pose(planning_component, goal_pose, ctx.tip_link)
            plan_result = planning_component.plan()
        except Exception as e:
            return OriginPathResult(
                status="failed",
                method=method,
                total_time_s=float(total_time_s),
                planning_elapsed_total_s=float(planning_elapsed_total_s),
                segments=tuple(segments),
                planning_frame=str(planning_frame),
                planner_id="PTP",
                note=f"{from_label}->{to_label} planning call failed: {e}",
            )

        planning_elapsed_s = float(perf_counter() - t0)
        planning_elapsed_total_s += planning_elapsed_s

        trajectory = _extract_trajectory(plan_result)
        if trajectory is None:
            return OriginPathResult(
                status="failed",
                method=method,
                total_time_s=float(total_time_s),
                planning_elapsed_total_s=float(planning_elapsed_total_s),
                segments=tuple(segments),
                planning_frame=str(planning_frame),
                planner_id="PTP",
                note=f"{from_label}->{to_label} plan result has no trajectory",
            )

        try:
            dt = float(_extract_trajectory_duration_s(trajectory))
        except Exception as e:
            return OriginPathResult(
                status="failed",
                method=method,
                total_time_s=float(total_time_s),
                planning_elapsed_total_s=float(planning_elapsed_total_s),
                segments=tuple(segments),
                planning_frame=str(planning_frame),
                planner_id="PTP",
                note=f"{from_label}->{to_label} cannot read trajectory duration: {e}",
            )

        q_next = _extract_last_joint_positions(trajectory, ctx.joint_names)
        if q_next is None:
            return OriginPathResult(
                status="failed",
                method=method,
                total_time_s=float(total_time_s),
                planning_elapsed_total_s=float(planning_elapsed_total_s),
                segments=tuple(segments),
                planning_frame=str(planning_frame),
                planner_id="PTP",
                note=f"{from_label}->{to_label} cannot read final joint positions from trajectory",
            )

        segments.append(
            OriginSegmentResult(
                seg_idx_1based=i,
                from_label=from_label,
                to_label=to_label,
                trajectory_time_s=float(dt),
                planning_elapsed_s=float(planning_elapsed_s),
                start_joint_positions=[float(v) for v in current_q],
                end_joint_positions=[float(v) for v in q_next],
            )
        )

        total_time_s += float(dt)
        current_q = [float(v) for v in q_next]

    return OriginPathResult(
        status="ok",
        method=method,
        total_time_s=float(total_time_s),
        planning_elapsed_total_s=float(planning_elapsed_total_s),
        segments=tuple(segments),
        planning_frame=str(planning_frame),
        planner_id="PTP",
        note="direct point-to-point planning through targets without IK candidate selection",
    )
