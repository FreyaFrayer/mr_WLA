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


def _is_robot_trajectory_or_msg(value: Any) -> bool:
    if value is None:
        return False
    # moveit_msgs/RobotTrajectory-like
    try:
        if getattr(value, "joint_trajectory", None) is not None:
            return True
    except Exception:
        pass
    # moveit.core.RobotTrajectory-like
    for method in ("get_robot_trajectory_msg", "getRobotTrajectoryMsg"):
        try:
            if callable(getattr(value, method, None)):
                return True
        except Exception:
            continue
    for attr in ("duration", "get_duration"):
        try:
            if getattr(value, attr, None) is not None:
                return True
        except Exception:
            continue
    return False


def _extract_trajectory_from_any(value: Any, visited: set[int]) -> Any | None:
    if value is None:
        return None

    value_id = id(value)
    if value_id in visited:
        return None
    visited.add(value_id)

    if _is_robot_trajectory_or_msg(value):
        return value

    if isinstance(value, dict):
        preferred_keys = (
            "trajectory",
            "planned_trajectory",
            "robot_trajectory",
            "traj",
            "result",
            "plan_result",
            "motion_plan_response",
        )
        for key in preferred_keys:
            if key in value:
                out = _extract_trajectory_from_any(value.get(key), visited)
                if out is not None:
                    return out
        for item in value.values():
            out = _extract_trajectory_from_any(item, visited)
            if out is not None:
                return out
        return None

    if isinstance(value, (list, tuple)):
        for item in value:
            out = _extract_trajectory_from_any(item, visited)
            if out is not None:
                return out
        return None

    preferred_attrs = (
        "trajectory",
        "planned_trajectory",
        "robot_trajectory",
        "traj",
        "result",
        "plan_result",
        "motion_plan_response",
    )
    for attr in preferred_attrs:
        try:
            nested = getattr(value, attr)
        except Exception:
            continue
        if callable(nested):
            try:
                nested = nested()
            except TypeError:
                continue
            except Exception:
                continue
        out = _extract_trajectory_from_any(nested, visited)
        if out is not None:
            return out

    # Last resort: probe trajectory/result-like attribute names on wrapper objects.
    try:
        dynamic_attrs = [
            str(name)
            for name in dir(value)
            if not str(name).startswith("_")
            and ("traj" in str(name).lower() or "result" in str(name).lower())
        ]
    except Exception:
        dynamic_attrs = []

    for attr in dynamic_attrs[:24]:
        if attr in preferred_attrs:
            continue
        try:
            nested = getattr(value, attr)
        except Exception:
            continue
        if callable(nested):
            try:
                nested = nested()
            except TypeError:
                continue
            except Exception:
                continue
        out = _extract_trajectory_from_any(nested, visited)
        if out is not None:
            return out
    return None


def _extract_trajectory(plan_result: Any) -> Any | None:
    return _extract_trajectory_from_any(plan_result, visited=set())


def _describe_plan_result(plan_result: Any) -> str:
    if plan_result is None:
        return "type=None"

    type_name = type(plan_result).__name__
    if isinstance(plan_result, (list, tuple)):
        elem_types = [type(v).__name__ for v in list(plan_result)[:4]]
        return f"type={type_name}, len={len(plan_result)}, elem_types={elem_types}"
    if isinstance(plan_result, dict):
        keys = [str(k) for k in list(plan_result.keys())[:8]]
        return f"type={type_name}, keys={keys}"

    known_attrs = []
    for name in (
        "trajectory",
        "planned_trajectory",
        "robot_trajectory",
        "traj",
        "result",
        "plan_result",
        "error_code",
        "planning_time",
    ):
        try:
            getattr(plan_result, name)
            known_attrs.append(name)
        except Exception:
            continue
    if len(known_attrs) > 0:
        return f"type={type_name}, attrs={known_attrs}"
    return f"type={type_name}"


def _extract_plan_error_detail(plan_result: Any) -> str:
    if plan_result is None:
        return ""

    try:
        if isinstance(plan_result, dict):
            err = plan_result.get("error_code", None)
            planning_time = plan_result.get("planning_time", None)
        else:
            err = getattr(plan_result, "error_code", None)
            planning_time = getattr(plan_result, "planning_time", None)
    except Exception:
        return ""

    parts: List[str] = []
    if err is not None:
        try:
            val = getattr(err, "val", None)
            if val is None:
                val = int(err)
            parts.append(f"error_code={int(val)}")
        except Exception:
            parts.append(f"error_code={type(err).__name__}")

    if planning_time is not None:
        try:
            parts.append(f"planning_time={float(planning_time):.6f}s")
        except Exception:
            pass

    return ", ".join(parts)


def _extract_trajectory_msg(robot_trajectory: Any) -> Any | None:
    # Already a moveit_msgs/RobotTrajectory-like message.
    try:
        if getattr(robot_trajectory, "joint_trajectory", None) is not None:
            return robot_trajectory
    except Exception:
        pass

    for attr in ("robot_trajectory_msg", "trajectory_msg", "msg"):
        try:
            maybe_msg = getattr(robot_trajectory, attr)
            if maybe_msg is not None and getattr(maybe_msg, "joint_trajectory", None) is not None:
                return maybe_msg
        except Exception:
            continue

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
    out_from_msg = _duration_from_msg_s(robot_trajectory)
    if out_from_msg is not None:
        return float(out_from_msg)

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

    out_from_msg2 = _duration_from_msg_s(traj_msg)
    if out_from_msg2 is None:
        raise RuntimeError("trajectory message has no valid time_from_start")
    return float(out_from_msg2)


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


def _set_goal_state_from_robot_state(planning_component: Any, goal_state: Any) -> None:
    fn = getattr(planning_component, "set_goal_state", None)
    if fn is None:
        raise RuntimeError("PlanningComponent.set_goal_state not available")

    for kwargs in (
        {"robot_state": goal_state},
        {"goal_state": goal_state},
        {},
    ):
        try:
            if len(kwargs) == 0:
                out = fn(goal_state)
            else:
                out = fn(**kwargs)
            if out is False:
                raise RuntimeError("set_goal_state returned False")
            return
        except TypeError:
            continue
        except Exception as e:
            raise RuntimeError(f"set_goal_state failed: {e}") from e

    raise RuntimeError("set_goal_state (robot_state) signature is unsupported")


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
    tip_quat_xyzw: Sequence[float] | None = None,
) -> OriginPathResult:
    """Compute origin path time by planner point-to-point calls without IK-choice DP.

    For each segment p{i-1} -> p{i}:
    - set explicit start state from previous segment end
    - plan to Cartesian pose target at p{i} (target-specific orientation, or fallback)
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

    fallback_quat_xyzw = (0.0, 0.0, 0.0, 1.0)
    if tip_quat_xyzw is not None:
        try:
            raw = [float(v) for v in tip_quat_xyzw]
        except Exception:
            raw = []
        if len(raw) != 4:
            return OriginPathResult(
                status="failed",
                method=method,
                total_time_s=0.0,
                planning_elapsed_total_s=0.0,
                segments=tuple(),
                planning_frame=str(planning_frame),
                planner_id="",
                note=f"tip_quat_xyzw length must be 4 when provided, got {len(raw)}",
            )
        fallback_quat_xyzw = (raw[0], raw[1], raw[2], raw[3])

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
        goal_pose = _make_pose_stamped(
            point=target,
            quat_xyzw=target.normalized_quat_xyzw(fallback_xyzw=fallback_quat_xyzw),
            frame_id=planning_frame,
        )

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
            err_detail = _extract_plan_error_detail(plan_result)
            return OriginPathResult(
                status="failed",
                method=method,
                total_time_s=float(total_time_s),
                planning_elapsed_total_s=float(planning_elapsed_total_s),
                segments=tuple(segments),
                planning_frame=str(planning_frame),
                planner_id="PTP",
                note=(
                    f"{from_label}->{to_label} plan result has no trajectory "
                    f"({_describe_plan_result(plan_result)}"
                    f"{', ' + err_detail if err_detail else ''})"
                ),
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
        note="direct point-to-point planning through target poses without IK candidate selection",
    )


def compute_joint_target_path_time(
    *,
    ctx: RobotContext,
    start_q: Sequence[float],
    joint_targets: Sequence[Sequence[float]],
) -> OriginPathResult:
    """Compute planner timing by planning through explicit joint targets."""

    method = "planner_joint_to_joint"
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

    if len(joint_targets) == 0:
        return OriginPathResult(
            status="ok",
            method=method,
            total_time_s=0.0,
            planning_elapsed_total_s=0.0,
            segments=tuple(),
            planning_frame=str(planning_frame),
            planner_id="PTP",
            note="no joint targets",
        )

    current_q = [float(v) for v in start_q]
    total_time_s = 0.0
    planning_elapsed_total_s = 0.0
    segments: List[OriginSegmentResult] = []

    for i, q_goal in enumerate(joint_targets, start=1):
        from_label = f"p{i-1}"
        to_label = f"p{i}"
        q_goal_list = [float(v) for v in q_goal]
        if len(q_goal_list) != int(ctx.dof):
            return OriginPathResult(
                status="failed",
                method=method,
                total_time_s=float(total_time_s),
                planning_elapsed_total_s=float(planning_elapsed_total_s),
                segments=tuple(segments),
                planning_frame=str(planning_frame),
                planner_id="PTP",
                note=(
                    f"{from_label}->{to_label} goal joint length mismatch: "
                    f"got {len(q_goal_list)}, expected dof={int(ctx.dof)}"
                ),
            )

        start_state = make_robot_state_from_joints(ctx, current_q)
        goal_state = make_robot_state_from_joints(ctx, q_goal_list)

        t0 = perf_counter()
        try:
            _set_start_state(planning_component, start_state)
            _set_goal_state_from_robot_state(planning_component, goal_state)
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
            err_detail = _extract_plan_error_detail(plan_result)
            return OriginPathResult(
                status="failed",
                method=method,
                total_time_s=float(total_time_s),
                planning_elapsed_total_s=float(planning_elapsed_total_s),
                segments=tuple(segments),
                planning_frame=str(planning_frame),
                planner_id="PTP",
                note=(
                    f"{from_label}->{to_label} plan result has no trajectory "
                    f"({_describe_plan_result(plan_result)}"
                    f"{', ' + err_detail if err_detail else ''})"
                ),
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
            # Fall back to requested joint target if trajectory message omitted positions.
            q_next = [float(v) for v in q_goal_list]

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
        note="joint-target point-to-point planning through selected solutions",
    )
