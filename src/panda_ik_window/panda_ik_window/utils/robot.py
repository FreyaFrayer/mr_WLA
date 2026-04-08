from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from moveit.core.robot_state import RobotState
from moveit.planning import MoveItPy

# Fallback limits for Panda arm joints.
# Used only when a MoveIt Python binding does not expose velocity/acceleration
# in active_joint_model_bounds.
_PANDA_FALLBACK_DYNAMICS = {
    "panda_joint1": (2.1750, 3.75),
    "panda_joint2": (2.1750, 1.875),
    "panda_joint3": (2.1750, 2.5),
    "panda_joint4": (2.1750, 3.125),
    "panda_joint5": (2.6100, 3.75),
    "panda_joint6": (2.6100, 5.0),
    "panda_joint7": (2.6100, 5.0),
}

# Position limits for Panda arm joints (rad), used as a safety fallback when
# bindings return mismatched/misordered bounds.
_PANDA_FALLBACK_POSITION_BOUNDS = {
    "panda_joint1": (-2.8973, 2.8973),
    "panda_joint2": (-1.7628, 1.7628),
    "panda_joint3": (-2.8973, 2.8973),
    "panda_joint4": (-3.0718, -0.0698),
    "panda_joint5": (-2.8973, 2.8973),
    "panda_joint6": (-0.0175, 3.7525),
    "panda_joint7": (-2.8973, 2.8973),
}


@dataclass(frozen=True)
class JointLimit:
    min_position: float
    max_position: float
    max_velocity: float  # rad/s
    max_acceleration: float  # rad/s^2

    def clamp(self, v: float) -> float:
        return float(min(max(v, self.min_position), self.max_position))


@dataclass(frozen=True)
class RobotContext:
    moveit_py: MoveItPy
    robot_model: object
    group: str
    tip_link: str
    joint_names: Sequence[str]
    joint_limits: Sequence[JointLimit]

    @property
    def dof(self) -> int:
        return int(len(self.joint_names))

    @property
    def velocity_limits(self) -> np.ndarray:
        return np.array([jl.max_velocity for jl in self.joint_limits], dtype=float)

    @property
    def acceleration_limits(self) -> np.ndarray:
        return np.array([jl.max_acceleration for jl in self.joint_limits], dtype=float)

    @property
    def position_bounds(self) -> List[Tuple[float, float]]:
        return [(jl.min_position, jl.max_position) for jl in self.joint_limits]


def _enforce_state_bounds(state: RobotState, *, robot_model: object, group: str) -> None:
    """Best-effort call into MoveIt RobotState bounds enforcement.

    MoveItPy method signatures vary across versions/bindings, so we probe a few.
    """
    jmg = None
    try:
        jmg = robot_model.get_joint_model_group(str(group))
    except Exception:
        jmg = None

    for method_name in ("enforce_bounds", "enforceBounds"):
        fn = getattr(state, method_name, None)
        if fn is None:
            continue

        candidates = [
            ((), {}),
            ((str(group),), {}),
            ((), {"group_name": str(group)}),
            ((), {"joint_model_group": jmg}) if jmg is not None else None,
            ((jmg,), {}) if jmg is not None else None,
        ]
        for candidate in candidates:
            if candidate is None:
                continue
            args, kwargs = candidate
            try:
                out = fn(*args, **kwargs)
                if out is False:
                    continue
                return
            except TypeError:
                continue
            except Exception:
                continue


def _infer_tip_link_from_jmg(jmg) -> str:
    if not getattr(jmg, "link_model_names", None):
        raise RuntimeError("JointModelGroup.link_model_names is empty; cannot infer tip_link.")
    # heuristic: the last link is the tip link of the kinematic chain
    return str(list(jmg.link_model_names)[-1])


def _extract_joint_limits_from_jmg(
    jmg,
) -> List[JointLimit]:
    """
    Try to read joint limits from MoveItPy bindings.

    We rely on `jmg.active_joint_model_bounds` (similar to the provided IK sampler).
    """
    limits: List[JointLimit] = []
    joint_names = list(getattr(jmg, "active_joint_model_names", getattr(jmg, "joint_model_names", [])))

    try:
        bounds = list(jmg.active_joint_model_bounds)
    except Exception as e:
        raise RuntimeError(
            f"Failed to read joint bounds from MoveIt (active_joint_model_bounds): {e}"
        ) from e

    if len(bounds) == 0:
        raise RuntimeError(
            "MoveIt returned empty active_joint_model_bounds; cannot continue without joint limits."
        )

    used_fallback = False

    for i, b in enumerate(bounds):
        joint_name = str(joint_names[i]) if i < len(joint_names) else f"joint[{i}]"

        # Position bounds
        lo = -math.pi
        hi = math.pi
        try:
            if bool(getattr(b, "position_bounded")):
                lo = float(getattr(b, "min_position"))
                hi = float(getattr(b, "max_position"))
        except Exception:
            pass

        # Velocity limit (required)
        vmax = None
        for attr in ("max_velocity", "max_velocity_", "velocity", "max_velocity_limit"):
            if hasattr(b, attr):
                try:
                    v = float(getattr(b, attr))
                    if v > 1e-6:
                        vmax = v
                        break
                except Exception:
                    pass
        if vmax is None:
            fb = _PANDA_FALLBACK_DYNAMICS.get(joint_name)
            if fb is not None and float(fb[0]) > 1e-6:
                vmax = float(fb[0])
                used_fallback = True
            else:
                raise RuntimeError(
                    f"Failed to read a valid max_velocity from MoveIt bounds for {joint_name}."
                )

        # Acceleration limit (required)
        amax = None
        for attr in ("max_acceleration", "max_acceleration_", "acceleration", "max_acceleration_limit"):
            if hasattr(b, attr):
                try:
                    a = float(getattr(b, attr))
                    if a > 1e-6:
                        amax = a
                        break
                except Exception:
                    pass
        if amax is None:
            fb = _PANDA_FALLBACK_DYNAMICS.get(joint_name)
            if fb is not None and float(fb[1]) > 1e-6:
                amax = float(fb[1])
                used_fallback = True
            else:
                raise RuntimeError(
                    f"Failed to read a valid max_acceleration from MoveIt bounds for {joint_name}."
                )

        limits.append(
            JointLimit(
                min_position=lo,
                max_position=hi,
                max_velocity=float(vmax),
                max_acceleration=float(amax),
            )
        )

    if used_fallback:
        print(
            "[robot] WARN: MoveIt bounds missed velocity/acceleration limits for some joints; "
            "used Panda fallback dynamics for missing values."
        )

    return limits


def load_robot_context(
    *,
    node_name: str = "panda_ik_window",
    group: str = "panda_arm",
    tip_link: str = "",
) -> RobotContext:
    """
    Create MoveItPy instance, load robot model, and extract group metadata.

    Notes
    -----
    MoveItPy expects robot_description/semantic/kinematics params injected via launch.
    """
    moveit_py = MoveItPy(node_name=node_name)
    robot_model = moveit_py.get_robot_model()

    if not robot_model.has_joint_model_group(group):
        raise RuntimeError(
            f"Group '{group}' not found. Available groups: {robot_model.joint_model_group_names}"
        )

    jmg = robot_model.get_joint_model_group(group)
    joint_names = list(getattr(jmg, "active_joint_model_names", jmg.joint_model_names))

    if not tip_link.strip():
        tip_link = _infer_tip_link_from_jmg(jmg)

    joint_limits = _extract_joint_limits_from_jmg(jmg)
    if len(joint_limits) != len(joint_names):
        # last resort: align lengths
        m = min(len(joint_limits), len(joint_names))
        joint_limits = joint_limits[:m]
        joint_names = joint_names[:m]

    return RobotContext(
        moveit_py=moveit_py,
        robot_model=robot_model,
        group=group,
        tip_link=tip_link,
        joint_names=joint_names,
        joint_limits=joint_limits,
    )


def make_robot_state_from_named(ctx: RobotContext, named_state: str) -> RobotState:
    state = RobotState(ctx.robot_model)
    state.set_to_default_values()
    state.set_to_default_values(ctx.group, named_state)
    state.update()
    return state


def make_robot_state_from_joints(ctx: RobotContext, joint_positions: Sequence[float]) -> RobotState:
    if len(joint_positions) != ctx.dof:
        raise ValueError(f"Expected {ctx.dof} joint values, got {len(joint_positions)}")

    q = np.array(list(map(float, joint_positions)), dtype=float)
    # clamp to bounds (avoid invalid seeds)
    for i, jl in enumerate(ctx.joint_limits):
        q[i] = jl.clamp(q[i])
        if i < len(ctx.joint_names):
            name = str(ctx.joint_names[i])
            pos_fallback = _PANDA_FALLBACK_POSITION_BOUNDS.get(name, None)
            if pos_fallback is not None:
                lo_fb, hi_fb = pos_fallback
                q[i] = float(min(max(q[i], float(lo_fb)), float(hi_fb)))

    state = RobotState(ctx.robot_model)
    state.set_to_default_values()
    state.set_joint_group_positions(ctx.group, q)
    state.update()
    _enforce_state_bounds(state, robot_model=ctx.robot_model, group=ctx.group)
    state.update()
    return state


def parse_joint_positions(text: str, expected_dof: int) -> Optional[List[float]]:
    """
    Parse joint positions from a string like:
    "0.0,-0.785,0.0,-2.356,0.0,1.571,0.785" (commas/spaces both OK)
    """
    if not text or not text.strip():
        return None

    s = text.strip().replace("[", "").replace("]", "").replace(";", ",")
    parts = []
    for token in s.replace(",", " ").split():
        try:
            parts.append(float(token))
        except ValueError:
            return None

    if len(parts) != expected_dof:
        return None
    return parts
