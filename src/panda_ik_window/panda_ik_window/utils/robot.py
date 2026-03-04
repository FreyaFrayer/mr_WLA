from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from moveit.core.robot_state import RobotState
from moveit.planning import MoveItPy


@dataclass(frozen=True)
class JointLimit:
    min_position: float
    max_position: float
    max_velocity: float  # rad/s (fallback to a sane default if unknown)
    max_acceleration: float  # rad/s^2 (fallback to a sane default if unknown)

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
    def acceleration_limits(self) -> np.ndarray:
        return np.array([jl.max_acceleration for jl in self.joint_limits], dtype=float)

    @property
    def position_bounds(self) -> List[Tuple[float, float]]:
        return [(jl.min_position, jl.max_position) for jl in self.joint_limits]


def _infer_tip_link_from_jmg(jmg) -> str:
    if not getattr(jmg, "link_model_names", None):
        raise RuntimeError("JointModelGroup.link_model_names is empty; cannot infer tip_link.")
    # heuristic: the last link is the tip link of the kinematic chain
    return str(list(jmg.link_model_names)[-1])


def _extract_joint_limits_from_jmg(
    jmg,
    *,
    default_vel: float = 2.0,
    default_acc: float = 4.0,
) -> List[JointLimit]:
    """
    Try to read joint limits from MoveItPy bindings.

    We rely on `jmg.active_joint_model_bounds` (similar to the provided IK sampler).
    If velocity/acceleration fields are missing, fall back to `default_vel` / `default_acc`.
    """
    limits: List[JointLimit] = []
    bounds = None
    try:
        bounds = list(jmg.active_joint_model_bounds)
    except Exception:
        bounds = None

    if bounds is None or len(bounds) == 0:
        # fallback: use wide bounds
        # Panda 7DoF typical range is within [-2.9, 2.9], but we avoid hardcoding if possible.
        # Still, we keep it safe.
        for _ in range(int(len(getattr(jmg, "joint_model_names", [])))):
            limits.append(
                JointLimit(
                    min_position=-math.pi,
                    max_position=math.pi,
                    max_velocity=float(default_vel),
                    max_acceleration=float(default_acc),
                )
            )
        return limits

    for b in bounds:
        # Position bounds
        lo = -math.pi
        hi = math.pi
        try:
            if bool(getattr(b, "position_bounded")):
                lo = float(getattr(b, "min_position"))
                hi = float(getattr(b, "max_position"))
        except Exception:
            pass

        # Velocity limit (if available)
        vmax = float(default_vel)
        for attr in ("max_velocity", "max_velocity_", "velocity", "max_velocity_limit"):
            if hasattr(b, attr):
                try:
                    v = float(getattr(b, attr))
                    if v > 1e-6:
                        vmax = v
                        break
                except Exception:
                    pass

        # Acceleration limit (if available)
        amax = float(default_acc)
        for attr in ("max_acceleration", "max_acceleration_", "acceleration", "max_acceleration_limit"):
            if hasattr(b, attr):
                try:
                    a = float(getattr(b, attr))
                    if a > 1e-6:
                        amax = a
                        break
                except Exception:
                    pass

        limits.append(JointLimit(min_position=lo, max_position=hi, max_velocity=vmax, max_acceleration=amax))

    return limits


def load_robot_context(
    *,
    node_name: str = "panda_ik_window",
    group: str = "panda_arm",
    tip_link: str = "",
    default_vel: float = 2.0,
    default_acc: float = 4.0,
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

    joint_limits = _extract_joint_limits_from_jmg(jmg, default_vel=float(default_vel), default_acc=float(default_acc))
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

    state = RobotState(ctx.robot_model)
    state.set_to_default_values()
    state.set_joint_group_positions(ctx.group, q)
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
