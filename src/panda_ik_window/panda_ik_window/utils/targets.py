from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from moveit.core.robot_state import RobotState

from ..types import TargetPoint
from .robot import RobotContext


@dataclass(frozen=True)
class WorkspaceBounds:
    """
    Simple axis-aligned workspace bounds (meters).
    Used only to reject obviously bad samples (e.g., below the table).
    """
    x_min: float = 0.15
    x_max: float = 0.75
    y_min: float = -0.55
    y_max: float = 0.55
    z_min: float = 0.05
    z_max: float = 0.85

    def contains(self, p: TargetPoint) -> bool:
        return (self.x_min <= p.x <= self.x_max and
                self.y_min <= p.y <= self.y_max and
                self.z_min <= p.z <= self.z_max)


def _euclidean(a: TargetPoint, b: TargetPoint) -> float:
    dx = float(a.x - b.x)
    dy = float(a.y - b.y)
    dz = float(a.z - b.z)
    return float((dx * dx + dy * dy + dz * dz) ** 0.5)


def sample_one_reachable_point_fk(
    ctx: RobotContext,
    *,
    rng: np.random.Generator,
    existing_points: Sequence[TargetPoint] = (),
    min_separation_m: float = 0.06,
    workspace: Optional[WorkspaceBounds] = None,
    max_attempts: int = 2000,
) -> TargetPoint:
    """Sample a single *FK-reachable* Cartesian point (position only).

    This is similar to :func:`sample_reachable_points`, but designed for
    **sequential acceptance** (e.g., resampling when IK fails).

    Key idea
    --------
    We sample a random joint vector within the group bounds, run FK, and keep
    the end-effector position. This guarantees kinematic reachability.

    Notes
    -----
    - We use a NumPy RNG to keep reproducibility stable.
    - Minimal separation to previously accepted points is enforced.
    """
    workspace = workspace or WorkspaceBounds()

    # Sample directly from joint limits (stable + reproducible).
    lows = np.array([jl.min_position for jl in ctx.joint_limits], dtype=float)
    highs = np.array([jl.max_position for jl in ctx.joint_limits], dtype=float)
    if lows.shape[0] != highs.shape[0] or int(lows.shape[0]) != int(ctx.dof):
        raise RuntimeError("Joint limit shape mismatch; cannot sample FK points reliably.")

    for _ in range(int(max_attempts)):
        q = rng.uniform(lows, highs).astype(float).tolist()

        state = RobotState(ctx.robot_model)
        state.set_to_default_values()
        state.set_joint_group_positions(ctx.group, q)
        state.update()

        pose = state.get_pose(ctx.tip_link)
        p = TargetPoint(
            x=float(pose.position.x),
            y=float(pose.position.y),
            z=float(pose.position.z),
        )

        if not workspace.contains(p):
            continue

        ok = True
        for q_prev in existing_points:
            if _euclidean(p, q_prev) < float(min_separation_m):
                ok = False
                break
        if not ok:
            continue

        return p

    raise RuntimeError(
        f"Failed to sample a FK-reachable point within {max_attempts} attempts. "
        f"Try relaxing workspace/min_separation."
    )


def sample_reachable_points(
    ctx: RobotContext,
    n: int,
    *,
    seed: int = 7,
    min_separation_m: float = 0.06,
    workspace: Optional[WorkspaceBounds] = None,
    max_attempts: int = 5000,
) -> List[TargetPoint]:
    """
    Sample `n` reachable Cartesian points by:
      1) sampling a random joint configuration within the planning group
      2) computing forward kinematics to obtain end-effector pose
      3) keeping only the position part

    This guarantees kinematic reachability because each point comes from a valid FK.

    Parameters
    ----------
    min_separation_m:
        Minimal distance between points to avoid duplicates.
    workspace:
        Optional AABB bounds to keep points in a reasonable region.
    """
    if n <= 0:
        return []

    rng = np.random.default_rng(int(seed))
    points: List[TargetPoint] = []
    workspace = workspace or WorkspaceBounds()

    for _ in range(int(max_attempts)):
        if len(points) >= n:
            break
        try:
            p = sample_one_reachable_point_fk(
                ctx,
                rng=rng,
                existing_points=points,
                min_separation_m=float(min_separation_m),
                workspace=workspace,
                max_attempts=50,
            )
        except RuntimeError:
            continue
        points.append(p)

    if len(points) < n:
        raise RuntimeError(
            f"Failed to sample {n} reachable points within {max_attempts} attempts. "
            f"Got {len(points)} points. Try relaxing workspace/min_separation."
        )

    return points
