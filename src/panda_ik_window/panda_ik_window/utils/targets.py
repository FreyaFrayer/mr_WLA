from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from moveit.core.robot_state import RobotState

from ..types import TargetPoint
from .robot import RobotContext

PATH_PATTERN_CHOICES = ("trend", "switching", "random")

_TREND_COHERENCE_MIN = 0.50
_TREND_AVG_TURN_MAX_RAD = float(np.deg2rad(130.0))
_TREND_CONSEC_COS_MIN = -0.7
_SWITCH_TURN_MIN_RAD = float(np.deg2rad(150.0))
_SWITCH_CONSEC_COS_MAX = -0.8
_EPS = 1e-9


def normalize_path_pattern(raw: str) -> str:
    """Normalize user-facing path pattern names to internal canonical ids."""
    key = "" if raw is None else str(raw).strip().lower().replace("_", "-")
    if key == "":
        return "random"

    alias = {
        "trend": "trend",
        "direction-consistent-trend": "trend",
        "direction-consistent": "trend",
        "switching": "switching",
        "multi-directional-switching": "switching",
        "multi-directional": "switching",
        "random": "random",
        "unstructured": "random",
        "directionally-unstructured": "random",
    }

    out = alias.get(key)
    if out is None:
        raise ValueError(f"Invalid path pattern: {raw!r}. Choose from: {PATH_PATTERN_CHOICES}.")
    return out


def _path_metrics(points: Sequence[TargetPoint]) -> tuple[int, float, float, float, np.ndarray]:
    """Return (H, C, theta_avg, theta_max, consecutive_dot_products)."""
    if len(points) < 2:
        return 0, 0.0, 0.0, 0.0, np.zeros((0,), dtype=float)

    xyz = np.asarray([[float(p.x), float(p.y), float(p.z)] for p in points], dtype=float)
    d = np.diff(xyz, axis=0)
    norms = np.linalg.norm(d, axis=1)
    if d.shape[0] == 0 or np.any(norms <= _EPS):
        return 0, 0.0, 0.0, 0.0, np.zeros((0,), dtype=float)

    u = d / norms[:, None]
    H = int(u.shape[0])
    coherence = float(np.linalg.norm(np.sum(u, axis=0)) / float(H))

    if H < 2:
        return H, coherence, 0.0, 0.0, np.zeros((0,), dtype=float)

    dots = np.sum(u[:-1] * u[1:], axis=1)
    dots = np.clip(dots, -1.0, 1.0).astype(float)
    theta = np.arccos(dots)
    theta_avg = float(np.mean(theta))
    theta_max = float(np.max(theta))
    return H, coherence, theta_avg, theta_max, dots


def _is_trend_pattern(H: int, coherence: float, theta_avg: float, dots: np.ndarray) -> bool:
    if H < 2:
        return False
    if coherence < _TREND_COHERENCE_MIN:
        return False
    if theta_avg > _TREND_AVG_TURN_MAX_RAD:
        return False
    return bool(dots.size == 0 or np.all(dots >= _TREND_CONSEC_COS_MIN))


def _is_switching_pattern(H: int, theta_max: float, dots: np.ndarray) -> bool:
    if H < 2:
        return False
    if theta_max >= _SWITCH_TURN_MIN_RAD:
        return True
    return bool(dots.size > 0 and np.any(dots <= _SWITCH_CONSEC_COS_MAX))


def _match_path_pattern(
    *,
    pattern: str,
    path_anchor: Optional[TargetPoint],
    existing_points: Sequence[TargetPoint],
    candidate: TargetPoint,
) -> bool:
    points: List[TargetPoint] = []
    if path_anchor is not None:
        points.append(path_anchor)
    points.extend(existing_points)
    points.append(candidate)

    H, coherence, theta_avg, theta_max, dots = _path_metrics(points)
    is_trend = _is_trend_pattern(H, coherence, theta_avg, dots)
    is_switching = _is_switching_pattern(H, theta_max, dots)

    if pattern == "trend":
        if H < 2:
            return True
        return is_trend

    if pattern == "switching":
        if H < 2:
            return True
        return is_switching

    if pattern == "random":
        # Random mode is intentionally unconstrained by direction class.
        # It may look trend-like, switching-like, or unstructured.
        return True

    raise ValueError(f"Unsupported path pattern: {pattern!r}")


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
    path_anchor: Optional[TargetPoint] = None,
    path_pattern: str = "random",
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
    - `path_pattern` follows Path Pattern Definition: trend / switching / random.
    """
    workspace = workspace or WorkspaceBounds()
    pattern = normalize_path_pattern(path_pattern)

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

        if not _match_path_pattern(
            pattern=pattern,
            path_anchor=path_anchor,
            existing_points=existing_points,
            candidate=p,
        ):
            continue

        return p

    raise RuntimeError(
        f"Failed to sample a FK-reachable point within {max_attempts} attempts "
        f"(path_pattern={pattern!r}). Try relaxing workspace/min_separation."
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
