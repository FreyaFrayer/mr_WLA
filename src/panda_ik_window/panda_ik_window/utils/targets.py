from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from moveit.core.robot_state import RobotState

from ..types import TargetPoint
from .robot import RobotContext

PATH_PATTERN_CHOICES = ("trend", "switching", "random", "trend_plus")

_TREND_COHERENCE_MIN = 0.50
_TREND_AVG_TURN_MAX_RAD = float(np.deg2rad(130.0))
_TREND_CONSEC_COS_MIN = -0.7
_SWITCH_TURN_MIN_RAD = float(np.deg2rad(150.0))
_SWITCH_CONSEC_COS_MAX = -0.8
_TREND_PLUS_MIN_STREAK = 3
_TREND_PLUS_STREAK_VARIANTS = 3  # 3/4/5 trend points then force a switch
_TREND_PLUS_SAFE_MARGIN_RATIO = 0.20
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
        "trend-plus": "trend_plus",
        "trend_plus": "trend_plus",
        "trendplus": "trend_plus",
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


def _is_switch_turn(prev2: TargetPoint, prev1: TargetPoint, curr: TargetPoint) -> bool:
    """Classify a local turn using the same thresholds as switching mode."""
    v1 = np.asarray(
        [float(prev1.x - prev2.x), float(prev1.y - prev2.y), float(prev1.z - prev2.z)],
        dtype=float,
    )
    v2 = np.asarray(
        [float(curr.x - prev1.x), float(curr.y - prev1.y), float(curr.z - prev1.z)],
        dtype=float,
    )
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 <= _EPS or n2 <= _EPS:
        return False

    cosv = float(np.dot(v1, v2) / (n1 * n2))
    cosv = float(np.clip(cosv, -1.0, 1.0))
    turn = float(np.arccos(cosv))
    return bool(turn >= _SWITCH_TURN_MIN_RAD or cosv <= _SWITCH_CONSEC_COS_MAX)


def _count_switches_and_trend_streak(points: Sequence[TargetPoint]) -> tuple[int, int]:
    """Replay accepted points and return (num_switch_points, trailing_trend_points)."""
    if len(points) <= 1:
        return 0, 0

    switch_count = 0
    trend_streak = 0
    for idx in range(1, len(points)):
        # First accepted point has no turn context; treat as trend.
        if idx < 2:
            is_switch = False
        else:
            is_switch = _is_switch_turn(points[idx - 2], points[idx - 1], points[idx])

        if is_switch:
            switch_count += 1
            trend_streak = 0
        else:
            trend_streak += 1

    return int(switch_count), int(trend_streak)


def _trend_plus_streak_cap(switch_count: int) -> int:
    """Cycle 3 -> 4 -> 5 trend points between switch points."""
    phase = int(max(0, switch_count)) % int(_TREND_PLUS_STREAK_VARIANTS)
    return int(_TREND_PLUS_MIN_STREAK + phase)


def _match_path_pattern(
    *,
    pattern: str,
    path_anchor: Optional[TargetPoint],
    existing_points: Sequence[TargetPoint],
    candidate: TargetPoint,
    workspace: Optional["WorkspaceBounds"] = None,
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

    if pattern == "trend_plus":
        ws = workspace if workspace is not None else WorkspaceBounds()
        history: List[TargetPoint] = []
        if path_anchor is not None:
            history.append(path_anchor)
        history.extend(existing_points)

        switch_count, trend_streak = _count_switches_and_trend_streak(history)
        streak_cap = _trend_plus_streak_cap(switch_count)

        has_turn_context = len(history) >= 2
        candidate_is_switch = (
            _is_switch_turn(history[-2], history[-1], candidate) if has_turn_context else False
        )

        last_selected = history[-1] if len(history) > 0 else None
        last_in_danger = (last_selected is not None and ws.in_danger_zone(last_selected))

        # Trigger conditions:
        # 1) periodic anti-stall switch after 3~5 trend points;
        # 2) if previous selected point is already in danger zone, force turning back.
        force_switch = bool(trend_streak >= streak_cap or last_in_danger)
        if not force_switch:
            if H < 2:
                return True
            return is_trend

        if last_in_danger and not ws.in_safe_zone(candidate):
            return False

        if not has_turn_context:
            # For very early points (insufficient turn context), only keep the safety rule.
            return True

        return candidate_is_switch

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

    def _safe_bounds(self, margin_ratio: float = _TREND_PLUS_SAFE_MARGIN_RATIO) -> tuple[float, float, float, float, float, float]:
        ratio = float(max(0.0, min(0.45, margin_ratio)))
        sx = float(self.x_max - self.x_min)
        sy = float(self.y_max - self.y_min)
        sz = float(self.z_max - self.z_min)
        mx = ratio * sx
        my = ratio * sy
        mz = ratio * sz
        return (
            float(self.x_min + mx), float(self.x_max - mx),
            float(self.y_min + my), float(self.y_max - my),
            float(self.z_min + mz), float(self.z_max - mz),
        )

    def in_safe_zone(self, p: TargetPoint, *, margin_ratio: float = _TREND_PLUS_SAFE_MARGIN_RATIO) -> bool:
        if not self.contains(p):
            return False
        x_min, x_max, y_min, y_max, z_min, z_max = self._safe_bounds(margin_ratio=margin_ratio)
        return (x_min <= p.x <= x_max and
                y_min <= p.y <= y_max and
                z_min <= p.z <= z_max)

    def in_danger_zone(self, p: TargetPoint, *, margin_ratio: float = _TREND_PLUS_SAFE_MARGIN_RATIO) -> bool:
        if not self.contains(p):
            return False
        return not self.in_safe_zone(p, margin_ratio=margin_ratio)


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
    - `path_pattern` supports: trend / switching / random / trend_plus.
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
            workspace=workspace,
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
