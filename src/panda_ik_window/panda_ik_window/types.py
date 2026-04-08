from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple


def normalize_quat_xyzw(
    q: Sequence[float] | None,
    *,
    fallback_xyzw: Sequence[float] = (0.0, 0.0, 0.0, 1.0),
) -> Tuple[float, float, float, float]:
    """Normalize a quaternion in (x, y, z, w) order.

    If input is missing/invalid/degenerate, return a normalized fallback.
    """
    def _norm_or_default(raw: Sequence[float]) -> Tuple[float, float, float, float]:
        vals = tuple(float(v) for v in raw[:4]) if len(raw) >= 4 else (0.0, 0.0, 0.0, 1.0)
        if len(vals) != 4:
            vals = (0.0, 0.0, 0.0, 1.0)
        x, y, z, w = vals
        if not all(math.isfinite(v) for v in (x, y, z, w)):
            return (0.0, 0.0, 0.0, 1.0)
        n = math.sqrt(x * x + y * y + z * z + w * w)
        if n <= 1e-12:
            return (0.0, 0.0, 0.0, 1.0)
        return (x / n, y / n, z / n, w / n)

    fallback = _norm_or_default(tuple(float(v) for v in fallback_xyzw))
    if q is None:
        return fallback
    try:
        raw = tuple(float(v) for v in q)
    except Exception:
        return fallback
    if len(raw) < 4:
        return fallback

    x, y, z, w = raw[:4]
    if not all(math.isfinite(v) for v in (x, y, z, w)):
        return fallback
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n <= 1e-12:
        return fallback
    return (x / n, y / n, z / n, w / n)


@dataclass(frozen=True)
class TargetPoint:
    """A reachable Cartesian target pose in the planning frame (usually base frame)."""
    x: float
    y: float
    z: float
    qx: float = 0.0
    qy: float = 0.0
    qz: float = 0.0
    qw: float = 1.0

    def as_list(self) -> List[float]:
        return [float(self.x), float(self.y), float(self.z)]

    def as_quat_xyzw(self, *, normalized: bool = False) -> List[float]:
        if normalized:
            qx, qy, qz, qw = self.normalized_quat_xyzw()
            return [qx, qy, qz, qw]
        return [float(self.qx), float(self.qy), float(self.qz), float(self.qw)]

    def normalized_quat_xyzw(
        self,
        *,
        fallback_xyzw: Sequence[float] = (0.0, 0.0, 0.0, 1.0),
    ) -> Tuple[float, float, float, float]:
        return normalize_quat_xyzw(
            (float(self.qx), float(self.qy), float(self.qz), float(self.qw)),
            fallback_xyzw=fallback_xyzw,
        )

    def with_orientation(self, quat_xyzw: Sequence[float]) -> "TargetPoint":
        qx, qy, qz, qw = normalize_quat_xyzw(quat_xyzw)
        return TargetPoint(
            x=float(self.x),
            y=float(self.y),
            z=float(self.z),
            qx=float(qx),
            qy=float(qy),
            qz=float(qz),
            qw=float(qw),
        )


@dataclass(frozen=True)
class IKSolution:
    """
    One IK solution for a target point.

    Notes
    -----
    - `attempt` is the internal sampling attempt counter (useful for debugging / reproducibility).
    - `index` is the 0-based index within the JSON list (stable and convenient for reporting).
    """
    index: int
    attempt: int
    sampled_yaw_rad: float
    joint_names: Sequence[str]
    joint_positions: Sequence[float]
