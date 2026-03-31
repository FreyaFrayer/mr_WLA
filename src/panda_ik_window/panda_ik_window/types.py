from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence


@dataclass(frozen=True)
class TargetPoint:
    """A reachable Cartesian target point in the planning frame (usually base frame), meters."""
    x: float
    y: float
    z: float

    def as_list(self) -> List[float]:
        return [float(self.x), float(self.y), float(self.z)]


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
