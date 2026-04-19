from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class TimeModelInfo:
    requested: str = "trapezoid"
    effective: str = "trapezoid"
    note: str = ""


def _estimate_time_trapezoid_s(
    q_from: Sequence[float],
    q_to: Sequence[float],
    *,
    max_vel_rad_s: Sequence[float],
    max_acc_rad_s2: Sequence[float],
    eps: float = 1e-9,
) -> float:
    """Rest-to-rest minimal time under per-joint v/a limits (triangle/trapezoid profile)."""

    q_from = np.asarray(q_from, dtype=float)
    q_to = np.asarray(q_to, dtype=float)
    vmax = np.asarray(max_vel_rad_s, dtype=float)
    amax = np.asarray(max_acc_rad_s2, dtype=float)

    if q_from.shape != q_to.shape:
        raise ValueError(f"q_from shape {q_from.shape} != q_to shape {q_to.shape}")
    if vmax.shape[0] != q_from.shape[0]:
        raise ValueError(f"vmax length {vmax.shape[0]} != dof {q_from.shape[0]}")
    if amax.shape[0] != q_from.shape[0]:
        raise ValueError(f"amax length {amax.shape[0]} != dof {q_from.shape[0]}")

    vmax = np.maximum(vmax, float(eps))
    amax = np.maximum(amax, float(eps))

    dq = np.abs(q_to - q_from)
    dcrit = (vmax * vmax) / amax

    t_tri = 2.0 * np.sqrt(dq / amax)
    t_trap = (dq / vmax) + (vmax / amax)
    t_joint = np.where(dq <= dcrit, t_tri, t_trap)
    return float(np.max(t_joint))


def _estimate_time_matrix_trapezoid_s(
    prev: np.ndarray,
    curr: np.ndarray,
    *,
    max_vel_rad_s: np.ndarray,
    max_acc_rad_s2: np.ndarray,
    eps: float = 1e-9,
) -> np.ndarray:
    """Vectorized trapezoid/triangle segment-time matrix."""

    prev = np.asarray(prev, dtype=float)
    curr = np.asarray(curr, dtype=float)
    vmax = np.asarray(max_vel_rad_s, dtype=float)
    amax = np.asarray(max_acc_rad_s2, dtype=float)

    if prev.ndim != 2 or curr.ndim != 2:
        raise ValueError("prev/curr must be 2D arrays")
    if prev.shape[1] != curr.shape[1]:
        raise ValueError(f"dof mismatch: prev {prev.shape}, curr {curr.shape}")
    dof = int(prev.shape[1])
    if vmax.shape[0] != dof or amax.shape[0] != dof:
        raise ValueError(f"limit length mismatch: vmax {vmax.shape}, amax {amax.shape}, dof {dof}")

    vmax = np.maximum(vmax.reshape((1, 1, dof)), float(eps))
    amax = np.maximum(amax.reshape((1, 1, dof)), float(eps))

    dq = np.abs(prev[:, None, :] - curr[None, :, :])  # (Kprev, Kcurr, dof)
    dcrit = (vmax * vmax) / amax

    t_tri = 2.0 * np.sqrt(dq / amax)
    t_trap = (dq / vmax) + (vmax / amax)
    t_joint = np.where(dq <= dcrit, t_tri, t_trap)
    return np.max(t_joint, axis=2)


class SegmentTimeModel:
    """Trapezoid-only segment-time computation."""

    def __init__(
        self,
        *,
        max_vel_rad_s: Sequence[float],
        max_acc_rad_s2: Sequence[float],
        note: str = "",
    ) -> None:
        self._vmax = np.asarray(max_vel_rad_s, dtype=float)
        self._amax = np.asarray(max_acc_rad_s2, dtype=float)
        self._info = TimeModelInfo(note=str(note))

    @property
    def info(self) -> TimeModelInfo:
        return self._info

    @property
    def vmax(self) -> np.ndarray:
        return self._vmax

    @property
    def amax(self) -> np.ndarray:
        return self._amax

    def segment_time_s(self, q_from: Sequence[float], q_to: Sequence[float]) -> float:
        return _estimate_time_trapezoid_s(
            q_from,
            q_to,
            max_vel_rad_s=self._vmax,
            max_acc_rad_s2=self._amax,
        )

    def segment_time_matrix_s(self, prev: np.ndarray, curr: np.ndarray) -> np.ndarray:
        return _estimate_time_matrix_trapezoid_s(
            prev,
            curr,
            max_vel_rad_s=self._vmax,
            max_acc_rad_s2=self._amax,
        )
