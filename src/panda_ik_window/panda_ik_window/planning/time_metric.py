from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Optional, Sequence, Tuple

import numpy as np


TimeModelName = Literal["auto", "totg", "trapezoid"]


@dataclass
class TotgSettings:
    """Configuration for MoveIt TOTG time parameterization.

    Notes
    -----
    - vel_scale / acc_scale are global scaling factors (1.0 means use the full joint limits).
    - The other parameters are passed to MoveIt TOTG if supported by the installed bindings.
    """

    vel_scale: float = 1.0
    acc_scale: float = 1.0
    path_tolerance: float = 0.1
    resample_dt: float = 0.1
    min_angle_change: float = 0.001


@dataclass
class TimeModelInfo:
    requested: str
    effective: str
    note: str = ""
    totg_available: bool = False
    totg_failures: int = 0


def _estimate_time_trapezoid_s(
    q_from: Sequence[float],
    q_to: Sequence[float],
    *,
    max_vel_rad_s: Sequence[float],
    max_acc_rad_s2: Sequence[float],
    eps: float = 1e-9,
) -> float:
    """Rest-to-rest minimal time under per-joint v/a limits (triangle/trapezoid profile).

    For each joint i:

    - Let d = |dq_i|.
    - If d <= v^2/a: triangular profile, t = 2*sqrt(d/a)
    - Else: trapezoidal profile, t = d/v + v/a

    The segment time is the *synchronised* time:
        t = max_i t_i

    This matches the common MoveIt assumption for point-to-point joint motion when
    start/end velocities are 0 ("stop at each waypoint").
    """

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
    """Vectorized trapezoid/triangle segment-time matrix.

    prev: (Kprev, dof)
    curr: (Kcurr, dof)
    returns: (Kprev, Kcurr)
    """

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


class _TotgEngine:
    """A small wrapper around MoveIt Python bindings for TOTG.

    This tries to keep allocations low by reusing RobotState / RobotTrajectory objects.
    """

    def __init__(
        self,
        *,
        moveit_py,
        robot_model,
        group: str,
        joint_names: Sequence[str],
        settings: TotgSettings,
    ) -> None:
        self._moveit_py = moveit_py
        self._robot_model = robot_model
        self._group = str(group)
        self._joint_names = list(joint_names)
        self._settings = settings

        # Lazy imports (so trapezoid mode works without ROS msg imports during unit tests)
        try:
            from moveit.core.robot_state import RobotState  # noqa: F401
            from moveit.core.robot_trajectory import RobotTrajectory  # noqa: F401
            from moveit_msgs.msg import RobotTrajectory as RobotTrajectoryMsg  # noqa: F401
            from trajectory_msgs.msg import JointTrajectoryPoint  # noqa: F401
            from builtin_interfaces.msg import Duration  # noqa: F401
        except Exception as e:  # pragma: no cover
            raise RuntimeError(f"TOTG imports unavailable: {e}")

        from moveit.core.robot_state import RobotState
        from moveit.core.robot_trajectory import RobotTrajectory

        self._RobotState = RobotState
        self._RobotTrajectory = RobotTrajectory

        # Reusable objects
        self._state = RobotState(self._robot_model)
        self._state.set_to_default_values()
        self._traj = RobotTrajectory(self._robot_model)
        # Some bindings expose this property; ignore failures.
        try:
            self._traj.joint_model_group_name = self._group
        except Exception:
            pass

        # Validate API availability early
        if not hasattr(self._traj, "set_robot_trajectory_msg"):
            raise RuntimeError("RobotTrajectory.set_robot_trajectory_msg not available in this MoveItPy build")

        if not hasattr(self._traj, "apply_totg_time_parameterization"):
            raise RuntimeError("RobotTrajectory.apply_totg_time_parameterization not available in this MoveItPy build")

    def duration_s(self, q_from: Sequence[float], q_to: Sequence[float]) -> Optional[float]:
        """Return duration in seconds, or None if TOTG failed."""
        q_from = [float(v) for v in q_from]
        q_to = [float(v) for v in q_to]

        # Local imports to keep module import light
        from moveit_msgs.msg import RobotTrajectory as RobotTrajectoryMsg
        from trajectory_msgs.msg import JointTrajectoryPoint
        from builtin_interfaces.msg import Duration

        # Update start state
        self._state.set_to_default_values()
        self._state.set_joint_group_positions(self._group, q_from)
        self._state.update()

        msg = RobotTrajectoryMsg()
        msg.joint_trajectory.joint_names = list(self._joint_names)

        p0 = JointTrajectoryPoint()
        p0.positions = list(q_from)
        p0.velocities = [0.0] * len(q_from)
        p0.time_from_start = Duration(sec=0, nanosec=0)

        p1 = JointTrajectoryPoint()
        p1.positions = list(q_to)
        p1.velocities = [0.0] * len(q_to)
        # Placeholder (must be monotonic). TOTG will overwrite timing anyway.
        p1.time_from_start = Duration(sec=1, nanosec=0)

        msg.joint_trajectory.points = [p0, p1]

        try:
            self._traj.set_robot_trajectory_msg(self._state, msg)
        except Exception:
            # Some builds may use a different method name.
            try:
                getattr(self._traj, "setRobotTrajectoryMsg")(self._state, msg)  # type: ignore[attr-defined]
            except Exception:
                return None

        fn = getattr(self._traj, "apply_totg_time_parameterization", None)
        if fn is None:
            return None

        s = self._settings
        ok = False
        try:
            # Newer bindings (MoveItPy docs)
            ok = bool(fn(float(s.vel_scale), float(s.acc_scale), float(s.path_tolerance), float(s.resample_dt), float(s.min_angle_change)))
        except TypeError:
            # Older bindings may accept fewer params
            try:
                ok = bool(fn(float(s.vel_scale), float(s.acc_scale)))
            except TypeError:
                try:
                    ok = bool(fn())
                except Exception:
                    ok = False
        except Exception:
            ok = False

        if not ok:
            return None

        # Duration access varies; prefer the documented property.
        try:
            return float(getattr(self._traj, "duration"))
        except Exception:
            pass
        try:
            return float(getattr(self._traj, "get_duration")())
        except Exception:
            return None


class SegmentTimeModel:
    """Unified segment-time computation with three modes.

    Modes
    -----
    - trapezoid: fast, deterministic, uses velocity + acceleration limits (rest-to-rest)
    - totg: calls MoveIt TOTG time parameterization (rest-to-rest)
    - auto: prefer totg if available, otherwise fall back to trapezoid
    """

    def __init__(
        self,
        *,
        model: TimeModelName,
        max_vel_rad_s: Sequence[float],
        max_acc_rad_s2: Sequence[float],
        # TOTG context
        moveit_py=None,
        robot_model=None,
        group: str = "",
        joint_names: Sequence[str] = (),
        totg_settings: Optional[TotgSettings] = None,
    ) -> None:
        self._vmax = np.asarray(max_vel_rad_s, dtype=float)
        self._amax = np.asarray(max_acc_rad_s2, dtype=float)
        self._model_req: str = str(model)

        if totg_settings is None:
            totg_settings = TotgSettings()
        self._totg_settings = totg_settings

        self._totg_engine: Optional[_TotgEngine] = None
        self._info = TimeModelInfo(requested=self._model_req, effective="trapezoid")

        # Decide effective model
        if model == "trapezoid":
            self._info.effective = "trapezoid"
            return

        if model == "totg" or model == "auto":
            # Try build TOTG engine
            try:
                if moveit_py is None or robot_model is None or not str(group):
                    raise RuntimeError("moveit_py/robot_model/group missing")
                self._totg_engine = _TotgEngine(
                    moveit_py=moveit_py,
                    robot_model=robot_model,
                    group=str(group),
                    joint_names=joint_names,
                    settings=totg_settings,
                )
                self._info.totg_available = True
            except Exception as e:
                self._totg_engine = None
                self._info.totg_available = False
                self._info.note = f"TOTG unavailable: {e}"

            if self._totg_engine is not None:
                self._info.effective = "totg"
                return

            # Auto fall back, totg mode should be strict
            if model == "totg":
                raise RuntimeError(self._info.note or "TOTG unavailable")

            self._info.effective = "trapezoid"
            if self._info.note:
                self._info.note += " | fallback=trapezoid"
            else:
                self._info.note = "fallback=trapezoid"

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
        if self._info.effective == "totg" and self._totg_engine is not None:
            t = self._totg_engine.duration_s(q_from, q_to)
            if t is not None and math.isfinite(float(t)):
                return float(t)

            # TOTG runtime failure: fall back to trapezoid (even if requested was totg).
            self._info.totg_failures += 1
            return _estimate_time_trapezoid_s(
                q_from,
                q_to,
                max_vel_rad_s=self._vmax,
                max_acc_rad_s2=self._amax,
            )

        return _estimate_time_trapezoid_s(
            q_from,
            q_to,
            max_vel_rad_s=self._vmax,
            max_acc_rad_s2=self._amax,
        )

    def segment_time_matrix_s(self, prev: np.ndarray, curr: np.ndarray) -> np.ndarray:
        """Compute (Kprev, Kcurr) matrix.

        - trapezoid: vectorized
        - totg: computed by nested loops (slow)
        """

        if self._info.effective != "totg" or self._totg_engine is None:
            return _estimate_time_matrix_trapezoid_s(
                prev,
                curr,
                max_vel_rad_s=self._vmax,
                max_acc_rad_s2=self._amax,
            )

        prev = np.asarray(prev, dtype=float)
        curr = np.asarray(curr, dtype=float)
        Kprev = int(prev.shape[0])
        Kcurr = int(curr.shape[0])
        out = np.zeros((Kprev, Kcurr), dtype=float)

        # Nested loop is expensive; keep it tight.
        for i in range(Kprev):
            qi = prev[i]
            for j in range(Kcurr):
                out[i, j] = self.segment_time_s(qi, curr[j])
        return out
