from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class JointTrapezoidProfile:
    sign: float
    distance: float
    accel: float
    v_peak: float
    t_acc: float
    t_cruise: float
    duration: float


def _build_joint_min_time_profile(
    delta: float,
    *,
    vmax: float,
    amax: float,
    eps: float = 1e-9,
) -> JointTrapezoidProfile:
    sign = 1.0 if delta >= 0.0 else -1.0
    dist = float(abs(delta))
    vmax = max(float(vmax), float(eps))
    amax = max(float(amax), float(eps))

    if dist <= eps:
        return JointTrapezoidProfile(
            sign=sign,
            distance=0.0,
            accel=amax,
            v_peak=0.0,
            t_acc=0.0,
            t_cruise=0.0,
            duration=0.0,
        )

    d_crit = (vmax * vmax) / amax
    if dist <= d_crit:
        # Triangle profile.
        t_acc = math.sqrt(dist / amax)
        v_peak = amax * t_acc
        t_cruise = 0.0
        duration = 2.0 * t_acc
    else:
        # Trapezoid profile.
        t_acc = vmax / amax
        v_peak = vmax
        t_cruise = (dist - d_crit) / vmax
        duration = (2.0 * t_acc) + t_cruise

    return JointTrapezoidProfile(
        sign=sign,
        distance=dist,
        accel=amax,
        v_peak=v_peak,
        t_acc=t_acc,
        t_cruise=t_cruise,
        duration=duration,
    )


def _sample_distance_on_profile(profile: JointTrapezoidProfile, t: np.ndarray) -> np.ndarray:
    """Sample traveled distance (>=0) for a min-time trapezoid/triangle profile."""
    t = np.asarray(t, dtype=float)
    out = np.zeros_like(t, dtype=float)

    if profile.distance <= 0.0:
        return out

    ta = profile.t_acc
    tc = profile.t_cruise
    td0 = ta + tc
    T = profile.duration
    a = profile.accel
    v = profile.v_peak

    tt = np.clip(t, 0.0, T)

    m1 = tt <= ta
    out[m1] = 0.5 * a * tt[m1] * tt[m1]

    m2 = (tt > ta) & (tt <= td0)
    out[m2] = (0.5 * a * ta * ta) + (v * (tt[m2] - ta))

    m3 = tt > td0
    dt = tt[m3] - td0
    out[m3] = (0.5 * a * ta * ta) + (v * tc) + (v * dt) - (0.5 * a * dt * dt)

    return np.minimum(np.maximum(out, 0.0), profile.distance)


def sample_synchronized_trapezoid_segment(
    q_from: Sequence[float],
    q_to: Sequence[float],
    *,
    max_vel_rad_s: Sequence[float],
    max_acc_rad_s2: Sequence[float],
    sample_dt: float = 0.02,
    min_samples: int = 3,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Sample a synchronized rest-to-rest motion with trapezoid velocity model.

    Implementation notes
    --------------------
    1. Build each joint's min-time triangle/trapezoid profile under (vmax, amax).
    2. Let global segment duration T = max(T_i).
    3. Slow down faster joints by time-stretching: u_i = t * (T_i / T).
       This keeps trapezoid/triangle shape while respecting limits.

    Returns
    -------
    times_s: (M,)
    q_samples: (M, dof)
    duration_s: float
    """

    q0 = np.asarray(q_from, dtype=float)
    q1 = np.asarray(q_to, dtype=float)
    vmax = np.asarray(max_vel_rad_s, dtype=float)
    amax = np.asarray(max_acc_rad_s2, dtype=float)

    if q0.shape != q1.shape:
        raise ValueError(f"q_from shape {q0.shape} != q_to shape {q1.shape}")
    if q0.ndim != 1:
        raise ValueError(f"q_from/q_to must be 1D, got {q0.ndim}D")
    dof = int(q0.shape[0])
    if vmax.shape != (dof,) or amax.shape != (dof,):
        raise ValueError(f"limit length mismatch, dof={dof}, vmax={vmax.shape}, amax={amax.shape}")

    sample_dt = max(float(sample_dt), float(eps))
    min_samples = max(int(min_samples), 2)

    profiles = [
        _build_joint_min_time_profile(
            float(q1[i] - q0[i]),
            vmax=float(vmax[i]),
            amax=float(amax[i]),
            eps=eps,
        )
        for i in range(dof)
    ]

    duration = float(max((p.duration for p in profiles), default=0.0))
    if duration <= eps:
        times = np.array([0.0, 0.0], dtype=float)
        samples = np.vstack([q0, q0])
        return times, samples, 0.0

    n_samples = max(int(math.ceil(duration / sample_dt)) + 1, min_samples)
    times = np.linspace(0.0, duration, num=n_samples, dtype=float)
    qs = np.empty((n_samples, dof), dtype=float)

    for i, p in enumerate(profiles):
        if p.distance <= eps or p.duration <= eps:
            qs[:, i] = q0[i]
            continue

        local_t = times * (p.duration / duration)
        traveled = _sample_distance_on_profile(p, local_t)
        qs[:, i] = q0[i] + (p.sign * traveled)

    # Ensure exact endpoint consistency against numeric drift.
    qs[0, :] = q0
    qs[-1, :] = q1

    return times, qs, duration
