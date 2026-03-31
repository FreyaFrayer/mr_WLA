from __future__ import annotations

import json
import math
import time

from dataclasses import dataclass

@dataclass
class DeterministicRNG:
    """Local deterministic RNG (no global random state).

    This is designed to make IK sampling reproducible across runs when `seed` is fixed.
    """

    seed: int

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(int(self.seed))

    def uniform01(self) -> float:
        return float(self._rng.random())

    def uniform(self, low: float, high: float) -> float:
        return float(self._rng.uniform(float(low), float(high)))

    def uniform_vec(self, low: np.ndarray, high: np.ndarray) -> np.ndarray:
        # numpy supports vector low/high broadcast
        return np.asarray(self._rng.uniform(low, high), dtype=float)

    def normal_vec(self, n: int, *, mean: float = 0.0, std: float = 1.0) -> np.ndarray:
        return np.asarray(self._rng.normal(float(mean), float(std), int(n)), dtype=float)

    def shuffle_inplace(self, xs: List[float]) -> None:
        if len(xs) <= 1:
            return
        order = self._rng.permutation(len(xs))
        xs[:] = [xs[i] for i in order]
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from geometry_msgs.msg import Pose

from moveit.core.robot_state import RobotState

from ..types import IKSolution, TargetPoint
from ..utils.robot import RobotContext


def quat_multiply(
    q1: Tuple[float, float, float, float],
    q2: Tuple[float, float, float, float],
) -> Tuple[float, float, float, float]:
    """Hamilton product: q = q1 ⊗ q2, quaternion format (x, y, z, w)."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    return (x, y, z, w)


def quat_normalize(q: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    x, y, z, w = q
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-12:
        return (0.0, 0.0, 0.0, 1.0)
    return (x / n, y / n, z / n, w / n)


def quat_from_yaw(yaw: float) -> Tuple[float, float, float, float]:
    """Rotation around +Z axis by yaw (radians), quaternion (x,y,z,w)."""
    half = 0.5 * yaw
    return (0.0, 0.0, math.sin(half), math.cos(half))


def build_pose(point: TargetPoint, quat_xyzw: Tuple[float, float, float, float]) -> Pose:
    pose = Pose()
    pose.position.x = float(point.x)
    pose.position.y = float(point.y)
    pose.position.z = float(point.z)
    pose.orientation.x = float(quat_xyzw[0])
    pose.orientation.y = float(quat_xyzw[1])
    pose.orientation.z = float(quat_xyzw[2])
    pose.orientation.w = float(quat_xyzw[3])
    return pose


def hash_joint_vector(q: np.ndarray, resolution: float) -> Tuple[int, ...]:
    """Quantize joint vector to create a stable uniqueness key."""
    return tuple(int(round(float(v) / resolution)) for v in q)


def clamp_to_group_bounds(q: np.ndarray, jmg) -> np.ndarray:
    """
    Clamp q to the active joint bounds of JointModelGroup (if available).
    If bounds unavailable / mismatch, return q unchanged.
    """
    try:
        bounds = list(jmg.active_joint_model_bounds)
    except Exception:
        return q

    if len(bounds) != int(q.shape[0]):
        return q

    qq = np.array(q, dtype=float, copy=True)
    for i, b in enumerate(bounds):
        try:
            if bool(getattr(b, "position_bounded")):
                lo = float(getattr(b, "min_position"))
                hi = float(getattr(b, "max_position"))
                if lo <= hi:
                    if qq[i] < lo:
                        qq[i] = lo
                    elif qq[i] > hi:
                        qq[i] = hi
        except Exception:
            pass
    return qq


def infer_feature_link_for_uniformity(jmg, *, fallback_tip_link: str) -> str:
    """Pick a link whose motion best reflects the 7-DoF redundancy ("elbow"-like).

    We avoid hard-coding Panda-specific names to keep this sampler usable for
    other 7-DoF arms / different URDF naming conventions.

    Heuristic
    ---------
    Use the *middle* link in the kinematic chain, excluding base and tip when
    possible. For Panda, this is typically around ``panda_link4``.
    """
    try:
        link_names = list(getattr(jmg, "link_model_names", []))
    except Exception:
        link_names = []

    if not link_names:
        return str(fallback_tip_link)

    n = int(len(link_names))
    if n >= 3:
        mid = n // 2
        mid = max(1, min(n - 2, mid))
        return str(link_names[mid])

    return str(link_names[n // 2])


def get_link_xyz(state: RobotState, link_name: str) -> np.ndarray:
    """Get link position (x,y,z) in the planning frame."""
    pose = state.get_pose(str(link_name))
    return np.array([float(pose.position.x), float(pose.position.y), float(pose.position.z)], dtype=float)


def compute_nullspace_direction(state: RobotState, group: str, tip_link: str, rng: DeterministicRNG) -> Optional[np.ndarray]:
    """
    Compute one (randomized) nullspace direction in joint space from Jacobian J(q).

    For Panda arm (7-DoF), this is typically 1D nullspace.
    """
    ref = np.zeros(3, dtype=float)
    try:
        # Prefer explicit link_name overload if available
        J = state.get_jacobian(group, tip_link, ref, False)
    except TypeError:
        # Fallback: group + reference point overload
        try:
            J = state.get_jacobian(group, ref)
        except Exception:
            return None
    except Exception:
        return None

    if not isinstance(J, np.ndarray):
        J = np.array(J, dtype=float)
    if J.ndim != 2:
        return None

    dof = int(J.shape[1])
    if dof <= 0:
        return None

    try:
        _U, S, Vt = np.linalg.svd(J, full_matrices=True)
    except np.linalg.LinAlgError:
        return None

    tol = 1e-6
    rank = int(np.sum(S > tol))
    if rank >= dof:
        return None

    null_basis = Vt.T[:, rank:]  # (dof, null_dim)
    if null_basis.size == 0:
        return None

    weights = rng.normal_vec(int(null_basis.shape[1]), mean=0.0, std=1.0)
    n = null_basis @ weights
    n_norm = float(np.linalg.norm(n))
    if n_norm < 1e-12:
        n = null_basis[:, -1]
        n_norm = float(np.linalg.norm(n))
        if n_norm < 1e-12:
            return None

    # Canonicalize sign for reproducibility (nullspace vectors have sign ambiguity).
    try:
        k = int(np.argmax(np.abs(n)))
        if k >= 0 and float(n[k]) < 0.0:
            n = -n
    except Exception:
        pass

    return (n / n_norm).reshape((-1,))


def stratified_yaw(space_idx: int, num_spaces: int, yaw_range: float, rng: DeterministicRNG) -> float:
    """
    Stratified random sampling of yaw within [-yaw_range/2, +yaw_range/2].
    """
    if num_spaces <= 1:
        return (rng.uniform01() - 0.5) * float(yaw_range)

    bin_w = float(yaw_range) / float(num_spaces)
    return (-0.5 * float(yaw_range)) + (float(space_idx) + rng.uniform01()) * bin_w


def sample_ik_solutions(
    ctx: RobotContext,
    *,
    target_point: TargetPoint,
    nominal_tip_quat_xyzw: Tuple[float, float, float, float],
    named_start_for_seeding: str = "ready",
    num_solutions: int = 200,
    num_spaces: int = 20,
    max_attempts: int = 20000,
    ik_timeout_s: float = 0.05,
    yaw_range_rad: float = 2.0 * math.pi,
    nullspace_step: float = 0.20,
    nullspace_jitter: float = 0.02,
    uniq_resolution_rad: float = 1e-3,
    seed: int = 7,
) -> Dict:
    """
    Sample many IK solutions for one Cartesian point.

    This is adapted from the provided `ik_sampler_space.py` implementation,
    keeping the same "yaw spaces + nullspace exploration" idea.

    Returns
    -------
    A dict with:
      - meta: dict
      - solutions: List[dict]  (JSON-friendly)
      - solutions_obj: List[IKSolution]  (typed objects, convenient for search)
    """
    rng = DeterministicRNG(int(seed))

    # NOTE: This benchmark supports up to 3000 IK solutions per point.
    # Keep a hard cap to prevent accidental huge JSON / memory blowups.
    num_solutions = int(num_solutions)
    num_solutions = max(1, min(3000, num_solutions))

    num_spaces = int(num_spaces)
    num_spaces = max(1, min(num_solutions, num_spaces))

    nullspace_step = float(nullspace_step) if float(nullspace_step) > 1e-9 else 0.20
    nullspace_jitter = float(nullspace_jitter) if float(nullspace_jitter) >= 0.0 else 0.0

    solutions_per_space_target = int(math.ceil(float(num_solutions) / float(num_spaces)))
    solutions_per_space_target = max(1, solutions_per_space_target)

    robot_model = ctx.robot_model
    group = ctx.group
    tip_link = ctx.tip_link

    jmg = robot_model.get_joint_model_group(group)
    joint_names = list(ctx.joint_names)
    dof = len(joint_names)

    # "Visual" uniformity target: measure arc-length on an elbow-like link.
    # (This keeps sampling approximately uniform in the physical world, instead
    # of being uniform in joint-space.)
    feature_link = infer_feature_link_for_uniformity(jmg, fallback_tip_link=tip_link)

    # Deterministic joint seed sampling (avoid MoveIt internal RNG).
    lows = np.array([jl.min_position for jl in ctx.joint_limits], dtype=float)
    highs = np.array([jl.max_position for jl in ctx.joint_limits], dtype=float)
    if lows.shape[0] != highs.shape[0] or int(lows.shape[0]) != dof:
        # Fallback to a safe range if limits are not aligned.
        lows = np.full((dof,), -math.pi, dtype=float)
        highs = np.full((dof), +math.pi, dtype=float)

    # Handle any degenerate bounds
    span = highs - lows
    span = np.where(span > 1e-12, span, 2.0 * math.pi)
    highs = lows + span

    q_nominal = quat_normalize(tuple(map(float, nominal_tip_quat_xyzw)))

    uniq_keys = set()
    solutions: List[Dict] = []
    solutions_obj: List[IKSolution] = []

    t0 = time.time()
    attempts = 0
    successes = 0

    space_attempt_budget = max(50, int(float(max_attempts) / float(num_spaces)))

    for space_idx in range(num_spaces):
        if len(solutions) >= num_solutions or attempts >= max_attempts:
            break

        remaining_total = num_solutions - len(solutions)
        goal_in_space = min(solutions_per_space_target, remaining_total)
        if goal_in_space <= 0:
            break

        yaw = stratified_yaw(space_idx, num_spaces, float(yaw_range_rad), rng)
        q_yaw = quat_from_yaw(yaw)
        q_target = quat_normalize(quat_multiply(q_nominal, q_yaw))
        pose_target = build_pose(target_point, q_target)

        space_attempts_start = attempts
        added_in_space = 0

        # 1) find a base IK solution for this yaw space
        base_state: Optional[RobotState] = None
        base_q: Optional[np.ndarray] = None

        while (attempts < max_attempts) and ((attempts - space_attempts_start) < space_attempt_budget):
            attempts += 1

            state = RobotState(robot_model)
            state.set_to_default_values()
            q_seed = rng.uniform_vec(lows, highs)
            state.set_joint_group_positions(group, q_seed)

            ok = state.set_from_ik(group, pose_target, tip_link, float(ik_timeout_s))
            if not ok:
                continue

            successes += 1
            state.update()
            base_state = state
            base_q = np.array(state.get_joint_group_positions(group), dtype=float)
            break

        if base_state is None or base_q is None:
            continue

        # 2) Uniform nullspace exploration (predictor-corrector tracing)
        # --------------------------------------------------------------
        # Goal: instead of scanning a *fixed* nullspace direction at the base point,
        # we *track the self-motion manifold* by recomputing the Jacobian nullspace
        # at each step. Then we select solutions uniformly by the arc-length of an
        # elbow-like link in Cartesian space.

        # Candidate format: (q: np.ndarray, attempt: int, feature_xyz: np.ndarray)
        candidates: List[Tuple[np.ndarray, int, np.ndarray]] = []

        # Base candidate
        try:
            p_base = get_link_xyz(base_state, feature_link)
        except Exception:
            # Fallback if the inferred feature link is not available
            p_base = get_link_xyz(base_state, tip_link)
        candidates.append((base_q, int(attempts), p_base))

        # If there is no redundancy, fall back to random seeding (same as before).
        n0 = compute_nullspace_direction(base_state, group, tip_link, rng)

        def _budget_ok() -> bool:
            return (
                attempts < max_attempts
                and (attempts - space_attempts_start) < space_attempt_budget
            )

        def _project(seed_q: np.ndarray) -> Optional[Tuple[RobotState, np.ndarray, np.ndarray]]:
            """Project a seed onto the IK manifold; return (state, q, feature_xyz)."""
            nonlocal attempts, successes

            if not _budget_ok():
                return None

            attempts += 1
            st = RobotState(robot_model)
            st.set_to_default_values()
            st.set_joint_group_positions(group, seed_q)

            ok = st.set_from_ik(group, pose_target, tip_link, float(ik_timeout_s))
            if not ok:
                return None

            successes += 1
            st.update()
            q = np.array(st.get_joint_group_positions(group), dtype=float)
            try:
                p = get_link_xyz(st, feature_link)
            except Exception:
                p = get_link_xyz(st, tip_link)
            return st, q, p

        def _trace_direction(
            *,
            sign: float,
            q_start: np.ndarray,
            st_start: RobotState,
            p_start: np.ndarray,
            local_seen: set,
            trace_step: float,
            min_step: float,
            max_points: int,
        ) -> Tuple[List[Tuple[np.ndarray, int, np.ndarray]], bool]:
            """Trace the self-motion manifold in one direction using predictor-corrector."""
            out: List[Tuple[np.ndarray, int, np.ndarray]] = []
            closed = False

            q_cur = np.array(q_start, dtype=float, copy=True)
            st_cur = st_start
            n_prev: Optional[np.ndarray] = None

            # Closure detection thresholds (heuristic, tuned for Panda-scale motion)
            close_p_tol = 0.010  # 1 cm in Cartesian feature space
            close_q_tol = 0.10   # 0.1 rad in joint-space (helps avoid false positives)
            min_points_before_close = 15

            for _step_idx in range(int(max_points)):
                if not _budget_ok():
                    break

                nvec = compute_nullspace_direction(st_cur, group, tip_link, rng)
                if nvec is None or int(nvec.shape[0]) != int(dof):
                    break

                # Keep tangent direction continuous (avoid sign flips from SVD)
                if n_prev is not None:
                    try:
                        if float(np.dot(nvec, n_prev)) < 0.0:
                            nvec = -nvec
                    except Exception:
                        pass
                n_prev = np.array(nvec, dtype=float, copy=True)

                # Optional tiny jitter (helps avoid solver stagnation but shouldn't jump branches)
                jitter = np.zeros((dof,), dtype=float)
                if float(nullspace_jitter) > 0.0:
                    j = rng.normal_vec(int(dof), mean=0.0, std=1.0)
                    jn = float(np.linalg.norm(j))
                    if jn > 1e-12:
                        # Keep jitter small relative to the trace step
                        jitter_mag = min(float(nullspace_jitter), 0.05) * 0.25
                        jitter = (j / jn) * float(jitter_mag)

                ds = float(trace_step)
                got = False

                # Simple backtracking line-search if we hit limits / IK failure
                for _bt in range(8):
                    if not _budget_ok():
                        break

                    seed_q = q_cur + float(sign) * ds * nvec + jitter

                    # If we stepped outside joint limits, reduce step.
                    if np.any(seed_q < (lows - 1e-9)) or np.any(seed_q > (highs + 1e-9)):
                        ds *= 0.5
                        if ds < float(min_step):
                            break
                        continue

                    seed_q = clamp_to_group_bounds(seed_q, jmg)

                    proj = _project(seed_q)
                    if proj is None:
                        ds *= 0.5
                        if ds < float(min_step):
                            break
                        continue

                    st_new, q_new, p_new = proj

                    # Progress / uniqueness
                    if float(np.linalg.norm(q_new - q_cur)) < 1e-6:
                        # No progress -> stop this direction
                        ds *= 0.5
                        if ds < float(min_step):
                            break
                        continue

                    key = hash_joint_vector(q_new, float(uniq_resolution_rad))
                    if key in local_seen:
                        closed = True
                        got = False
                        break
                    local_seen.add(key)

                    out.append((q_new, int(attempts), p_new))

                    # Update current
                    q_cur = q_new
                    st_cur = st_new

                    # Soft closure: returned near the start in both feature and joint space.
                    if (
                        len(out) >= int(min_points_before_close)
                        and float(np.linalg.norm(p_new - p_start)) < float(close_p_tol)
                        and float(np.linalg.norm(q_new - q_start)) < float(close_q_tol)
                    ):
                        closed = True

                    got = True
                    break

                if not got:
                    break
                if closed:
                    break

            return out, closed

        # Try tracing only if nullspace exists at the base.
        if n0 is not None and int(getattr(n0, "shape", [0])[0]) == int(dof):
            # Trace step in joint-space. We keep it conservative for stability.
            trace_step = max(0.005, min(0.05, float(nullspace_step) * 0.25))
            min_step = max(1e-4, float(trace_step) * 0.05)

            # Local uniqueness (avoid looping in the manifold trace)
            local_seen = {hash_joint_vector(base_q, float(uniq_resolution_rad))}

            # We trace enough points to allow good arc-length resampling.
            # (Bounded by the per-space attempt budget anyway.)
            max_points = max(80, int(goal_in_space) * 20)
            max_points = min(int(max_points), 5000)

            pos_list, pos_closed = _trace_direction(
                sign=+1.0,
                q_start=base_q,
                st_start=base_state,
                p_start=p_base,
                local_seen=local_seen,
                trace_step=float(trace_step),
                min_step=float(min_step),
                max_points=int(max_points),
            )

            # If we closed a loop going forward, we already covered the circle.
            if bool(pos_closed):
                # Drop the last point if it's extremely close to the base (avoid near-duplicate)
                if pos_list:
                    try:
                        if float(np.linalg.norm(pos_list[-1][2] - p_base)) < 0.006:
                            pos_list = pos_list[:-1]
                    except Exception:
                        pass
                candidates.extend(pos_list)
            else:
                # Open manifold segment: trace both directions and stitch
                neg_list, _neg_closed = _trace_direction(
                    sign=-1.0,
                    q_start=base_q,
                    st_start=base_state,
                    p_start=p_base,
                    local_seen=local_seen,
                    trace_step=float(trace_step),
                    min_step=float(min_step),
                    max_points=int(max_points),
                )

                # Path order: (neg reversed) -> base -> pos
                candidates = list(reversed(neg_list)) + candidates + pos_list

        # If tracing produced too few candidates, top up with random IK samples (old behavior).
        if len(candidates) < goal_in_space:
            while (
                len(candidates) < goal_in_space
                and _budget_ok()
            ):
                q_seed = rng.uniform_vec(lows, highs)
                proj = _project(q_seed)
                if proj is None:
                    continue
                _st, q_new, p_new = proj
                key = hash_joint_vector(q_new, float(uniq_resolution_rad))
                # Keep this *local* too; we only want distinct candidates for selection.
                # (Global uniqueness is applied below.)
                if any(key == hash_joint_vector(c[0], float(uniq_resolution_rad)) for c in candidates):
                    continue
                candidates.append((q_new, int(attempts), p_new))

        # 3) Select uniformly by arc-length in feature space, then commit to global set.
        if not candidates:
            continue

        # Determine whether we likely traced a loop (circle) or a segment.
        traced_closed = False
        if len(candidates) >= 30:
            try:
                if float(np.linalg.norm(candidates[0][2] - candidates[-1][2])) < 0.010:
                    traced_closed = True
            except Exception:
                traced_closed = False

        feats = [c[2] for c in candidates]

        def _select_uniform_indices(
            feats_xyz: List[np.ndarray],
            *,
            goal: int,
            closed: bool,
            base_index: int,
        ) -> List[int]:
            n = int(len(feats_xyz))
            goal = int(goal)
            if n <= 0 or goal <= 0:
                return []
            if goal == 1:
                return [int(max(0, min(n - 1, base_index)))]

            if n <= goal:
                idxs = list(range(n))
            else:
                pts = np.stack([np.asarray(p, dtype=float).reshape((3,)) for p in feats_xyz], axis=0)
                seg = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
                s = np.concatenate(([0.0], np.cumsum(seg)))
                L = float(s[-1])
                if L < 1e-12:
                    idxs = list(np.linspace(0, n - 1, goal).round().astype(int))
                else:
                    if bool(closed):
                        # Treat as a loop: sample on [0, L) to avoid duplicating the start.
                        targets = [float(i) * (L / float(goal)) for i in range(goal)]
                    else:
                        targets = np.linspace(0.0, L, goal).tolist()

                    idxs = []
                    for t in targets:
                        j = int(np.searchsorted(s, float(t), side="right") - 1)
                        if j < 0:
                            j = 0
                        if j > n - 1:
                            j = n - 1
                        if j + 1 < n:
                            if abs(float(s[j + 1]) - float(t)) < abs(float(s[j]) - float(t)):
                                j = j + 1
                        idxs.append(int(j))

            # Ensure base is included (important for reproducibility)
            base_index = int(max(0, min(n - 1, base_index)))
            if base_index not in idxs:
                pts = np.stack([np.asarray(p, dtype=float).reshape((3,)) for p in feats_xyz], axis=0)
                pb = pts[base_index]
                best_k = 0
                best_d = float("inf")
                for k, idx in enumerate(idxs):
                    d = float(np.linalg.norm(pts[int(idx)] - pb))
                    if d < best_d:
                        best_d = d
                        best_k = int(k)
                idxs[best_k] = int(base_index)

            # Unique while preserving order
            out: List[int] = []
            seen = set()
            for idx in idxs:
                ii = int(idx)
                if ii not in seen:
                    seen.add(ii)
                    out.append(ii)

            # Fill if we lost too many due to coarse resolution
            if len(out) < goal:
                fallback = list(np.linspace(0, n - 1, goal).round().astype(int))
                for ii in fallback:
                    jj = int(ii)
                    if jj not in seen:
                        seen.add(jj)
                        out.append(jj)
                    if len(out) >= goal:
                        break
            if len(out) < goal:
                for jj in range(n):
                    if jj not in seen:
                        seen.add(jj)
                        out.append(int(jj))
                    if len(out) >= goal:
                        break

            return out[:goal]

        # base is the first element in the candidate list
        base_idx_local = 0
        # In the open-segment stitch case, base may not be at index 0.
        # We detect it by searching for the closest feature point to p_base.
        if len(candidates) > 1:
            try:
                dists = [float(np.linalg.norm(p - p_base)) for p in feats]
                base_idx_local = int(np.argmin(np.asarray(dists, dtype=float)))
            except Exception:
                base_idx_local = 0

        selected_local = _select_uniform_indices(
            feats,
            goal=int(goal_in_space),
            closed=bool(traced_closed),
            base_index=int(base_idx_local),
        )

        # Commit selected solutions (keep their original attempt IDs).
        for idx_local in selected_local:
            if added_in_space >= goal_in_space:
                break
            if len(solutions) >= num_solutions:
                break

            q_sel, attempt_sel, _p_sel = candidates[int(idx_local)]
            key = hash_joint_vector(q_sel, float(uniq_resolution_rad))
            if key in uniq_keys:
                continue
            uniq_keys.add(key)

            sol_dict = {
                "index": int(len(solutions)),
                "attempt": int(attempt_sel),
                "target_point": {"x": float(target_point.x), "y": float(target_point.y), "z": float(target_point.z)},
                "tip_link": str(tip_link),
                "sampled_yaw_rad": float(yaw),
                "joint_names": joint_names,
                "joint_positions": [float(v) for v in np.asarray(q_sel, dtype=float).tolist()],
            }
            solutions.append(sol_dict)
            solutions_obj.append(
                IKSolution(
                    index=sol_dict["index"],
                    attempt=sol_dict["attempt"],
                    sampled_yaw_rad=sol_dict["sampled_yaw_rad"],
                    joint_names=joint_names,
                    joint_positions=sol_dict["joint_positions"],
                )
            )
            added_in_space += 1

    dt = time.time() - t0

    meta = {
        "group": str(group),
        "tip_link": str(tip_link),
        "named_start_for_seeding": str(named_start_for_seeding),
        "target_point": {"x": float(target_point.x), "y": float(target_point.y), "z": float(target_point.z)},
        "requested": int(num_solutions),
        "found": int(len(solutions)),
        "attempts": int(attempts),
        "ik_successes": int(successes),
        "uniq_resolution_rad": float(uniq_resolution_rad),
        "yaw_range_rad": float(yaw_range_rad),
        "ik_timeout_s": float(ik_timeout_s),
        "seed": int(seed),
        "rng": {
            "engine": "numpy.default_rng",
            "deterministic_joint_seeds": True,
            "note": "Avoid MoveIt internal RNG (RobotState.set_to_random_positions).",
        },
        "num_spaces": int(num_spaces),
        "nullspace_step": float(nullspace_step),
        "nullspace_jitter": float(nullspace_jitter),
        "elapsed_s": float(dt),
    }

    return {"meta": meta, "solutions": solutions, "solutions_obj": solutions_obj}


def save_ik_json(path: str, payload: Dict) -> None:
    """
    Save JSON in a stable format (UTF-8, readable).
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {"meta": payload["meta"], "solutions": payload["solutions"]},
            f,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
