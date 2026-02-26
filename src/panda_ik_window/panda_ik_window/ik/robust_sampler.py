from __future__ import annotations

import time
from typing import Dict, List, Sequence, Tuple

import numpy as np

from ..types import IKSolution, TargetPoint
from ..utils.robot import RobotContext
from .sampler_space import sample_ik_solutions


def _hash_joint_vector(q: Sequence[float], resolution: float) -> Tuple[int, ...]:
    """Quantize a joint vector into an integer tuple key."""
    res = float(resolution)
    if res <= 0.0:
        res = 1e-3
    return tuple(int(round(float(v) / res)) for v in q)


def sample_ik_solutions_multi_pass(
    ctx: RobotContext,
    *,
    target_point: TargetPoint,
    nominal_tip_quat_xyzw: Tuple[float, float, float, float],
    named_start_for_seeding: str = "ready",
    requested: int = 200,
    passes: int = 3,
    pass_seed_stride: int = 100_000,
    # per-pass sampling params (same as sampler_space.sample_ik_solutions)
    num_spaces: int = 20,
    max_attempts: int = 20_000,
    ik_timeout_s: float = 0.05,
    yaw_range_rad: float = 2.0 * np.pi,
    nullspace_step: float = 0.20,
    nullspace_jitter: float = 0.02,
    uniq_resolution_rad: float = 1e-3,
    seed: int = 7,
) -> Dict:
    """Robust IK sampling with *multiple passes* (different seeds), merged by uniqueness.

    Motivation
    ----------
    A single sampling run may return < requested solutions due to limited attempts,
    unlucky seeds, or poor coverage. This wrapper runs several passes and merges
    unique solutions across passes.

    Returns
    -------
    dict with keys: meta, solutions, solutions_obj

    Notes
    -----
    - `requested` is the final target count (usually 200).
    - If uniqueness saturation prevents reaching `requested`, this function returns
      fewer solutions and sets meta['reason'].
    """

    requested = int(requested)
    if requested <= 0:
        requested = 200

    passes = int(passes)
    if passes < 1:
        passes = 1

    uniq_keys = set()
    merged_solutions: List[Dict] = []
    pass_metas: List[Dict] = []

    attempts_total = 0
    ik_successes_total = 0

    t0 = time.time()

    for pass_idx in range(passes):
        if len(merged_solutions) >= requested:
            break

        remaining = requested - len(merged_solutions)
        # Request the remaining count for this pass. The underlying sampler enforces
        # a hard cap (currently 3000) to keep runtime / output size bounded.
        per_pass_request = max(1, int(remaining))

        seed_pass = int(seed) + int(pass_idx) * int(pass_seed_stride)

        payload = sample_ik_solutions(
            ctx,
            target_point=target_point,
            nominal_tip_quat_xyzw=nominal_tip_quat_xyzw,
            named_start_for_seeding=named_start_for_seeding,
            num_solutions=per_pass_request,
            num_spaces=int(num_spaces),
            max_attempts=int(max_attempts),
            ik_timeout_s=float(ik_timeout_s),
            yaw_range_rad=float(yaw_range_rad),
            nullspace_step=float(nullspace_step),
            nullspace_jitter=float(nullspace_jitter),
            uniq_resolution_rad=float(uniq_resolution_rad),
            seed=seed_pass,
        )

        meta_pass = dict(payload.get("meta", {}))
        meta_pass["pass_index"] = int(pass_idx)
        meta_pass["seed_pass"] = int(seed_pass)
        pass_metas.append(meta_pass)

        attempts_total += int(meta_pass.get("attempts", 0))
        ik_successes_total += int(meta_pass.get("ik_successes", 0))

        for sol in payload.get("solutions", []):
            q = sol.get("joint_positions", [])
            key = _hash_joint_vector(q, float(uniq_resolution_rad))
            if key in uniq_keys:
                continue
            uniq_keys.add(key)

            sol_new = dict(sol)
            sol_new["pass_index"] = int(pass_idx)
            sol_new["pass_seed"] = int(seed_pass)
            # Make attempt id unique across passes for easier debugging
            try:
                sol_new["attempt_global"] = int(pass_idx) * int(max_attempts) + int(sol.get("attempt", 0))
            except Exception:
                sol_new["attempt_global"] = int(sol.get("attempt", 0))

            merged_solutions.append(sol_new)

            if len(merged_solutions) >= requested:
                break

    elapsed = float(time.time() - t0)

    # Trim and re-index to be stable
    merged_solutions = merged_solutions[:requested]
    solutions_obj: List[IKSolution] = []
    for idx, sol in enumerate(merged_solutions):
        sol["index"] = int(idx)
        solutions_obj.append(
            IKSolution(
                index=int(idx),
                attempt=int(sol.get("attempt_global", sol.get("attempt", 0))),
                sampled_yaw_rad=float(sol.get("sampled_yaw_rad", 0.0)),
                joint_names=list(sol.get("joint_names", [])),
                joint_positions=list(sol.get("joint_positions", [])),
            )
        )

    found = int(len(merged_solutions))

    reason = "ok"
    if found == 0:
        reason = "no_ik_solution_found"
    elif found < requested:
        reason = "insufficient_unique_solutions_under_constraints"

    meta = {
        "group": str(ctx.group),
        "tip_link": str(ctx.tip_link),
        "named_start_for_seeding": str(named_start_for_seeding),
        "target_point": {"x": float(target_point.x), "y": float(target_point.y), "z": float(target_point.z)},
        "requested": int(requested),
        "found": int(found),
        "passes_requested": int(passes),
        "passes_used": int(len(pass_metas)),
        "pass_metas": pass_metas,
        "attempts_total": int(attempts_total),
        "ik_successes_total": int(ik_successes_total),
        "uniq_resolution_rad": float(uniq_resolution_rad),
        "yaw_range_rad": float(yaw_range_rad),
        "ik_timeout_s": float(ik_timeout_s),
        "num_spaces": int(num_spaces),
        "nullspace_step": float(nullspace_step),
        "nullspace_jitter": float(nullspace_jitter),
        "seed": int(seed),
        "pass_seed_stride": int(pass_seed_stride),
        "elapsed_total_s": float(elapsed),
        "reason": str(reason),
    }

    return {"meta": meta, "solutions": merged_solutions, "solutions_obj": solutions_obj}
