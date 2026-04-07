from __future__ import annotations

import argparse
import json
import math
import shutil
from functools import wraps
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Dict, List

import numpy as np

from ..ik.sampler_space import sample_ik_solutions, save_ik_json
from ..ik.robust_sampler import sample_ik_solutions_multi_pass
from ..planning.origin_time import compute_origin_path_time
from ..planning.search import window_path_receding_horizon
from ..planning.time_metric import SegmentTimeModel, TotgSettings
from ..types import IKSolution, TargetPoint
from ..utils.reporting import write_robot_info_txt, write_targets_json
from ..utils.robot import (
    load_robot_context,
    make_robot_state_from_joints,
    make_robot_state_from_named,
    parse_joint_positions,
)
from ..utils.targets import (
    PATH_PATTERN_CHOICES,
    WorkspaceBounds,
    normalize_path_pattern,
    sample_one_reachable_point_fk,
)
from ..utils.timestamp import prepare_data_paths


def _timed_call(fn: Callable[..., Any]) -> Callable[..., tuple[Any, float]]:
    """Decorator: return (result, elapsed_seconds)."""

    @wraps(fn)
    def _wrapped(*args: Any, **kwargs: Any) -> tuple[Any, float]:
        t0 = perf_counter()
        out = fn(*args, **kwargs)
        return out, float(perf_counter() - t0)

    return _wrapped


def _load_reused_candidates(
    *,
    reuse_dir: Path,
    num_points: int,
    requested: int,
    dof: int,
) -> tuple[str, List[float], List[TargetPoint], List[List[IKSolution]], List[Dict], List[Dict]]:
    """Load targets + IK candidates from an existing run directory.

    Expected files in ``reuse_dir``:
    - ``targets.json``
    - ``p1.json`` ... ``pN.json``
    """

    root = Path(reuse_dir).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"reuse-candidates-dir does not exist or is not a directory: {root}")

    targets_path = root / "targets.json"
    if not targets_path.exists():
        raise FileNotFoundError(f"Missing targets.json in reuse-candidates-dir: {targets_path}")

    with targets_path.open("r", encoding="utf-8") as f:
        targets_payload = json.load(f)
    if not isinstance(targets_payload, dict):
        raise ValueError(f"Invalid targets.json format (expected object): {targets_path}")

    meta = targets_payload.get("meta", {})
    if not isinstance(meta, dict):
        meta = {}
    start_label = str(meta.get("start_label", "reused"))

    p0 = targets_payload.get("p0", {})
    if not isinstance(p0, dict):
        p0 = {}
    start_q_raw = p0.get("joint_positions", [])
    if not isinstance(start_q_raw, list):
        raise ValueError(f"Invalid p0.joint_positions in {targets_path}")
    start_q = [float(v) for v in start_q_raw]
    if len(start_q) != int(dof):
        raise ValueError(
            f"p0.joint_positions length mismatch in {targets_path}: got {len(start_q)}, expected dof={int(dof)}"
        )

    targets_raw = targets_payload.get("targets", [])
    if not isinstance(targets_raw, list):
        raise ValueError(f"Invalid targets field in {targets_path}")
    if len(targets_raw) < int(num_points):
        raise ValueError(
            f"Not enough targets in {targets_path}: got {len(targets_raw)}, need at least {int(num_points)}"
        )

    targets: List[TargetPoint] = []
    for i in range(1, int(num_points) + 1):
        obj = targets_raw[i - 1]
        if not isinstance(obj, dict):
            raise ValueError(f"targets[{i-1}] is not an object in {targets_path}")
        try:
            targets.append(TargetPoint(x=float(obj["x"]), y=float(obj["y"]), z=float(obj["z"])))
        except Exception as e:
            raise ValueError(f"Invalid target point targets[{i-1}] in {targets_path}: {e}") from e

    solutions_by_point: List[List[IKSolution]] = []
    ik_meta_by_point: List[Dict] = []
    ik_solve_timing_by_point: List[Dict] = []

    for i in range(1, int(num_points) + 1):
        p_path = root / f"p{i}.json"
        if not p_path.exists():
            raise FileNotFoundError(f"Missing IK file in reuse-candidates-dir: {p_path}")

        with p_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid IK JSON format (expected object): {p_path}")

        meta_i = payload.get("meta", {})
        if not isinstance(meta_i, dict):
            meta_i = {}
        sols_raw = payload.get("solutions", [])
        if not isinstance(sols_raw, list):
            raise ValueError(f"Invalid solutions array in {p_path}")
        if len(sols_raw) < int(requested):
            raise RuntimeError(
                f"{p_path.name}: only {len(sols_raw)} solutions, requested={int(requested)}. "
                f"Please regenerate candidates with enough num_solutions."
            )

        sols_i: List[IKSolution] = []
        for j, sol in enumerate(sols_raw[: int(requested)]):
            if not isinstance(sol, dict):
                raise ValueError(f"Invalid solution entry at {p_path.name}[{j}]")
            q_raw = sol.get("joint_positions", [])
            if not isinstance(q_raw, list):
                raise ValueError(f"Invalid joint_positions at {p_path.name}[{j}]")
            q = [float(v) for v in q_raw]
            if len(q) != int(dof):
                raise ValueError(
                    f"Joint vector length mismatch at {p_path.name}[{j}]: got {len(q)}, expected dof={int(dof)}"
                )

            joint_names = sol.get("joint_names", [])
            if not isinstance(joint_names, list):
                joint_names = []

            sols_i.append(
                IKSolution(
                    index=int(j),
                    attempt=int(sol.get("attempt", j + 1)),
                    sampled_yaw_rad=float(sol.get("sampled_yaw_rad", 0.0)),
                    joint_names=list(joint_names),
                    joint_positions=q,
                )
            )

        solutions_by_point.append(sols_i)

        meta_out = dict(meta_i)
        meta_out.update(
            {
                "reused": True,
                "reuse_source_dir": str(root),
                "reused_file": str(p_path.name),
                "requested": int(requested),
                "found": int(len(sols_i)),
            }
        )
        ik_meta_by_point.append(meta_out)

        solve_elapsed_s = 0.0
        try:
            solve_elapsed_s = float(meta_i.get("point_solve_elapsed_s", 0.0))
            if not math.isfinite(solve_elapsed_s):
                solve_elapsed_s = 0.0
        except Exception:
            solve_elapsed_s = 0.0

        ik_solve_timing_by_point.append(
            {
                "point": f"p{i}",
                "solve_elapsed_s": float(solve_elapsed_s),
                "resample_trials_used": int(meta_i.get("resample_trials_used", 0)),
                "requested": int(requested),
                "found": int(len(sols_i)),
                "reused": True,
            }
        )

    return (
        start_label,
        start_q,
        targets,
        solutions_by_point,
        ik_meta_by_point,
        ik_solve_timing_by_point,
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="ik_window",
        description=(
            "MoveIt2 + Panda: generate an IK dataset (p0->p1..pN) with a fixed seed, "
            "then evaluate window policies for the requested window_size(s)."
        ),
    )
    # MoveIt / robot
    p.add_argument("--group", type=str, default="panda_arm", help="Planning group name (default: panda_arm).")
    p.add_argument("--tip-link", type=str, default="", help="Tip link; empty -> infer from group.")
    p.add_argument(
        "--named-start",
        type=str,
        default="random",
        help="Named start state in SRDF. Use 'random' (default) for seeded random start.",
    )
    p.add_argument(
        "--p0",
        type=str,
        default="",
        help=(
            "Start joint positions, e.g. '0,-0.7,0,-2.3,0,1.6,0.8'. "
            "Empty -> named-start (or random if named-start=random)."
        ),
    )

    # Experiment setting
    p.add_argument("--num-points", type=int, default=8, help="Number of target points n (max: 30).")
    p.add_argument("--seed", type=int, default=7, help="Random seed for dataset generation.")
    p.add_argument(
        "--path-pattern",
        type=str,
        default="random",
        choices=list(PATH_PATTERN_CHOICES),
        help=(
            "Path pattern for sampled target sequence: "
            "trend=direction-consistent, switching=multi-directional jump, "
            "random=no directional class constraint, "
            "trend_plus=trend with periodic anti-stall switch and danger-zone return-to-safe switch."
        ),
    )

    # Window policy evaluation
    p.add_argument(
        "--window-size",
        "--window_size",
        type=str,
        default="all",
        help=(
            "Window size(s) to evaluate. Accepts a single int (e.g. '3') or a list "
            "(e.g. '1,3,8' or '[1,3,8]'). 'all' means sweep ws=1..num_points (default)."
        ),
    )


    # DP / metric computation device (only affects trapezoid DP; TOTG is CPU)
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        help=(
            "Device for DP when time_model=trapezoid: auto/cpu/cuda/cuda:0... "
            "(auto uses CUDA if available)."
        ),
    )
    p.add_argument(
        "--dp-block-size",
        type=int,
        default=128,
        help="DP block size (caps peak memory; larger can be faster). Default: 128.",
    )

    # IK sampling (per point)
    p.add_argument(
        "--num-solutions",
        type=int,
        default=100,
        help="IK solutions per target point m (max: 3000).",
    )
    p.add_argument("--num-spaces", type=int, default=20, help="Yaw spaces (default: 20).")
    p.add_argument("--max-attempts", type=int, default=10000, help="Max IK attempts per point (default: 20000).")
    p.add_argument("--ik-timeout", type=float, default=0.05, help="IK timeout per attempt (s).")
    p.add_argument("--yaw-range", type=float, default=2.0 * math.pi, help="Yaw perturbation range (rad).")
    p.add_argument("--nullspace-step", type=float, default=0.20, help="Nullspace step (rad).")
    p.add_argument("--nullspace-jitter", type=float, default=0.02, help="Nullspace jitter (rad).")
    p.add_argument("--uniq-resolution", type=float, default=1e-3, help="Uniq quantization (rad).")

    # Segment time model (stop at each waypoint)
    p.add_argument(
        "--time-model",
        type=str,
        default="trapezoid",
        choices=["auto", "totg", "trapezoid"],
        help="Segment time model: auto=prefer MoveIt TOTG, fallback to trapezoid; "
        "totg=force MoveIt TOTG; trapezoid=analytic rest-to-rest model.",
    )
    # TOTG parameters (used when time-model is auto/totg)
    p.add_argument("--totg-vel-scale", type=float, default=1.0, help="TOTG velocity scaling factor.")
    p.add_argument("--totg-acc-scale", type=float, default=1.0, help="TOTG acceleration scaling factor.")
    p.add_argument("--totg-path-tolerance", type=float, default=0.1, help="TOTG path tolerance.")
    p.add_argument("--totg-resample-dt", type=float, default=0.1, help="TOTG resample dt (s).")
    p.add_argument("--totg-min-angle-change", type=float, default=0.001, help="TOTG min angle change (rad).")

    # Robust sampling / auto-resample (to avoid 0-solution points)
    p.add_argument(
        "--resample-max",
        type=int,
        default=100,
        help="Max resampling trials per point when IK is infeasible / insufficient.",
    )
    p.add_argument(
        "--topup-passes",
        type=int,
        default=3,
        help="How many IK sampling passes to merge (different seeds) for one point.",
    )
    p.add_argument(
        "--precheck-attempts",
        type=int,
        default=1000,
        help="Quick feasibility check budget (attempts) before doing full sampling.",
    )
    p.add_argument(
        "--precheck-num-spaces",
        type=int,
        default=8,
        help="Quick feasibility check yaw spaces (smaller is faster).",
    )

    # Output
    p.add_argument("--data-root", type=str, default="data_three", help="Data root directory (default: ./data).")
    p.add_argument(
        "--reuse-candidates-dir",
        type=str,
        default="",
        help=(
            "Optional existing run directory that contains targets.json + p*.json. "
            "If set, skip random target/IK sampling and reuse those candidates for evaluation."
        ),
    )

    # Workspace filter for sampled points (optional but helpful)
    p.add_argument("--ws-x", type=float, nargs=2, default=[-0.75, 0.75], metavar=("X_MIN", "X_MAX"))
    p.add_argument("--ws-y", type=float, nargs=2, default=[-0.55, 0.55], metavar=("Y_MIN", "Y_MAX"))
    p.add_argument("--ws-z", type=float, nargs=2, default=[0.05, 0.85], metavar=("Z_MIN", "Z_MAX"))
    p.add_argument("--min-sep", type=float, default=0.06, help="Min separation between target points (m).")

    # Important: ignore ROS 2 launch args like --ros-args/--params-file
    args, _unknown = p.parse_known_args()
    return args


def main() -> None:
    args = _parse_args()

    n = int(args.num_points)
    if n < 1:
        n = 1
    if n > 30:
        n = 30

    data_paths = prepare_data_paths(args.data_root)
    print(f"[run] timestamp = {data_paths.timestamp}")
    print(f"[run] data dir   = {data_paths.run_dir}")

    ctx = None
    try:
        # 1) Load robot context
        ctx = load_robot_context(
            node_name="panda_ik_window",
            group=str(args.group),
            tip_link=str(args.tip_link),
        )

        # Save robot info early (independent of sampled points)
        write_robot_info_txt(
            data_paths.robot_info_txt,
            group=ctx.group,
            tip_link=ctx.tip_link,
            joint_names=ctx.joint_names,
            joint_limits=[(jl.min_position, jl.max_position, jl.max_velocity, jl.max_acceleration) for jl in ctx.joint_limits],
        )

        requested = int(args.num_solutions)
        if requested < 1:
            requested = 1
        if requested > 3000:
            requested = 3000
        resample_max = int(args.resample_max)
        topup_passes = int(args.topup_passes)
        precheck_attempts = int(args.precheck_attempts)
        precheck_spaces = int(args.precheck_num_spaces)
        path_pattern = normalize_path_pattern(str(args.path_pattern))
        reuse_dir_raw = str(args.reuse_candidates_dir).strip()
        reuse_dir: Path | None = None
        candidate_source_mode = "sampled"
        if reuse_dir_raw:
            reuse_dir = Path(reuse_dir_raw).expanduser().resolve()
            candidate_source_mode = "reused"

        targets: List[TargetPoint] = []
        solutions_by_point: List[List[IKSolution]] = []
        ik_meta_by_point: List[Dict] = []
        ik_solve_timing_by_point: List[Dict] = []

        if reuse_dir is not None:
            print(f"[select] candidate_mode = reused")
            print(f"[select] reuse-candidates-dir = {reuse_dir}")
            print(f"[select] path_pattern argument is ignored in reuse mode: {path_pattern}")

            (start_label, start_q, targets, solutions_by_point, ik_meta_by_point, ik_solve_timing_by_point) = (
                _load_reused_candidates(
                    reuse_dir=reuse_dir,
                    num_points=int(n),
                    requested=int(requested),
                    dof=int(ctx.dof),
                )
            )

            for i in range(1, n + 1):
                src_path = reuse_dir / f"p{i}.json"
                dst_path = data_paths.ik_json_for_point(i)
                if src_path.resolve() != dst_path.resolve():
                    shutil.copy2(src_path, dst_path)

            print(
                f"[select] reused candidates loaded: points={len(targets)}, "
                f"solutions_per_point={requested}"
            )
        else:
            # Build start state (p0)
            p0_list = parse_joint_positions(str(args.p0), ctx.dof)
            named_start_raw = str(args.named_start).strip()
            named_start_low = named_start_raw.lower()

            if p0_list is not None:
                start_state = make_robot_state_from_joints(ctx, p0_list)
                start_label = "custom"
                named_start_for_seeding = "custom"
            elif named_start_low in {"", "random", "rand", "rng"}:
                rng_start = np.random.default_rng(int(args.seed) + 1_234_567)
                lows = np.array([jl.min_position for jl in ctx.joint_limits], dtype=float)
                highs = np.array([jl.max_position for jl in ctx.joint_limits], dtype=float)
                q0 = rng_start.uniform(lows, highs).astype(float).tolist()
                start_state = make_robot_state_from_joints(ctx, q0)
                start_label = "random_seeded"
                named_start_for_seeding = "ready"
                print("[run] start state: random_seeded (deterministic by --seed)")
            else:
                start_state = make_robot_state_from_named(ctx, named_start_raw)
                start_label = named_start_raw
                named_start_for_seeding = named_start_raw

            start_q = list(map(float, start_state.get_joint_group_positions(ctx.group)))

            # Nominal tip orientation from start state (used as base quaternion for sampling yaw)
            tip_pose = start_state.get_pose(ctx.tip_link)
            q_nominal = (
                float(tip_pose.orientation.x),
                float(tip_pose.orientation.y),
                float(tip_pose.orientation.z),
                float(tip_pose.orientation.w),
            )
            p0_tip = TargetPoint(
                x=float(tip_pose.position.x),
                y=float(tip_pose.position.y),
                z=float(tip_pose.position.z),
            )

            # Workspace bounds (used for candidate FK sampling)
            ws = WorkspaceBounds(
                x_min=float(args.ws_x[0]), x_max=float(args.ws_x[1]),
                y_min=float(args.ws_y[0]), y_max=float(args.ws_y[1]),
                z_min=float(args.ws_z[0]), z_max=float(args.ws_z[1]),
            )

            rng_points = np.random.default_rng(int(args.seed))
            print(f"[select] candidate_mode = sampled")
            print(f"[select] path_pattern = {path_pattern}")

            @_timed_call
            def _solve_one_point(point_idx_1based: int) -> tuple[TargetPoint, Dict, int]:
                best_payload = None
                best_target = None
                best_found = -1
                trials_used = 0

                for trial in range(1, resample_max + 1):
                    trials_used = trial

                    # sample one FK-reachable point (position only), enforce separation
                    tp = sample_one_reachable_point_fk(
                        ctx,
                        rng=rng_points,
                        existing_points=targets,
                        path_anchor=p0_tip,
                        path_pattern=path_pattern,
                        min_separation_m=float(args.min_sep),
                        workspace=ws,
                        max_attempts=2000,
                    )

                    # quick feasibility check (fast reject for orientation-infeasible points)
                    pre_payload = sample_ik_solutions(
                        ctx,
                        target_point=tp,
                        nominal_tip_quat_xyzw=q_nominal,
                        named_start_for_seeding=str(named_start_for_seeding),
                        num_solutions=50,
                        num_spaces=max(1, min(int(args.num_spaces), int(precheck_spaces))),
                        max_attempts=max(200, int(precheck_attempts)),
                        ik_timeout_s=float(args.ik_timeout),
                        yaw_range_rad=float(args.yaw_range),
                        nullspace_step=float(args.nullspace_step),
                        nullspace_jitter=float(args.nullspace_jitter),
                        uniq_resolution_rad=float(args.uniq_resolution),
                        seed=int(args.seed) + 50_000 * point_idx_1based + trial,
                    )
                    pre_found = int(pre_payload.get("meta", {}).get("found", 0))
                    if pre_found <= 0:
                        continue

                    # full sampling with multi-pass merge
                    payload = sample_ik_solutions_multi_pass(
                        ctx,
                        target_point=tp,
                        nominal_tip_quat_xyzw=q_nominal,
                        named_start_for_seeding=str(named_start_for_seeding),
                        requested=requested,
                        passes=topup_passes,
                        pass_seed_stride=100_000,
                        num_spaces=int(args.num_spaces),
                        max_attempts=int(args.max_attempts),
                        ik_timeout_s=float(args.ik_timeout),
                        yaw_range_rad=float(args.yaw_range),
                        nullspace_step=float(args.nullspace_step),
                        nullspace_jitter=float(args.nullspace_jitter),
                        uniq_resolution_rad=float(args.uniq_resolution),
                        seed=int(args.seed) + 200_000 * point_idx_1based + 1000 * trial,
                    )

                    found = int(payload.get("meta", {}).get("found", 0))
                    payload["meta"].update(
                        {
                            "resample_trial_final": int(trial),
                            "resample_trials_used": int(trials_used),
                            "resample_max": int(resample_max),
                            "precheck_found": int(pre_found),
                            "precheck_attempts": int(pre_payload.get("meta", {}).get("attempts", 0)),
                            "precheck_ik_successes": int(pre_payload.get("meta", {}).get("ik_successes", 0)),
                            "precheck_max_attempts": int(max(200, int(precheck_attempts))),
                            "precheck_num_spaces": int(max(1, min(int(args.num_spaces), int(precheck_spaces)))),
                        }
                    )

                    if found > best_found:
                        best_found = found
                        best_payload = payload
                        best_target = tp

                    if found >= requested:
                        break

                if best_payload is None or best_target is None or best_found <= 0:
                    raise RuntimeError(
                        f"Failed to find any IK-solvable point for p{point_idx_1based} after {resample_max} trials. "
                        f"Consider relaxing workspace bounds, increasing --ik-timeout/--max-attempts, "
                        f"or reducing --num-points."
                    )

                if best_found < requested:
                    raise RuntimeError(
                        f"p{point_idx_1based}: only found {best_found}/{requested} unique IK solutions after "
                        f"{resample_max} trials (best candidate). Consider increasing --max-attempts, "
                        f"--ik-timeout, --topup-passes, relaxing --uniq-resolution, or shrinking --num-solutions."
                    )

                best_payload["meta"].update({"accepted_with_shortfall": False, "shortfall": 0})
                return best_target, best_payload, int(best_found)

            for i in range(1, n + 1):
                print(f"\n[select] searching a solvable target for p{i} ...")
                (solve_out, solve_elapsed_s) = _solve_one_point(i)
                best_target, best_payload, best_found = solve_out
                best_payload["meta"]["point_solve_elapsed_s"] = float(solve_elapsed_s)

                targets.append(best_target)
                solutions_by_point.append(list(best_payload["solutions_obj"]))
                ik_meta_by_point.append(dict(best_payload["meta"]))
                ik_solve_timing_by_point.append(
                    {
                        "point": f"p{i}",
                        "solve_elapsed_s": float(solve_elapsed_s),
                        "resample_trials_used": int(best_payload["meta"].get("resample_trials_used", 0)),
                        "requested": int(requested),
                        "found": int(best_found),
                    }
                )

                out_path = data_paths.ik_json_for_point(i)
                save_ik_json(str(out_path), best_payload)
                print(
                    f"[select] p{i}: ({best_target.x:.3f}, {best_target.y:.3f}, {best_target.z:.3f}) "
                    f"found {best_found}/{requested} "
                    f"(trials_used={best_payload['meta'].get('resample_trials_used')}, "
                    f"solve_elapsed={solve_elapsed_s:.4f}s) -> {out_path.name}"
                )

        # Save run metadata (targets only contain coordinates, per requirement)
        write_targets_json(
            data_paths.targets_json,
            timestamp=data_paths.timestamp,
            group=ctx.group,
            tip_link=ctx.tip_link,
            start_label=start_label,
            start_joint_positions=start_q,
            targets=targets,
        )

        origin_start_state = make_robot_state_from_joints(ctx, start_q)
        origin_tip_pose = origin_start_state.get_pose(ctx.tip_link)
        origin_tip_quat = (
            float(origin_tip_pose.orientation.x),
            float(origin_tip_pose.orientation.y),
            float(origin_tip_pose.orientation.z),
            float(origin_tip_pose.orientation.w),
        )

        print("\n[eval] origin path baseline (planner point-to-point, no IK selection) ...")
        origin_result = compute_origin_path_time(
            ctx=ctx,
            start_q=start_q,
            targets=targets,
            tip_quat_xyzw=origin_tip_quat,
        )
        if str(origin_result.status) == "ok":
            print(
                f"[eval] origin total_time_s = {float(origin_result.total_time_s):.6f} "
                f"(segments={len(origin_result.segments)}, "
                f"planning_elapsed_total_s={float(origin_result.planning_elapsed_total_s):.6f})"
            )
        else:
            print(f"[eval] WARN origin planning failed: {origin_result.note}")

        def _make_time_model() -> SegmentTimeModel:
            return SegmentTimeModel(
                model=str(args.time_model),
                max_vel_rad_s=ctx.velocity_limits,
                max_acc_rad_s2=ctx.acceleration_limits,
                moveit_py=ctx.moveit_py,
                robot_model=ctx.robot_model,
                group=ctx.group,
                joint_names=ctx.joint_names,
                totg_settings=TotgSettings(
                    vel_scale=float(args.totg_vel_scale),
                    acc_scale=float(args.totg_acc_scale),
                    path_tolerance=float(args.totg_path_tolerance),
                    resample_dt=float(args.totg_resample_dt),
                    min_angle_change=float(args.totg_min_angle_change),
                ),
            )


        # 6) Policy evaluation (window search)
        # For this package, one run evaluates ONLY the requested window_size(s).
        time_model = _make_time_model()

        def _path_segments_payload(res) -> List[Dict]:
            return [
                {
                    "from": seg.from_label,
                    "to": seg.to_label,
                    "time_s": float(seg.best_time_s),
                    "solution_index_0based": int(seg.best_solution.index),
                    "solution_index_1based": int(seg.best_solution.index) + 1,
                    "solution_id": f"p{seg.seg_idx_1based}_{int(seg.best_solution.index)+1}",
                    "attempt": int(seg.best_solution.attempt),
                    "joint_positions": [float(v) for v in seg.best_solution.joint_positions],
                }
                for seg in res.segments
            ]

        def _selection_timing_payload(res) -> List[Dict]:
            out: List[Dict] = []
            for rec in list(getattr(res, "window_selection_stats", ())):
                out.append(
                    {
                        "segment_index_1based": int(rec["segment_index_1based"]),
                        "point": str(rec["point"]),
                        "window_size": int(rec["window_size"]),
                        "remain_points": int(rec["remain_points"]),
                        "window_layer_sizes": [int(v) for v in rec["window_layer_sizes"]],
                        "theoretical_paths": int(rec["theoretical_paths"]),
                        "selected_solution_index_0based": int(rec["selected_solution_index_0based"]),
                        "selected_solution_index_1based": int(rec["selected_solution_index_1based"]),
                        "selected_solution_id": str(rec["selected_solution_id"]),
                        "selection_elapsed_s": float(rec["selection_elapsed_s"]),
                    }
                )
            return out

        def _origin_segments_payload() -> List[Dict]:
            out: List[Dict] = []
            for seg in list(origin_result.segments):
                out.append(
                    {
                        "segment_index_1based": int(seg.seg_idx_1based),
                        "from": str(seg.from_label),
                        "to": str(seg.to_label),
                        "trajectory_time_s": float(seg.trajectory_time_s),
                        "planning_elapsed_s": float(seg.planning_elapsed_s),
                        "start_joint_positions": [float(v) for v in seg.start_joint_positions],
                        "end_joint_positions": [float(v) for v in seg.end_joint_positions],
                    }
                )
            return out

        def _origin_joint_positions_by_point_payload() -> List[Dict]:
            point_to_joint_positions: Dict[str, List[float]] = {
                "p0": [float(v) for v in start_q],
            }

            for seg in list(origin_result.segments):
                point_to_joint_positions[str(seg.from_label)] = [float(v) for v in seg.start_joint_positions]
                point_to_joint_positions[str(seg.to_label)] = [float(v) for v in seg.end_joint_positions]

            def _point_sort_key(label: str) -> tuple[int, str]:
                if label.startswith("p"):
                    idx_txt = label[1:]
                    if idx_txt.isdigit():
                        return int(idx_txt), str(label)
                return 10**9, str(label)

            out: List[Dict] = []
            for point_label in sorted(point_to_joint_positions.keys(), key=_point_sort_key):
                joint_positions = [float(v) for v in point_to_joint_positions[point_label]]
                out.append(
                    {
                        "point": str(point_label),
                        "joint_positions": joint_positions,
                        "joint_positions_by_name": {
                            str(joint_name): float(joint_pos)
                            for joint_name, joint_pos in zip(ctx.joint_names, joint_positions)
                        },
                    }
                )
            return out

        def _parse_window_sizes_arg(s: str, *, n_points: int) -> tuple[str, List[int], List[int]]:
            """Parse user input window_size into a list.

            Accepts:
              - single int: "3"
              - comma/space separated: "1,3,8" / "1 3 8"
              - JSON list: "[1,3,8]"
              - special: "all" -> [1..n_points]

            Returns:
              (input_str, requested_list, evaluated_list)
            Where evaluated_list is clamped into [1, n_points] and de-duplicated.
            """
            import json
            import re

            inp = "" if s is None else str(s)
            raw = inp.strip()
            if raw == "" or raw.lower() in {"all", "sweep", "*"}:
                req = list(range(1, int(n_points) + 1))
            else:
                # Try single integer.
                if re.fullmatch(r"[+-]?\d+", raw):
                    req = [int(raw)]
                # Try JSON list.
                elif raw.startswith("[") and raw.endswith("]"):
                    try:
                        obj = json.loads(raw)
                        if isinstance(obj, list):
                            req = [int(v) for v in obj]
                        else:
                            req = [int(obj)]
                    except Exception as e:
                        raise ValueError(f"Invalid --window-size JSON: {raw!r} ({e})")
                else:
                    # Comma / whitespace separated.
                    parts = [p for p in re.split(r"[\s,]+", raw) if p.strip()]
                    if not parts:
                        raise ValueError(f"Invalid --window-size: {raw!r}")
                    req = [int(p) for p in parts]

            # Clamp + de-duplicate (keep order)
            n_points = max(1, int(n_points))
            out: List[int] = []
            for ws in req:
                w = int(ws)
                if w < 1:
                    w = 1
                if w > n_points:
                    w = n_points
                if w not in out:
                    out.append(w)

            if not out:
                raise ValueError("--window-size parsed to an empty list")
            return inp, req, out

        window_size_input, window_sizes_requested, ws_list = _parse_window_sizes_arg(
            str(args.window_size), n_points=int(n)
        )

        # (Optional) stable order in summary: keep user's order.
        window_results_by_ws: Dict[str, Dict] = {}
        window_total_time_s_by_ws: Dict[str, float] = {}
        window_selection_timing_by_ws: Dict[str, List[Dict]] = {}
        window_selection_timing_total_s_by_ws: Dict[str, float] = {}

        for ws in ws_list:
            print(f"\n[eval] window policy (ws={ws}/{n}) ...")
            res = window_path_receding_horizon(
                start_q=start_q,
                solutions_by_point=solutions_by_point,
                requested_per_point=requested,
                time_model=time_model,
                window_size=int(ws),
                block_size=int(args.dp_block_size),
                device=str(args.device),
            )

            seg_times = [float(seg.best_time_s) for seg in res.segments]
            cum_times = [float(v) for v in np.cumsum(np.asarray(seg_times, dtype=float))]
            selection_timing = _selection_timing_payload(res)
            selection_timing_total_s = float(sum(float(v["selection_elapsed_s"]) for v in selection_timing))

            window_total_time_s_by_ws[str(ws)] = float(res.total_time_s)
            window_selection_timing_by_ws[str(ws)] = [dict(v) for v in selection_timing]
            window_selection_timing_total_s_by_ws[str(ws)] = float(selection_timing_total_s)
            window_results_by_ws[str(ws)] = {
                "window_size": int(ws),
                "segment_times_s": seg_times,
                "cumulative_times_s": cum_times,
                "selection_timing": [dict(v) for v in selection_timing],
                "selection_timing_by_point": selection_timing,
                "selection_timing_total_s": float(selection_timing_total_s),
                "selection_timing_note": (
                    "Only full-window decisions are recorded. "
                    "Tail points with remain_points <= window_size are not recorded."
                ),
                "final_path": {
                    "total_time_s": float(res.total_time_s),
                    "segments": _path_segments_payload(res),
                },
            }

        # Build unified summary.json (keeps the same overall structure as panda_ik_global_window,
        # but records results by window_size instead of separate greedy/global/window blocks).
        import json

        # Represent window_size as either an int or a list[int] to match the user's input style.
        # - If user passed a single integer (e.g. "3"), write int.
        # - Otherwise (e.g. "1,3,8" / "[1,3,8]" / "all"), write a list.
        import re

        raw_ws = str(window_size_input).strip()
        if re.fullmatch(r"[+-]?\d+", raw_ws):
            window_size_requested_field = int(window_sizes_requested[0]) if window_sizes_requested else int(raw_ws)
            window_size_effective_field = int(ws_list[0]) if ws_list else int(n)
        else:
            window_size_requested_field = [int(v) for v in window_sizes_requested]
            window_size_effective_field = [int(v) for v in ws_list]

        ik_solve_timing_payload = {
            "by_point": [dict(v) for v in ik_solve_timing_by_point],
            "total_s": float(sum(float(v["solve_elapsed_s"]) for v in ik_solve_timing_by_point)),
            "note": (
                "Elapsed wall time for finding requested num_solutions at each point."
                if str(candidate_source_mode) != "reused"
                else "Candidates are reused from reuse-candidates-dir; values are copied from source meta when available."
            ),
        }

        summary = {
            "format": "panda_ik_window_summary",
            "format_version": 3,
            "meta": {
                "timestamp": str(data_paths.timestamp),
                "seed": int(args.seed),
                "group": str(ctx.group),
                "tip_link": str(ctx.tip_link),
                "num_points": int(n),
                "num_solutions": int(requested),
                "path_pattern": str(path_pattern),
                "candidate_source_mode": str(candidate_source_mode),
                "reuse_candidates_dir": str(reuse_dir) if reuse_dir is not None else "",
                "time_model": {
                    "requested": str(time_model.info.requested),
                    "effective": str(time_model.info.effective),
                    "totg_available": bool(time_model.info.totg_available),
                    "totg_failures": int(time_model.info.totg_failures),
                    "note": str(time_model.info.note),
                },
                # New in v3:
                "window_size_input": str(window_size_input),
                "window_size_requested": window_size_requested_field,
                "window_size": window_size_effective_field,
                "window_sizes_evaluated": [int(ws) for ws in ws_list],
                "note": (
                    "ws=1 is equivalent to greedy; ws=num_points is equivalent to global. "
                    "This run evaluates ONLY the requested window_size(s)."
                ),
            },
            "units": {"cartesian_position": "m", "joint_position": "rad", "time": "s"},
            "start": {
                "name": "p0",
                "label": str(start_label),
                "joint_positions": [float(v) for v in start_q],
            },
            "targets": [
                {"name": f"p{i}", "x": float(p.x), "y": float(p.y), "z": float(p.z)}
                for i, p in enumerate(targets, start=1)
            ],
            "ik_files": {f"p{i}": data_paths.ik_json_for_point(i).name for i in range(1, n + 1)},
            "ik_sampling_meta": {f"p{i}": dict(meta) for i, meta in enumerate(ik_meta_by_point, start=1)},
            "ik_solve_timing": dict(ik_solve_timing_payload),
            "origin": {
                "status": str(origin_result.status),
                "method": str(origin_result.method),
                "planner_id": str(origin_result.planner_id),
                "planning_frame": str(origin_result.planning_frame),
                "joint_names": [str(name) for name in ctx.joint_names],
                "joint_positions_by_point": _origin_joint_positions_by_point_payload(),
                "segment_times_s": [float(seg.trajectory_time_s) for seg in origin_result.segments],
                "cumulative_times_s": [
                    float(v)
                    for v in np.cumsum(
                        np.asarray([float(seg.trajectory_time_s) for seg in origin_result.segments], dtype=float)
                    )
                ],
                "segments": _origin_segments_payload(),
                "total_time_s": float(origin_result.total_time_s),
                "planning_elapsed_total_s": float(origin_result.planning_elapsed_total_s),
                "note": str(origin_result.note),
            },
            "window": {
                # Echo the user input (int or list[int]) for convenience.
                "window_size": window_size_effective_field,
                "window_sizes": [int(ws) for ws in ws_list],
                "ws_min": int(min(ws_list)) if len(ws_list) > 0 else 1,
                "ws_max": int(max(ws_list)) if len(ws_list) > 0 else int(n),
                # Keep mapping form for easy parsing.
                "total_time_s_by_ws": dict(window_total_time_s_by_ws),
                "selection_timing_by_ws": dict(window_selection_timing_by_ws),
                "selection_timing_total_s_by_ws": dict(window_selection_timing_total_s_by_ws),
                "results_by_ws": dict(window_results_by_ws),
                # Also provide an ordered list form.
                "results": [dict(window_results_by_ws[str(ws)]) for ws in ws_list],
            },
        }

        with open(data_paths.summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, sort_keys=True)

        print(f"[eval] summary -> {data_paths.summary_json.name}")
        txt_lines: List[str] = []
        txt_lines.append("panda_ik_window summary")
        txt_lines.append(f"timestamp: {data_paths.timestamp}")
        txt_lines.append(f"seed: {int(args.seed)}")
        txt_lines.append(f"path_pattern: {path_pattern}")
        txt_lines.append(f"candidate_source_mode: {candidate_source_mode}")
        if reuse_dir is not None:
            txt_lines.append(f"reuse_candidates_dir: {reuse_dir}")
        txt_lines.append(f"time_model.requested: {time_model.info.requested}")
        txt_lines.append(f"time_model.effective: {time_model.info.effective}")

        data_paths.summary_txt.write_text("\n".join(txt_lines) + "\n", encoding="utf-8")
        print(f"[eval] summary -> {data_paths.summary_txt.name}")

        print("\n[done] bye.")
    finally:
        # Always shut down MoveItPy to avoid class_loader warnings on abrupt exit.
        if ctx is not None:
            try:
                ctx.moveit_py.shutdown()
            except Exception:
                pass


if __name__ == "__main__":
    main()
