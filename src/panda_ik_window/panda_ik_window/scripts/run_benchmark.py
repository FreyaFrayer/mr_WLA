from __future__ import annotations

import argparse
import math
from typing import Dict, List

import numpy as np

from ..ik.sampler_space import sample_ik_solutions, save_ik_json
from ..ik.robust_sampler import sample_ik_solutions_multi_pass
from ..planning.search import dp_backend_info, window_path_receding_horizon
from ..planning.time_metric import SegmentTimeModel, TotgSettings
from ..types import IKSolution, TargetPoint
from ..utils.reporting import write_robot_info_txt, write_targets_json
from ..utils.robot import (
    load_robot_context,
    make_robot_state_from_joints,
    make_robot_state_from_named,
    parse_joint_positions,
)
from ..utils.targets import WorkspaceBounds, sample_one_reachable_point_fk
from ..utils.timestamp import prepare_data_paths


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
    p.add_argument("--named-start", type=str, default="ready", help="Named start state in SRDF (default: ready).")
    p.add_argument(
        "--p0",
        type=str,
        default="",
        help=(
            "Start joint positions, e.g. '0,-0.7,0,-2.3,0,1.6,0.8'. "
            "Empty -> named-start."
        ),
    )

    # Experiment setting
    p.add_argument("--num-points", type=int, default=8, help="Number of target points n (max: 30).")
    p.add_argument("--seed", type=int, default=7, help="Random seed for dataset generation.")

    # Window policy evaluation
    # NOTE: Now accepts ONLY a single integer window size (ws), and ws must be > 1.
    # Greedy baseline (ws=1) will be evaluated automatically and recorded in summary.json.
    p.add_argument(
        "--window-size",
        "--window_size",
        type=int,
        default=3,
        help=(
            "Window size ws (>1). Greedy baseline (ws=1) is evaluated automatically for comparison. "
            "Default: 3."
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

    p.add_argument(
        "--dp-dtype",
        type=str,
        default="float64",
        choices=["float64", "float32", "auto"],
        help=(
            "DP numeric dtype for the CUDA torch backend (trapezoid only). "
            "float64 is the most accurate (default). float32 is faster and often increases GPU utilization. "
            "auto selects float32 on CUDA."
        ),
    )

    p.add_argument(
        "--eval-workers",
        type=int,
        default=1,
        help=(
            "Parallel workers for evaluating multiple window_size values. "
            "Uses a thread pool and is enabled only when effective time_model is trapezoid. "
            "Default: 1 (sequential)."
        ),
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

    # Workspace filter for random points (optional but helpful)
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

        # 2) Build start state (p0)
        p0_list = parse_joint_positions(str(args.p0), ctx.dof)
        if p0_list is None:
            start_state = make_robot_state_from_named(ctx, str(args.named_start))
            start_label = str(args.named_start)
        else:
            start_state = make_robot_state_from_joints(ctx, p0_list)
            start_label = "custom"

        start_q = list(map(float, start_state.get_joint_group_positions(ctx.group)))

        # 3) Nominal tip orientation from start state (used as base quaternion for sampling yaw)
        tip_pose = start_state.get_pose(ctx.tip_link)
        q_nominal = (
            float(tip_pose.orientation.x),
            float(tip_pose.orientation.y),
            float(tip_pose.orientation.z),
            float(tip_pose.orientation.w),
        )

        # 4) Workspace bounds (used for candidate FK sampling)
        ws = WorkspaceBounds(
            x_min=float(args.ws_x[0]), x_max=float(args.ws_x[1]),
            y_min=float(args.ws_y[0]), y_max=float(args.ws_y[1]),
            z_min=float(args.ws_z[0]), z_max=float(args.ws_z[1]),
        )

        # Save robot info early (independent of sampled points)
        write_robot_info_txt(
            data_paths.robot_info_txt,
            group=ctx.group,
            tip_link=ctx.tip_link,
            joint_names=ctx.joint_names,
            joint_limits=[(jl.min_position, jl.max_position, jl.max_velocity, jl.max_acceleration) for jl in ctx.joint_limits],
        )

        # 4.5) Segment time model (independent of sampled points)
        time_model = SegmentTimeModel(
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

        # Print whether GPU will be used (only affects trapezoid DP; TOTG is CPU).
        dp_info = dp_backend_info(time_model=time_model, device=str(args.device))
        gpu_used = bool(dp_info.get("gpu_used", False))
        gpu_zh = "是" if gpu_used else "否"
        extra = ""
        if gpu_used:
            dev_eff = str(dp_info.get("device_effective", "cuda"))
            name = dp_info.get("gpu_name")
            extra = dev_eff + (f", {name}" if name else "")
        else:
            note = str(dp_info.get("note", "")).strip()
            extra = note
        if extra:
            print(f"[run] GPU 是否被使用 = {gpu_zh} ({extra})")
        else:
            print(f"[run] GPU 是否被使用 = {gpu_zh}")

        print(f"[run] DP dtype        = {str(args.dp_dtype)}")
        print(f"[run] eval_workers    = {int(max(1, int(args.eval_workers)))}")

        requested = int(args.num_solutions)
        if requested < 1:
            requested = 1
        if requested > 3000:
            requested = 3000
        resample_max = int(args.resample_max)
        topup_passes = int(args.topup_passes)
        precheck_attempts = int(args.precheck_attempts)
        precheck_spaces = int(args.precheck_num_spaces)

        # 5) Sequentially sample p1..pn, but *only accept* points that are IK-solvable
        # under the 'ready' nominal orientation (+ yaw perturbation).
        targets: List[TargetPoint] = []
        solutions_by_point: List[List[IKSolution]] = []
        ik_meta_by_point: List[Dict] = []

        rng_points = np.random.default_rng(int(args.seed))

        for i in range(1, n + 1):
            print(f"\n[select] searching a solvable target for p{i} ...")

            best_payload = None
            best_target = None
            best_found = -1
            trials_used = 0

            for trial in range(1, resample_max + 1):
                trials_used = trial

                # 5.1) sample one FK-reachable point (position only), enforce separation
                tp = sample_one_reachable_point_fk(
                    ctx,
                    rng=rng_points,
                    existing_points=targets,
                    min_separation_m=float(args.min_sep),
                    workspace=ws,
                    max_attempts=2000,
                )

                # 5.2) quick feasibility check (fast reject for orientation-infeasible points)
                pre_payload = sample_ik_solutions(
                    ctx,
                    target_point=tp,
                    nominal_tip_quat_xyzw=q_nominal,
                    named_start_for_seeding=str(args.named_start),
                    num_solutions=50,
                    num_spaces=max(1, min(int(args.num_spaces), int(precheck_spaces))),
                    max_attempts=max(200, int(precheck_attempts)),
                    ik_timeout_s=float(args.ik_timeout),
                    yaw_range_rad=float(args.yaw_range),
                    nullspace_step=float(args.nullspace_step),
                    nullspace_jitter=float(args.nullspace_jitter),
                    uniq_resolution_rad=float(args.uniq_resolution),
                    seed=int(args.seed) + 50_000 * i + trial,
                )
                pre_found = int(pre_payload.get("meta", {}).get("found", 0))
                if pre_found <= 0:
                    # Not solvable under current orientation constraints
                    continue

                # 5.3) full sampling with multi-pass merge (try to reach requested=200)
                payload = sample_ik_solutions_multi_pass(
                    ctx,
                    target_point=tp,
                    nominal_tip_quat_xyzw=q_nominal,
                    named_start_for_seeding=str(args.named_start),
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
                    seed=int(args.seed) + 200_000 * i + 1000 * trial,
                )

                found = int(payload.get("meta", {}).get("found", 0))

                # Attach selection meta (useful for debugging + summary)
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
                    # Great, we can stop resampling for this point.
                    break

            if best_payload is None or best_target is None or best_found <= 0:
                raise RuntimeError(
                    f"Failed to find any IK-solvable point for p{i} after {resample_max} trials. "
                    f"Consider relaxing workspace bounds, increasing --ik-timeout/--max-attempts, "
                    f"or reducing --num-points."
                )

            # Strict: the spec requires finding exactly m solutions for each point.
            if best_found < requested:
                raise RuntimeError(
                    f"p{i}: only found {best_found}/{requested} unique IK solutions after "
                    f"{resample_max} trials (best candidate). Consider increasing --max-attempts, "
                    f"--ik-timeout, --topup-passes, relaxing --uniq-resolution, or shrinking --num-solutions."
                )

            best_payload["meta"].update({"accepted_with_shortfall": False, "shortfall": 0})

            targets.append(best_target)
            solutions_by_point.append(list(best_payload["solutions_obj"]))
            ik_meta_by_point.append(dict(best_payload["meta"]))

            out_path = data_paths.ik_json_for_point(i)
            save_ik_json(str(out_path), best_payload)
            print(
                f"[select] p{i}: ({best_target.x:.3f}, {best_target.y:.3f}, {best_target.z:.3f}) "
                f"found {best_found}/{requested} (trials_used={best_payload['meta'].get('resample_trials_used')}) -> {out_path.name}"
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

        # 6) Policy evaluation (window search)
        # This run evaluates:
        #   - the requested window_size (ws > 1)
        #   - plus a greedy baseline (ws=1) for comparison (recorded in summary.json)

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

        # Validate / clamp the single requested window size.
        window_size_input = str(args.window_size)
        ws_requested = int(args.window_size)

        if int(n) < 2:
            raise ValueError(
                "num_points must be >= 2 when using --window-size > 1 (greedy baseline is ws=1)"
            )

        if ws_requested <= 1:
            raise ValueError(
                "--window-size must be an integer > 1 (greedy baseline ws=1 is evaluated automatically)"
            )

        ws_effective = int(ws_requested)
        if ws_effective > int(n):
            print(f"[eval][warn] window_size={ws_requested} > num_points={n}, clamped to {n}")
            ws_effective = int(n)

        # This run has exactly ONE requested/evaluated window size (ws_effective).
        ws_list: List[int] = [int(ws_effective)]
        window_size_requested_field = int(ws_requested)
        window_size_effective_field = int(ws_effective)

        # Main-window results (for ws_effective)
        window_results_by_ws: Dict[str, Dict] = {}
        window_total_time_s_by_ws: Dict[str, float] = {}

        # Optional: precompute all joint-position tensors on GPU once and reuse across
        # different window_size evaluations (saves time and improves GPU utilization).
        Q_all_torch = None
        if bool(gpu_used) and str(time_model.info.effective) == "trapezoid":
            try:
                import torch  # type: ignore

                dev_eff = str(dp_info.get("device_effective", "cuda"))
                torch_device = torch.device(dev_eff)
                dp_dtype_raw = str(args.dp_dtype).strip().lower()
                if dp_dtype_raw in {"auto", "float32"}:
                    dtype_torch = torch.float32
                else:
                    dtype_torch = torch.float64

                Q_all_torch = []
                for layer in solutions_by_point:
                    q_np = np.asarray([s.joint_positions for s in layer], dtype=float)
                    Q_all_torch.append(torch.as_tensor(q_np, dtype=dtype_torch, device=torch_device))

                print(
                    f"[eval] precomputed Q tensors on {dev_eff} (dtype={dtype_torch}) for {len(Q_all_torch)} layer(s)"
                )
            except Exception as e:
                Q_all_torch = None
                print(f"[eval][warn] failed to precompute Q tensors on GPU: {e}")

        def _eval_one_ws(ws: int) -> tuple[int, Dict, float]:
            res = window_path_receding_horizon(
                start_q=start_q,
                solutions_by_point=solutions_by_point,
                requested_per_point=requested,
                time_model=time_model,
                window_size=int(ws),
                block_size=int(args.dp_block_size),
                device=str(args.device),
                dp_dtype=str(args.dp_dtype),
                _precomputed_Q_all_torch=Q_all_torch,
            )

            seg_times = [float(seg.best_time_s) for seg in res.segments]
            cum_times = [float(v) for v in np.cumsum(np.asarray(seg_times, dtype=float))]

            payload = {
                "window_size": int(ws),
                "segment_times_s": seg_times,
                "cumulative_times_s": cum_times,
                "final_path": {
                    "total_time_s": float(res.total_time_s),
                    "segments": _path_segments_payload(res),
                },
            }
            return int(ws), payload, float(res.total_time_s)

        # Evaluate the requested window_size (ws_effective).
        print(f"\n[eval] window policy (ws={window_size_effective_field}/{n}) ...")
        ws_i, payload_main, total_main_s = _eval_one_ws(int(window_size_effective_field))
        window_total_time_s_by_ws[str(ws_i)] = float(total_main_s)
        window_results_by_ws[str(ws_i)] = payload_main

        # Evaluate greedy baseline (ws=1) for comparison.
        print(f"\n[eval] greedy baseline (ws=1/{n}) ...")
        _greedy_ws_i, greedy_payload, greedy_total_s = _eval_one_ws(1)

        # Build unified summary.json (keeps the same overall structure as panda_ik_global_window,
        # but records results by window_size instead of separate greedy/global/window blocks).
        import json

        summary = {
            "format": "panda_ik_window_summary",
            "format_version": 2,
            "meta": {
                "timestamp": str(data_paths.timestamp),
                "seed": int(args.seed),
                "group": str(ctx.group),
                "tip_link": str(ctx.tip_link),
                "num_points": int(n),
                "num_solutions": int(requested),
                "time_model": {
                    "requested": str(time_model.info.requested),
                    "effective": str(time_model.info.effective),
                    "totg_available": bool(time_model.info.totg_available),
                    "totg_failures": int(time_model.info.totg_failures),
                    "note": str(time_model.info.note),
                },
                "dp": {
                    "device_requested": str(args.device),
                    "device_effective": str(dp_info.get("device_effective", "cpu")),
                    "gpu_used": bool(dp_info.get("gpu_used", False)),
                    "gpu_name": dp_info.get("gpu_name"),
                    "dp_block_size": int(args.dp_block_size),
                    "dp_dtype": str(args.dp_dtype),
                },
                # Keep the user request for traceability (even though we clamp to [2..n]).
                "eval_workers": int(max(1, int(args.eval_workers))),
                # New in v2:
                "window_size_input": str(window_size_input),
                "window_size_requested": int(window_size_requested_field),
                "window_size": int(window_size_effective_field),
                "window_sizes_evaluated": [int(ws) for ws in ws_list],
                # New: baseline comparison (greedy)
                "comparison": {
                    "greedy_window_size": 1,
                    "included": True,
                },
                "note": (
                    "ws=1 is equivalent to greedy; ws=num_points is equivalent to global. "
                    "This run evaluates the requested ws (>1) and also evaluates greedy (ws=1) as a baseline."
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
            "window": {
                # Echo the user input (int or list[int]) for convenience.
                "window_size": int(window_size_effective_field),
                "window_sizes": [int(ws) for ws in ws_list],
                "ws_min": int(min(ws_list)) if len(ws_list) > 0 else 2,
                "ws_max": int(max(ws_list)) if len(ws_list) > 0 else int(window_size_effective_field),
                # Keep mapping form for easy parsing.
                "total_time_s_by_ws": dict(window_total_time_s_by_ws),
                "results_by_ws": dict(window_results_by_ws),
                # Also provide an ordered list form.
                "results": [dict(window_results_by_ws[str(ws)]) for ws in ws_list],
                # New: greedy baseline comparison (ws=1)
                "greedy": dict(greedy_payload),
                "greedy_total_time_s": float(greedy_total_s),
            },
        }

        with open(data_paths.summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, sort_keys=True)

        print(f"[eval] summary -> {data_paths.summary_json.name}")

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
