from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from panda_ik_window.collision.dataset_loader import (
    load_index_rows,
    load_meta,
    load_npz_arrays,
    resolve_dataset_npz,
)
from panda_ik_window.collision.state_collision import build_state_collision_checker
from panda_ik_window.collision.trapezoid_sampler import sample_synchronized_trapezoid_segment


@dataclass(frozen=True)
class RunnerConfig:
    dataset: str
    output_dir: Optional[str]
    output_prefix: str
    sample_dt: float
    min_samples: int
    max_samples: int
    max_candidates: int
    progress_every: int
    preferred_ws: int
    group: str
    tip_link: str
    node_name: str
    details_jsonl: Optional[str]


def _parse_args() -> RunnerConfig:
    p = argparse.ArgumentParser(
        description=(
            "Verify self-collision for each q_cur -> candidate(next) pair in dataset_ws3_top50_sort50 "
            "with synchronized trapezoid velocity sampling."
        )
    )

    p.add_argument(
        "--dataset",
        type=str,
        default="dataset_ws3_top50_sort50",
        help="Dataset directory or dataset_ws*.npz path. Name-only input is auto-searched under cwd/Desktop.",
    )
    p.add_argument("--preferred-ws", type=int, default=3)

    p.add_argument("--output-dir", type=str, nargs="?", default="", const="")
    p.add_argument("--output-prefix", type=str, default="self_collision_trapezoid")
    p.add_argument("--details-jsonl", type=str, nargs="?", default="", const="")

    p.add_argument("--sample-dt", type=float, default=0.02, help="Sampling step (s) along each segment.")
    p.add_argument("--min-samples", type=int, default=5, help="Minimum samples per segment (including endpoints).")

    p.add_argument("--max-samples", type=int, default=0, help="0 means all samples.")
    p.add_argument("--max-candidates", type=int, default=0, help="0 means all candidate slots.")
    p.add_argument("--progress-every", type=int, default=200)

    p.add_argument("--group", type=str, default="panda_arm")
    p.add_argument("--tip-link", type=str, nargs="?", default="", const="")
    p.add_argument("--node-name", type=str, default="panda_ws3_self_collision_check")

    # ros2 launch (Node action) injects ROS-specific args like:
    #   --ros-args --params-file <tmp.yaml>
    # Use parse_known_args so this script can run both as a plain CLI tool
    # and as a ROS2 launched node.
    args, unknown = p.parse_known_args()

    if unknown:
        # Keep strictness for non-ROS unknown args while allowing ROS extras.
        ros_tokens = {
            "--ros-args",
            "--params-file",
            "--remap",
            "-r",
            "--log-level",
            "--log-config-file",
            "--enclave",
            "--disable-rosout-logs",
            "--enable-rosout-logs",
        }
        filtered = [u for u in unknown if u not in ros_tokens and not u.startswith("__")]
        # Value tokens following ROS flags (e.g. /tmp/launch_params_xxx) have no prefix,
        # so only reject when they look like real CLI options.
        hard_unknown = [u for u in filtered if u.startswith("-")]
        if hard_unknown:
            p.error(f"unrecognized arguments: {' '.join(hard_unknown)}")
    return RunnerConfig(
        dataset=args.dataset,
        output_dir=args.output_dir or None,
        output_prefix=str(args.output_prefix),
        sample_dt=float(args.sample_dt),
        min_samples=int(args.min_samples),
        max_samples=int(args.max_samples),
        max_candidates=int(args.max_candidates),
        progress_every=max(int(args.progress_every), 1),
        preferred_ws=int(args.preferred_ws),
        group=str(args.group),
        tip_link=str(args.tip_link),
        node_name=str(args.node_name),
        details_jsonl=args.details_jsonl or None,
    )


def _bool_ratio(numer: int, denom: int) -> float:
    if denom <= 0:
        return 0.0
    return float(numer) / float(denom)


def run(cfg: RunnerConfig) -> dict:
    t0 = time.time()
    from moveit.core.robot_state import RobotState
    from panda_ik_window.utils.robot import load_robot_context

    ds = resolve_dataset_npz(cfg.dataset, preferred_ws=cfg.preferred_ws)
    arrays = load_npz_arrays(ds)

    q_cur = arrays.q_cur
    q_cand_next = arrays.q_cand_next
    cand_mask_next = arrays.cand_mask_next

    n_total = int(q_cur.shape[0])
    k_total = int(q_cand_next.shape[1])

    n_run = n_total if cfg.max_samples <= 0 else min(n_total, cfg.max_samples)
    k_run = k_total if cfg.max_candidates <= 0 else min(k_total, cfg.max_candidates)

    q_cur = q_cur[:n_run]
    q_cand_next = q_cand_next[:n_run, :k_run, :]
    cand_mask_next = cand_mask_next[:n_run, :k_run]

    ctx = load_robot_context(
        node_name=cfg.node_name,
        group=cfg.group,
        tip_link=cfg.tip_link,
    )

    checker, reason = build_state_collision_checker(ctx, group=ctx.group)
    if checker is None:
        raise RuntimeError(f"Self-collision checker unavailable: {reason}")

    vmax = ctx.velocity_limits
    amax = ctx.acceleration_limits

    coll_any = np.zeros((n_run, k_run), dtype=bool)
    coll_mid = np.zeros((n_run, k_run), dtype=bool)
    coll_start = np.zeros((n_run, k_run), dtype=bool)
    coll_end = np.zeros((n_run, k_run), dtype=bool)
    unknown = np.zeros((n_run, k_run), dtype=bool)
    first_coll_idx = np.full((n_run, k_run), fill_value=-1, dtype=np.int32)
    segment_duration_s = np.zeros((n_run, k_run), dtype=np.float32)
    segment_num_samples = np.zeros((n_run, k_run), dtype=np.int32)

    state = RobotState(ctx.robot_model)
    state.set_to_default_values()

    details_fh = None
    if cfg.details_jsonl:
        details_path = Path(cfg.details_jsonl)
        details_path.parent.mkdir(parents=True, exist_ok=True)
        details_fh = details_path.open("w", encoding="utf-8")

    try:
        for i in range(n_run):
            if (i + 1) % cfg.progress_every == 0 or i == 0:
                elapsed = time.time() - t0
                print(f"[check] sample {i + 1}/{n_run} elapsed={elapsed:.1f}s")

            q0 = q_cur[i]
            valid_slots = np.nonzero(cand_mask_next[i])[0]
            if valid_slots.size == 0:
                continue

            for k in valid_slots.tolist():
                q1 = q_cand_next[i, k]
                times_s, q_samples, seg_t = sample_synchronized_trapezoid_segment(
                    q0,
                    q1,
                    max_vel_rad_s=vmax,
                    max_acc_rad_s2=amax,
                    sample_dt=cfg.sample_dt,
                    min_samples=cfg.min_samples,
                )

                segment_duration_s[i, k] = float(seg_t)
                segment_num_samples[i, k] = int(len(times_s))

                pair_any = False
                pair_mid = False
                pair_start = False
                pair_end = False
                pair_unknown = False
                pair_first_idx = -1

                for si, q in enumerate(q_samples):
                    state.set_joint_group_positions(ctx.group, np.asarray(q, dtype=float))
                    state.update()
                    flag = checker(state)
                    if flag is None:
                        pair_unknown = True
                        break

                    if bool(flag):
                        pair_any = True
                        if pair_first_idx < 0:
                            pair_first_idx = int(si)

                        if si == 0:
                            pair_start = True
                        elif si == len(q_samples) - 1:
                            pair_end = True
                        else:
                            pair_mid = True

                coll_any[i, k] = bool(pair_any)
                coll_mid[i, k] = bool(pair_mid)
                coll_start[i, k] = bool(pair_start)
                coll_end[i, k] = bool(pair_end)
                unknown[i, k] = bool(pair_unknown)
                first_coll_idx[i, k] = int(pair_first_idx)

                if details_fh is not None and (pair_any or pair_unknown):
                    rec = {
                        "i": int(i),
                        "k": int(k),
                        "unknown": bool(pair_unknown),
                        "self_collision": bool(pair_any),
                        "mid_collision": bool(pair_mid),
                        "start_collision": bool(pair_start),
                        "end_collision": bool(pair_end),
                        "first_collision_sample_idx": int(pair_first_idx),
                        "segment_duration_s": float(seg_t),
                        "segment_num_samples": int(len(times_s)),
                    }
                    details_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    finally:
        if details_fh is not None:
            details_fh.close()

    evaluated_mask = np.asarray(cand_mask_next, dtype=bool)

    valid_pairs = int(np.count_nonzero(evaluated_mask))
    any_pairs = int(np.count_nonzero(coll_any & evaluated_mask))
    mid_pairs = int(np.count_nonzero(coll_mid & evaluated_mask))
    unknown_pairs = int(np.count_nonzero(unknown & evaluated_mask))

    elapsed_total = time.time() - t0

    out_dir = Path(cfg.output_dir) if cfg.output_dir else Path(ds.dataset_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_npz = out_dir / f"{cfg.output_prefix}.npz"
    np.savez_compressed(
        str(out_npz),
        evaluated_mask=evaluated_mask,
        self_collision=coll_any,
        mid_self_collision=coll_mid,
        start_self_collision=coll_start,
        end_self_collision=coll_end,
        unknown=unknown,
        first_collision_sample_idx=first_coll_idx,
        segment_duration_s=segment_duration_s,
        segment_num_samples=segment_num_samples,
    )

    # Persist colliding (sample, candidate) pairs in JSON.
    # This can be large for full-dataset runs.
    pair_idx = np.argwhere(coll_any & evaluated_mask)
    collision_pairs: list[dict] = []
    for i, k in pair_idx.tolist():
        rec = {
            "i": int(i),
            "k": int(k),
            "first_collision_sample_idx": int(first_coll_idx[i, k]),
            "mid_collision": bool(coll_mid[i, k]),
            "start_collision": bool(coll_start[i, k]),
            "end_collision": bool(coll_end[i, k]),
        }
        collision_pairs.append(rec)

    out_collision_pairs = out_dir / "collision_pairs.json"
    with out_collision_pairs.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "count": int(len(collision_pairs)),
                "pairs": collision_pairs,
            },
            f,
            ensure_ascii=False,
        )

    idx_rows = load_index_rows(ds.index_jsonl, limit=n_run)
    meta = load_meta(ds.meta_json)

    summary = {
        "dataset": {
            "dataset_dir": ds.dataset_dir,
            "npz_path": ds.npz_path,
            "index_jsonl": ds.index_jsonl,
            "meta_json": ds.meta_json,
            "ws": int(ds.ws),
            "dof": int(ds.dof),
            "k_max": int(ds.k_max),
            "num_samples_total": int(ds.num_samples),
        },
        "run": {
            "n_run": int(n_run),
            "k_run": int(k_run),
            "sample_dt": float(cfg.sample_dt),
            "min_samples": int(cfg.min_samples),
            "elapsed_s": float(elapsed_total),
            "group": str(ctx.group),
            "tip_link": str(ctx.tip_link),
            "joint_names": list(map(str, ctx.joint_names)),
            "vmax_rad_s": [float(v) for v in vmax.tolist()],
            "amax_rad_s2": [float(a) for a in amax.tolist()],
        },
        "counts": {
            "valid_pairs": int(valid_pairs),
            "self_collision_pairs": int(any_pairs),
            "mid_collision_pairs": int(mid_pairs),
            "unknown_pairs": int(unknown_pairs),
            "self_collision_ratio": float(_bool_ratio(any_pairs, valid_pairs)),
            "mid_collision_ratio": float(_bool_ratio(mid_pairs, valid_pairs)),
            "unknown_ratio": float(_bool_ratio(unknown_pairs, valid_pairs)),
        },
        "outputs": {
            "result_npz": str(out_npz),
            "collision_pairs_json": str(out_collision_pairs),
            "details_jsonl": str(cfg.details_jsonl) if cfg.details_jsonl else None,
        },
        "config": asdict(cfg),
        "index_preview": idx_rows[:5] if idx_rows else None,
        "meta_preview": {
            "source": meta.get("source") if isinstance(meta, dict) else None,
            "filter": meta.get("filter") if isinstance(meta, dict) else None,
        },
    }

    out_json = out_dir / f"{cfg.output_prefix}_summary.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[check] done. result_npz={out_npz}")
    print(f"[check] collision_pairs_json={out_collision_pairs}")
    print(f"[check] summary_json={out_json}")
    print(
        "[check] valid_pairs={} self_collision_pairs={} mid_collision_pairs={} unknown_pairs={}".format(
            valid_pairs,
            any_pairs,
            mid_pairs,
            unknown_pairs,
        )
    )

    return summary


def main() -> None:
    cfg = _parse_args()
    run(cfg)


if __name__ == "__main__":
    main()
