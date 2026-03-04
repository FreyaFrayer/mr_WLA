#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
order_count_next_rank.py

What it does
------------
Scan ws3/seed* folders (natural sort: seed1, seed2, ..., seed10, ...).
For each seed, read:
  - summary.json (final chosen IK index per point for a fixed window_size)
  - p1.json ... p{num_points}.json (IK solution sets per point)

For i = 1..(num_points - window_size) (default 18-3=15), compute a "next-point rank":

  1) The current point configuration is fixed as the chosen solution at p_i (from summary.json): q_i*
  2) For every candidate solution at the next point p_{i+1}: {q_{i+1}^k},
     compute the segment time from q_i* to q_{i+1}^k under per-joint vmax/amax limits.
     - Joint time uses the closed-form triangular/trapezoidal profile.
     - The 7-DoF segment time is max over joints (the slowest joint dominates).
  3) Sort the next-point candidates by this segment time ascending.
  4) Record the 1-based rank of the chosen next solution q_{i+1}* in that ordering.

Finally, aggregate ranks across all seeds/transitions and print:
  - total transitions
  - count / percentage for rank <= 50, 10, 3, 1

CLI arguments
-------------
--root        Root directory containing seed* subfolders (default: ws3)
--num-points  Number of points (default: 18)
--window-size Window size (default: 3)
--dof         DoF (default: 7)
--amax        Per-joint max acceleration list (default: all ones)
--vmax        Per-joint max velocity list (default: all ones)
"""


from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

EPS = 1e-9


def parse_float_list(s: str, dof: int) -> np.ndarray:
    """Parse comma/space separated floats into (dof,) array."""
    parts = re.split(r"[\s,]+", s.strip())
    parts = [p for p in parts if p != ""]
    vals = [float(p) for p in parts]
    if len(vals) == 1:
        vals = vals * dof
    if len(vals) != dof:
        raise ValueError(f"Expected {dof} values, got {len(vals)} from {s!r}")
    arr = np.asarray(vals, dtype=np.float64)
    return np.maximum(arr, EPS)


def read_json(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"JSON decode failed: {path} ({e})") from e


def seed_sort_key(p: Path) -> Tuple[int, str]:
    """Sort seed folders as seed1, seed2, ..., seed10, ..."""
    m = re.match(r"^seed(\d+)$", p.name)
    if m:
        return (int(m.group(1)), p.name)
    m2 = re.search(r"(\d+)", p.name)
    if m2:
        return (int(m2.group(1)), p.name)
    return (10**18, p.name)


def resolve_run_dir(seed_dir: Path) -> Optional[Path]:
    """Return a directory that contains summary.json and p*.json.

    Supports layouts:
      seedX/summary.json
      seedX/<timestamp>/summary.json
      seedX/**/summary.json

    If multiple runs exist, pick the latest by (parent folder name, mtime).
    """
    if (seed_dir / "summary.json").is_file():
        return seed_dir

    candidates = list(seed_dir.rglob("summary.json"))
    if not candidates:
        return None

    def key(p: Path) -> Tuple[str, float]:
        parent = p.parent
        try:
            mt = parent.stat().st_mtime
        except Exception:
            mt = 0.0
        return (parent.name, mt)

    candidates.sort(key=key)
    return candidates[-1].parent


def load_point_q(run_dir: Path, i: int, summary: Optional[dict]) -> np.ndarray:
    """Load joint_positions for point p{i} as (K,dof) float64 array."""
    fname = None
    if summary is not None:
        ik_files = summary.get("ik_files")
        if isinstance(ik_files, dict):
            fname = ik_files.get(f"p{i}")
    if not fname:
        fname = f"p{i}.json"

    p_path = run_dir / fname
    obj = read_json(p_path)

    sols = obj.get("solutions", [])
    Q = np.asarray([s["joint_positions"] for s in sols], dtype=np.float64)
    if Q.ndim != 2:
        raise RuntimeError(f"Bad joint_positions shape in {p_path}")
    return Q


def time_from_single_to_many(q_from: np.ndarray, q_to_all: np.ndarray, vmax: np.ndarray, amax: np.ndarray) -> np.ndarray:
    """Compute segment time from one config to many configs: (K,)"""
    # dq: (K, dof)
    dq = np.abs(q_to_all - q_from.reshape((1, -1)))

    dcrit = (vmax * vmax) / amax  # (dof,)
    t_tri = 2.0 * np.sqrt(dq / amax.reshape((1, -1)))
    t_trap = (dq / vmax.reshape((1, -1))) + (vmax / amax).reshape((1, -1))
    t_joint = np.where(dq <= dcrit.reshape((1, -1)), t_tri, t_trap)

    return np.max(t_joint, axis=1)  # max over joints


def chosen_indices_from_summary(summary: dict, window_size: int) -> Dict[int, int]:
    """Extract chosen solution_index_0based for each point p1..pN from summary.json.

    In your dataset, summary['window']['results_by_ws'][str(ws)]['final_path']['segments']
    contains a list of length N, where each segment has:
      - to: 'p{i}'
      - solution_index_0based: chosen solution index at that point
    """
    w = summary.get("window", {})

    rbw = w.get("results_by_ws")
    if isinstance(rbw, dict) and str(window_size) in rbw:
        item = rbw[str(window_size)]
        final_path = item.get("final_path", {})
        segs = final_path.get("segments") if isinstance(final_path, dict) else None
        if isinstance(segs, list) and segs:
            out: Dict[int, int] = {}
            for seg in segs:
                to = seg.get("to", "")
                m = re.match(r"^p(\d+)$", str(to))
                if not m:
                    continue
                i = int(m.group(1))
                idx0 = seg.get("solution_index_0based")
                if idx0 is None:
                    idx1 = seg.get("solution_index_1based")
                    if idx1 is not None:
                        idx0 = int(idx1) - 1
                if idx0 is None:
                    continue
                out[i] = int(idx0)
            if out:
                return out

    # fallback
    results = w.get("results")
    if isinstance(results, dict):
        final_path = results.get("final_path", {})
        segs = final_path.get("segments") if isinstance(final_path, dict) else None
        if isinstance(segs, list) and segs:
            out = {}
            for seg in segs:
                to = seg.get("to", "")
                m = re.match(r"^p(\d+)$", str(to))
                if not m:
                    continue
                i = int(m.group(1))
                idx0 = seg.get("solution_index_0based")
                if idx0 is None:
                    idx1 = seg.get("solution_index_1based")
                    if idx1 is not None:
                        idx0 = int(idx1) - 1
                if idx0 is None:
                    continue
                out[i] = int(idx0)
            if out:
                return out

    raise RuntimeError(
        "Cannot find chosen indices in summary.json. "
        "Expected summary['window']['results_by_ws'][str(window_size)]['final_path']['segments'][...]['solution_index_0based']."
    )


@dataclass
class SeedResult:
    seed_name: str
    ranks_next: List[int]  # rank of chosen next solution among next candidates


def process_seed(
    seed_dir: Path,
    *,
    num_points: int,
    window_size: int,
    vmax: np.ndarray,
    amax: np.ndarray,
) -> Optional[SeedResult]:
    run_dir = resolve_run_dir(seed_dir)
    if run_dir is None:
        print(f"[WARN] skip {seed_dir.name}: no summary.json found")
        return None

    summary_path = run_dir / "summary.json"
    summary = read_json(summary_path)

    chosen = chosen_indices_from_summary(summary, window_size=window_size)

    limit = int(num_points) - int(window_size)
    if limit <= 0:
        raise ValueError(f"num_points={num_points} must be > window_size={window_size}")

    # preload all points' solution sets
    Q_by_i: Dict[int, np.ndarray] = {}
    for i in range(1, int(num_points) + 1):
        Q_by_i[i] = load_point_q(run_dir, i, summary)

    ranks: List[int] = []

    # transitions: p_i -> p_{i+1}, i=1..limit
    for i in range(1, limit + 1):
        Qcur = Q_by_i[i]
        Qnxt = Q_by_i[i + 1]

        idx_cur = chosen.get(i)
        idx_nxt = chosen.get(i + 1)
        if idx_cur is None or idx_nxt is None:
            raise RuntimeError(f"Missing chosen index for p{i} or p{i+1} in {summary_path}")

        if not (0 <= idx_cur < Qcur.shape[0]):
            raise RuntimeError(f"Chosen index out of range at p{i}: {idx_cur} (K={Qcur.shape[0]})")
        if not (0 <= idx_nxt < Qnxt.shape[0]):
            raise RuntimeError(f"Chosen index out of range at p{i+1}: {idx_nxt} (K={Qnxt.shape[0]})")

        q_from = Qcur[idx_cur]
        scores = time_from_single_to_many(q_from, Qnxt, vmax=vmax, amax=amax)  # (K_next,)

        order = np.argsort(scores, kind="mergesort")  # stable sort
        pos = int(np.where(order == int(idx_nxt))[0][0])
        ranks.append(pos + 1)

    return SeedResult(seed_name=seed_dir.name, ranks_next=ranks)


def print_stats(ranks: Sequence[int]) -> None:
    total = int(len(ranks))
    if total == 0:
        print("[WARN] no ranks computed")
        return

    def line(thr: int) -> str:
        cnt = int(sum(1 for r in ranks if int(r) <= thr))
        pct = 100.0 * cnt / total
        return f"rank<= {thr:<2d}: {cnt:>7d} / {total:<7d}  ({pct:6.2f}%)"

    ranks_np = np.asarray(list(ranks), dtype=np.int32)
    print("\n[Summary]")
    print(f"total_transitions: {total}  (i=1..num_points-window_size per seed)")
    print(line(50))
    print(line(10))
    print(line(3))
    print(line(1))

    print("\n[Rank distribution (sanity check)]")
    print(
        "min/median/mean/max: "
        f"{int(ranks_np.min())} / {int(np.median(ranks_np))} / {float(ranks_np.mean()):.2f} / {int(ranks_np.max())}"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="ws3", help="Folder that contains seed* subfolders")
    ap.add_argument("--num-points", type=int, default=18)
    ap.add_argument("--window-size", type=int, default=3)
    ap.add_argument("--dof", type=int, default=7)
    ap.add_argument("--amax", type=str, default="1,1,1,1,1,1,1", help="Joint max acceleration list")
    ap.add_argument("--vmax", type=str, default="1,1,1,1,1,1,1", help="Joint max velocity list")

    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"root not found: {root}")

    dof = int(args.dof)
    vmax = parse_float_list(args.vmax, dof)
    amax = parse_float_list(args.amax, dof)

    seed_dirs = [p for p in root.iterdir() if p.is_dir() and p.name.startswith("seed")]
    seed_dirs.sort(key=seed_sort_key)

    all_ranks: List[int] = []

    for sd in seed_dirs:
        print(f"[Seed] {sd.name}")
        try:
            res = process_seed(
                sd,
                num_points=int(args.num_points),
                window_size=int(args.window_size),
                vmax=vmax,
                amax=amax,
            )
        except Exception as e:
            print(f"[WARN] {sd.name} failed: {e}")
            continue

        if res is None:
            continue

        all_ranks.extend(res.ranks_next)
        print(f"[Done] {sd.name}  transitions={len(res.ranks_next)}")

    print_stats(all_ranks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
