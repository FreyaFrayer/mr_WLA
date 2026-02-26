#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch runner for panda_ik_window (ROS2).

It will:
1) Loop fixed num_points=np (default: 8)
2) For each seed in --seeds (default: 7,8,9), run ONE ros2 launch call.
   - Each run generates a dataset (depends on seed)
   - Then evaluates window_size ws=1..np (depends only on ws)
3) Parse each run's data_root/<timestamp>/summary.json
4) Collect total_time_s for each ws into CSV with:
   - rows: ws1..wsN
   - columns: seed values
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence


def _parse_int_list(s: str) -> List[int]:
    """Parse comma-separated ints, e.g. '7,8,9'."""
    out: List[int] = []
    for part in s.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _now_str() -> str:
    """Human-readable local time with timezone offset (e.g. 2026-01-26 14:03:21+08:00)."""
    return datetime.now().astimezone().isoformat(sep=" ", timespec="seconds")


def _find_latest_run_dir(data_root: Path) -> Path:
    """
    Find latest <data_root>/<timestamp>/ directory.

    Benchmark timestamps use: YYYYMMDD_HHMMSS
    Lexicographic order works for 'latest'.
    """
    if not data_root.exists():
        raise FileNotFoundError(f"data_root not found: {data_root}")
    dirs = [d for d in data_root.iterdir() if d.is_dir()]
    if not dirs:
        raise FileNotFoundError(f"No run dirs under: {data_root}")
    return sorted(dirs, key=lambda d: d.name)[-1]


def _wait_for_summary_json(data_root: Path, timeout_s: float = 120.0) -> Path:
    """Wait until <data_root>/<latest_ts>/summary.json exists."""
    deadline = time.time() + timeout_s
    last_err: Exception | None = None
    while time.time() < deadline:
        try:
            run_dir = _find_latest_run_dir(data_root)
            summary_path = run_dir / "summary.json"
            if summary_path.exists():
                return summary_path
        except Exception as e:
            last_err = e
        time.sleep(0.2)
    raise TimeoutError(f"Timeout waiting for summary.json under {data_root}. Last error: {last_err}")


def _extract_total_time_by_ws(summary_json: Path, num_points_expected: int) -> List[float]:
    """Return [T(ws=1), T(ws=2), ..., T(ws=N)] from one run summary.json."""
    with summary_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    win = data.get("window", {})
    mapping = win.get("total_time_s_by_ws", {})

    out: List[float] = []
    for ws in range(1, int(num_points_expected) + 1):
        v = math.nan
        try:
            if isinstance(mapping, dict):
                if str(ws) in mapping:
                    v = float(mapping[str(ws)])
            else:
                v = math.nan
        except Exception:
            v = math.nan
        out.append(float(v))
    return out


def _run_ros2_launch(
    *,
    pkg: str,
    launch_file: str,
    num_points: int,
    seed: int,
    time_model: str,
    data_root: Path,
    num_solutions: int | None,
    log_file: Path | None,
    extra_launch_args: Sequence[str],
) -> int:
    """Run one ros2 launch call. Return process return code."""
    cmd = [
        "ros2",
        "launch",
        pkg,
        launch_file,
        f"num_points:={num_points}",
        f"seed:={seed}",
        f"time_model:={time_model}",
        f"data_root:={str(data_root)}",
    ]
    if num_solutions is not None:
        cmd.append(f"num_solutions:={int(num_solutions)}")
    cmd.extend(extra_launch_args)

    print("\n[batch] running:")
    print(" ".join(cmd))

    if log_file is None:
        proc = subprocess.run(cmd)
        return int(proc.returncode)

    _ensure_dir(log_file.parent)
    with log_file.open("w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    return int(proc.returncode)


def _write_total_time_csv(
    *,
    out_csv: Path,
    seeds: Sequence[int],
    total_time_by_seed: Dict[int, List[float]],
    num_points: int,
) -> None:
    """
    Write CSV with:
      - rows: ws1..wsN
      - columns: seed values
      - values: total_time_s
    """
    _ensure_dir(out_csv.parent)

    header = ["ws"] + [str(s) for s in seeds]

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for ws in range(1, int(num_points) + 1):
            row_name = f"ws{ws}"
            row: List[str] = [row_name]
            for s in seeds:
                vals = total_time_by_seed.get(s)
                v = math.nan
                if vals is not None and (ws - 1) < len(vals):
                    v = float(vals[ws - 1])
                row.append(str(v))
            w.writerow(row)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Batch experiments for panda_ik_window: collect total_time_s by window_size into CSV.",
    )
    ap.add_argument("--pkg", default="panda_ik_window", help="ROS2 package name.")
    ap.add_argument("--launch", default="ik_benchmark.launch.py", help="Launch file name.")
    ap.add_argument("--num-points", type=int, default=8, help="num_points (default: 8).")

    ap.add_argument(
        "--seeds",
        type=str,
        default="7",
        help="Comma-separated seed list (default: 7,8,9).",
    )

    ap.add_argument("--time-model", default="trapezoid", choices=["auto", "totg", "trapezoid"], help="Time model.")
    ap.add_argument("--num-solutions", type=int, default=None, help="Override num_solutions (optional).")

    ap.add_argument("--base-data-root", type=str, default="data_three/batch_data_window", help="Base dir for raw outputs.")
    ap.add_argument("--csv-dir", type=str, default="data_three/batch_csv_window", help="Dir for aggregated CSV outputs.")
    ap.add_argument("--log-dir", type=str, default="data_three/batch_logs_window", help="Dir for per-run ros2 logs.")
    ap.add_argument("--no-logs", action="store_true", help="Print ros2 output to console instead of logs.")
    ap.add_argument(
        "--extra",
        type=str,
        default="",
        help="Extra launch args appended verbatim (e.g. 'totg_vel_scale:=0.5').",
    )

    args = ap.parse_args()

    num_points = int(args.num_points)
    seeds = _parse_int_list(str(args.seeds))
    if not seeds:
        raise ValueError("--seeds is empty")

    time_model = str(args.time_model)
    num_solutions = None if args.num_solutions is None else int(args.num_solutions)

    base_data_root = Path(args.base_data_root)
    csv_dir = Path(args.csv_dir)
    log_dir = Path(args.log_dir)
    extra_args = [a for a in str(args.extra).split() if a.strip()]

    print("[batch] settings")
    print(f"  num_points     = {num_points}")
    print(f"  seeds          = {seeds}")
    print(f"  time_model     = {time_model}")
    print(f"  num_solutions  = {num_solutions if num_solutions is not None else '(launch default)'}")
    print(f"  base_data_root = {base_data_root}")
    print(f"  csv_dir        = {csv_dir}")
    print(f"  log_dir        = {log_dir} (enabled={not args.no_logs})")
    if extra_args:
        print(f"  extra launch args = {extra_args}")

    total_rounds = len(seeds)
    round_idx = 0

    total_time_by_seed: Dict[int, List[float]] = {}

    print("\n" + "=" * 80)
    print(f"[batch] group: num_points={num_points}, time_model={time_model}")
    print("=" * 80)

    for seed in seeds:
        round_idx += 1
        print(
            f"\n[batch] 当前轮次/总轮次: {round_idx}/{total_rounds} | "
            f"seed={seed} | START @ {_now_str()}"
        )

        # Unique data_root for each (np, seed); benchmark still creates inner <timestamp> dir.
        run_data_root = base_data_root / f"np{num_points}" / f"seed{seed}"

        log_file = None
        if not args.no_logs:
            log_file = log_dir / f"np{num_points}_seed{seed}.log"

        rc = _run_ros2_launch(
            pkg=str(args.pkg),
            launch_file=str(args.launch),
            num_points=num_points,
            seed=seed,
            time_model=time_model,
            data_root=run_data_root,
            num_solutions=num_solutions,
            log_file=log_file,
            extra_launch_args=extra_args,
        )
        if rc != 0:
            print(f"[batch][{_now_str()}][ERROR] ros2 launch failed (returncode={rc}) for seed={seed}")
            total_time_by_seed[seed] = [math.nan] * num_points
            continue

        try:
            summary_path = _wait_for_summary_json(run_data_root, timeout_s=180.0)
            totals = _extract_total_time_by_ws(summary_path, num_points_expected=num_points)
            total_time_by_seed[seed] = totals

            end_ts = _now_str()
            print(f"[batch][{end_ts}] summary: {summary_path}")
            print(f"[batch][{end_ts}] total_time_s(ws=1..{num_points}) = {totals}")
        except Exception as e:
            print(f"[batch][{_now_str()}][ERROR] Failed to parse summary for seed={seed}: {e}")
            total_time_by_seed[seed] = [math.nan] * num_points

    # One CSV per group
    out_csv = csv_dir / f"np{num_points}_time_{time_model}.csv"
    _write_total_time_csv(
        out_csv=out_csv,
        seeds=seeds,
        total_time_by_seed=total_time_by_seed,
        num_points=num_points,
    )
    print(f"\n[batch] CSV written: {out_csv}\n")

    print("[batch] all done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
