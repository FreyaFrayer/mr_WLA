#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch runner for panda_ik_window (ROS2).

It will:
1) Use a fixed num_points=np (default: 8)
2) Use ONE window_size (single integer, default: 3, and must be > 1)
3) For each seed in --seeds (default: 7), run ONE ros2 launch call.
   - Each run generates a dataset (depends on seed)
   - Then evaluates the requested window_size (ws>1)
   - And also evaluates greedy baseline (ws=1) automatically for comparison
4) Parse each run's data_root/<timestamp>/summary.json
5) Collect total_time_s into CSV with:
   - rows: greedy_ws1, ws{window_size}
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


def _extract_total_times(summary_json: Path, *, window_size: int) -> tuple[float, float]:
    """Return (T_greedy_ws1, T_requested_ws) from one run summary.json."""
    with summary_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    win = data.get("window", {})

    # Requested ws total
    main_total = math.nan
    try:
        mapping = win.get("total_time_s_by_ws", {}) if isinstance(win, dict) else {}
        if isinstance(mapping, dict) and str(int(window_size)) in mapping:
            main_total = float(mapping[str(int(window_size))])
        else:
            res_by_ws = win.get("results_by_ws", {}) if isinstance(win, dict) else {}
            if isinstance(res_by_ws, dict):
                res = res_by_ws.get(str(int(window_size)))
                if isinstance(res, dict):
                    fp = res.get("final_path", {})
                    if isinstance(fp, dict) and "total_time_s" in fp:
                        main_total = float(fp["total_time_s"])
    except Exception:
        main_total = math.nan

    # Greedy baseline total (ws=1)
    greedy_total = math.nan
    try:
        if isinstance(win, dict) and "greedy_total_time_s" in win:
            greedy_total = float(win["greedy_total_time_s"])
        elif isinstance(win, dict):
            g = win.get("greedy", {})
            if isinstance(g, dict):
                fp = g.get("final_path", {})
                if isinstance(fp, dict) and "total_time_s" in fp:
                    greedy_total = float(fp["total_time_s"])
    except Exception:
        greedy_total = math.nan

    return float(greedy_total), float(main_total)


def _run_ros2_launch(
    *,
    pkg: str,
    launch_file: str,
    num_points: int,
    seed: int,
    window_size: int,
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
        f"window_size:={int(window_size)}",
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
    greedy_time_by_seed: Dict[int, float],
    main_time_by_seed: Dict[int, float],
    window_size: int,
) -> None:
    """
    Write CSV with:
      - rows: greedy_ws1, ws{window_size}
      - columns: seed values
      - values: total_time_s
    """
    _ensure_dir(out_csv.parent)

    header = ["metric"] + [str(s) for s in seeds]

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)

        # greedy baseline
        row_g: List[str] = ["greedy_ws1"]
        for s in seeds:
            row_g.append(str(float(greedy_time_by_seed.get(int(s), math.nan))))
        w.writerow(row_g)

        # requested ws
        row_w: List[str] = [f"ws{int(window_size)}"]
        for s in seeds:
            row_w.append(str(float(main_time_by_seed.get(int(s), math.nan))))
        w.writerow(row_w)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Batch experiments for panda_ik_window: collect total_time_s by window_size into CSV.",
    )
    ap.add_argument("--pkg", default="panda_ik_window", help="ROS2 package name.")
    ap.add_argument("--launch", default="ik_benchmark.launch.py", help="Launch file name.")
    ap.add_argument("--num-points", type=int, default=8, help="num_points (default: 8).")

    ap.add_argument(
        "--window-size",
        "--window_size",
        type=int,
        default=3,
        help="Window size ws (>1). Greedy baseline (ws=1) will be evaluated automatically. Default: 3.",
    )

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
    window_size = int(args.window_size)
    if window_size <= 1:
        raise ValueError("--window-size must be an integer > 1")
    if num_points < 2:
        raise ValueError("--num-points must be >= 2 when using --window-size > 1")
    if window_size > num_points:
        print(f"[batch][warn] window_size={window_size} > num_points={num_points}, clamped to {num_points}")
        window_size = int(num_points)
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
    print(f"  window_size    = {window_size} (greedy baseline ws=1 included)")
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

    greedy_time_by_seed: Dict[int, float] = {}
    main_time_by_seed: Dict[int, float] = {}

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
            window_size=window_size,
            time_model=time_model,
            data_root=run_data_root,
            num_solutions=num_solutions,
            log_file=log_file,
            extra_launch_args=extra_args,
        )
        if rc != 0:
            print(f"[batch][{_now_str()}][ERROR] ros2 launch failed (returncode={rc}) for seed={seed}")
            greedy_time_by_seed[seed] = math.nan
            main_time_by_seed[seed] = math.nan
            continue

        try:
            summary_path = _wait_for_summary_json(run_data_root, timeout_s=180.0)
            g_total, w_total = _extract_total_times(summary_path, window_size=window_size)
            greedy_time_by_seed[seed] = float(g_total)
            main_time_by_seed[seed] = float(w_total)

            end_ts = _now_str()
            print(f"[batch][{end_ts}] summary: {summary_path}")
            print(
                f"[batch][{end_ts}] total_time_s: greedy(ws=1)={float(g_total):.6f} | "
                f"ws={int(window_size)}={float(w_total):.6f}"
            )
        except Exception as e:
            print(f"[batch][{_now_str()}][ERROR] Failed to parse summary for seed={seed}: {e}")
            greedy_time_by_seed[seed] = math.nan
            main_time_by_seed[seed] = math.nan

    # One CSV per group
    out_csv = csv_dir / f"np{num_points}_ws{int(window_size)}_time_{time_model}.csv"
    _write_total_time_csv(
        out_csv=out_csv,
        seeds=seeds,
        greedy_time_by_seed=greedy_time_by_seed,
        main_time_by_seed=main_time_by_seed,
        window_size=window_size,
    )
    print(f"\n[batch] CSV written: {out_csv}\n")

    print("[batch] all done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
