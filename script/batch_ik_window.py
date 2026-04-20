#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch runner for panda_ik_window (ROS2).

This script will:

1) Fix num_points (default: 16)
2) Loop window_size over the user-provided list (default: 3..7 inclusive)
3) For each window_size, run seeds in [seed-start .. seed-end] (inclusive)
4) For each run, execute:
      ros2 launch panda_ik_window ik_benchmark.launch.py num_points:=16 seed:=7 window_size:=3
   (plus data_root:=... and optional extra launch args)
5) Parse each run's data_root/<timestamp>/summary.json
6) Collect one aggregated CSV with one row per seed:
   - seed
   - solve_total_time_s
   - origin_time
   - ws1_time
   - ws1_planner_time
   - ws{K}_time
   - ws{K}_planner_time
   - ws{K}_opt_rate_vs_ws1
   - ws{K}_planner_opt_rate_vs_origin

Notes on summary parsing
------------------------
- solve_total_time_s comes from:
      ik_solve_timing.total_s
- origin_time comes from:
      origin.total_time_s
- ws total time comes from:
      window.total_time_s_by_ws
- ws planner replay time comes from:
      trapezoid_solutions_true_plan.total_time_s_by_ws

All outputs are kept under data_window/ by default:
  - raw run outputs: data_window/np{N}/ws{W}/seed{S}/<timestamp>/*
  - csv:             data_window/batch_csv/
  - per-run logs:    data_window/batch_logs/
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _now_str() -> str:
    """Human-readable local time with timezone offset (e.g. 2026-01-26 14:03:21+08:00)."""
    return datetime.now().astimezone().isoformat(sep=" ", timespec="seconds")


def _find_latest_run_dir(data_root: Path) -> Optional[Path]:
    """
    Find latest <data_root>/<timestamp>/ directory.

    Benchmark timestamps use: YYYYMMDD_HHMMSS
    Lexicographic order works for 'latest'.
    """
    if not data_root.exists():
        return None
    dirs = [d for d in data_root.iterdir() if d.is_dir()]
    if not dirs:
        return None
    return sorted(dirs, key=lambda d: d.name)[-1]


def _wait_for_new_summary_json(
    data_root: Path,
    prev_latest_dirname: Optional[str],
    timeout_s: float = 90.0,
) -> Path:
    """
    Wait until a *new* <data_root>/<timestamp>/summary.json appears.

    This avoids accidentally reading an old summary.json when re-running the same (ws, seed)
    combination into an existing data_root.
    """
    deadline = time.time() + timeout_s
    last_err: Exception | None = None

    while time.time() < deadline:
        try:
            latest_dir = _find_latest_run_dir(data_root)
            if latest_dir is None:
                time.sleep(0.2)
                continue

            if prev_latest_dirname is not None and latest_dir.name == prev_latest_dirname:
                time.sleep(0.2)
                continue

            summary_path = latest_dir / "summary.json"
            if summary_path.exists():
                return summary_path
        except Exception as e:
            last_err = e
        time.sleep(0.2)

    raise TimeoutError(
        f"Timeout waiting for *new* summary.json under {data_root}. Last error: {last_err}"
    )


def _parse_window_sizes_arg(spec: str, *, num_points: int) -> List[int]:
    """
    Parse --window-size into a list of ints.

    Supported:
      - single int: "3"
      - comma/space separated: "1,3,8" / "1 3 8"
      - JSON list: "[1,3,8]"
      - range: "3-7" (inclusive)
      - special: "all" -> [1..num_points]

    Output is de-duplicated (keep order) and clamped to [1, num_points].
    """
    import json as _json

    raw = "" if spec is None else str(spec).strip()
    if raw == "" or raw.lower() in {"all", "sweep", "*"}:
        req = list(range(1, int(num_points) + 1))
    else:
        # Try range "a-b" (inclusive)
        m = re.fullmatch(r"\s*([+-]?\d+)\s*-\s*([+-]?\d+)\s*", raw)
        if m is not None:
            a = int(m.group(1))
            b = int(m.group(2))
            step = 1 if a <= b else -1
            req = list(range(a, b + step, step))
        # Try single integer
        elif re.fullmatch(r"[+-]?\d+", raw):
            req = [int(raw)]
        # Try JSON list
        elif raw.startswith("[") and raw.endswith("]"):
            try:
                obj = _json.loads(raw)
                if isinstance(obj, list):
                    req = [int(v) for v in obj]
                else:
                    req = [int(obj)]
            except Exception as e:
                raise ValueError(f"Invalid --window-size JSON: {raw!r} ({e})") from e
        else:
            # Comma / whitespace separated.
            parts = [p for p in re.split(r"[\s,]+", raw) if p.strip()]
            if not parts:
                raise ValueError(f"Invalid --window-size: {raw!r}")
            req = [int(p) for p in parts]

    n = max(1, int(num_points))
    out: List[int] = []
    for ws in req:
        w = int(ws)
        if w < 1:
            w = 1
        if w > n:
            w = n
        if w not in out:
            out.append(w)

    if not out:
        raise ValueError("--window-size parsed to an empty list")
    return out


def _extract_summary_metrics(
    summary_json: Path,
) -> tuple[float, float, Dict[int, float], Dict[int, float]]:
    """Return solve/origin/window/planner-replay totals from one run summary.json."""
    with summary_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    solve_total_time_s = math.nan
    ik_solve_timing = data.get("ik_solve_timing", {})
    if isinstance(ik_solve_timing, dict):
        try:
            solve_total_time_s = float(ik_solve_timing.get("total_s", math.nan))
        except Exception:
            solve_total_time_s = math.nan

    origin_time_s = math.nan
    origin = data.get("origin", {})
    if isinstance(origin, dict):
        try:
            origin_time_s = float(origin.get("total_time_s", math.nan))
        except Exception:
            origin_time_s = math.nan

    totals_by_ws: Dict[int, float] = {}
    window = data.get("window", {})
    if isinstance(window, dict):
        total_time_s_by_ws = window.get("total_time_s_by_ws", {}) or {}
        if isinstance(total_time_s_by_ws, dict):
            for ws_key, total_s in total_time_s_by_ws.items():
                try:
                    totals_by_ws[int(ws_key)] = float(total_s)
                except Exception:
                    continue

        if not totals_by_ws:
            results_by_ws = window.get("results_by_ws", {}) or {}
            if isinstance(results_by_ws, dict):
                for ws_key, payload in results_by_ws.items():
                    if not isinstance(payload, dict):
                        continue
                    raw_total = payload.get("total_time_s", None)
                    if raw_total is None:
                        final_path = payload.get("final_path", {})
                        if isinstance(final_path, dict):
                            raw_total = final_path.get("total_time_s", None)
                    try:
                        totals_by_ws[int(ws_key)] = float(raw_total)
                    except Exception:
                        continue

    planner_totals_by_ws: Dict[int, float] = {}
    planner_replay = data.get("trapezoid_solutions_true_plan", {})
    if isinstance(planner_replay, dict):
        planner_total_time_s_by_ws = planner_replay.get("total_time_s_by_ws", {}) or {}
        if isinstance(planner_total_time_s_by_ws, dict):
            for ws_key, total_s in planner_total_time_s_by_ws.items():
                try:
                    planner_totals_by_ws[int(ws_key)] = float(total_s)
                except Exception:
                    continue

        if not planner_totals_by_ws:
            replay_results_by_ws = planner_replay.get("results_by_ws", {}) or {}
            if isinstance(replay_results_by_ws, dict):
                for ws_key, payload in replay_results_by_ws.items():
                    if not isinstance(payload, dict):
                        continue
                    try:
                        planner_totals_by_ws[int(ws_key)] = float(payload.get("total_time_s", math.nan))
                    except Exception:
                        continue

    return solve_total_time_s, origin_time_s, totals_by_ws, planner_totals_by_ws


def _run_ros2_launch(
    *,
    pkg: str,
    launch_file: str,
    num_points: int,
    seed: int,
    window_size: int,
    path_pattern: str,
    data_root: Path,
    log_file: Optional[Path],
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
        f"window_size:={window_size}",
        f"path_pattern:={path_pattern}",
        f"data_root:={str(data_root)}",
    ]
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


def _relative_improvement_against_ws1(ws1_time: float, ws_time: float) -> float:
    """Return relative improvement over ws=1: (ws1 - ws) / ws1."""
    if not math.isfinite(float(ws1_time)) or float(ws1_time) <= 0.0:
        return math.nan
    if not math.isfinite(float(ws_time)):
        return math.nan
    return float((float(ws1_time) - float(ws_time)) / float(ws1_time))


def _relative_planner_improvement_against_origin(origin_time: float, planner_time: float) -> float:
    """Return planner replay improvement over origin: (origin - planner) / origin."""
    if not math.isfinite(float(origin_time)) or float(origin_time) <= 0.0:
        return math.nan
    if not math.isfinite(float(planner_time)):
        return math.nan
    return float((float(origin_time) - float(planner_time)) / float(origin_time))


def _write_summary_csv(
    *,
    out_csv: Path,
    seeds: Sequence[int],
    window_sizes: Sequence[int],
    summary_by_seed: Dict[int, Dict[str, object]],
) -> None:
    """Write one aggregated CSV row per seed."""
    _ensure_dir(out_csv.parent)

    ordered_all_ws = [int(ws) for ws in window_sizes]
    ordered_other_ws = [int(ws) for ws in window_sizes if int(ws) != 1]
    header = ["seed", "solve_total_time_s", "origin_time"]
    for ws in ordered_all_ws:
        header.append(f"ws{int(ws)}_time")
        header.append(f"ws{int(ws)}_planner_time")
    for ws in ordered_other_ws:
        header.append(f"ws{int(ws)}_opt_rate_vs_ws1")
        header.append(f"ws{int(ws)}_planner_opt_rate_vs_origin")

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for seed in seeds:
            payload = summary_by_seed.get(int(seed), {})
            solve_total_time_s = math.nan
            origin_time_s = math.nan
            total_time_s_by_ws: Dict[int, float] = {}
            planner_time_s_by_ws: Dict[int, float] = {}

            if isinstance(payload, dict):
                try:
                    solve_total_time_s = float(payload.get("solve_total_time_s", math.nan))
                except Exception:
                    solve_total_time_s = math.nan
                try:
                    origin_time_s = float(payload.get("origin_time_s", math.nan))
                except Exception:
                    origin_time_s = math.nan
                raw_totals = payload.get("total_time_s_by_ws", {})
                if isinstance(raw_totals, dict):
                    for ws_key, total_s in raw_totals.items():
                        try:
                            total_time_s_by_ws[int(ws_key)] = float(total_s)
                        except Exception:
                            continue
                raw_planner_totals = payload.get("planner_time_s_by_ws", {})
                if isinstance(raw_planner_totals, dict):
                    for ws_key, total_s in raw_planner_totals.items():
                        try:
                            planner_time_s_by_ws[int(ws_key)] = float(total_s)
                        except Exception:
                            continue

            ws1_time = float(total_time_s_by_ws.get(1, math.nan))
            row: List[object] = [int(seed), solve_total_time_s, origin_time_s]
            for ws in ordered_all_ws:
                ws_time = float(total_time_s_by_ws.get(int(ws), math.nan))
                row.append(ws_time)
                row.append(float(planner_time_s_by_ws.get(int(ws), math.nan)))
            for ws in ordered_other_ws:
                ws_time = float(total_time_s_by_ws.get(int(ws), math.nan))
                row.append(_relative_improvement_against_ws1(ws1_time, ws_time))
                ws_planner_time = float(planner_time_s_by_ws.get(int(ws), math.nan))
                row.append(_relative_planner_improvement_against_origin(origin_time_s, ws_planner_time))
            w.writerow(row)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Batch experiments for panda_ik_window: sweep window_size & seed, collect per-seed total timing summary into CSV.",
    )

    ap.add_argument("--pkg", default="panda_ik_window", help="ROS2 package name.")
    ap.add_argument("--launch", default="ik_benchmark.launch.py", help="Launch file name.")

    ap.add_argument("--num-points", type=int, default=16, help="Fixed num_points (default: 16).")

    ap.add_argument(
        "--window-size",
        "--window_size",
        type=str,
        default="3",
        help=(
            "Window size(s) to sweep. "
            "Accepts single int '3', range '3-7', list '1,3,8', JSON '[1,3,8]', or 'all'. "
            "Default: '3-7'."
        ),
    )

    ap.add_argument("--seed-start", type=int, default=7, help="Seed start (inclusive).")
    ap.add_argument("--seed-end", type=int, default=10, help="Seed end (inclusive).")
    ap.add_argument(
        "--path-pattern",
        type=str,
        default="random",
        choices=["trend", "switching", "random"],
        help="Path pattern for target generation.",
    )

    # All outputs are under this dir (as requested)
    ap.add_argument(
        "--base-data-root",
        type=str,
        default="data_window",
        help="Base dir for raw outputs (default: data_window).",
    )
    ap.add_argument(
        "--csv-dir",
        type=str,
        default="data_window/batch_csv",
        help="Dir for aggregated CSV outputs (default: data_window/batch_csv).",
    )
    ap.add_argument(
        "--log-dir",
        type=str,
        default="data_window/batch_logs",
        help="Dir for per-run ros2 logs (default: data_window/batch_logs).",
    )
    ap.add_argument("--no-logs", action="store_true", help="Print ros2 output to console instead of logs.")

    ap.add_argument(
        "--extra",
        type=str,
        default="",
        help="Extra launch args appended verbatim (e.g. 'num_solutions:=150').",
    )

    args = ap.parse_args()

    num_points = int(args.num_points)
    window_sizes = _parse_window_sizes_arg(str(args.window_size), num_points=num_points)
    seeds = list(range(int(args.seed_start), int(args.seed_end) + 1))
    path_pattern = str(args.path_pattern)

    base_data_root = Path(args.base_data_root)
    csv_dir = Path(args.csv_dir)
    log_dir = Path(args.log_dir)

    extra_args = [a for a in args.extra.split() if a.strip()]

    total_rounds = len(window_sizes) * len(seeds)
    round_idx = 0
    summary_by_seed: Dict[int, Dict[str, object]] = {
        int(seed): {
            "solve_total_time_s": math.nan,
            "origin_time_s": math.nan,
            "total_time_s_by_ws": {int(ws): math.nan for ws in window_sizes},
            "planner_time_s_by_ws": {int(ws): math.nan for ws in window_sizes},
        }
        for seed in seeds
    }

    print("[batch] settings")
    print(f"  num_points      = {num_points}")
    print(f"  window_sizes    = {window_sizes} (from --window-size={args.window_size!r})")
    print(f"  seeds           = {seeds}")
    print(f"  path_pattern    = {path_pattern}")
    print(f"  base_data_root  = {base_data_root}")
    print(f"  csv_dir         = {csv_dir}")
    print(f"  log_dir         = {log_dir} (enabled={not args.no_logs})")
    if extra_args:
        print(f"  extra args      = {extra_args}")

    for ws in window_sizes:
        print("\n" + "=" * 80)
        print(f"[batch] group: num_points={num_points}, window_size={ws}")
        print("=" * 80)

        for seed in seeds:
            round_idx += 1
            print(
                f"\n[batch] 当前轮次/总轮次: {round_idx}/{total_rounds} | "
                f"ws={ws}, seed={seed} | START @ {_now_str()}"
            )

            # Unique data_root per (ws, seed); benchmark creates inner <timestamp> dir.
            run_data_root = base_data_root / f"np{num_points}" / f"ws{ws}" / f"seed{seed}"

            # record previous latest dir to avoid reading an old summary.json
            prev_latest = _find_latest_run_dir(run_data_root)
            prev_latest_name = prev_latest.name if prev_latest is not None else None

            log_file = None
            if not args.no_logs:
                log_file = log_dir / f"np{num_points}_ws{ws}_seed{seed}.log"

            rc = _run_ros2_launch(
                pkg=str(args.pkg),
                launch_file=str(args.launch),
                num_points=num_points,
                seed=seed,
                window_size=int(ws),
                path_pattern=path_pattern,
                data_root=run_data_root,
                log_file=log_file,
                extra_launch_args=extra_args,
            )

            if rc != 0:
                print(
                    f"[batch][{_now_str()}][ERROR] ros2 launch failed (returncode={rc}) "
                    f"for ws={ws}, seed={seed}"
                )
                summary_by_seed[int(seed)]["origin_time_s"] = math.nan
                summary_by_seed[int(seed)]["total_time_s_by_ws"][int(ws)] = math.nan
                summary_by_seed[int(seed)]["planner_time_s_by_ws"][int(ws)] = math.nan
                continue

            try:
                summary_path = _wait_for_new_summary_json(
                    run_data_root,
                    prev_latest_dirname=prev_latest_name,
                    timeout_s=120.0,
                )
                solve_total_time_s, origin_time_s, total_time_s_by_ws, planner_time_s_by_ws = _extract_summary_metrics(summary_path)
                if math.isnan(float(summary_by_seed[int(seed)]["solve_total_time_s"])) or int(ws) == 1:
                    summary_by_seed[int(seed)]["solve_total_time_s"] = float(solve_total_time_s)
                if math.isnan(float(summary_by_seed[int(seed)]["origin_time_s"])) or int(ws) == 1:
                    summary_by_seed[int(seed)]["origin_time_s"] = float(origin_time_s)
                summary_by_seed[int(seed)]["total_time_s_by_ws"][int(ws)] = float(
                    total_time_s_by_ws.get(int(ws), math.nan)
                )
                summary_by_seed[int(seed)]["planner_time_s_by_ws"][int(ws)] = float(
                    planner_time_s_by_ws.get(int(ws), math.nan)
                )

                end_ts = _now_str()
                print(f"[batch][{end_ts}] summary: {summary_path}")
                print(f"[batch][{end_ts}] solve_total_time_s = {solve_total_time_s}")
                print(f"[batch][{end_ts}] origin_time_s = {origin_time_s}")
                print(
                    f"[batch][{end_ts}] ws={ws} total_time_s = "
                    f"{summary_by_seed[int(seed)]['total_time_s_by_ws'][int(ws)]}"
                )
                print(
                    f"[batch][{end_ts}] ws={ws} planner_time_s = "
                    f"{summary_by_seed[int(seed)]['planner_time_s_by_ws'][int(ws)]}"
                )
            except Exception as e:
                print(
                    f"[batch][{_now_str()}][ERROR] Failed to parse summary for ws={ws}, seed={seed}: {e}"
                )
                summary_by_seed[int(seed)]["origin_time_s"] = math.nan
                summary_by_seed[int(seed)]["total_time_s_by_ws"][int(ws)] = math.nan
                summary_by_seed[int(seed)]["planner_time_s_by_ws"][int(ws)] = math.nan

    window_slug = "_".join(str(int(ws)) for ws in window_sizes)
    out_csv = csv_dir / f"np{num_points}_ws{window_slug}_summary.csv"
    _write_summary_csv(
        out_csv=out_csv,
        seeds=seeds,
        window_sizes=window_sizes,
        summary_by_seed=summary_by_seed,
    )
    print(f"\n[batch] CSV written: {out_csv}\n")

    print("[batch] all done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
