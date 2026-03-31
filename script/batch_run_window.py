#!/usr/local/anaconda3/bin/python3.12
# -*- coding: utf-8 -*-
"""
Batch runner for panda_ik_window (ROS2).

This script will:

1) Fix num_points (default: 8)
2) Loop window_size over the user-provided list (default: 3..7 inclusive)
3) For each window_size, run seeds in [seed-start .. seed-end] (inclusive)
4) For each run, execute:
      ros2 launch panda_ik_window ik_benchmark.launch.py \
        num_points:=8 seed:=7 window_size:=3 path_pattern:=random \
        data_root:=... \
        time_model:=totg device:=cuda dp_block_size:=256 \
        (plus optional --extra launch args)
5) Parse each run's data_root/<timestamp>/summary.json
6) Collect per-point values into CSV with:
   - rows: r1..rN
   - columns: seed (seed-start..seed-end)
   One CSV per window_size group.

Notes on summary parsing
------------------------
- If summary.json contains:
      optimization_rate.values = [{"point":"p1","r_i":...}, ...]
  we will use it.
- Otherwise (default panda_ik_window summary format), we fall back to:
      window.results_by_ws[str(window_size)].segment_times_s
  (i.e. per-segment time in seconds).

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


def _extract_rates(summary_json: Path, num_points_expected: int, window_size: int) -> List[float]:
    """
    Return [r1..rN] from one run summary.json.

    Priority:
      1) optimization_rate.values[*].r_i (if present)
      2) window.results_by_ws[ws].segment_times_s (fallback)
    """
    with summary_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # (1) optimization_rate.values (user-provided/legacy/custom field)
    vals = data.get("optimization_rate", {}).get("values", [])
    if isinstance(vals, list) and vals:

        def _point_index(v: Dict) -> int:
            p = str(v.get("point", ""))
            if p.startswith("p"):
                try:
                    return int(p[1:])
                except Exception:
                    return 10**9
            return 10**9

        vals_sorted = sorted(vals, key=_point_index)
        rates = [float(v.get("r_i", math.nan)) for v in vals_sorted]
    else:
        # (2) fallback: segment_times_s in panda_ik_window summary
        ws_key = str(int(window_size))
        window = data.get("window", {})
        results_by_ws = {}
        if isinstance(window, dict):
            results_by_ws = window.get("results_by_ws", {}) or {}
        res = None
        if isinstance(results_by_ws, dict):
            res = results_by_ws.get(ws_key)

        if not isinstance(res, dict):
            raise ValueError(
                f"No optimization_rate.values AND no window.results_by_ws['{ws_key}'] in {summary_json}"
            )

        seg_times = res.get("segment_times_s", [])
        if not isinstance(seg_times, list) or not seg_times:
            raise ValueError(f"No segment_times_s for ws={ws_key} in {summary_json}")

        rates = [float(x) for x in seg_times]

    # Normalize length to num_points_expected
    if len(rates) < num_points_expected:
        rates = rates + [math.nan] * (num_points_expected - len(rates))
    elif len(rates) > num_points_expected:
        rates = rates[:num_points_expected]

    return rates


def _extract_last_override(extra_args: Sequence[str], key: str) -> tuple[Optional[str], List[str]]:
    """
    From extra_args, extract the last occurrence of f"{key}:=..."
    Return (override_value_or_None, extra_args_without_that_key).
    Keep the relative order of the remaining args.
    """
    override: Optional[str] = None
    kept: List[str] = []
    prefix = f"{key}:="
    for a in extra_args:
        if a.startswith(prefix):
            override = a.split(":=", 1)[1]
        else:
            kept.append(a)
    return override, kept


def _run_ros2_launch(
    *,
    pkg: str,
    launch_file: str,
    num_points: int,
    seed: int,
    window_size: int,
    path_pattern: str,
    data_root: Path,
    # time_model: str,
    # device: str,
    # dp_block_size: int,
    log_file: Optional[Path],
    # extra_launch_args: Sequence[str],
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
        # GPU/DP args (as requested)
        # f"time_model:={time_model}",
        # f"device:={device}",
        # f"dp_block_size:={int(dp_block_size)}",
    ]
    # cmd.extend(extra_launch_args)

    print("\n[batch] running:")
    print(" ".join(cmd))

    if log_file is None:
        proc = subprocess.run(cmd)
        return int(proc.returncode)

    _ensure_dir(log_file.parent)
    with log_file.open("w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    return int(proc.returncode)


def _write_rates_csv(
    *,
    out_csv: Path,
    seeds: Sequence[int],
    rates_by_seed: Dict[int, List[float]],
    num_points: int,
) -> None:
    """
    Write CSV with:
      - rows: r1..rN
      - columns: seed values
    """
    _ensure_dir(out_csv.parent)

    header = ["rate"] + [str(s) for s in seeds]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i in range(1, num_points + 1):
            row_name = f"r{i}"
            row: List[str] = [row_name]
            for s in seeds:
                vals = rates_by_seed.get(s)
                v = math.nan
                if vals is not None and (i - 1) < len(vals):
                    v = float(vals[i - 1])
                row.append(str(v))
            w.writerow(row)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Batch experiments for panda_ik_window: sweep window_size & seed, collect per-point values into CSV.",
    )

    ap.add_argument("--pkg", default="panda_ik_window", help="ROS2 package name.")
    ap.add_argument("--launch", default="ik_benchmark.launch.py", help="Launch file name.")

    ap.add_argument("--num-points", type=int, default=8, help="Fixed num_points (default: 8).")

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
        "--path_pattern",
        type=str,
        default="random",
        choices=["trend", "switching", "random"],
        help="Launch arg path_pattern (default: random).",
    )

    # GPU / DP args (match your requested ros2 launch invocation)
    ap.add_argument(
        "--time-model",
        "--time_model",
        type=str,
        default="totg",
        help="Launch arg time_model (default: totg).",
    )
    ap.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Launch arg device (default: cuda). Use 'cpu' or 'auto' if needed.",
    )
    ap.add_argument(
        "--dp-block-size",
        "--dp_block_size",
        type=int,
        default=256,
        help="Launch arg dp_block_size (default: 256).",
    )

    # All outputs are under this dir
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
        help="Extra launch args appended verbatim (e.g. 'num_solutions:=150'). "
             "If you also provide time_model/device/dp_block_size here, it will override the dedicated flags.",
    )

    args = ap.parse_args()

    num_points = int(args.num_points)
    window_sizes = _parse_window_sizes_arg(str(args.window_size), num_points=num_points)
    seeds = list(range(int(args.seed_start), int(args.seed_end) + 1))
    path_pattern = str(args.path_pattern)

    base_data_root = Path(args.base_data_root)
    csv_dir = Path(args.csv_dir)
    log_dir = Path(args.log_dir)

    raw_extra_args = [a for a in args.extra.split() if a.strip()]

    # Allow --extra to override time_model/device/dp_block_size cleanly (avoid duplicate keys)
    time_model_override, extra_1 = _extract_last_override(raw_extra_args, "time_model")
    device_override, extra_2 = _extract_last_override(extra_1, "device")
    dp_bs_override, extra_3 = _extract_last_override(extra_2, "dp_block_size")

    time_model = str(time_model_override) if time_model_override is not None else str(args.time_model)
    device = str(device_override) if device_override is not None else str(args.device)
    dp_block_size = int(dp_bs_override) if dp_bs_override is not None else int(args.dp_block_size)

    extra_args = extra_3

    total_rounds = len(window_sizes) * len(seeds)
    round_idx = 0

    print("[batch] settings")
    print(f"  num_points      = {num_points}")
    print(f"  window_sizes    = {window_sizes} (from --window-size={args.window_size!r})")
    print(f"  seeds           = {seeds}")
    print(f"  path_pattern    = {path_pattern}")
    print(f"  time_model      = {time_model}")
    print(f"  device          = {device}")
    print(f"  dp_block_size   = {dp_block_size}")
    print(f"  base_data_root  = {base_data_root}")
    print(f"  csv_dir         = {csv_dir}")
    print(f"  log_dir         = {log_dir} (enabled={not args.no_logs})")
    if extra_args:
        print(f"  extra args      = {extra_args}")

    for ws in window_sizes:
        rates_by_seed: Dict[int, List[float]] = {}

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
                # time_model=time_model,
                # device=device,
                # dp_block_size=dp_block_size,
                log_file=log_file,
                # extra_launch_args=extra_args,
            )

            if rc != 0:
                print(
                    f"[batch][{_now_str()}][ERROR] ros2 launch failed (returncode={rc}) "
                    f"for ws={ws}, seed={seed}"
                )
                rates_by_seed[seed] = [math.nan] * num_points
                continue

            try:
                summary_path = _wait_for_new_summary_json(
                    run_data_root,
                    prev_latest_dirname=prev_latest_name,
                    timeout_s=120.0,
                )
                rates = _extract_rates(summary_path, num_points_expected=num_points, window_size=int(ws))
                rates_by_seed[seed] = rates

                end_ts = _now_str()
                print(f"[batch][{end_ts}] summary: {summary_path}")
                print(f"[batch][{end_ts}] r[1..{num_points}] = {rates}")
            except Exception as e:
                print(
                    f"[batch][{_now_str()}][ERROR] Failed to parse summary for ws={ws}, seed={seed}: {e}"
                )
                rates_by_seed[seed] = [math.nan] * num_points

        # One CSV per window_size group
        out_csv = csv_dir / f"np{num_points}_ws{ws}.csv"
        _write_rates_csv(
            out_csv=out_csv,
            seeds=seeds,
            rates_by_seed=rates_by_seed,
            num_points=num_points,
        )
        print(f"\n[batch] CSV written: {out_csv}\n")

    print("[batch] all done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
