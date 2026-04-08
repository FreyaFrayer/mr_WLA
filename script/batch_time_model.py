#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch compare time models (totg vs trapezoid) for panda_ik_window.

Outputs in summary-dir:
  - *_summary.txt
  - time.csv (per seed + ws>1: origin, greedy_totg(ws=1), trapezoid(ws), trapezoid_sol_totg(ws), totg(ws), trapezoid_sol_true_plan(ws))

Example:
  python3 script/batch_time_model.py \
    --num-points 3 \
    --window-size "1,3" \
    --seed-start 10 --seed-end 109
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
from typing import Dict, List, Optional, Sequence, Tuple


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _now_str() -> str:
    return datetime.now().astimezone().isoformat(sep=" ", timespec="seconds")


def _find_latest_run_dir(data_root: Path) -> Optional[Path]:
    if not data_root.exists():
        return None
    dirs = [d for d in data_root.iterdir() if d.is_dir()]
    if not dirs:
        return None
    return sorted(dirs, key=lambda d: d.name)[-1]


def _wait_for_new_summary_json(
    data_root: Path,
    prev_latest_dirname: Optional[str],
    timeout_s: float,
) -> Path:
    deadline = time.time() + float(timeout_s)
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
    import json as _json

    raw = "" if spec is None else str(spec).strip()
    if raw == "" or raw.lower() in {"all", "sweep", "*"}:
        req = list(range(1, int(num_points) + 1))
    else:
        m = re.fullmatch(r"\s*([+-]?\d+)\s*-\s*([+-]?\d+)\s*", raw)
        if m is not None:
            a = int(m.group(1))
            b = int(m.group(2))
            step = 1 if a <= b else -1
            req = list(range(a, b + step, step))
        elif re.fullmatch(r"[+-]?\d+", raw):
            req = [int(raw)]
        elif raw.startswith("[") and raw.endswith("]"):
            obj = _json.loads(raw)
            if isinstance(obj, list):
                req = [int(v) for v in obj]
            else:
                req = [int(obj)]
        else:
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


def _safe_float(v) -> float:
    try:
        return float(v)
    except Exception:
        return float("nan")


def _is_finite(v: float) -> bool:
    return isinstance(v, float) and math.isfinite(v)


def _sum_finite(xs: Sequence[float]) -> float:
    s = 0.0
    for x in xs:
        if _is_finite(float(x)):
            s += float(x)
    return float(s)


def _filter_extra_args(extra_args: Sequence[str], blocked_keys: Sequence[str]) -> Tuple[List[str], List[str]]:
    blocked_prefixes = tuple(f"{k}:=" for k in blocked_keys)
    kept: List[str] = []
    dropped: List[str] = []
    for a in extra_args:
        if a.startswith(blocked_prefixes):
            dropped.append(a)
        else:
            kept.append(a)
    return kept, dropped


def _run_ros2_launch(
    *,
    pkg: str,
    launch_file: str,
    num_points: int,
    seed: int,
    window_size_spec: str,
    path_pattern: str,
    time_model: str,
    data_root: Path,
    reuse_candidates_dir: Optional[Path],
    log_file: Optional[Path],
    extra_launch_args: Sequence[str],
) -> int:
    cmd = [
        "ros2",
        "launch",
        pkg,
        launch_file,
        f"num_points:={int(num_points)}",
        f"seed:={int(seed)}",
        f"window_size:={str(window_size_spec)}",
        f"path_pattern:={str(path_pattern)}",
        f"time_model:={str(time_model)}",
        f"data_root:={str(data_root)}",
    ]
    if reuse_candidates_dir is not None:
        cmd.append(f"reuse_candidates_dir:={str(reuse_candidates_dir)}")
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


def _extract_model_metrics(summary_json: Path, *, ws_ref: int, ws_cmp: int) -> Dict:
    with summary_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    window = data.get("window", {})
    if not isinstance(window, dict):
        window = {}

    totals_map = window.get("total_time_s_by_ws", {})
    if not isinstance(totals_map, dict):
        totals_map = {}

    results_by_ws = window.get("results_by_ws", {})
    if not isinstance(results_by_ws, dict):
        results_by_ws = {}

    selection_by_ws = window.get("selection_timing_by_ws", {})
    if not isinstance(selection_by_ws, dict):
        selection_by_ws = {}

    origin = data.get("origin", {})
    if not isinstance(origin, dict):
        origin = {}
    origin_status = str(origin.get("status", "")).strip().lower()
    origin_total_time_s_raw = _safe_float(origin.get("total_time_s", math.nan))
    if origin_status == "ok":
        origin_total_time_s = float(origin_total_time_s_raw)
    else:
        # Origin baseline can fail at planning stage; in that case many summaries carry 0.0.
        # Treat non-ok as unavailable in batch CSV to avoid misleading zeros.
        origin_total_time_s = float("nan")

    trapezoid_solutions_totg = data.get("trapezoid_solutions_totg", {})
    if not isinstance(trapezoid_solutions_totg, dict):
        trapezoid_solutions_totg = {}
    trap_totg_total_by_ws_raw = trapezoid_solutions_totg.get("total_time_s_by_ws", {})
    if not isinstance(trap_totg_total_by_ws_raw, dict):
        trap_totg_total_by_ws_raw = {}
    trap_totg_total_by_ws: Dict[str, float] = {}
    for k, v in trap_totg_total_by_ws_raw.items():
        key = str(k).strip()
        if not key:
            continue
        trap_totg_total_by_ws[key] = _safe_float(v)

    trapezoid_solutions_true_plan = data.get("trapezoid_solutions_true_plan", {})
    if not isinstance(trapezoid_solutions_true_plan, dict):
        trapezoid_solutions_true_plan = {}
    trap_true_plan_total_by_ws_raw = trapezoid_solutions_true_plan.get("total_time_s_by_ws", {})
    if not isinstance(trap_true_plan_total_by_ws_raw, dict):
        trap_true_plan_total_by_ws_raw = {}
    trap_true_plan_total_by_ws: Dict[str, float] = {}
    for k, v in trap_true_plan_total_by_ws_raw.items():
        key = str(k).strip()
        if not key:
            continue
        trap_true_plan_total_by_ws[key] = _safe_float(v)

    window_total_time_s_by_ws: Dict[str, float] = {}
    for k, v in totals_map.items():
        key = str(k).strip()
        if not key:
            continue
        window_total_time_s_by_ws[key] = _safe_float(v)

    def _extract_ws(ws: int) -> Dict:
        key = str(int(ws))
        total_time_s = _safe_float(totals_map.get(key, math.nan))

        res = results_by_ws.get(key, {})
        if not isinstance(res, dict):
            res = {}

        final_path = res.get("final_path", {})
        if not isinstance(final_path, dict):
            final_path = {}
        segs = final_path.get("segments", [])
        if not isinstance(segs, list):
            segs = []
        selected_solution_ids: List[str] = []
        selected_solution_details: List[Dict] = []
        selected_by_point: Dict[str, Dict] = {}
        for seg in segs:
            if isinstance(seg, dict):
                point = str(seg.get("to", "")).strip()
                sid = str(seg.get("solution_id", "")).strip()
                if sid:
                    selected_solution_ids.append(sid)
                q_raw = seg.get("joint_positions", [])
                q_rad: List[float] = []
                if isinstance(q_raw, list):
                    for v in q_raw:
                        fv = _safe_float(v)
                        if _is_finite(fv):
                            q_rad.append(float(fv))
                q_deg = [float(v * 180.0 / math.pi) for v in q_rad]

                detail = {
                    "point": point,
                    "selected_solution_id": sid,
                    "joint_positions_rad": q_rad,
                    "joint_positions_deg": q_deg,
                }
                selected_solution_details.append(detail)
                if point:
                    selected_by_point[point] = detail

        recs = selection_by_ws.get(key, None)
        if not isinstance(recs, list):
            recs = res.get("selection_timing_by_point", [])
        if not isinstance(recs, list):
            recs = []

        recs_out: List[Dict] = []
        for rec in recs:
            if not isinstance(rec, dict):
                continue
            point = str(rec.get("point", ""))
            sid = str(rec.get("selected_solution_id", ""))
            sel_detail = selected_by_point.get(point, {})
            q_rad = sel_detail.get("joint_positions_rad", [])
            q_deg = sel_detail.get("joint_positions_deg", [])
            if not sid:
                sid = str(sel_detail.get("selected_solution_id", ""))
            recs_out.append(
                {
                    "point": point,
                    "selected_solution_id": sid,
                    "selection_elapsed_s": _safe_float(rec.get("selection_elapsed_s", math.nan)),
                    "joint_positions_rad": list(q_rad) if isinstance(q_rad, list) else [],
                    "joint_positions_deg": list(q_deg) if isinstance(q_deg, list) else [],
                }
            )

        selection_total_s = _safe_float(res.get("selection_timing_total_s", math.nan))
        if not _is_finite(selection_total_s):
            selection_total_s = _sum_finite([float(r.get("selection_elapsed_s", math.nan)) for r in recs_out])

        return {
            "total_time_s": float(total_time_s),
            "selected_solution_ids": selected_solution_ids,
            "selected_solution_details": selected_solution_details,
            "selection_by_point": recs_out,
            "selection_total_s": float(selection_total_s),
        }

    ws_ref_data = _extract_ws(int(ws_ref))
    ws_cmp_data = _extract_ws(int(ws_cmp))

    t_ref = float(ws_ref_data["total_time_s"])
    t_cmp = float(ws_cmp_data["total_time_s"])
    if _is_finite(t_ref) and _is_finite(t_cmp) and t_ref > 0.0:
        opt_pct = float((t_ref - t_cmp) / t_ref * 100.0)
    else:
        opt_pct = float("nan")

    return {
        "status": "ok",
        "summary_path": str(summary_json),
        "origin_status": str(origin_status),
        "origin_total_time_s": float(origin_total_time_s),
        "window_total_time_s_by_ws": dict(window_total_time_s_by_ws),
        "trapezoid_solutions_totg_total_time_s_by_ws": dict(trap_totg_total_by_ws),
        "trapezoid_solutions_true_plan_total_time_s_by_ws": dict(trap_true_plan_total_by_ws),
        "ws": {
            str(int(ws_ref)): ws_ref_data,
            str(int(ws_cmp)): ws_cmp_data,
        },
        "optimization_pct_cmp_vs_ref": float(opt_pct),
        "optimization_definition": f"(T_ws{int(ws_ref)}-T_ws{int(ws_cmp)})/T_ws{int(ws_ref)}*100%",
    }


def _fmt_num(v: float, prec: int = 6) -> str:
    if _is_finite(float(v)):
        return f"{float(v):.{prec}f}"
    return "nan"


def _fmt_vec(xs: Sequence[float], *, prec: int = 3) -> str:
    vals: List[str] = []
    for x in xs:
        vals.append(_fmt_num(_safe_float(x), prec=prec))
    return "[" + ", ".join(vals) + "]"


def _fmt_csv_num(v: float) -> str:
    if _is_finite(float(v)):
        return f"{float(v):.12g}"
    return "nan"


def _write_time_csv(
    *,
    path: Path,
    seeds: Sequence[int],
    window_sizes: Sequence[int],
    results: Dict[int, Dict[str, Dict]],
) -> None:
    ws_targets = [int(ws) for ws in window_sizes if int(ws) != 1]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "seed",
                "ws",
                "origin_time_s",
                "greedy_totg_time_s",
                "trapezoid_time_s",
                "trapezoid_sol_totg_time_s",
                "totg_time_s",
                "trapezoid_sol_true_plan_time_s",
            ]
        )

        for seed in seeds:
            by_model = results.get(int(seed), {})
            trap_item = by_model.get("trapezoid", {})
            if not isinstance(trap_item, dict):
                trap_item = {}
            totg_item = by_model.get("totg", {})
            if not isinstance(totg_item, dict):
                totg_item = {}

            trap_ok = str(trap_item.get("status", "")) == "ok"
            totg_ok = str(totg_item.get("status", "")) == "ok"

            origin_time_s = float("nan")
            greedy_totg_time_s = float("nan")
            window_totals: Dict[str, float] = {}
            trap_totg_totals: Dict[str, float] = {}
            trap_true_plan_totals: Dict[str, float] = {}
            ws_payload: Dict[str, Dict] = {}
            if trap_ok:
                origin_time_s = _safe_float(trap_item.get("origin_total_time_s", math.nan))
                window_totals_raw = trap_item.get("window_total_time_s_by_ws", {})
                if isinstance(window_totals_raw, dict):
                    window_totals = dict(window_totals_raw)
                trap_totg_totals_raw = trap_item.get("trapezoid_solutions_totg_total_time_s_by_ws", {})
                if isinstance(trap_totg_totals_raw, dict):
                    trap_totg_totals = dict(trap_totg_totals_raw)
                trap_true_plan_totals_raw = trap_item.get("trapezoid_solutions_true_plan_total_time_s_by_ws", {})
                if isinstance(trap_true_plan_totals_raw, dict):
                    trap_true_plan_totals = dict(trap_true_plan_totals_raw)
                ws_payload_raw = trap_item.get("ws", {})
                if isinstance(ws_payload_raw, dict):
                    ws_payload = dict(ws_payload_raw)

            totg_window_totals: Dict[str, float] = {}
            totg_ws_payload: Dict[str, Dict] = {}
            if totg_ok:
                totg_window_totals_raw = totg_item.get("window_total_time_s_by_ws", {})
                if isinstance(totg_window_totals_raw, dict):
                    totg_window_totals = dict(totg_window_totals_raw)
                totg_ws_payload_raw = totg_item.get("ws", {})
                if isinstance(totg_ws_payload_raw, dict):
                    totg_ws_payload = dict(totg_ws_payload_raw)

                greedy_totg_time_s = _safe_float(totg_window_totals.get("1", math.nan))
                if not _is_finite(greedy_totg_time_s):
                    ws1 = totg_ws_payload.get("1", {})
                    if isinstance(ws1, dict):
                        greedy_totg_time_s = _safe_float(ws1.get("total_time_s", math.nan))

            for ws in ws_targets:
                key = str(int(ws))

                trapezoid_time_s = float("nan")
                if trap_ok:
                    trapezoid_time_s = _safe_float(window_totals.get(key, math.nan))
                    if not _is_finite(trapezoid_time_s):
                        ws_item = ws_payload.get(key, {})
                        if isinstance(ws_item, dict):
                            trapezoid_time_s = _safe_float(ws_item.get("total_time_s", math.nan))

                trap_sol_totg_time_s = float("nan")
                if trap_ok:
                    trap_sol_totg_time_s = _safe_float(trap_totg_totals.get(key, math.nan))

                trap_sol_true_plan_time_s = float("nan")
                if trap_ok:
                    trap_sol_true_plan_time_s = _safe_float(trap_true_plan_totals.get(key, math.nan))

                totg_time_s = float("nan")
                if totg_ok:
                    totg_time_s = _safe_float(totg_window_totals.get(key, math.nan))
                    if not _is_finite(totg_time_s):
                        totg_ws_item = totg_ws_payload.get(key, {})
                        if isinstance(totg_ws_item, dict):
                            totg_time_s = _safe_float(totg_ws_item.get("total_time_s", math.nan))

                writer.writerow(
                    [
                        int(seed),
                        int(ws),
                        _fmt_csv_num(origin_time_s),
                        _fmt_csv_num(greedy_totg_time_s),
                        _fmt_csv_num(trapezoid_time_s),
                        _fmt_csv_num(trap_sol_totg_time_s),
                        _fmt_csv_num(totg_time_s),
                        _fmt_csv_num(trap_sol_true_plan_time_s),
                    ]
                )


def _write_summary_txt(
    *,
    path: Path,
    num_points: int,
    ws_ref: int,
    ws_cmp: int,
    seeds: Sequence[int],
    window_size_spec: str,
    path_pattern: str,
    candidate_mode: str,
    results: Dict[int, Dict[str, Dict]],
) -> None:
    lines: List[str] = []
    lines.append("panda_ik_window time-model comparison summary")
    lines.append(f"generated_at: {_now_str()}")
    lines.append(f"num_points: {int(num_points)}")
    lines.append(f"window_size: {str(window_size_spec)}")
    lines.append(f"path_pattern: {str(path_pattern)}")
    lines.append(f"candidate_mode: {str(candidate_mode)}")
    lines.append(f"ws_ref: {int(ws_ref)}")
    lines.append(f"ws_cmp: {int(ws_cmp)}")
    lines.append(f"optimization: (T_ws{int(ws_ref)}-T_ws{int(ws_cmp)})/T_ws{int(ws_ref)}*100%")
    lines.append("")

    for seed in seeds:
        lines.append("=" * 80)
        lines.append(f"seed: {int(seed)}")
        by_model = results.get(int(seed), {})
        for model in ("totg", "trapezoid"):
            item = by_model.get(model, {})
            status = str(item.get("status", "missing"))
            lines.append(f"[model={model}] status={status}")
            if status != "ok":
                lines.append(f"  reason: {item.get('reason', 'unknown')}")
                continue
            lines.append(f"  candidate_source_mode: {item.get('candidate_source_mode', 'unknown')}")
            if str(item.get("reuse_candidates_dir", "")).strip():
                lines.append(f"  reuse_candidates_dir: {item.get('reuse_candidates_dir')}")
            if str(item.get("run_dir", "")).strip():
                lines.append(f"  run_dir: {item.get('run_dir')}")

            ws_ref_data = item["ws"][str(int(ws_ref))]
            ws_cmp_data = item["ws"][str(int(ws_cmp))]
            lines.append(f"  summary_json: {item.get('summary_path')}")
            lines.append(
                f"  total_time_s: ws{int(ws_ref)}={_fmt_num(float(ws_ref_data['total_time_s']))}, "
                f"ws{int(ws_cmp)}={_fmt_num(float(ws_cmp_data['total_time_s']))}"
            )
            lines.append(
                f"  optimization_pct(ws{int(ws_cmp)} vs ws{int(ws_ref)}): "
                f"{_fmt_num(float(item.get('optimization_pct_cmp_vs_ref', math.nan)), prec=3)}%"
            )

            for ws in (int(ws_ref), int(ws_cmp)):
                ws_data = item["ws"][str(ws)]
                lines.append(
                    f"  selection(ws={ws}) total_s={_fmt_num(float(ws_data['selection_total_s']))}, "
                    f"selected_solution_ids={ws_data.get('selected_solution_ids', [])}"
                )
                details = ws_data.get("selected_solution_details", [])
                if isinstance(details, list) and details:
                    lines.append("  selected_solution_joint_angles:")
                    for d in details:
                        if not isinstance(d, dict):
                            continue
                        p = str(d.get("point", ""))
                        sid = str(d.get("selected_solution_id", ""))
                        q_rad = d.get("joint_positions_rad", [])
                        q_deg = d.get("joint_positions_deg", [])
                        if not isinstance(q_rad, list):
                            q_rad = []
                        if not isinstance(q_deg, list):
                            q_deg = []
                        lines.append(
                            "    "
                            f"{p}: {sid}, "
                            f"q_rad={_fmt_vec(q_rad, prec=6)}, "
                            f"q_deg={_fmt_vec(q_deg, prec=3)}"
                        )
                recs = ws_data.get("selection_by_point", [])
                if not isinstance(recs, list) or len(recs) == 0:
                    lines.append("    (no per-point selection timing records)")
                    continue
                for rec in recs:
                    q_deg = rec.get("joint_positions_deg", [])
                    if not isinstance(q_deg, list):
                        q_deg = []
                    lines.append(
                        "    "
                        f"{str(rec.get('point', ''))}: "
                        f"{str(rec.get('selected_solution_id', ''))}, "
                        f"elapsed={_fmt_num(_safe_float(rec.get('selection_elapsed_s', math.nan)))}s, "
                        f"q_deg={_fmt_vec(q_deg, prec=3)}"
                    )

        # Optional side-by-side aid
        totg_item = by_model.get("totg", {})
        trap_item = by_model.get("trapezoid", {})
        if str(totg_item.get("status")) == "ok" and str(trap_item.get("status")) == "ok":
            t_totg_ref = _safe_float(totg_item["ws"][str(int(ws_ref))]["total_time_s"])
            t_trap_ref = _safe_float(trap_item["ws"][str(int(ws_ref))]["total_time_s"])
            t_totg_cmp = _safe_float(totg_item["ws"][str(int(ws_cmp))]["total_time_s"])
            t_trap_cmp = _safe_float(trap_item["ws"][str(int(ws_cmp))]["total_time_s"])
            lines.append(
                f"[cross-model] ws{int(ws_ref)}: totg-trapezoid={_fmt_num(t_totg_ref - t_trap_ref)}s, "
                f"ws{int(ws_cmp)}: totg-trapezoid={_fmt_num(t_totg_cmp - t_trap_cmp)}s"
            )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Compare time_model=totg vs trapezoid over seed sweep. "
            "Writes one summary.txt with per-seed ws metrics and selection data."
        )
    )
    ap.add_argument("--pkg", default="panda_ik_window", help="ROS2 package name.")
    ap.add_argument("--launch", default="ik_benchmark.launch.py", help="Launch file name.")
    ap.add_argument("--num-points", type=int, default=3, help="num_points (default: 3).")
    ap.add_argument("--window-size", "--window_size", type=str, default="1,3", help="Window size spec.")
    ap.add_argument("--seed-start", type=int, default=10, help="Seed start (inclusive).")
    ap.add_argument("--seed-end", type=int, default=109, help="Seed end (inclusive).")
    ap.add_argument(
        "--path-pattern",
        "--path_pattern",
        type=str,
        default="random",
        choices=["trend", "switching", "random"],
        help="Launch arg path_pattern (default: random).",
    )
    ap.add_argument(
        "--candidate-mode",
        type=str,
        default="shared",
        choices=["shared", "independent"],
        help=(
            "How to compare time models per seed: "
            "shared=trapezoid reuses candidates generated in totg run (default), "
            "independent=each model samples its own candidates."
        ),
    )
    ap.add_argument(
        "--base-data-root",
        type=str,
        default="data_window/batch_time_model_data",
        help="Base dir for raw run outputs.",
    )
    ap.add_argument(
        "--summary-dir",
        type=str,
        default="data_window/batch_time_model",
        help="Directory for final summary.txt.",
    )
    ap.add_argument(
        "--log-dir",
        type=str,
        default="data_window/batch_time_model_logs",
        help="Directory for per-run ros2 logs.",
    )
    ap.add_argument("--no-logs", action="store_true", help="Print ros2 output to console.")
    ap.add_argument("--wait-timeout", type=float, default=180.0, help="Wait timeout for summary.json (s).")
    ap.add_argument(
        "--extra",
        type=str,
        default="",
        help="Extra launch args appended verbatim. Protected keys are filtered out.",
    )

    args = ap.parse_args()

    num_points = int(args.num_points)
    window_sizes = _parse_window_sizes_arg(str(args.window_size), num_points=num_points)
    if 1 not in window_sizes or 3 not in window_sizes:
        raise ValueError("--window-size must include both 1 and 3 for this comparison script.")
    ws_ref = 1
    ws_cmp = 3
    window_size_launch = ",".join(str(ws) for ws in window_sizes)

    s0 = int(args.seed_start)
    s1 = int(args.seed_end)
    if s1 < s0:
        raise ValueError("--seed-end must be >= --seed-start")
    seeds = list(range(s0, s1 + 1))

    path_pattern = str(args.path_pattern)
    models = ("totg", "trapezoid")

    base_data_root = Path(args.base_data_root)
    summary_dir = Path(args.summary_dir)
    log_dir = Path(args.log_dir)
    raw_extra_args = [a for a in str(args.extra).split() if a.strip()]
    extra_args, dropped = _filter_extra_args(
        raw_extra_args,
        blocked_keys=(
            "num_points",
            "seed",
            "window_size",
            "path_pattern",
            "time_model",
            "data_root",
            "reuse_candidates_dir",
        ),
    )

    print("[batch] settings")
    print(f"  num_points      = {num_points}")
    print(f"  window_size     = {window_size_launch} (from {args.window_size!r})")
    print(f"  seeds           = [{s0}..{s1}] (count={len(seeds)})")
    print(f"  models          = {models}")
    print(f"  path_pattern    = {path_pattern}")
    print(f"  candidate_mode  = {args.candidate_mode}")
    print(f"  base_data_root  = {base_data_root}")
    print(f"  summary_dir     = {summary_dir}")
    print(f"  log_dir         = {log_dir} (enabled={not args.no_logs})")
    if extra_args:
        print(f"  extra args      = {extra_args}")
    if dropped:
        print(f"  filtered extra  = {dropped}")

    results: Dict[int, Dict[str, Dict]] = {}
    total_rounds = len(seeds) * len(models)
    round_idx = 0

    for seed in seeds:
        results[int(seed)] = {}
        shared_candidate_dir: Optional[Path] = None
        for model_idx, model in enumerate(models):
            round_idx += 1
            print(
                f"\n[batch] round {round_idx}/{total_rounds} | seed={seed} model={model} | START @ {_now_str()}"
            )

            reuse_for_run: Optional[Path] = None
            if str(args.candidate_mode) == "shared" and int(model_idx) > 0:
                if shared_candidate_dir is None:
                    results[int(seed)][str(model)] = {
                        "status": "skipped",
                        "reason": "shared candidate dir not available from first model run",
                    }
                    print(
                        f"[batch][ERROR] skip seed={seed}, model={model}: "
                        "shared candidate dir not available"
                    )
                    continue
                reuse_for_run = shared_candidate_dir

            run_data_root = base_data_root / f"np{num_points}" / f"seed{seed}" / f"model_{model}"
            prev_latest = _find_latest_run_dir(run_data_root)
            prev_latest_name = prev_latest.name if prev_latest is not None else None

            log_file = None
            if not args.no_logs:
                log_file = log_dir / f"np{num_points}_seed{seed}_{model}.log"

            rc = _run_ros2_launch(
                pkg=str(args.pkg),
                launch_file=str(args.launch),
                num_points=num_points,
                seed=seed,
                window_size_spec=window_size_launch,
                path_pattern=path_pattern,
                time_model=model,
                data_root=run_data_root,
                reuse_candidates_dir=reuse_for_run,
                log_file=log_file,
                extra_launch_args=extra_args,
            )
            if rc != 0:
                results[int(seed)][str(model)] = {
                    "status": "launch_failed",
                    "reason": f"ros2 launch return code={rc}",
                }
                print(f"[batch][ERROR] launch failed: seed={seed}, model={model}, rc={rc}")
                continue

            try:
                summary_path = _wait_for_new_summary_json(
                    run_data_root,
                    prev_latest_dirname=prev_latest_name,
                    timeout_s=float(args.wait_timeout),
                )
                metrics = _extract_model_metrics(summary_path, ws_ref=ws_ref, ws_cmp=ws_cmp)
                metrics["run_dir"] = str(summary_path.parent)
                metrics["candidate_source_mode"] = "reused" if reuse_for_run is not None else "sampled"
                metrics["reuse_candidates_dir"] = str(reuse_for_run) if reuse_for_run is not None else ""
                results[int(seed)][str(model)] = metrics
                print(f"[batch] summary: {summary_path}")
                print(
                    f"[batch] seed={seed} model={model} "
                    f"T(ws{ws_ref})={_fmt_num(metrics['ws'][str(ws_ref)]['total_time_s'])} "
                    f"T(ws{ws_cmp})={_fmt_num(metrics['ws'][str(ws_cmp)]['total_time_s'])} "
                    f"opt={_fmt_num(metrics['optimization_pct_cmp_vs_ref'], prec=3)}%"
                )
                if int(model_idx) == 0 and str(args.candidate_mode) == "shared":
                    shared_candidate_dir = summary_path.parent
            except Exception as e:
                results[int(seed)][str(model)] = {
                    "status": "parse_failed",
                    "reason": str(e),
                }
                print(f"[batch][ERROR] parse failed: seed={seed}, model={model}: {e}")

    _ensure_dir(summary_dir)
    summary_txt = summary_dir / f"np{num_points}_ws{ws_ref}_{ws_cmp}_seed{s0}_{s1}_summary.txt"
    _write_summary_txt(
        path=summary_txt,
        num_points=num_points,
        ws_ref=ws_ref,
        ws_cmp=ws_cmp,
        seeds=seeds,
        window_size_spec=window_size_launch,
        path_pattern=path_pattern,
        candidate_mode=str(args.candidate_mode),
        results=results,
    )
    time_csv = summary_dir / "time.csv"
    _write_time_csv(
        path=time_csv,
        seeds=seeds,
        window_sizes=window_sizes,
        results=results,
    )
    print(f"\n[batch] summary written: {summary_txt}")
    print(f"[batch] time csv written: {time_csv}")
    print("[batch] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
