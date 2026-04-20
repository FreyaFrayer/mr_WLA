#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot aggregated CSV files produced by script/batch_run_window.py.

Default output:
  - input CSVs: data_window/batch_csv/*.csv
  - output PNGs: data_window/batch_plots/*_overview.png
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _maybe_float(value: object) -> float:
    if value is None:
        return math.nan
    text = str(value).strip()
    if text == "":
        return math.nan
    try:
        return float(text)
    except Exception:
        return math.nan


def _finite_pairs(xs: Sequence[float], ys: Sequence[float]) -> tuple[List[float], List[float]]:
    out_x: List[float] = []
    out_y: List[float] = []
    for x, y in zip(xs, ys):
        try:
            xf = float(x)
            yf = float(y)
        except Exception:
            continue
        if math.isfinite(xf) and math.isfinite(yf):
            out_x.append(xf)
            out_y.append(yf)
    return out_x, out_y


def _finite_values(values: Sequence[float]) -> List[float]:
    out: List[float] = []
    for value in values:
        try:
            num = float(value)
        except Exception:
            continue
        if math.isfinite(num):
            out.append(num)
    return out


def _nanmean(values: Sequence[float]) -> float:
    finite = _finite_values(values)
    if not finite:
        return math.nan
    return float(sum(finite) / len(finite))


def _parse_window_sizes(fieldnames: Sequence[str]) -> List[int]:
    window_sizes: List[int] = []
    for name in fieldnames:
        match = re.fullmatch(r"ws(\d+)_time", name)
        if match is None:
            continue
        ws = int(match.group(1))
        if ws not in window_sizes:
            window_sizes.append(ws)
    return window_sizes


def _read_batch_summary_csv(csv_path: Path) -> Dict[str, object]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        window_sizes = _parse_window_sizes(fieldnames)

        rows: List[Dict[str, float]] = []
        for idx, row in enumerate(reader):
            record: Dict[str, float] = {
                "seed": _maybe_float(row.get("seed")),
                "origin_time": _maybe_float(row.get("origin_time")),
                "solve_total_time_s": _maybe_float(row.get("solve_total_time_s")),
                "_order": float(idx),
            }
            for ws in window_sizes:
                record[f"ws{ws}_time"] = _maybe_float(row.get(f"ws{ws}_time"))
                record[f"ws{ws}_planner_time"] = _maybe_float(row.get(f"ws{ws}_planner_time"))
                record[f"ws{ws}_opt_rate_vs_ws1"] = _maybe_float(row.get(f"ws{ws}_opt_rate_vs_ws1"))
                record[f"ws{ws}_planner_opt_rate_vs_origin"] = _maybe_float(
                    row.get(f"ws{ws}_planner_opt_rate_vs_origin")
                )
            rows.append(record)

    rows.sort(
        key=lambda row: (
            0 if math.isfinite(float(row["seed"])) else 1,
            float(row["seed"]) if math.isfinite(float(row["seed"])) else float(row["_order"]),
        )
    )

    seeds = [
        float(row["seed"]) if math.isfinite(float(row["seed"])) else float(i + 1)
        for i, row in enumerate(rows)
    ]
    return {
        "csv_path": csv_path,
        "rows": rows,
        "seeds": seeds,
        "window_sizes": window_sizes,
    }


def _plot_line_series(ax: plt.Axes, seeds: Sequence[float], values: Sequence[float], label: str, color: object) -> None:
    xs, ys = _finite_pairs(seeds, values)
    if not xs:
        return
    ax.plot(xs, ys, marker="o", linewidth=1.8, markersize=4.5, label=label, color=color)


def plot_batch_summary_csv(csv_path: Path | str, out_dir: Path | str = "data_window/batch_plots", dpi: int = 180) -> Path:
    csv_path = Path(csv_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    parsed = _read_batch_summary_csv(csv_path)
    rows = parsed["rows"]
    seeds = parsed["seeds"]
    window_sizes = parsed["window_sizes"]

    if not rows:
        raise ValueError(f"CSV has no data rows: {csv_path}")

    total_colors = plt.cm.tab10(range(max(len(window_sizes), 1)))
    planner_colors = plt.cm.Set2(range(max(len(window_sizes), 1)))

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    ax_total, ax_planner, ax_box, ax_gain = axes.flatten()

    origin_values = [float(row["origin_time"]) for row in rows]
    _plot_line_series(ax_total, seeds, origin_values, "origin", "#222222")
    for idx, ws in enumerate(window_sizes):
        values = [float(row[f"ws{ws}_time"]) for row in rows]
        _plot_line_series(ax_total, seeds, values, f"ws={ws}", total_colors[idx])
    ax_total.set_title("Total Time by Seed")
    ax_total.set_xlabel("seed")
    ax_total.set_ylabel("time (s)")
    ax_total.grid(True, alpha=0.25)
    if ax_total.lines:
        ax_total.legend(loc="best", fontsize=9)

    _plot_line_series(ax_planner, seeds, origin_values, "origin", "#222222")
    for idx, ws in enumerate(window_sizes):
        values = [float(row[f"ws{ws}_planner_time"]) for row in rows]
        _plot_line_series(ax_planner, seeds, values, f"ws={ws} planner", planner_colors[idx])
    ax_planner.set_title("Planner Replay Time by Seed")
    ax_planner.set_xlabel("seed")
    ax_planner.set_ylabel("time (s)")
    ax_planner.grid(True, alpha=0.25)
    if ax_planner.lines:
        ax_planner.legend(loc="best", fontsize=9)

    box_data: List[List[float]] = []
    box_labels: List[str] = []
    box_colors: List[str] = []
    origin_box = _finite_values(origin_values)
    if origin_box:
        box_data.append(origin_box)
        box_labels.append("origin")
        box_colors.append("#d9d9d9")
    for idx, ws in enumerate(window_sizes):
        total_vals = _finite_values([float(row[f"ws{ws}_time"]) for row in rows])
        if total_vals:
            box_data.append(total_vals)
            box_labels.append(f"ws{ws} total")
            color = total_colors[idx]
            box_colors.append(matplotlib.colors.to_hex(color))
    for idx, ws in enumerate(window_sizes):
        planner_vals = _finite_values([float(row[f"ws{ws}_planner_time"]) for row in rows])
        if planner_vals:
            box_data.append(planner_vals)
            box_labels.append(f"ws{ws} planner")
            color = planner_colors[idx]
            box_colors.append(matplotlib.colors.to_hex(color))
    if box_data:
        artists = ax_box.boxplot(box_data, patch_artist=True, tick_labels=box_labels)
        for patch, color in zip(artists["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        ax_box.set_title("Distribution Overview")
        ax_box.set_ylabel("time (s)")
        ax_box.tick_params(axis="x", rotation=20)
        ax_box.grid(True, axis="y", alpha=0.25)
    else:
        ax_box.text(0.5, 0.5, "No finite timing data", ha="center", va="center", transform=ax_box.transAxes)
        ax_box.set_axis_off()

    gain_ws = [ws for ws in window_sizes if ws != 1]
    time_gain_means = [_nanmean([float(row[f"ws{ws}_opt_rate_vs_ws1"]) for row in rows]) * 100.0 for ws in gain_ws]
    planner_gain_means = [
        _nanmean([float(row[f"ws{ws}_planner_opt_rate_vs_origin"]) for row in rows]) * 100.0 for ws in gain_ws
    ]
    if gain_ws:
        xs = list(range(len(gain_ws)))
        width = 0.36
        bars1 = ax_gain.bar(
            [x - width / 2 for x in xs],
            time_gain_means,
            width=width,
            label="time vs ws1",
            color="#4c78a8",
        )
        bars2 = ax_gain.bar(
            [x + width / 2 for x in xs],
            planner_gain_means,
            width=width,
            label="planner vs origin",
            color="#72b7b2",
        )
        ax_gain.set_xticks(xs, [f"ws={ws}" for ws in gain_ws])
        ax_gain.axhline(0.0, color="#444444", linewidth=1.0)
        ax_gain.set_title("Mean Improvement Rate")
        ax_gain.set_ylabel("improvement (%)")
        ax_gain.grid(True, axis="y", alpha=0.25)
        ax_gain.legend(loc="best", fontsize=9)
        for bars in (bars1, bars2):
            for bar in bars:
                height = bar.get_height()
                if not math.isfinite(float(height)):
                    continue
                va = "bottom" if height >= 0 else "top"
                offset = 1.0 if height >= 0 else -1.0
                ax_gain.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + offset,
                    f"{height:.1f}%",
                    ha="center",
                    va=va,
                    fontsize=8,
                )
    else:
        ax_gain.text(0.5, 0.5, "No ws>1 improvement columns", ha="center", va="center", transform=ax_gain.transAxes)
        ax_gain.set_axis_off()

    ws_text = ", ".join(str(ws) for ws in window_sizes) if window_sizes else "none"
    fig.suptitle(
        f"{csv_path.stem}\nrows={len(rows)}, window_sizes=[{ws_text}]",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))

    out_path = out_dir / f"{csv_path.stem}_overview.png"
    fig.savefig(out_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    return out_path


def _collect_csv_paths(inputs: Sequence[str], recursive: bool) -> List[Path]:
    csv_paths: List[Path] = []
    for raw in inputs:
        path = Path(raw)
        if path.is_file() and path.suffix.lower() == ".csv":
            csv_paths.append(path)
            continue
        if path.is_dir():
            pattern = "**/*.csv" if recursive else "*.csv"
            csv_paths.extend(sorted(path.glob(pattern)))
            continue
        raise FileNotFoundError(f"Input does not exist or is not a CSV/dir: {path}")

    unique_paths: List[Path] = []
    seen: set[Path] = set()
    for path in csv_paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_paths.append(path)
    return unique_paths


def main() -> int:
    ap = argparse.ArgumentParser(description="Plot batch_run_window summary CSV files into PNG charts.")
    ap.add_argument(
        "inputs",
        nargs="*",
        default=["data_window/batch_csv"],
        help="CSV file(s) or directory(s). Default: data_window/batch_csv",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default="data_window/batch_plots",
        help="Output directory for PNG files (default: data_window/batch_plots).",
    )
    ap.add_argument("--dpi", type=int, default=180, help="PNG dpi (default: 180).")
    ap.add_argument("--recursive", action="store_true", help="Recursively search CSV files inside input directories.")
    args = ap.parse_args()

    csv_paths = _collect_csv_paths(args.inputs, recursive=bool(args.recursive))
    if not csv_paths:
        raise FileNotFoundError("No CSV files found to plot.")

    for csv_path in csv_paths:
        out_path = plot_batch_summary_csv(csv_path, out_dir=args.out_dir, dpi=int(args.dpi))
        print(f"[plot] wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
