from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from ..planning.search import PathResult
from ..types import TargetPoint


def _fmt_list(xs: Sequence[float], *, prec: int = 4) -> str:
    return "[" + ", ".join(f"{float(v):.{prec}f}" for v in xs) + "]"


def write_targets_json(path: Path, *, timestamp: str, group: str, tip_link: str,
                       start_label: str, start_joint_positions: Sequence[float],
                       targets: Sequence[TargetPoint]) -> None:
    import json
    payload = {
        "meta": {
            "timestamp": str(timestamp),
            "group": str(group),
            "tip_link": str(tip_link),
            "start_label": str(start_label),
        },
        "p0": {
            "joint_positions": [float(v) for v in start_joint_positions],
        },
        "targets": [
            {"name": f"p{i}", "x": float(p.x), "y": float(p.y), "z": float(p.z)}
            for i, p in enumerate(targets, start=1)
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, sort_keys=True)


def write_robot_info_txt(path: Path, *, group: str, tip_link: str,
                         joint_names: Sequence[str],
                         joint_limits: Sequence[tuple]) -> None:
    lines = []
    lines.append(f"group: {group}")
    lines.append(f"tip_link: {tip_link}")
    lines.append(f"dof: {len(joint_names)}")
    lines.append("")
    lines.append("Joint limits (position [rad], velocity [rad/s], acceleration [rad/s^2]):")
    for jn, lim in zip(joint_names, joint_limits):
        # Backward compatible: allow 3-tuple (lo, hi, vmax) or 4-tuple (lo, hi, vmax, amax)
        lo = float(lim[0])
        hi = float(lim[1])
        vmax = float(lim[2]) if len(lim) >= 3 else 0.0
        amax = float(lim[3]) if len(lim) >= 4 else float("nan")
        if math.isfinite(amax):
            lines.append(f"  - {jn}: [{lo:+.4f}, {hi:+.4f}], vmax={vmax:.4f}, amax={amax:.4f}")
        else:
            lines.append(f"  - {jn}: [{lo:+.4f}, {hi:+.4f}], vmax={vmax:.4f}, amax=unknown")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_human_readable_lines(
    *,
    title: str,
    targets: Sequence[TargetPoint],
    start_label: str,
    start_q: Sequence[float],
    result: PathResult,
    ik_metas: Optional[Sequence[Dict[str, Any]]] = None,
) -> list[str]:
    """Create a compact, stable, plain-text summary embedded in JSON.

    Users often want both: machine-readable fields + a quick human-readable view.
    """
    lines: list[str] = []
    lines.append(title)
    lines.append("")
    lines.append(f"method: {result.method}")
    lines.append(f"total_paths_theoretical: {result.total_paths_theoretical}")
    lines.append("")
    lines.append("Time model:")
    lines.append(f"  requested: {result.time_model.requested}")
    lines.append(f"  effective: {result.time_model.effective}")
    lines.append(f"  totg_available: {bool(result.time_model.totg_available)}")
    if str(result.time_model.note).strip():
        lines.append(f"  note: {str(result.time_model.note).strip()}")
    if int(result.time_model.totg_failures) > 0:
        lines.append(
            f"  totg_failures: {int(result.time_model.totg_failures)} (these edges used trapezoid fallback)"
        )
    lines.append("")
    lines.append(f"p0 ({start_label}) joint_positions = {_fmt_list(start_q, prec=4)}")
    lines.append("")

    if ik_metas is not None and len(ik_metas) > 0:
        lines.append("IK sampling overview:")
        for i, meta in enumerate(ik_metas, start=1):
            found = int(meta.get("found", 0))
            req = int(meta.get("requested", meta.get("requested_per_point", 200)))
            trials = meta.get("resample_trials_used", meta.get("resample_trial_final", "?"))
            passes_used = meta.get("passes_used", meta.get("passes", "?"))
            reason = str(meta.get("reason", "ok"))
            shortfall = int(meta.get("shortfall", max(0, req - found)))
            note = ""
            if found == 0:
                note = "  NOTE: no IK solution found (should not happen after filtering)."
            elif found < req:
                note = f"  NOTE: shortfall={shortfall}, reason={reason}"
            lines.append(
                f"  p{i}: found {found}/{req}  (resample_trials_used={trials}, passes_used={passes_used}){note}"
            )
        lines.append("")

    lines.append("Targets (Cartesian positions, meters):")
    for i, p in enumerate(targets, start=1):
        lines.append(f"  p{i}: ({p.x:.4f}, {p.y:.4f}, {p.z:.4f})")
    lines.append("")

    for seg in result.segments:
        i = seg.seg_idx_1based
        found = seg.found
        req = seg.requested
        sol = seg.best_solution
        lines.append(f"{seg.from_label}->{seg.to_label} success: {found}/{req}")
        lines.append(
            f"t{i} (min {seg.from_label}->{seg.to_label}): {seg.best_time_s:.4f}s "
            f"at p{i}_{sol.index + 1} (attempt={sol.attempt}) joint_positions = {_fmt_list(sol.joint_positions, prec=4)}"
        )
        lines.append("")

    lines.append(f"Total time (t1+t2+...+tn) = {result.total_time_s:.6f} s")
    return lines


def write_summary_json(
    path: Path,
    *,
    title: str,
    targets: Sequence[TargetPoint],
    start_label: str,
    start_q: Sequence[float],
    result: PathResult,
    ik_metas: Optional[Sequence[Dict[str, Any]]] = None,
    timestamp: Optional[str] = None,
) -> None:
    """Write summary in JSON format.

    The JSON contains:
    - structured fields for easy parsing
    - a `human_readable.lines` section similar to the old text report
    """
    import json

    # IK overview
    ik_overview = []
    if ik_metas is not None:
        for i, meta in enumerate(list(ik_metas), start=1):
            found = int(meta.get("found", 0))
            req = int(meta.get("requested", meta.get("requested_per_point", 200)))
            trials = meta.get("resample_trials_used", meta.get("resample_trial_final", None))
            passes_used = meta.get("passes_used", meta.get("passes", None))
            reason = str(meta.get("reason", "ok"))
            shortfall = int(meta.get("shortfall", max(0, req - found)))
            accepted_with_shortfall = bool(meta.get("accepted_with_shortfall", found < req))

            note = ""
            if found == 0:
                note = "no IK solution found (should not happen after filtering)"
            elif found < req:
                note = f"shortfall={shortfall}, reason={reason}"

            ik_overview.append(
                {
                    "point": f"p{i}",
                    "found": found,
                    "requested": req,
                    "accepted_with_shortfall": accepted_with_shortfall,
                    "shortfall": shortfall,
                    "reason": reason,
                    "resample_trials_used": trials,
                    "passes_used": passes_used,
                    # keep a compact note for quick reading
                    "note": note,
                }
            )

    # Targets
    targets_list = [
        {"name": f"p{i}", "x": float(p.x), "y": float(p.y), "z": float(p.z)}
        for i, p in enumerate(targets, start=1)
    ]

    # Segments
    segments = []
    for seg in result.segments:
        i = int(seg.seg_idx_1based)
        sol = seg.best_solution
        sol_id = f"p{i}_{int(sol.index) + 1}"
        segments.append(
            {
                "segment_index": i,
                "from": str(seg.from_label),
                "to": str(seg.to_label),
                "success": {"found": int(seg.found), "requested": int(seg.requested)},
                "best": {
                    "time_s": float(seg.best_time_s),
                    "solution_id": sol_id,
                    "solution_index_0based": int(sol.index),
                    "solution_index_1based": int(sol.index) + 1,
                    "attempt": int(sol.attempt),
                    "joint_positions": [float(v) for v in sol.joint_positions],
                },
                # keep an old-style compact string for readability
                "string": (
                    f"t{i} (min {seg.from_label}->{seg.to_label}): {float(seg.best_time_s):.4f}s "
                    f"at {sol_id} (attempt={int(sol.attempt)})"
                ),
            }
        )

    payload: Dict[str, Any] = {
        "format": "panda_ik_window_summary",
        "format_version": 1,
        "title": str(title),
        "meta": {
            "timestamp": str(timestamp) if timestamp is not None else None,
            "method": str(result.method),
            "total_paths_theoretical": int(result.total_paths_theoretical),
            "time_model": {
                "requested": str(result.time_model.requested),
                "effective": str(result.time_model.effective),
                "totg_available": bool(result.time_model.totg_available),
                "totg_failures": int(result.time_model.totg_failures),
                "note": str(result.time_model.note),
            },
        },
        "units": {
            "cartesian_position": "m",
            "joint_position": "rad",
            "time": "s",
        },
        "start": {
            "name": "p0",
            "label": str(start_label),
            "joint_positions": [float(v) for v in start_q],
        },
        "ik_sampling_overview": ik_overview,
        "targets": targets_list,
        "segments": segments,
        "total_time_s": float(result.total_time_s),
        "total_time_string": f"Total time (t1+t2+...+tn) = {float(result.total_time_s):.6f} s",
        "human_readable": {
            "lines": _build_human_readable_lines(
                title=title,
                targets=targets,
                start_label=start_label,
                start_q=start_q,
                result=result,
                ik_metas=ik_metas,
            )
        },
    }

    # Drop null timestamp for cleaner output
    if payload["meta"].get("timestamp") is None:
        payload["meta"].pop("timestamp", None)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, sort_keys=True)


# Backward-compatible alias (old API): still writes a TXT file.
def write_summary_txt(
    path: Path,
    *,
    title: str,
    targets: Sequence[TargetPoint],
    start_label: str,
    start_q: Sequence[float],
    result: PathResult,
    ik_metas: Optional[Sequence[Dict[str, Any]]] = None,
) -> None:  # pragma: no cover
    """Deprecated: text summary writer.

    Kept for compatibility. New runs should use :func:`write_summary_json`.
    """
    lines = _build_human_readable_lines(
        title=title,
        targets=targets,
        start_label=start_label,
        start_q=start_q,
        result=result,
        ik_metas=ik_metas,
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
