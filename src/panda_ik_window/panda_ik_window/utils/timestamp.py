from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


def make_timestamp(compact: bool = True) -> str:
    """
    Create a filesystem-friendly timestamp.

    Examples
    --------
    - compact=True  -> 20260113_153012
    - compact=False -> 2026-01-13_15-30-12
    """
    now = datetime.now()
    if compact:
        return now.strftime("%Y%m%d_%H%M%S")
    return now.strftime("%Y-%m-%d_%H-%M-%S")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


@dataclass(frozen=True)
class DataPaths:
    root_dir: Path
    timestamp: str
    run_dir: Path

    @property
    def targets_json(self) -> Path:
        return self.run_dir / "targets.json"

    @property
    def robot_info_txt(self) -> Path:
        return self.run_dir / "robot_info.txt"

    @property
    def summary_traditional_json(self) -> Path:
        return self.run_dir / "summary_tradition.json"

    @property
    def summary_enumeration_json(self) -> Path:
        return self.run_dir / "summary_enumeration.json"

    @property
    def summary_json(self) -> Path:
        """Unified summary required by the benchmark spec."""
        return self.run_dir / "summary.json"

    # Backward-compatible aliases (deprecated): keep the old property names but point to JSON.
    @property
    def summary_traditional_txt(self) -> Path:  # pragma: no cover
        return self.summary_traditional_json

    @property
    def summary_enumeration_txt(self) -> Path:  # pragma: no cover
        return self.summary_enumeration_json

    def ik_json_for_point(self, point_idx_1based: int) -> Path:
        return self.run_dir / f"p{int(point_idx_1based)}.json"


def prepare_data_paths(data_root: str = "data_three") -> DataPaths:
    """
    Prepare `data/<timestamp>` directory for one run.
    """
    ts = make_timestamp(compact=True)
    root_dir = ensure_dir(Path(data_root))
    run_dir = ensure_dir(root_dir / ts)
    return DataPaths(root_dir=root_dir, timestamp=ts, run_dir=run_dir)
