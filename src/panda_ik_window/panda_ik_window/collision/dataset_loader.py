from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


_REQUIRED_KEYS = ("q_cur", "q_cand_fut", "cand_mask")


@dataclass(frozen=True)
class LoadedDataset:
    dataset_dir: str
    npz_path: str
    index_jsonl: Optional[str]
    meta_json: Optional[str]
    ws: int
    dof: int
    k_max: int
    num_samples: int


@dataclass(frozen=True)
class LoadedArrays:
    q_cur: np.ndarray
    q_cand_next: np.ndarray
    cand_mask_next: np.ndarray
    order: Optional[np.ndarray]
    sortik: Optional[np.ndarray]


def _extract_ws_from_npz_name(npz_path: Path) -> Optional[int]:
    name = npz_path.name
    if not name.startswith("dataset_ws") or not name.endswith(".npz"):
        return None
    token = name[len("dataset_ws") : -len(".npz")]
    if token.isdigit():
        return int(token)
    return None


def _find_dataset_npz_in_dir(root: Path, preferred_ws: int) -> Path:
    files = sorted(root.rglob("dataset_ws*.npz"))
    if not files:
        raise FileNotFoundError(f"No dataset_ws*.npz found under: {root}")

    for p in files:
        ws = _extract_ws_from_npz_name(p)
        if ws == preferred_ws:
            return p
    return files[0]


def _search_dataset_candidates(dataset_name: str) -> list[Path]:
    candidates: list[Path] = []
    seen: set[str] = set()

    def _add_if_exists(p: Path) -> None:
        if not p.exists():
            return
        r = str(p.resolve())
        if r in seen:
            return
        seen.add(r)
        candidates.append(Path(r))

    hint = Path(dataset_name).expanduser()
    _add_if_exists(hint)

    roots = [
        Path.cwd(),
        Path.cwd().parent,
        Path.home() / "Desktop",
        Path.home() / "Desktop" / "data_window",
    ]
    for root in roots:
        _add_if_exists(root / dataset_name)

    return candidates


def resolve_dataset_npz(
    dataset: str,
    *,
    preferred_ws: int = 3,
) -> LoadedDataset:
    dataset_paths = _search_dataset_candidates(dataset)
    if not dataset_paths:
        raise FileNotFoundError(
            f"Dataset path not found: {dataset}. "
            "Provide --dataset as an existing directory or npz path."
        )

    errors: list[str] = []

    for dataset_path in dataset_paths:
        if dataset_path.is_file():
            if dataset_path.suffix.lower() != ".npz":
                errors.append(f"{dataset_path}: not an .npz file")
                continue
            npz_path = dataset_path
            data_dir = dataset_path.parent
        else:
            try:
                npz_path = _find_dataset_npz_in_dir(dataset_path, preferred_ws=preferred_ws)
            except Exception as e:
                errors.append(f"{dataset_path}: {e}")
                continue
            data_dir = npz_path.parent

        ws = _extract_ws_from_npz_name(npz_path) or int(preferred_ws)

        index_jsonl = data_dir / f"index_ws{ws}.jsonl"
        if not index_jsonl.exists():
            index_jsonl = None

        meta_json = data_dir / f"meta_ws{ws}.json"
        if not meta_json.exists():
            meta_json = None

        try:
            with np.load(str(npz_path), allow_pickle=False, mmap_mode="r") as arr:
                for k in _REQUIRED_KEYS:
                    if k not in arr.files:
                        raise KeyError(f"Missing key '{k}' in {npz_path}")

                q_cur = arr["q_cur"]
                q_cand_fut = arr["q_cand_fut"]
                cand_mask = arr["cand_mask"]

                if q_cur.ndim != 2:
                    raise ValueError(f"q_cur must be 2D, got shape={q_cur.shape}")
                if q_cand_fut.ndim != 4:
                    raise ValueError(f"q_cand_fut must be 4D, got shape={q_cand_fut.shape}")
                if cand_mask.ndim != 3:
                    raise ValueError(f"cand_mask must be 3D, got shape={cand_mask.shape}")

                n = int(q_cur.shape[0])
                dof = int(q_cur.shape[1])
                k_max = int(q_cand_fut.shape[2])

                if int(q_cand_fut.shape[0]) != n or int(cand_mask.shape[0]) != n:
                    raise ValueError("N mismatch among q_cur/q_cand_fut/cand_mask")
                if int(q_cand_fut.shape[3]) != dof:
                    raise ValueError("dof mismatch between q_cur and q_cand_fut")
                if int(cand_mask.shape[2]) != k_max:
                    raise ValueError("K mismatch between q_cand_fut and cand_mask")
        except Exception as e:
            errors.append(f"{npz_path}: {e}")
            continue

        return LoadedDataset(
            dataset_dir=str(data_dir),
            npz_path=str(npz_path),
            index_jsonl=str(index_jsonl) if index_jsonl is not None else None,
            meta_json=str(meta_json) if meta_json is not None else None,
            ws=int(ws),
            dof=int(dof),
            k_max=int(k_max),
            num_samples=int(n),
        )

    joined = "; ".join(errors) if errors else "no candidate paths"
    raise RuntimeError(f"Failed to resolve a valid dataset from '{dataset}'. Details: {joined}")


def load_npz_arrays(ds: LoadedDataset) -> LoadedArrays:
    with np.load(ds.npz_path, allow_pickle=False, mmap_mode="r") as arr:
        q_cur = np.asarray(arr["q_cur"], dtype=np.float64)
        q_cand_fut = np.asarray(arr["q_cand_fut"], dtype=np.float64)
        cand_mask = np.asarray(arr["cand_mask"], dtype=bool)

        order = np.asarray(arr["order"], dtype=np.int32) if "order" in arr.files else None
        sortik = np.asarray(arr["sortIK"], dtype=np.int32) if "sortIK" in arr.files else None

    q_cand_next = q_cand_fut[:, 0, :, :]
    cand_mask_next = cand_mask[:, 0, :]

    return LoadedArrays(
        q_cur=q_cur,
        q_cand_next=q_cand_next,
        cand_mask_next=cand_mask_next,
        order=order,
        sortik=sortik,
    )


def load_index_rows(index_jsonl: Optional[str], *, limit: Optional[int] = None) -> Optional[list[dict]]:
    if not index_jsonl:
        return None

    p = Path(index_jsonl)
    if not p.exists():
        return None

    rows: list[dict] = []
    with p.open("r", encoding="utf-8") as f:
        for line_i, line in enumerate(f):
            s = line.strip()
            if not s:
                continue
            try:
                rows.append(json.loads(s))
            except json.JSONDecodeError:
                rows.append({"line": line_i, "raw": s})

            if limit is not None and len(rows) >= int(limit):
                break

    return rows


def load_meta(meta_json: Optional[str]) -> Optional[dict]:
    if not meta_json:
        return None

    p = Path(meta_json)
    if not p.exists():
        return None

    with p.open("r", encoding="utf-8") as f:
        return json.load(f)
