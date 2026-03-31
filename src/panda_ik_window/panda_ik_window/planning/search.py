from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import wraps
from time import perf_counter
from typing import Any, Callable, List, Sequence, Tuple

import numpy as np

from ..types import IKSolution
from .time_metric import SegmentTimeModel, TimeModelInfo


def _try_import_torch() -> Any | None:
    """Lazy torch import.

    We keep torch optional so the package still works in CPU-only / non-torch
    environments. GPU acceleration is enabled only when:
      - torch is available
      - a CUDA device is available
      - time_model.effective == 'trapezoid'
    """

    try:
        import torch  # type: ignore

        return torch
    except Exception:
        return None


def _resolve_torch_device(torch: Any, device: str | None) -> tuple[Any, bool, str]:
    """Resolve device string to torch.device.

    Returns
    -------
    (torch_device, use_cuda, note)

    - If requested CUDA but unavailable, falls back to CPU (use_cuda=False).
    """

    raw = "auto" if device is None else str(device).strip()
    low = raw.lower()

    # Common aliases
    if low in {"gpu"}:
        low = "cuda"

    if low in {"", "auto"}:
        if bool(torch.cuda.is_available()):
            return torch.device("cuda"), True, "auto->cuda"
        return torch.device("cpu"), False, "auto->cpu (cuda unavailable)"

    # Explicit device
    try:
        dev = torch.device(low)
    except Exception:
        # Fallback to auto
        if bool(torch.cuda.is_available()):
            return torch.device("cuda"), True, f"invalid device={raw!r} -> cuda"
        return torch.device("cpu"), False, f"invalid device={raw!r} -> cpu"

    if dev.type == "cuda":
        if not bool(torch.cuda.is_available()):
            return torch.device("cpu"), False, f"requested {raw!r} but cuda unavailable -> cpu"
        # Validate index if present
        try:
            count = int(torch.cuda.device_count())
        except Exception:
            count = 1
        if dev.index is not None and count > 0 and int(dev.index) >= count:
            return torch.device("cuda"), True, f"requested {raw!r} but only {count} cuda device(s) -> cuda:0"
        return dev, True, f"use {dev}"

    return dev, False, f"use {dev}"


def _trapezoid_time_matrix_torch(
    *,
    torch: Any,
    prev: Any,
    curr: Any,
    vmax_1x1xd: Any,
    amax_1x1xd: Any,
    eps: float = 1e-9,
) -> Any:
    """Torch implementation of the trapezoid/triangle segment-time matrix.

    prev: (Kprev, dof)
    curr: (Kcurr, dof)
    returns: (Kprev, Kcurr)

    The formula matches _estimate_time_matrix_trapezoid_s in time_metric.py.
    """

    # Ensure numeric stability
    vmax = torch.clamp(vmax_1x1xd, min=float(eps))
    amax = torch.clamp(amax_1x1xd, min=float(eps))

    dq = torch.abs(prev[:, None, :] - curr[None, :, :])  # (Kprev, Kcurr, dof)
    dcrit = (vmax * vmax) / amax

    t_tri = 2.0 * torch.sqrt(dq / amax)
    t_trap = (dq / vmax) + (vmax / amax)
    t_joint = torch.where(dq <= dcrit, t_tri, t_trap)

    # (Kprev, Kcurr)
    return torch.amax(t_joint, dim=2)


def _timed_call(fn: Callable[..., Any]) -> Callable[..., tuple[Any, float]]:
    """Decorator: return (result, elapsed_seconds)."""

    @wraps(fn)
    def _wrapped(*args: Any, **kwargs: Any) -> tuple[Any, float]:
        t0 = perf_counter()
        out = fn(*args, **kwargs)
        return out, float(perf_counter() - t0)

    return _wrapped


@_timed_call
def _solve_window_dp(
    *,
    start_q: Sequence[float],
    layers: Sequence[Sequence[IKSolution]],
    time_model: SegmentTimeModel,
    block_size: int,
    device: str,
    precomputed_q_torch: Sequence[Any] | None,
) -> Tuple[Tuple[int, ...], float]:
    """Solve one window DP; timing is provided by @_timed_call."""
    return _optimal_path_indices_dp(
        start_q=start_q,
        layers=layers,
        time_model=time_model,
        block_size=block_size,
        device=device,
        _precomputed_Q_torch=precomputed_q_torch,
    )


def _optimal_path_indices_dp_numpy(
    *,
    start_q: Sequence[float],
    layers: Sequence[Sequence[IKSolution]],
    time_model: SegmentTimeModel,
    block_size: int = 128,
) -> Tuple[Tuple[int, ...], float]:
    """Numpy DP backend (original implementation)."""

    layers = [list(layer) for layer in layers]
    if len(layers) == 0:
        return tuple(), 0.0
    if any(len(layer) == 0 for layer in layers):
        bad = [i for i, layer in enumerate(layers) if len(layer) == 0]
        raise RuntimeError(f"Some window layers have 0 IK solutions: {bad}")

    q0 = np.asarray(start_q, dtype=float)

    # Convert to numpy arrays
    Q = [np.asarray([s.joint_positions for s in layer], dtype=float) for layer in layers]
    dof = int(Q[0].shape[1])

    # dp for layer 0: dp0[j] = cost(start -> layer0_j)
    t0 = time_model.segment_time_matrix_s(q0.reshape((1, dof)), Q[0])  # (1, K0)
    dp = t0.reshape((-1,))
    parents: List[np.ndarray] = [np.full((int(dp.shape[0]),), -1, dtype=int)]

    block_size = max(1, int(block_size))

    for li in range(1, len(Q)):
        prev = Q[li - 1]
        curr = Q[li]
        Kprev = int(prev.shape[0])
        Kcurr = int(curr.shape[0])

        dp_new = np.empty((Kcurr,), dtype=float)
        parent = np.empty((Kcurr,), dtype=int)

        for j0 in range(0, Kcurr, block_size):
            j1 = min(Kcurr, j0 + block_size)
            curr_block = curr[j0:j1]

            tmat = time_model.segment_time_matrix_s(prev, curr_block)  # (Kprev, B)
            totals = dp.reshape((-1, 1)) + tmat

            parent_block = np.argmin(totals, axis=0).astype(int)
            cols = np.arange(int(totals.shape[1]), dtype=int)
            dp_block = totals[parent_block, cols]

            dp_new[j0:j1] = dp_block
            parent[j0:j1] = parent_block

        parents.append(parent)
        dp = dp_new

    best_last = int(np.argmin(dp))
    best_time = float(dp[best_last])

    # backtrack
    idxs = [0] * len(Q)
    idxs[-1] = best_last
    for k in range(len(Q) - 1, 0, -1):
        idxs[k - 1] = int(parents[k][idxs[k]])

    return tuple(int(x) for x in idxs), best_time


def _optimal_path_indices_dp_torch(
    *,
    start_q: Sequence[float],
    Q_layers: Sequence[Any],
    time_model: SegmentTimeModel,
    torch: Any,
    torch_device: Any,
    block_size: int = 128,
    dtype: Any = None,
    eps: float = 1e-9,
) -> Tuple[Tuple[int, ...], float]:
    """Torch DP backend (CUDA when available).

    Q_layers are torch tensors on the same device, each shaped (K_i, dof).
    """

    if len(Q_layers) == 0:
        return tuple(), 0.0
    if any(int(q.shape[0]) == 0 for q in Q_layers):
        bad = [i for i, q in enumerate(Q_layers) if int(q.shape[0]) == 0]
        raise RuntimeError(f"Some window layers have 0 IK solutions: {bad}")

    if dtype is None:
        dtype = torch.float64

    dof = int(Q_layers[0].shape[1])

    # Limits on device (broadcastable)
    vmax = torch.as_tensor(time_model.vmax, dtype=dtype, device=torch_device).reshape((1, 1, dof))
    amax = torch.as_tensor(time_model.amax, dtype=dtype, device=torch_device).reshape((1, 1, dof))

    # Start state on device
    q0 = torch.as_tensor(np.asarray(start_q, dtype=float), dtype=dtype, device=torch_device).reshape((1, dof))

    # dp for layer 0
    t0 = _trapezoid_time_matrix_torch(
        torch=torch,
        prev=q0,
        curr=Q_layers[0],
        vmax_1x1xd=vmax,
        amax_1x1xd=amax,
        eps=eps,
    )  # (1, K0)

    dp = t0.reshape((-1,))  # (K0,)

    # Keep parents on CPU (ints), dp stays on GPU.
    parents_cpu: List[np.ndarray] = [np.full((int(dp.shape[0]),), -1, dtype=int)]

    block_size = max(1, int(block_size))

    for li in range(1, len(Q_layers)):
        prev = Q_layers[li - 1]
        curr = Q_layers[li]
        Kcurr = int(curr.shape[0])

        dp_new = torch.empty((Kcurr,), dtype=dtype, device=torch_device)
        parent_cpu = np.empty((Kcurr,), dtype=int)

        for j0 in range(0, Kcurr, block_size):
            j1 = min(Kcurr, j0 + block_size)
            curr_block = curr[j0:j1]

            tmat = _trapezoid_time_matrix_torch(
                torch=torch,
                prev=prev,
                curr=curr_block,
                vmax_1x1xd=vmax,
                amax_1x1xd=amax,
                eps=eps,
            )  # (Kprev, B)

            totals = tmat + dp.reshape((-1, 1))

            # values/indices: (B,)
            dp_block, parent_block = torch.min(totals, dim=0)

            dp_new[j0:j1] = dp_block
            parent_cpu[j0:j1] = parent_block.detach().to("cpu").to(torch.int64).numpy()

        parents_cpu.append(parent_cpu)
        dp = dp_new

    best_last = int(torch.argmin(dp).detach().to("cpu").item())
    best_time = float(dp[best_last].detach().to("cpu").item())

    # backtrack (on CPU)
    idxs = [0] * len(Q_layers)
    idxs[-1] = best_last
    for k in range(len(Q_layers) - 1, 0, -1):
        idxs[k - 1] = int(parents_cpu[k][idxs[k]])

    return tuple(int(x) for x in idxs), best_time


def _optimal_path_indices_dp(
    *,
    start_q: Sequence[float],
    layers: Sequence[Sequence[IKSolution]],
    time_model: SegmentTimeModel,
    block_size: int = 128,
    device: str = "auto",
    _precomputed_Q_torch: Sequence[Any] | None = None,
) -> Tuple[Tuple[int, ...], float]:
    """Solve a small layered shortest-path problem and return the best indices.

    This function automatically selects a backend:
      - CUDA Torch backend (trapezoid only) when available and requested
      - otherwise the original NumPy backend

    Parameters
    ----------
    start_q:
        Fixed start joint configuration (e.g. the already-committed IK of p{i-1}).
    layers:
        IK solution layers for consecutive points (e.g. [IK(p_i), IK(p_{i+1}), ...]).
    device:
        'auto'|'cpu'|'cuda'|'cuda:0'... (torch-style). Only affects trapezoid mode.

    Notes
    -----
    - TOTG mode is left untouched (CPU, nested loops) because MoveIt TOTG runs on CPU.
    - When _precomputed_Q_torch is provided, it must match `layers` and be already
      on the desired torch device.
    """

    # Only accelerate trapezoid (the default).
    if str(time_model.info.effective) != "trapezoid":
        return _optimal_path_indices_dp_numpy(
            start_q=start_q,
            layers=layers,
            time_model=time_model,
            block_size=block_size,
        )

    torch = _try_import_torch()
    if torch is None:
        return _optimal_path_indices_dp_numpy(
            start_q=start_q,
            layers=layers,
            time_model=time_model,
            block_size=block_size,
        )

    torch_device, use_cuda, _note = _resolve_torch_device(torch, device)
    if not use_cuda:
        # Keep CPU path by default if no CUDA is available/requested.
        return _optimal_path_indices_dp_numpy(
            start_q=start_q,
            layers=layers,
            time_model=time_model,
            block_size=block_size,
        )

    # Precompute Q tensors if not given.
    if _precomputed_Q_torch is not None:
        Q_layers = list(_precomputed_Q_torch)
    else:
        layers = [list(layer) for layer in layers]
        Q_np = [np.asarray([s.joint_positions for s in layer], dtype=float) for layer in layers]
        Q_layers = [torch.as_tensor(q, dtype=torch.float64, device=torch_device) for q in Q_np]

    return _optimal_path_indices_dp_torch(
        start_q=start_q,
        Q_layers=Q_layers,
        time_model=time_model,
        torch=torch,
        torch_device=torch_device,
        block_size=block_size,
        dtype=torch.float64,
    )


@dataclass(frozen=True)
class SegmentResult:
    seg_idx_1based: int
    from_label: str
    to_label: str
    found: int
    requested: int
    best_solution: IKSolution
    best_time_s: float


@dataclass(frozen=True)
class PathResult:
    method: str
    segments: Sequence[SegmentResult]
    total_time_s: float
    total_paths_theoretical: int
    time_model: TimeModelInfo
    window_selection_stats: Sequence[dict] = field(default_factory=tuple)


def greedy_path(
    *,
    start_q: Sequence[float],
    solutions_by_point: Sequence[Sequence[IKSolution]],
    requested_per_point: int,
    time_model: SegmentTimeModel,
    device: str = "auto",
) -> PathResult:
    """Traditional greedy baseline.

    Notes
    -----
    This is intentionally kept simple. GPU acceleration targets the DP-based
    methods (window/global). Greedy is usually not the bottleneck.
    """

    _ = device  # reserved for future use

    current_q = np.asarray(start_q, dtype=float)

    segments: List[SegmentResult] = []
    total = 0.0

    for i, sols in enumerate(solutions_by_point, start=1):
        sols = list(sols)
        if len(sols) == 0:
            raise RuntimeError(f"Point p{i} has 0 IK solutions; greedy cannot continue.")

        times = []
        for s in sols:
            t = time_model.segment_time_s(current_q, s.joint_positions)
            times.append(t)

        j_best = int(np.argmin(np.asarray(times, dtype=float)))
        best_sol = sols[j_best]
        best_t = float(times[j_best])

        segments.append(
            SegmentResult(
                seg_idx_1based=i,
                from_label=f"p{i-1}",
                to_label=f"p{i}",
                found=len(sols),
                requested=int(requested_per_point),
                best_solution=best_sol,
                best_time_s=best_t,
            )
        )
        total += best_t
        current_q = np.asarray(best_sol.joint_positions, dtype=float)

    total_paths = 1
    for sols in solutions_by_point:
        total_paths *= max(1, int(len(sols)))

    tm = time_model.info
    tm_copy = TimeModelInfo(
        requested=str(tm.requested),
        effective=str(tm.effective),
        note=str(tm.note),
        totg_available=bool(tm.totg_available),
        totg_failures=int(tm.totg_failures),
    )

    return PathResult(
        method="greedy",
        segments=segments,
        total_time_s=float(total),
        total_paths_theoretical=int(total_paths),
        time_model=tm_copy,
    )


def window_path_receding_horizon(
    *,
    start_q: Sequence[float],
    solutions_by_point: Sequence[Sequence[IKSolution]],
    requested_per_point: int,
    time_model: SegmentTimeModel,
    window_size: int,
    block_size: int = 128,
    device: str = "auto",
) -> PathResult:
    """Sliding-window lookahead (receding horizon) path selection.

    Updated window logic (per spec)
    -------------------------------
    Let ws = window_size, and let `remain` be the number of unsolved target points
    (p_i..p_n).

    - If remain > ws:
        Solve the shortest path within the fixed-size window
            [p_i, p_{i+1}, ..., p_{i+ws-1}]
        and **commit only the first decision** (the IK choice at p_i).

    - Once we reach the tail where remain <= ws:
        Solve **one** shortest-path problem over **all remaining points** and
        commit the entire remaining path at once, then terminate.

    GPU acceleration
    ----------------
    When the effective time model is "trapezoid" (default) and a CUDA device is
    available, the DP inside the window solver is executed with PyTorch on GPU.
    This keeps the overall logic and output format unchanged.
    """

    layers_all = [list(layer) for layer in solutions_by_point]
    if any(len(layer) == 0 for layer in layers_all):
        bad = [i + 1 for i, layer in enumerate(layers_all) if len(layer) == 0]
        raise RuntimeError(f"Some points have 0 IK solutions: {bad}")

    n = int(len(layers_all))
    w = int(window_size)
    if w < 1:
        w = 1

    # Optional: precompute Q tensors on GPU once per call (huge speedup for repeated windows).
    torch = None
    torch_device = None
    use_cuda = False
    Q_all_torch: List[Any] | None = None

    if str(time_model.info.effective) == "trapezoid":
        torch = _try_import_torch()
        if torch is not None:
            torch_device, use_cuda, _note = _resolve_torch_device(torch, device)
            if use_cuda:
                # Build per-layer joint-position tensors on GPU.
                Q_all_torch = []
                for layer in layers_all:
                    q_np = np.asarray([s.joint_positions for s in layer], dtype=float)
                    Q_all_torch.append(torch.as_tensor(q_np, dtype=torch.float64, device=torch_device))

    current_q = np.asarray(start_q, dtype=float)
    segments: List[SegmentResult] = []
    selection_stats: List[dict] = []
    total = 0.0

    i0 = 0  # 0-based for p_{i0+1}
    while i0 < n:
        remain = n - i0

        # Tail case: solve the remaining points in ONE DP and finish.
        if remain <= w:
            window_layers = layers_all[i0:n]

            if Q_all_torch is not None:
                Q_window = Q_all_torch[i0:n]
                (idxs, _best_window_time), _elapsed_s = _solve_window_dp(
                    start_q=current_q,
                    layers=window_layers,
                    time_model=time_model,
                    block_size=block_size,
                    device=str(device),
                    precomputed_q_torch=Q_window,
                )
            else:
                (idxs, _best_window_time), _elapsed_s = _solve_window_dp(
                    start_q=current_q,
                    layers=window_layers,
                    time_model=time_model,
                    block_size=block_size,
                    device=str(device),
                    precomputed_q_torch=None,
                )

            # Commit ALL remaining decisions (p_{i0+1}..p_n) at once.
            for k, (layer, chosen_idx) in enumerate(zip(window_layers, idxs)):
                seg_idx_1based = i0 + k + 1
                chosen_sol = layer[int(chosen_idx)]
                dt = float(time_model.segment_time_s(current_q, chosen_sol.joint_positions))

                segments.append(
                    SegmentResult(
                        seg_idx_1based=seg_idx_1based,
                        from_label=f"p{seg_idx_1based - 1}",
                        to_label=f"p{seg_idx_1based}",
                        found=len(layer),
                        requested=int(requested_per_point),
                        best_solution=chosen_sol,
                        best_time_s=dt,
                    )
                )
                total += dt
                current_q = np.asarray(chosen_sol.joint_positions, dtype=float)

            break

        # Main case: fixed-size window, commit only the first step.
        window_layers = layers_all[i0 : i0 + w]
        layer_sizes = [int(len(layer)) for layer in window_layers]
        theoretical_paths = 1
        for sz in layer_sizes:
            theoretical_paths *= max(1, int(sz))

        if Q_all_torch is not None:
            Q_window = Q_all_torch[i0 : i0 + w]
            (idxs, _best_window_time), elapsed_s = _solve_window_dp(
                start_q=current_q,
                layers=window_layers,
                time_model=time_model,
                block_size=block_size,
                device=str(device),
                precomputed_q_torch=Q_window,
            )
        else:
            (idxs, _best_window_time), elapsed_s = _solve_window_dp(
                start_q=current_q,
                layers=window_layers,
                time_model=time_model,
                block_size=block_size,
                device=str(device),
                precomputed_q_torch=None,
            )

        chosen_idx = int(idxs[0])
        chosen_sol = window_layers[0][chosen_idx]

        # Record the solve time for full-size window decisions only.
        selection_stats.append(
            {
                "segment_index_1based": int(i0 + 1),
                "point": f"p{i0 + 1}",
                "window_size": int(w),
                "remain_points": int(remain),
                "window_layer_sizes": [int(v) for v in layer_sizes],
                "theoretical_paths": int(theoretical_paths),
                "selected_solution_index_0based": int(chosen_idx),
                "selected_solution_index_1based": int(chosen_idx) + 1,
                "selected_solution_id": f"p{i0 + 1}_{int(chosen_idx) + 1}",
                "selection_elapsed_s": float(elapsed_s),
            }
        )

        dt = float(time_model.segment_time_s(current_q, chosen_sol.joint_positions))
        segments.append(
            SegmentResult(
                seg_idx_1based=i0 + 1,
                from_label=f"p{i0}",
                to_label=f"p{i0 + 1}",
                found=len(window_layers[0]),
                requested=int(requested_per_point),
                best_solution=chosen_sol,
                best_time_s=dt,
            )
        )
        total += dt
        current_q = np.asarray(chosen_sol.joint_positions, dtype=float)
        i0 += 1

    total_paths = 1
    for layer in layers_all:
        total_paths *= max(1, int(len(layer)))

    tm = time_model.info
    tm_copy = TimeModelInfo(
        requested=str(tm.requested),
        effective=str(tm.effective),
        note=str(tm.note),
        totg_available=bool(tm.totg_available),
        totg_failures=int(tm.totg_failures),
    )

    return PathResult(
        method="window",
        segments=segments,
        total_time_s=float(total),
        total_paths_theoretical=int(total_paths),
        time_model=tm_copy,
        window_selection_stats=tuple(selection_stats),
    )


def global_optimal_path(
    *,
    start_q: Sequence[float],
    solutions_by_point: Sequence[Sequence[IKSolution]],
    requested_per_point: int,
    time_model: SegmentTimeModel,
    block_size: int = 128,
    device: str = "auto",
) -> PathResult:
    """Global method: DP shortest path across all points.

    This method can also use the same CUDA DP backend as the window solver when
    time_model.effective == 'trapezoid'.
    """

    q0 = np.asarray(start_q, dtype=float)
    layers = [list(layer) for layer in solutions_by_point]
    if any(len(layer) == 0 for layer in layers):
        bad = [i + 1 for i, layer in enumerate(layers) if len(layer) == 0]
        raise RuntimeError(f"Some points have 0 IK solutions: {bad}")

    # Solve DP indices (use GPU when available).
    Q_all_torch: List[Any] | None = None
    if str(time_model.info.effective) == "trapezoid":
        torch = _try_import_torch()
        if torch is not None:
            torch_device, use_cuda, _note = _resolve_torch_device(torch, device)
            if use_cuda:
                Q_all_torch = []
                for layer in layers:
                    q_np = np.asarray([s.joint_positions for s in layer], dtype=float)
                    Q_all_torch.append(torch.as_tensor(q_np, dtype=torch.float64, device=torch_device))

    if Q_all_torch is not None:
        idxs, _best_time = _optimal_path_indices_dp(
            start_q=q0,
            layers=layers,
            time_model=time_model,
            block_size=block_size,
            device=str(device),
            _precomputed_Q_torch=Q_all_torch,
        )
    else:
        idxs, _best_time = _optimal_path_indices_dp(
            start_q=q0,
            layers=layers,
            time_model=time_model,
            block_size=block_size,
            device=str(device),
        )

    # Build PathResult
    segments: List[SegmentResult] = []
    current_q = q0.copy()
    total = 0.0
    for i, (layer, chosen_idx) in enumerate(zip(layers, idxs), start=1):
        sol = layer[int(chosen_idx)]
        t = float(time_model.segment_time_s(current_q, sol.joint_positions))
        segments.append(
            SegmentResult(
                seg_idx_1based=i,
                from_label=f"p{i-1}",
                to_label=f"p{i}",
                found=len(layer),
                requested=int(requested_per_point),
                best_solution=sol,
                best_time_s=t,
            )
        )
        total += t
        current_q = np.asarray(sol.joint_positions, dtype=float)

    total_paths = 1
    for layer in layers:
        total_paths *= max(1, int(len(layer)))

    tm = time_model.info
    tm_copy = TimeModelInfo(
        requested=str(tm.requested),
        effective=str(tm.effective),
        note=str(tm.note),
        totg_available=bool(tm.totg_available),
        totg_failures=int(tm.totg_failures),
    )

    return PathResult(
        method="global",
        segments=segments,
        total_time_s=float(total),
        total_paths_theoretical=int(total_paths),
        time_model=tm_copy,
        window_selection_stats=tuple(),
    )
