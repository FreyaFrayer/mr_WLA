from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from moveit.core.robot_state import RobotState
    from panda_ik_window.utils.robot import RobotContext
else:
    RobotState = Any
    RobotContext = Any


def _normalize_collision_check_output(out: Any) -> Optional[bool]:
    if isinstance(out, bool):
        return bool(out)
    if isinstance(out, (int, np.integer)):
        return bool(int(out))

    if isinstance(out, tuple) and len(out) > 0:
        head = out[0]
        if isinstance(head, bool):
            return bool(head)
        if isinstance(head, (int, np.integer)):
            return bool(int(head))

    for attr in ("collision", "is_collision", "colliding", "is_colliding"):
        try:
            if hasattr(out, attr):
                val = getattr(out, attr)
                if isinstance(val, bool):
                    return bool(val)
                if isinstance(val, (int, np.integer)):
                    return bool(int(val))
        except Exception:
            continue

    return None


def _query_state_collision_on_target(target: Any, *, state: RobotState, group: str) -> Optional[bool]:
    for method in ("is_state_colliding", "isStateColliding"):
        fn = getattr(target, method, None)
        if fn is None:
            continue

        candidates = [
            ((), {"robot_state": state, "joint_model_group_name": str(group)}),
            ((), {"robot_state": state, "group_name": str(group)}),
            ((), {"robot_state": state, "group": str(group)}),
            ((), {"robot_state": state}),
            ((state, str(group)), {}),
            ((state,), {}),
        ]

        for args, kwargs in candidates:
            try:
                out = fn(*args, **kwargs)
            except TypeError:
                continue
            except Exception:
                continue

            flag = _normalize_collision_check_output(out)
            if flag is not None:
                return bool(flag)

    return None


def build_state_collision_checker(ctx: RobotContext, *, group: str) -> Tuple[Optional[Callable[[RobotState], Optional[bool]]], str]:
    """Build a best-effort state self-collision checker from MoveItPy APIs.

    Returns
    -------
    checker:
        Callable(state)->bool|None where True=colliding, False=collision-free,
        None=unknown (API unavailable at runtime)
    reason:
        Non-empty when checker is unavailable.
    """

    psm = None
    for attr in ("get_planning_scene_monitor", "getPlanningSceneMonitor", "planning_scene_monitor"):
        try:
            value = getattr(ctx.moveit_py, attr)
            psm = value() if callable(value) else value
            if psm is not None:
                break
        except Exception:
            continue

    if psm is None:
        return None, "planning_scene_monitor not available from MoveItPy"

    def _check(state: RobotState) -> Optional[bool]:
        state.update()

        # Fast path: psm directly exposes collision query.
        flag = _query_state_collision_on_target(psm, state=state, group=group)
        if flag is not None:
            return bool(flag)

        # Try read lock APIs.
        for lock_name in ("read_only", "readOnly", "locked_planning_scene_ro", "lockedPlanningSceneRO"):
            lock_fn = getattr(psm, lock_name, None)
            if lock_fn is None:
                continue
            try:
                lock = lock_fn()
            except Exception:
                continue

            try:
                with lock as scene:
                    flag2 = _query_state_collision_on_target(scene, state=state, group=group)
                    if flag2 is not None:
                        return bool(flag2)
            except Exception:
                continue

        # Try plain planning-scene getter APIs.
        for getter in ("get_planning_scene", "getPlanningScene", "planning_scene"):
            try:
                value = getattr(psm, getter)
                scene = value() if callable(value) else value
            except Exception:
                continue
            if scene is None:
                continue

            flag3 = _query_state_collision_on_target(scene, state=state, group=group)
            if flag3 is not None:
                return bool(flag3)

        return None

    return _check, ""
