# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick, Zeyu Ma

import logging
from typing import Callable

import bpy
import numpy as np
import procfunc as pf

from infinigen2.util.errors import RejectedScene

__all__ = [
    "random_walk",
    "random_walk_step_fn",
    "walk_loop",
]

logger = logging.getLogger(__name__)

StepFn = Callable[
    [pf.RNG, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, float]
]


def random_walk_step_fn(
    bbox: tuple[np.ndarray, np.ndarray],
    margin: float = 0.05,
    speed_mps_range: tuple[float, float] = (2.0, 3.0),
    loc_step_range: tuple[float, float] = (1.0, 4.0),
    rot_std_deg: tuple[float, float, float] = (15.0, 15.0, 30.0),
    roll_range_deg: tuple[float, float] = (-25.0, 25.0),
    pitch_range_deg: tuple[float, float] = (45.0, 135.0),
    height_range: tuple[float, float] | None = None,
) -> StepFn:
    bbox_min = np.asarray(bbox[0]) + margin
    bbox_max = np.asarray(bbox[1]) - margin
    roll_range_rad = np.deg2rad(roll_range_deg)
    pitch_range_rad = np.deg2rad(pitch_range_deg)
    if height_range is not None:
        bbox_min[2] = max(bbox_min[2], height_range[0])
        bbox_max[2] = min(bbox_max[2], height_range[1])

    def step(
        rng: pf.RNG,
        curr_loc: np.ndarray,
        curr_rot: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Return (next_loc, next_rot, duration_seconds)."""
        dist = float(rng.uniform(*loc_step_range))
        direction = rng.normal(0.0, np.ones(3))
        direction = direction / (np.linalg.norm(direction) + 1e-8)
        next_loc = np.clip(curr_loc + direction * dist, bbox_min, bbox_max)

        rot_jitter = np.deg2rad(rng.normal(0.0, np.asarray(rot_std_deg)))
        next_rot = curr_rot + rot_jitter
        next_rot[0] = np.clip(next_rot[0], *pitch_range_rad)
        next_rot[1] = np.clip(next_rot[1], *roll_range_rad)

        speed = float(rng.uniform(*speed_mps_range))
        duration = dist / max(speed, 1e-4)
        return next_loc, next_rot, duration

    return step


def random_walk(
    rng: pf.RNG,
    obj: pf.Object,
    bbox: tuple[np.ndarray, np.ndarray],
    frame_start: int,
    frame_end: int,
    accept_fn: Callable[[], bool] | None = None,
    max_retries: int = 20,
    failure_mode: str = "error",
    **sampler_kwargs,
) -> pf.Object | None:
    """Animate any pre-posed *obj* along a random walk within *bbox*.

    Initial posing is the caller's responsibility; this only drives the walk.
    Returns *obj* if a complete path was found, else None.
    """
    sampler = random_walk_step_fn(bbox=bbox, **sampler_kwargs)
    return walk_loop(
        rng=rng,
        obj=obj,
        sampler=sampler,
        accept_fn=accept_fn,
        frame_start=frame_start,
        frame_end=frame_end,
        max_retries=max_retries,
        failure_mode=failure_mode,
    )


def _delete_keyframes(obj: bpy.types.Object, frame: int) -> None:
    if obj.animation_data is None or obj.animation_data.action is None:
        return
    data_paths = {
        fc.data_path for fc in obj.animation_data.action.fcurves if fc.data_path
    }
    for data_path in data_paths:
        obj.keyframe_delete(data_path=data_path, frame=frame)


def _validate_trajectory(
    obj: bpy.types.Object,
    accept_fn: Callable,
    frame_start: int,
    frame_end: int,
) -> bool:
    for f in range(frame_start, frame_end + 1):
        bpy.context.scene.frame_set(f)
        if not accept_fn():
            return False
    return True


def walk_loop(
    rng: pf.RNG,
    obj: pf.Object,
    sampler: StepFn,
    frame_start: int,
    frame_end: int,
    accept_fn: Callable[[], bool] | None = None,
    max_retries: int = 20,
    failure_mode: str = "error",
) -> pf.Object | None:
    """Animate *obj* along a random walk from *frame_start* to *frame_end*.

    The object must already be positioned at a valid initial pose.
    A None *accept_fn* accepts every proposed trajectory.
    On exhaustion *failure_mode* selects "error" (raise), "warn" (log and
    return None) or "return" (return None silently).
    Returns *obj* if a complete path was found, else None.
    """
    if failure_mode not in ("error", "warn", "return"):
        raise ValueError(f"unknown failure_mode {failure_mode!r}")
    fps = bpy.context.scene.render.fps / bpy.context.scene.render.fps_base
    bl_obj = obj.item()

    curr_loc = np.array(bl_obj.location)
    curr_rot = np.array(bl_obj.rotation_euler)

    bl_obj.keyframe_insert("location", frame=frame_start)
    bl_obj.keyframe_insert("rotation_euler", frame=frame_start)

    # Stack: list of (loc, rot, frame, retries_used)
    stack: list[tuple[np.ndarray, np.ndarray, int, int]] = [
        (curr_loc.copy(), curr_rot.copy(), frame_start, 0)
    ]

    def _fail(msg: str) -> None:
        if failure_mode == "error":
            raise RejectedScene(msg)
        if failure_mode == "warn":
            logger.warning(msg)
        return None

    while stack[-1][2] < frame_end:
        curr_loc, curr_rot, curr_frame, retries = stack[-1]

        if retries >= max_retries:
            _delete_keyframes(bl_obj, curr_frame)
            stack.pop()
            if not stack:
                return _fail("Random walk exhausted all backtracking options")
            ploc, prot, pframe, pretries = stack[-1]
            stack[-1] = (ploc, prot, pframe, pretries + 1)
            pf.ops.object.set_transform(obj, location=ploc, rotation_euler=prot)
            continue

        stack[-1] = (curr_loc, curr_rot, curr_frame, retries + 1)

        next_loc, next_rot, duration_s = sampler(rng, curr_loc, curr_rot)
        step_frames = max(1, int(duration_s * fps))
        next_frame = min(frame_end, curr_frame + step_frames)

        pf.ops.object.set_transform(obj, location=next_loc, rotation_euler=next_rot)
        bl_obj.keyframe_insert("location", frame=next_frame)
        bl_obj.keyframe_insert("rotation_euler", frame=next_frame)

        if accept_fn is None or _validate_trajectory(
            bl_obj, accept_fn, curr_frame + 1, next_frame
        ):
            stack.append((next_loc, next_rot, next_frame, 0))
            logger.debug(
                "Keyframed frame %d/%d (stack depth %d)",
                next_frame,
                frame_end,
                len(stack),
            )
        else:
            _delete_keyframes(bl_obj, next_frame)
            pf.ops.object.set_transform(obj, location=curr_loc, rotation_euler=curr_rot)

    return obj
