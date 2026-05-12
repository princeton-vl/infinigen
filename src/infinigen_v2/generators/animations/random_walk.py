import logging
from dataclasses import dataclass
from typing import Callable

import bpy
import numpy as np
import procfunc as pf

from infinigen_v2.util.errors import RejectedScene

logger = logging.getLogger(__name__)


@dataclass
class RandomWalkSampler:
    bbox: tuple[np.ndarray, np.ndarray]
    margin: float = 0.05
    speed_mps_range: tuple[float, float] = (2.0, 3.0)
    loc_step_range: tuple[float, float] = (1.0, 4.0)
    rot_std_deg: tuple[float, float, float] = (15.0, 15.0, 30.0)
    roll_range_deg: tuple[float, float] = (-25.0, 25.0)
    pitch_range_deg: tuple[float, float] = (45.0, 135.0)
    height_range: tuple[float, float] | None = None
    loc_bias: np.ndarray | None = None

    def __post_init__(self):
        self._bbox_min = np.asarray(self.bbox[0]) + self.margin
        self._bbox_max = np.asarray(self.bbox[1]) - self.margin
        self._roll_range_rad = np.deg2rad(self.roll_range_deg)
        self._pitch_range_rad = np.deg2rad(self.pitch_range_deg)
        if self.height_range is not None:
            self._bbox_min[2] = max(self._bbox_min[2], self.height_range[0])
            self._bbox_max[2] = min(self._bbox_max[2], self.height_range[1])

    def __call__(
        self,
        rng: pf.RNG,
        curr_loc: np.ndarray,
        curr_rot: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Return (next_loc, next_rot, duration_seconds)."""
        dist = float(rng.uniform(*self.loc_step_range))
        direction = rng.normal(0.0, np.ones(3))
        if self.loc_bias is not None:
            direction = direction + self.loc_bias
        direction = direction / (np.linalg.norm(direction) + 1e-8)
        next_loc = np.clip(curr_loc + direction * dist, self._bbox_min, self._bbox_max)

        rot_jitter = np.deg2rad(rng.normal(0.0, np.asarray(self.rot_std_deg)))
        next_rot = curr_rot + rot_jitter
        next_rot[0] = np.clip(next_rot[0], *self._pitch_range_rad)
        next_rot[1] = np.clip(next_rot[1], *self._roll_range_rad)

        speed = float(rng.uniform(*self.speed_mps_range))
        duration = dist / max(speed, 1e-4)
        return next_loc, next_rot, duration


def _delete_keyframes(obj: bpy.types.Object, frame: int) -> None:
    if obj.animation_data is None or obj.animation_data.action is None:
        return
    for fc in obj.animation_data.action.fcurves:
        if fc.data_path:
            obj.keyframe_delete(data_path=fc.data_path, frame=frame)


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
    sampler: RandomWalkSampler,
    accept_fn: Callable[[], bool],
    frame_start: int,
    frame_end: int,
    max_retries: int = 20,
    failure_mode: str = "error",
) -> bool:
    """Animate *obj* along a random walk from *frame_start* to *frame_end*.

    The object must already be positioned at a valid initial pose.
    Returns True if a complete path was found.
    """
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

    def _fail(msg: str) -> bool:
        if failure_mode == "error":
            raise RejectedScene(msg)
        if failure_mode == "warn":
            logger.warning(msg)
        return False

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

        if _validate_trajectory(bl_obj, accept_fn, curr_frame + 1, next_frame):
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

    return True
