# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

import logging
from typing import Callable

import numpy as np
import procfunc as pf

import infinigen2.scenes.placement.collision as ccol
from infinigen2.animations.random_walk import random_walk
from infinigen2.cameras.util import (
    AcceptPred,
    _propose_pose_in_bbox,
    camera_collision_check,
    pose_and_filter,
    total_bbox,
)
from infinigen2.scenes.placement.retry import repeat_attempts
from infinigen2.util.errors import RejectedScene

__all__ = [
    "random_walk_camera",
]

logger = logging.getLogger(__name__)


def _camera_pose_rand(
    bbox: tuple[np.ndarray, np.ndarray],
    margin: float,
    pitch_range_deg: tuple[float, float],
    roll_range_deg: tuple[float, float],
    height_range: tuple[float, float] | None,
) -> Callable[[pf.RNG], tuple]:
    bbox_min = np.asarray(bbox[0]) + margin
    bbox_max = np.asarray(bbox[1]) - margin
    if height_range is not None:
        bbox_min[2] = max(bbox_min[2], height_range[0])
        bbox_max[2] = min(bbox_max[2], height_range[1])
    pitch_range_rad = np.deg2rad(pitch_range_deg)
    roll_range_rad = np.deg2rad(roll_range_deg)

    def pose_rand(r: pf.RNG) -> tuple:
        loc, rot = _propose_pose_in_bbox(r, bbox, margin)
        loc = np.clip(loc, bbox_min, bbox_max)
        rot = list(rot)
        rot[0] = float(np.clip(rot[0], *pitch_range_rad))
        rot[1] = float(np.clip(rot[1], *roll_range_rad))
        return loc, tuple(rot)

    return pose_rand


def random_walk_camera(
    rng: pf.RNG,
    colliders: ccol.CollisionSet,
    objects: list[pf.MeshObject],
    camera: pf.CameraObject | None = None,
    frame_start: int = 1,
    frame_end: int = 1,
    focal_length_mm: float = 15,
    margin: float = 0.05,
    accept_pred: AcceptPred | None = None,
    max_tries: int = 20,
    max_retries: int = 20,
    speed_mps_range: tuple[float, float] = (1.33, 2.0),
    loc_step_range: tuple[float, float] = (1.0, 4.0),
    rot_std_deg: tuple[float, float, float] = (15.0, 15.0, 30.0),
    roll_range_deg: tuple[float, float] = (-25.0, 25.0),
    pitch_range_deg: tuple[float, float] = (45.0, 135.0),
    height_range: tuple[float, float] | None = (0.5, 2.2),
    bbox: tuple[np.ndarray, np.ndarray] | None = None,
) -> pf.CameraObject:
    if bbox is None:
        bbox = total_bbox(objects)
    pred = accept_pred or camera_collision_check
    if camera is None:
        camera = pf.ops.primitives.perspective_camera(focal_length_mm=focal_length_mm)

    pose_rand = _camera_pose_rand(
        bbox, margin, pitch_range_deg, roll_range_deg, height_range
    )

    def place_and_walk(r: pf.RNG) -> pf.CameraObject | None:
        placed = pose_and_filter(r, camera, pose_rand, colliders, accept_pred)
        if placed is None:
            return None
        return random_walk(
            r,
            camera,
            bbox,
            frame_start=frame_start,
            frame_end=frame_end,
            accept_fn=lambda: pred(camera, colliders),
            max_retries=max_retries,
            failure_mode="return",
            margin=margin,
            speed_mps_range=speed_mps_range,
            loc_step_range=loc_step_range,
            rot_std_deg=rot_std_deg,
            roll_range_deg=roll_range_deg,
            pitch_range_deg=pitch_range_deg,
            height_range=height_range,
        )

    if repeat_attempts(place_and_walk, rng, attempts=max_tries) is None:
        raise RejectedScene(
            f"Could not place random-walk camera after {max_tries} attempts"
        )
    return camera
