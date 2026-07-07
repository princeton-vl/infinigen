# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

import numpy as np
import procfunc as pf

import infinigen2.scenes.collision_collection as ccol

from .random_walk import random_walk_camera
from .rrt import rrt_camera, rrt_camera_fast
from .util import AcceptPred, _place_camera_in_bbox, attach_stereo_right, total_bbox

__all__ = [
    "stereo_camera_rig_rand",
    "stereo_cameras_in_bbox_rand",
    "stereo_random_walk_camera",
]

DEFAULT_BASELINE_RANGE = (0.03, 0.4)


def _sample_baseline(rng: pf.RNG, baseline: float | None) -> float:
    if baseline is not None:
        return baseline
    return float(pf.random.uniform(rng, *DEFAULT_BASELINE_RANGE))


@pf.tracer.generator
def stereo_camera_rig_rand(
    rng: pf.RNG,
    focal_length_mm: float = 15,
    baseline: float | None = None,
) -> list[pf.CameraObject]:
    baseline = _sample_baseline(rng, baseline)
    camera_left = pf.ops.primitives.perspective_camera(focal_length_mm=focal_length_mm)
    return attach_stereo_right(camera_left, baseline, focal_length_mm)


@pf.tracer.grammar
def stereo_cameras_in_bbox_rand(
    rng: pf.RNG,
    objects: list[pf.MeshObject],
    colliders: ccol.CollisionSet,
    frame_start: int = 1,
    frame_end: int = 1,
    margin: float = 0.05,
    max_tries: int = 100,
    focal_length_mm: float = 15,
    baseline: float | None = None,
    accept_pred: AcceptPred | None = None,
) -> list[pf.CameraObject]:
    cameras = stereo_camera_rig_rand(
        rng, focal_length_mm=focal_length_mm, baseline=baseline
    )
    _place_camera_in_bbox(
        rng,
        cameras[0],
        total_bbox(objects),
        colliders,
        frame_start,
        frame_end,
        margin,
        max_tries,
        accept_pred=accept_pred,
    )
    return cameras


def _stereo_rrt_camera(
    rng: pf.RNG,
    colliders: ccol.CollisionSet,
    objects: list[pf.MeshObject],
    focal_length_mm: float = 15,
    baseline: float | None = None,
    accept_pred: AcceptPred | None = None,
    **kwargs,
) -> list[pf.CameraObject]:
    baseline = _sample_baseline(rng, baseline)
    camera_left = rrt_camera(
        rng,
        colliders,
        objects,
        focal_length_mm=focal_length_mm,
        accept_pred=accept_pred,
        **kwargs,
    )
    return attach_stereo_right(camera_left, baseline, focal_length_mm)


def stereo_random_walk_camera(
    rng: pf.RNG,
    colliders: ccol.CollisionSet,
    objects: list[pf.MeshObject],
    frame_start: int = 1,
    frame_end: int = 1,
    focal_length_mm: float = 15,
    margin: float = 0.05,
    baseline: float | None = None,
    accept_pred: AcceptPred | None = None,
    max_retries: int = 20,
    speed_mps_range: tuple[float, float] = (2.0, 3.0),
    loc_step_range: tuple[float, float] = (1.0, 4.0),
    rot_std_deg: tuple[float, float, float] = (15.0, 15.0, 30.0),
    roll_range_deg: tuple[float, float] = (-25.0, 25.0),
    pitch_range_deg: tuple[float, float] = (45.0, 135.0),
    height_range: tuple[float, float] | None = None,
    loc_bias: np.ndarray | None = None,
    bbox: tuple[np.ndarray, np.ndarray] | None = None,
) -> list[pf.CameraObject]:
    baseline = _sample_baseline(rng, baseline)
    camera_left = random_walk_camera(
        rng,
        colliders,
        objects,
        frame_start=frame_start,
        frame_end=frame_end,
        focal_length_mm=focal_length_mm,
        margin=margin,
        accept_pred=accept_pred,
        max_retries=max_retries,
        speed_mps_range=speed_mps_range,
        loc_step_range=loc_step_range,
        rot_std_deg=rot_std_deg,
        roll_range_deg=roll_range_deg,
        pitch_range_deg=pitch_range_deg,
        height_range=height_range,
        loc_bias=loc_bias,
        bbox=bbox,
    )
    return attach_stereo_right(camera_left, baseline, focal_length_mm)


def _stereo_rrt_camera_fast(
    rng: pf.RNG,
    colliders: ccol.CollisionSet,
    objects: list[pf.MeshObject],
    focal_length_mm: float = 15,
    baseline: float | None = None,
    accept_pred: AcceptPred | None = None,
    **kwargs,
) -> list[pf.CameraObject]:
    baseline = _sample_baseline(rng, baseline)
    camera_left = rrt_camera_fast(
        rng,
        colliders,
        objects,
        focal_length_mm=focal_length_mm,
        accept_pred=accept_pred,
        **kwargs,
    )
    return attach_stereo_right(camera_left, baseline, focal_length_mm)
