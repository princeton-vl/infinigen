import logging

import numpy as np
import procfunc as pf

import infinigen_v2.generators.scenes.collision_collection as ccol
from infinigen_v2.generators.animations.random_walk import RandomWalkSampler, walk_loop
from infinigen_v2.generators.cameras.util import (
    AcceptPred,
    _propose_pose_in_bbox,
    camera_collision_check,
    total_bbox,
)
from infinigen_v2.util.errors import RejectedScene

logger = logging.getLogger(__name__)


def _init_camera_pose(
    rng: pf.RNG,
    camera: pf.CameraObject,
    colliders: ccol.CollisionSet,
    sampler: RandomWalkSampler,
    pred: AcceptPred,
    max_retries: int,
    bbox: tuple[np.ndarray, np.ndarray],
    margin: float,
) -> None:
    for _ in range(max_retries):
        loc, rot = _propose_pose_in_bbox(rng, bbox, margin)
        loc = np.clip(loc, sampler._bbox_min, sampler._bbox_max)
        rot = list(rot)
        rot[0] = float(np.clip(rot[0], *sampler._pitch_range_rad))
        rot[1] = float(np.clip(rot[1], *sampler._roll_range_rad))
        rot = tuple(rot)
        pf.ops.object.set_transform(camera, location=loc, rotation_euler=rot)
        if pred(camera, colliders):
            return
    raise RejectedScene("Could not find valid initial camera pose")


def random_walk_camera(
    rng: pf.RNG,
    colliders: ccol.CollisionSet,
    objects: list[pf.MeshObject],
    frame_start: int = 1,
    frame_end: int = 1,
    focal_length_mm: float = 15,
    margin: float = 0.05,
    accept_pred: AcceptPred | None = None,
    max_retries: int = 20,
    speed_mps_range: tuple[float, float] = (1.33, 2.0),
    loc_step_range: tuple[float, float] = (1.0, 4.0),
    rot_std_deg: tuple[float, float, float] = (15.0, 15.0, 30.0),
    roll_range_deg: tuple[float, float] = (-25.0, 25.0),
    pitch_range_deg: tuple[float, float] = (45.0, 135.0),
    height_range: tuple[float, float] | None = (0.5, 2.2),
    loc_bias: np.ndarray | None = None,
    bbox: tuple[np.ndarray, np.ndarray] | None = None,
) -> pf.CameraObject:
    if bbox is None:
        bbox = total_bbox(objects)
    pred = accept_pred or camera_collision_check
    sampler = RandomWalkSampler(
        bbox=bbox,
        margin=margin,
        speed_mps_range=speed_mps_range,
        loc_step_range=loc_step_range,
        rot_std_deg=rot_std_deg,
        roll_range_deg=roll_range_deg,
        pitch_range_deg=pitch_range_deg,
        height_range=height_range,
        loc_bias=loc_bias,
    )

    camera = pf.ops.primitives.perspective_camera(focal_length_mm=focal_length_mm)
    _init_camera_pose(rng, camera, colliders, sampler, pred, max_retries, bbox, margin)
    walk_loop(
        rng=rng,
        obj=camera,
        sampler=sampler,
        accept_fn=lambda: pred(camera, colliders),
        frame_start=frame_start,
        frame_end=frame_end,
        max_retries=max_retries,
    )
    return camera
