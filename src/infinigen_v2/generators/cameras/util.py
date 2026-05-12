import logging
from typing import Callable

import bpy
import numpy as np
import procfunc as pf

import infinigen_v2.generators.scenes.collision_collection as ccol
from infinigen_v2.generators.scenes.placement_utils import repeat_attempts
from infinigen_v2.util.errors import RejectedScene

logger = logging.getLogger(__name__)

AcceptPred = Callable[[pf.CameraObject, ccol.CollisionSet], bool]


def total_bbox(objects: list[pf.MeshObject]) -> tuple[np.ndarray, np.ndarray]:
    if len(objects) == 0:
        raise RejectedScene("Cannot compute camera bbox from empty object list")
    bs = []
    for obj in objects:
        logger.debug(f"{total_bbox.__name__} including object {obj.item().name}")
        bbox = pf.ops.attr.bbox_min_max(obj, global_coords=True)
        bs.append(bbox)
    all_min = np.minimum.reduce([b[0] for b in bs])
    all_max = np.maximum.reduce([b[1] for b in bs])
    return all_min, all_max


def camera_collision_check(
    camera: pf.CameraObject,
    colliders: ccol.CollisionSet,
    probe_offset: float = 0.0,
    probe_size: float = 0.75,
) -> bool:
    """Return True if camera pose is acceptable (no collision)."""
    bpy.context.view_layer.update()
    cam_obj = camera.item()
    probe_center = cam_obj.matrix_world @ pf.Vector((0, 0, probe_offset))
    probe_transform = np.array(cam_obj.matrix_world)
    probe_transform[:3, 3] = np.array(probe_center)
    if ccol.box_intersection_test(colliders, probe_transform, size=probe_size):
        logger.debug("Camera is too close to an object, rejecting")
        return False
    return True


def _propose_pose_in_bbox(
    r: pf.RNG,
    bbox: tuple,
    margin: float,
) -> tuple:
    all_min, all_max = bbox
    x = pf.random.uniform(r, all_min[0] + margin, all_max[0] - margin)
    y = pf.random.uniform(r, all_min[1] + margin, all_max[1] - margin)
    z = pf.random.clip_gaussian(r, 1.5, 0.4, all_min[2] + margin, all_max[2] - margin)
    yaw = pf.random.uniform(r, -np.pi, np.pi)
    pitch = pf.random.clip_gaussian(r, np.pi / 2, 0.3, np.pi / 4, 3 * np.pi / 4)
    roll = pf.random.clip_gaussian(r, 0.0, 0.05, -0.2, 0.2)
    return (x, y, z), (pitch, roll, yaw)


def pose_and_filter(
    r: pf.RNG,
    cam: pf.CameraObject,
    pose_distribution: Callable[[pf.RNG], tuple],
    colliders: ccol.CollisionSet | None,
    accept_pred: AcceptPred | None = None,
) -> tuple | None:
    loc, rot = pose_distribution(r)
    pf.ops.object.set_transform(cam, location=loc, rotation_euler=rot)
    if colliders is not None:
        pred = accept_pred or camera_collision_check
        if not pred(cam, colliders):
            return None
    return loc, rot


def _place_camera_in_bbox(
    rng: pf.RNG,
    cam: pf.CameraObject,
    bbox: tuple,
    colliders: ccol.CollisionSet,
    frame_start: int,
    frame_end: int,
    margin: float,
    max_tries: int,
    accept_pred: AcceptPred | None = None,
):
    def pose_distribution(r: pf.RNG) -> tuple:
        return _propose_pose_in_bbox(r, bbox, margin)

    logger.info(f"Collision set has {ccol.n_colliders(colliders)} colliders")

    n_frames = max(frame_end - frame_start + 1, 1)
    for i in range(n_frames):
        frame = frame_start + i
        result = repeat_attempts(
            pose_and_filter,
            rng,
            attempts=max_tries,
            cam=cam,
            pose_distribution=pose_distribution,
            colliders=colliders,
            accept_pred=accept_pred,
        )
        if result is None:
            raise RejectedScene(
                f"frame {frame}: could not place camera after {max_tries} tries"
            )
        cam.item().keyframe_insert("location", frame=frame)
        cam.item().keyframe_insert("rotation_euler", frame=frame)


def attach_stereo_right(
    camera_left: pf.CameraObject,
    baseline: float,
    focal_length_mm: float = 15,
) -> list[pf.CameraObject]:
    """Create a right camera parented to *camera_left* with a baseline offset."""
    camera_left.item().name = "CameraLeft"

    camera_right = pf.ops.primitives.perspective_camera(focal_length_mm=focal_length_mm)
    camera_right.item().name = "CameraRight"
    camera_right.item().parent = camera_left.item()
    camera_right.item().location = (baseline, 0, 0)

    return [camera_left, camera_right]
