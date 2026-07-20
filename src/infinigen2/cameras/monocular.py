# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick, Karhan Kayan

import numpy as np
import procfunc as pf

import infinigen2.scenes.placement.collision as ccol
from infinigen2.scenes.placement.retry import repeat_attempts
from infinigen2.util.errors import RejectedScene

from .util import (
    AcceptPred,
    _place_camera_in_bbox,
    camera_collision_check,
    total_bbox,
)

__all__ = [
    "linear_pan_camera_rand",
    "monocular_360_camera_rand",
    "monocular_camera_in_bbox_rand",
    "orbit_90_camera_rand",
]


@pf.tracer.grammar
def monocular_camera_in_bbox_rand(
    rng: pf.RNG,
    objects: list[pf.MeshObject],
    colliders: ccol.CollisionSet,
    bbox: tuple[np.ndarray, np.ndarray] | None = None,
    frame_start: int = 1,
    frame_end: int = 1,
    margin: float = 0.05,
    max_tries: int = 100,
    focal_length_mm: float = 15,
    accept_pred: AcceptPred | None = None,
) -> list[pf.CameraObject]:
    camera = pf.ops.primitives.perspective_camera(focal_length_mm=focal_length_mm)
    camera.item().name = "Camera"
    if bbox is None:
        bbox = total_bbox(objects)
    _place_camera_in_bbox(
        rng,
        camera,
        bbox,
        colliders,
        frame_start,
        frame_end,
        margin,
        max_tries,
        accept_pred=accept_pred,
    )
    return [camera]


def _uniform_in_box(r: pf.RNG, lo: pf.Vector, hi: pf.Vector) -> pf.Vector:
    """A point drawn uniformly inside the axis-aligned box [lo, hi]."""
    return pf.Vector(tuple(float(pf.random.uniform(r, lo[i], hi[i])) for i in range(3)))


def _linear_pan_attempt(
    r: pf.RNG,
    camera: pf.CameraObject,
    colliders: ccol.CollisionSet,
    box_lo: pf.Vector,
    box_hi: pf.Vector,
    max_length: float,
    frame_start: int,
    steps: int,
    forward_clearance: float,
) -> pf.CameraObject | None:
    """One linear-pan trajectory: a straight segment between two points sampled
    uniformly in the interior box, shortened to `max_length` if it would exceed
    the per-frame speed cap. Sets/checks/keyframes each pose in a single pass;
    returns None (for retry) the moment a pose fails the collision probe."""
    height_frac = float(pf.random.clip_gaussian(r, 0.5, 0.2, 0.2, 0.8))
    z = box_lo.z + height_frac * (box_hi.z - box_lo.z)
    lo_z = pf.Vector((box_lo.x, box_lo.y, z))
    hi_z = pf.Vector((box_hi.x, box_hi.y, z))
    start = _uniform_in_box(r, lo_z, hi_z)
    min_travel = min(2.0, max_length)
    end = _uniform_in_box(r, lo_z, hi_z)
    for _ in range(20):
        if (end - start).length >= min_travel:
            break
        end = _uniform_in_box(r, lo_z, hi_z)
    travel = end - start
    if travel.length > max_length:
        end = start + travel * (max_length / travel.length)

    yaw = float(pf.random.uniform(r, 0.0, 2 * np.pi))
    pitch = np.radians(float(pf.random.clip_gaussian(r, -10.0, 7.0, -20.0, 5.0)))
    rot = (np.pi / 2 + pitch, 0.0, yaw)

    for t in range(steps + 1):
        loc = start.lerp(end, t / steps)
        pf.ops.object.set_transform(camera, location=loc, rotation_euler=rot)
        if not camera_collision_check(
            camera, colliders, forward_clearance=forward_clearance
        ):
            return None
        camera.item().keyframe_insert("location", frame=frame_start + t)
        camera.item().keyframe_insert("rotation_euler", frame=frame_start + t)
    return camera


@pf.tracer.grammar
def linear_pan_camera_rand(
    rng: pf.RNG,
    objects: list[pf.MeshObject],
    colliders: ccol.CollisionSet,
    bbox: tuple[np.ndarray, np.ndarray] | None = None,
    frame_start: int = 0,
    frame_end: int = 72,
    focal_length_mm: float = 15,
    speed: float = 0.04,
    footprint_frac: float = 0.4,
    forward_clearance: float = 0.75,
    max_tries: int = 200,
) -> list[pf.CameraObject]:
    """Dolly travelling in a straight line between two points drawn uniformly in
    the room interior, at up to `speed` metres/frame, holding a random fixed yaw
    and slight downward pitch so the scene slides across the view."""
    if bbox is None:
        bbox = total_bbox(objects)
    bb_lo, bb_hi = bbox
    lo = pf.Vector(tuple(float(v) for v in bb_lo))
    hi = pf.Vector(tuple(float(v) for v in bb_hi))

    room = hi - lo
    center = (lo + hi) * 0.5
    box_lo = pf.Vector(
        (
            center.x - room.x * footprint_frac,
            center.y - room.y * footprint_frac,
            lo.z,
        )
    )
    box_hi = pf.Vector(
        (
            center.x + room.x * footprint_frac,
            center.y + room.y * footprint_frac,
            hi.z,
        )
    )
    steps = max(frame_end - frame_start, 1)
    max_length = speed * steps

    camera = pf.ops.primitives.perspective_camera(focal_length_mm=focal_length_mm)
    camera.item().name = "Camera"

    result = repeat_attempts(
        _linear_pan_attempt,
        rng,
        max_tries,
        camera=camera,
        colliders=colliders,
        box_lo=box_lo,
        box_hi=box_hi,
        max_length=max_length,
        frame_start=frame_start,
        steps=steps,
        forward_clearance=forward_clearance,
    )
    if result is None:
        raise RejectedScene(
            f"linear_pan: no collision-free trajectory after {max_tries} tries"
        )
    return [camera]


@pf.tracer.grammar
def monocular_360_camera_rand(
    objects: list[pf.MeshObject],
    camera: pf.CameraObject | None = None,
    bbox: tuple[np.ndarray, np.ndarray] | None = None,
    center: tuple[float, float] | None = None,
    radius: float | None = None,
    height: float | None = None,
    frame_start: int = 1,
    frame_end: int = 1,
    focal_length_mm: float = 15,
    total_angle_rad: float = 2 * np.pi,
) -> list[pf.CameraObject]:
    if camera is None:
        camera = pf.ops.primitives.perspective_camera(focal_length_mm=focal_length_mm)

    if bbox is None:
        bbox = total_bbox(objects)
    all_min, all_max = bbox
    dims = all_max - all_min
    if center is None:
        center = (all_min[0] + all_max[0]) / 2, (all_min[1] + all_max[1]) / 2
    if radius is None:
        radius = min(dims[0], dims[1]) * 0.4
    if height is None:
        height = (all_min[2] + all_max[2]) / 2

    n_frames = max(frame_end - frame_start + 1, 1)
    angles = np.linspace(0, total_angle_rad, n_frames, endpoint=False)
    for t, a in enumerate(angles):
        camera.item().location = (
            center[0] + radius * np.sin(a),
            center[1] - radius * np.cos(a),
            height,
        )
        camera.item().rotation_euler = (np.pi / 2, 0, a)
        camera.item().keyframe_insert("location", frame=frame_start + t)
        camera.item().keyframe_insert("rotation_euler", frame=frame_start + t)

    return [camera]


@pf.tracer.grammar
def orbit_90_camera_rand(
    objects: list[pf.MeshObject],
    bbox: tuple[np.ndarray, np.ndarray] | None = None,
    frame_start: int = 0,
    frame_end: int = 72,
    focal_length_mm: float = 15,
    height: float = 1.5,
) -> list[pf.CameraObject]:
    """Eye-level 90-degree orbit around the room centre over the full frame
    range, for spinout clips. Pair with exporter_frames to render shards (the
    orbit is deterministic, so every shard follows the identical path).

    When `bbox` is given, the orbit is computed from that interior box so
    exterior meshes (extruded wall slabs, skylight shafts) can't inflate the
    radius and push the camera through the walls; otherwise it falls back to the
    object bbox."""
    center = radius = None
    if bbox is not None:
        lo, hi = bbox
        center = ((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2)
        radius = 0.4 * min(hi[0] - lo[0], hi[1] - lo[1])
        height = min(1.5, 0.8 * (hi[2] - lo[2]))
    return monocular_360_camera_rand(
        objects=objects,
        center=center,
        radius=radius,
        height=height,
        frame_start=frame_start,
        frame_end=frame_end,
        focal_length_mm=focal_length_mm,
        total_angle_rad=np.pi / 2,
    )
