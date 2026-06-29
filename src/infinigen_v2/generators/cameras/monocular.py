import numpy as np
import procfunc as pf

import infinigen_v2.generators.scenes.collision_collection as ccol
from infinigen_v2.generators.scenes.placement_utils import repeat_attempts
from infinigen_v2.util.errors import RejectedScene

from .util import (
    AcceptPred,
    _place_camera_in_bbox,
    camera_collision_check,
    total_bbox,
)


@pf.tracer.grammar
def monocular_camera_in_bbox_distribution(
    rng: pf.RNG,
    objects: list[pf.MeshObject],
    colliders: ccol.CollisionSet,
    frame_start: int = 1,
    frame_end: int = 1,
    margin: float = 0.05,
    max_tries: int = 100,
    focal_length_mm: float = 15,
    accept_pred: AcceptPred | None = None,
) -> list[pf.CameraObject]:
    camera = pf.ops.primitives.perspective_camera(focal_length_mm=focal_length_mm)
    camera.item().name = "Camera"
    _place_camera_in_bbox(
        rng,
        camera,
        total_bbox(objects),
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
def linear_pan_camera_distribution(
    rng: pf.RNG,
    objects: list[pf.MeshObject],
    colliders: ccol.CollisionSet,
    dimensions: pf.Vector | None = None,
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
    if dimensions is not None:
        lo = pf.Vector((0.0, 0.0, 0.0))
        hi = pf.Vector((float(dimensions.x), float(dimensions.y), float(dimensions.z)))
    else:
        bb_lo, bb_hi = total_bbox(objects)
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
def monocular_360_camera_distribution(
    objects: list[pf.MeshObject],
    camera: pf.CameraObject | None = None,
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

    all_min, all_max = total_bbox(objects)
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
def orbit_90_camera_distribution(
    objects: list[pf.MeshObject],
    dimensions: pf.Vector | None = None,
    frame_start: int = 0,
    frame_end: int = 72,
    focal_length_mm: float = 15,
    height: float = 1.5,
) -> list[pf.CameraObject]:
    """Eye-level 90-degree orbit around the room centre over the full frame
    range, for spinout clips. Pair with exporter_frames to render shards (the
    orbit is deterministic, so every shard follows the identical path).

    When `dimensions` is given, the orbit is computed from the room interior so
    exterior meshes (extruded wall slabs, skylight shafts) can't inflate the
    radius and push the camera through the walls; otherwise it falls back to the
    object bbox."""
    center = radius = None
    if dimensions is not None:
        center = (dimensions.x / 2, dimensions.y / 2)
        radius = 0.4 * min(dimensions.x, dimensions.y)
        height = min(1.5, 0.8 * dimensions.z)
    return monocular_360_camera_distribution(
        objects=objects,
        center=center,
        radius=radius,
        height=height,
        frame_start=frame_start,
        frame_end=frame_end,
        focal_length_mm=focal_length_mm,
        total_angle_rad=np.pi / 2,
    )


@pf.tracer.grammar
def material_orbit_camera_distribution(
    radius: float = 4.48,
    height: float = 2.29,
    target_z: float = 0.75,
    frame_start: int = 0,
    frame_end: int = 72,
    lens_mm: float = 50,
    total_angle_rad: float = 2 * np.pi,
) -> list[pf.CameraObject]:
    """Full 360-degree orbit at fixed radius/height around the world Z axis while
    aiming at (0, 0, target_z), for material-preview turntables. Matches the
    material_sphere default camera (radius/height/lens). Geometry is fixed (not
    derived from the scene bbox) so the path is identical across materials/seeds
    and exporter-frame shards stitch seamlessly. lens_mm is deliberately not named
    focal_length_mm so the --focal_length_mm pipeline arg can't override it."""
    camera = pf.ops.primitives.perspective_camera(focal_length_mm=lens_mm)
    camera.item().name = "Camera"

    pitch = np.pi / 2 - np.arctan2(height - target_z, radius)
    n_frames = max(frame_end - frame_start + 1, 1)
    angles = np.linspace(0, total_angle_rad, n_frames, endpoint=False)
    for t, a in enumerate(angles):
        camera.item().location = (radius * np.sin(a), -radius * np.cos(a), height)
        camera.item().rotation_euler = (pitch, 0, a)
        camera.item().keyframe_insert("location", frame=frame_start + t)
        camera.item().keyframe_insert("rotation_euler", frame=frame_start + t)
    return [camera]
