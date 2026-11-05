# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

import logging
from collections.abc import Sequence
from typing import Callable, NamedTuple, Protocol, TypeVar, runtime_checkable

import numpy as np
import procfunc as pf
import shapely

from infinigen2.objects import sofa, storage, table
from infinigen2.scenes.placement import collision as ccol
from infinigen2.scenes.placement.snap import snap_to_plane

__all__ = [
    "BareMeshResult",
    "MeshResult",
    "back_face_grounded",
    "bbox_face_grounded",
    "center_inside",
    "clear_of",
    "inset_floor_point_rand",
    "jitter_object_rotation_rand",
    "random_bbox_poses_animation_rand",
    "retry_place",
    "side_table_object_rand",
    "snap_back_front",
    "snap_on_top",
    "snap_side_by_side",
    "snap_to_wall",
    "sofa_object_rand",
    "standalone_wall_planes",
    "storage_object_rand",
]

logger = logging.getLogger(__name__)

MR = TypeVar("MR", bound="MeshResult")


@runtime_checkable
class MeshResult(Protocol):
    @property
    def mesh(self) -> pf.MeshObject: ...


class BareMeshResult(NamedTuple):
    mesh: pf.MeshObject


def _yaw_about_point(
    location: tuple[float, float, float],
    center: np.ndarray,
    yaw: float,
) -> tuple[float, float, float]:
    """Rotate `location` by `yaw` about the vertical axis through `center`."""
    rot = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    x, y = rot @ (np.asarray(location[:2]) - center[:2]) + center[:2]
    return (x, y, location[2])


def jitter_object_rotation_rand(
    rng: pf.RNG,
    obj: pf.MeshObject,
    max_angle: float,
    colliders: ccol.CollisionSet,
    attempts: int = 12,
) -> None:
    """Retry yaw against every other object in `colliders`."""
    item = obj.item()
    location = tuple(item.location)
    rotation = tuple(item.rotation_euler)
    bbox_min, bbox_max = pf.ops.attr.bbox_min_max(obj, global_coords=True)
    center = (np.asarray(bbox_min) + np.asarray(bbox_max)) / 2
    other_objs = [other for other in colliders.objs if other.item() is not item]
    other_colliders = ccol.collision_set(other_objs, cache=colliders)
    for attempt_rng in rng.spawn(attempts):
        yaw = pf.random.uniform(attempt_rng, -max_angle, max_angle)
        pf.ops.object.set_transform(
            obj,
            location=_yaw_about_point(location, center, yaw),
            rotation_euler=(rotation[0], rotation[1], rotation[2] + yaw),
        )
        if not ccol.intersection_test(other_colliders, obj):
            return
    pf.ops.object.set_transform(obj, location=location, rotation_euler=rotation)


@pf.tracer.grammar
def random_bbox_poses_animation_rand(
    camera: pf.CameraObject,
    rng: pf.RNG,
    room_dimensions: pf.Vector,
    frame_start: int = 1,
    frame_end: int = 1,
    wall_margin: float = 0.5,
    ceil_margin: float = 0.3,
    floor_margin: float = 0.3,
) -> None:
    cam = camera.item()
    n_frames = max(frame_end - frame_start + 1, 1)
    frame_rngs = rng.spawn(n_frames)
    for i in range(n_frames):
        frame = frame_start + i
        r = frame_rngs[i]
        x = pf.random.uniform(r, wall_margin, room_dimensions.x - wall_margin)
        y = pf.random.uniform(r, wall_margin, room_dimensions.y - wall_margin)
        z = pf.random.clip_gaussian(
            r, 1.5, 0.4, floor_margin, room_dimensions.z - ceil_margin
        )
        yaw = pf.random.uniform(r, -np.pi, np.pi)
        pitch = pf.random.clip_gaussian(r, np.pi / 2, 0.3, np.pi / 4, 3 * np.pi / 4)
        roll = pf.random.clip_gaussian(r, 0.0, 0.05, -0.2, 0.2)
        cam.location = (x, y, z)
        cam.rotation_euler = (pitch, roll, yaw)
        cam.keyframe_insert("location", frame=frame)
        cam.keyframe_insert("rotation_euler", frame=frame)


@pf.tracer.grammar
def side_table_object_rand(rng: pf.RNG) -> MeshResult:
    rng_choice, rng_asset = rng.spawn(2)
    func = pf.control.choice(
        rng_choice,
        [
            (table.table_side_rand, 1.0),
            (table.table_side_circular_rand, 1.0),
            (storage.storage_table_side_rand, 1.0),
        ],
    )
    result = func(rng_asset)
    return result


def sofa_object_rand(rng: pf.RNG) -> sofa.SofaResult:
    rng_choice, rng_asset = rng.spawn(2)
    func = pf.control.choice(
        rng_choice,
        [
            (sofa.sofa_rand, 1.0),
            (sofa.sofa_with_base_rand, 1.0),
        ],
    )
    result = func(rng_asset)
    return result


@pf.tracer.grammar
def storage_object_rand(rng: pf.RNG) -> MeshResult:
    result = storage.storage_rand(rng)
    return result


def standalone_wall_planes(
    length: float = 15.0,
    height: float = 3.0,
) -> list[pf.MeshObject]:
    grid = pf.nodes.geo.mesh_grid(vertices_x=2, vertices_y=2)
    position = pf.nodes.geo.input_position()
    u = (position.x + 0.5) * length
    v = (position.y + 0.5) * height
    geometry = pf.nodes.geo.set_position(
        grid.mesh,
        position=pf.nodes.math.combine_xyz(y=u, z=v),
    )
    wall = pf.nodes.to_mesh_object(geometry)
    wall.item().name = standalone_wall_planes.__name__
    return [wall]


def center_inside(region: shapely.Polygon, obj: pf.MeshObject) -> bool:
    lower, upper = pf.ops.attr.bbox_min_max(obj, global_coords=True)
    center = shapely.Point((lower[0] + upper[0]) / 2, (lower[1] + upper[1]) / 2)
    return region.contains(center)


def _footprint(obj: pf.MeshObject) -> shapely.Polygon:
    lower, upper = pf.ops.attr.bbox_min_max(obj, global_coords=True)
    return shapely.box(lower[0], lower[1], upper[0], upper[1])


def clear_of(
    obj: pf.MeshObject, others: Sequence[pf.MeshObject], distance: float
) -> bool:
    footprint = _footprint(obj)
    return all(footprint.distance(_footprint(o)) >= distance for o in others)


def retry_place(
    rng: pf.RNG,
    child: MR,
    colliders: ccol.CollisionSet,
    place_fn: Callable[..., None],
    attempts: int = 7,
    accept_fn: Callable[[pf.MeshObject], bool] | None = None,
    **kwargs,
) -> MR | None:
    """Re-pose `child` with `place_fn` until it clears the existing `colliders`
    (and satisfies `accept_fn`, if given), or give up and return None. Reads
    `colliders` but never extends it; keep_non_colliding owns updates and
    intra-batch collisions.
    """
    for r in rng.spawn(attempts):
        place_fn(r, child, **kwargs)
        if ccol.intersection_test(colliders, child.mesh):
            continue
        if accept_fn is not None and not accept_fn(child.mesh):
            continue
        return child
    child.mesh.item().name = child.mesh.item().name + "_FAILED_PLACEMENT"
    return None


def snap_to_wall(
    rng: pf.RNG,
    child: MR,
    parents: list[pf.MeshObject],
    placement: float | None = None,
    margin: float | None = None,
    child_side: str | None = None,
) -> None:
    rng, rng_side = rng.spawn(2)
    if placement is None:
        placement = pf.random.uniform(rng, 0.1, 0.9)
    if margin is None:
        margin = pf.random.clip_gaussian(rng, 0.15, 0.1, 0.1, 0.4)
    if child_side is None:
        child_side = pf.control.choice(rng_side, [("back", 1.0), ("left", 1.0)])
    snap_to_plane(
        child=child.mesh,
        parent=rng.choice(parents),
        placement=placement,
        margin=margin,
        child_side=child_side,
        parent_side="front",
    )


def inset_floor_point_rand(
    rng: pf.RNG,
    room_dimensions: pf.Vector,
    half_size: tuple[float, float],
    margin: float = 0.0,
) -> tuple[float, float]:
    """A random floor point where a footprint of `half_size` stays `margin` inside
    the room, falling back to the room centre on an axis too small for it."""
    slack = [
        max(0.0, room_dimensions[i] / 2.0 - margin - half_size[i]) for i in range(2)
    ]
    x = room_dimensions.x / 2.0 + pf.random.uniform(rng, -slack[0], slack[0])
    y = room_dimensions.y / 2.0 + pf.random.uniform(rng, -slack[1], slack[1])
    return x, y


def snap_back_front(
    rng: pf.RNG,
    child: MR,
    parents: list[pf.MeshObject],
    placement: float | None = None,
    margin: float | None = None,
) -> None:
    snap_to_wall(
        rng, child, parents, placement=placement, margin=margin, child_side="back"
    )


def bbox_face_grounded(
    obj: pf.MeshObject,
    colliders: ccol.CollisionSet,
    side: str,
    max_distance: float,
) -> bool:
    bmin, bmax = pf.ops.attr.bbox_min_max(obj, global_coords=False)
    axis, sign = {
        "back": (0, -1.0),
        "front": (0, 1.0),
        "left": (1, -1.0),
        "right": (1, 1.0),
        "bottom": (2, -1.0),
    }[side]
    face_axes = [index for index in range(3) if index != axis]
    center = (np.asarray(bmin) + np.asarray(bmax)) / 2.0
    samples_local = np.tile(center, (5, 1))
    samples_local[:, axis] = bmin[axis] if sign < 0 else bmax[axis]
    samples_local[:4, face_axes[0]] = [
        bmin[face_axes[0]],
        bmin[face_axes[0]],
        bmax[face_axes[0]],
        bmax[face_axes[0]],
    ]
    samples_local[:4, face_axes[1]] = [
        bmin[face_axes[1]],
        bmax[face_axes[1]],
        bmin[face_axes[1]],
        bmax[face_axes[1]],
    ]
    mw = np.array(obj.item().matrix_world)
    samples_world = samples_local @ mw[:3, :3].T + mw[:3, 3]
    face_normal_local = np.zeros(3)
    face_normal_local[axis] = sign
    face_normal = mw[:3, :3] @ face_normal_local
    face_normal = face_normal / np.linalg.norm(face_normal)

    hits, ray_idx, _ = ccol.raycast(
        colliders, samples_world, np.tile(face_normal, (len(samples_world), 1))
    )
    grounded = np.zeros(len(samples_world), dtype=bool)
    for loc, ri in zip(hits, ray_idx, strict=False):
        if np.linalg.norm(loc - samples_world[ri]) <= max_distance:
            grounded[ri] = True
    return bool(grounded.all())


def back_face_grounded(
    obj: pf.MeshObject,
    colliders: ccol.CollisionSet,
    margin: float,
    eps: float = 0.5,
    side: str = "back",
) -> bool:
    return bbox_face_grounded(
        obj,
        colliders,
        side=side,
        max_distance=(1.0 + eps) * margin,
    )


def snap_side_by_side(rng: pf.RNG, child: MR, parents: list[pf.MeshObject]) -> None:
    rng, rng_sides = rng.spawn(2)
    sides = pf.control.choice(
        rng_sides, [(("left", "right"), 0.5), (("right", "left"), 0.5)]
    )
    snap_to_plane(
        child.mesh,
        parent=rng.choice(parents),
        placement=rng.uniform(0.05, 0.95),
        margin=pf.random.clip_gaussian(rng, 0.07, 0.06, 0.02, 0.25),
        child_side=sides[0],
        parent_side=sides[1],
    )


def snap_on_top(
    rng: pf.RNG,
    child: MR,
    parents: list[pf.MeshObject],
    xy_frac: tuple[float, float] = (0.5, 0.5),
) -> None:
    parent = rng.choice(list(parents))
    bbox_min, bbox_max = pf.ops.attr.bbox_min_max(parent, global_coords=True)
    child.mesh.item().location = (
        bbox_min[0] + (bbox_max[0] - bbox_min[0]) * xy_frac[0],
        bbox_min[1] + (bbox_max[1] - bbox_min[1]) * xy_frac[1],
        bbox_max[2] + 0.002,
    )
