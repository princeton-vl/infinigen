# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

import logging
import math
from typing import Callable, NamedTuple, Protocol, TypeVar, runtime_checkable

import numpy as np
import procfunc as pf
from procfunc.nodes import types as t

from infinigen2.objects import (
    bookcase,
    chair,
    lamp,
    rug,
    sofa,
    storage,
    table,
    triangle_shelf,
    vase,
)
from infinigen2.scenes.placement import collision as ccol
from infinigen2.scenes.placement.culling import (
    keep_non_colliding,
    keep_unobstructed,
    place_surrounding,
)
from infinigen2.scenes.placement.snap import snap_to_plane

__all__ = [
    "CenteredSofaSetupResult",
    "DiningSetupResult",
    "DiningTableSetupResult",
    "MeshResult",
    "SofaSetupResult",
    "WallSofaSetupResult",
    "arrange_dining_chairs",
    "back_face_grounded",
    "centered_sofa_setup_rand",
    "dining_setup_rand",
    "dining_table_setup_rand",
    "random_bbox_poses_animation_rand",
    "retry_place",
    "side_table_object_rand",
    "snap_back_front",
    "snap_on_top",
    "sofa_lamps_rand",
    "sofa_setup_rand",
    "storage_object_rand",
    "table_decoration_object_rand",
    "wall_sofa_setup_rand",
]

logger = logging.getLogger(__name__)

MR = TypeVar("MR", bound="MeshResult")


@runtime_checkable
class MeshResult(Protocol):
    @property
    def mesh(self) -> pf.MeshObject: ...


class _BareMeshResult(NamedTuple):
    mesh: pf.MeshObject


class DiningSetupResult(NamedTuple):
    dining_table: pf.MeshObject
    chairs: list[pf.MeshObject]
    all_objects: list[pf.MeshObject]


class WallSofaSetupResult(NamedTuple):
    sofas: list[MeshResult]
    coffee_tables: list[MeshResult]
    rugs: list[pf.MeshObject]
    all_objects: list[pf.MeshObject]


class CenteredSofaSetupResult(NamedTuple):
    sofas: list[MeshResult]
    coffee_tables: list[MeshResult]
    rugs: list[pf.MeshObject]
    all_objects: list[pf.MeshObject]


class SofaSetupResult(NamedTuple):
    sofas: list[MeshResult]
    coffee_tables: list[MeshResult]
    side_tables: list[MeshResult]
    rugs: list[pf.MeshObject]
    all_objects: list[pf.MeshObject]


class DiningTableSetupResult(NamedTuple):
    dining_tables: list[MeshResult]
    dining_chairs: list[MeshResult]
    rugs: list[pf.MeshObject]
    all_objects: list[pf.MeshObject]


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
def table_decoration_object_rand(
    rng: pf.RNG,
) -> MeshResult:
    func = pf.control.choice(
        rng,
        [
            (vase.vase_rand, 1.0),
        ],
    )
    result = func(rng)
    result.mesh.item().name = func.__name__
    return result


@pf.tracer.grammar
def side_table_object_rand(rng: pf.RNG) -> MeshResult:
    def triangle_shelf_sidetable_rand(
        rng: pf.RNG,
    ) -> triangle_shelf.TriangleShelfResult:
        dimensions = table.side_table_dimensions_rand(rng)
        return triangle_shelf.triangle_shelf_rand(rng, dimensions=dimensions)

    func = pf.control.choice(
        rng,
        [
            (table.side_table_rand, 1.0),
            (triangle_shelf_sidetable_rand, 100.0),
        ],
    )
    result = func(rng)
    result.mesh.item().name = func.__name__
    return result


@pf.tracer.grammar
def storage_object_rand(rng: pf.RNG) -> MeshResult:
    func = pf.control.choice(
        rng,
        [
            (bookcase.bookcase_rand, 1.0),
            (storage.shelves_rand, 1.0),
        ],
    )
    result = func(rng)
    result.mesh.item().name = func.__name__
    return result


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


def _snap_to_wall(
    rng: pf.RNG,
    child: MR,
    parents: list[pf.MeshObject],
    placement: float | None = None,
    margin: float | None = None,
    child_side: str | None = None,
) -> None:
    if placement is None:
        placement = pf.random.uniform(rng, 0.1, 0.9)
    if margin is None:
        margin = pf.random.clip_gaussian(rng, 0.15, 0.1, 0.1, 0.4)
    if child_side is None:
        child_side = pf.control.choice(rng, [("back", 1.0), ("left", 1.0)])
    snap_to_plane(
        child=child.mesh,
        parent=rng.choice(parents),
        placement=placement,
        margin=margin,
        child_side=child_side,
        parent_side="front",
    )


def snap_back_front(
    rng: pf.RNG,
    child: MR,
    parents: list[pf.MeshObject],
    placement: float | None = None,
    margin: float | None = None,
) -> None:
    _snap_to_wall(
        rng, child, parents, placement=placement, margin=margin, child_side="back"
    )


def back_face_grounded(
    obj: pf.MeshObject,
    colliders: ccol.CollisionSet,
    margin: float,
    eps: float = 0.5,
) -> bool:
    """True iff every corner of `obj`'s back (-X local) face has a collider
    directly behind it within (1 + eps) * margin. Rejects placements where part
    of the back overhangs a window/door alcove (no wall behind that corner).
    """
    bmin, bmax = pf.ops.attr.bbox_min_max(obj, global_coords=False)
    corners_local = np.array(
        [
            [bmin[0], bmin[1], bmin[2]],
            [bmin[0], bmin[1], bmax[2]],
            [bmin[0], bmax[1], bmin[2]],
            [bmin[0], bmax[1], bmax[2]],
        ]
    )
    mw = np.array(obj.item().matrix_world)
    corners_world = corners_local @ mw[:3, :3].T + mw[:3, 3]
    back_normal = mw[:3, :3] @ np.array([-1.0, 0.0, 0.0])
    back_normal = back_normal / np.linalg.norm(back_normal)

    hits, ray_idx, _ = ccol.raycast(
        colliders, corners_world, np.tile(back_normal, (4, 1))
    )
    threshold = (1.0 + eps) * margin
    grounded = np.zeros(4, dtype=bool)
    for loc, ri in zip(hits, ray_idx, strict=False):
        if np.linalg.norm(loc - corners_world[ri]) <= threshold:
            grounded[ri] = True
    return bool(grounded.all())


def _snap_side_by_side(rng: pf.RNG, child: MR, parents: list[pf.MeshObject]) -> None:
    sides = pf.control.choice(rng, [(("left", "right"), 0.5), (("right", "left"), 0.5)])
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


def _world_vert_bbox(obj: pf.MeshObject) -> tuple[np.ndarray, np.ndarray]:
    bmin, bmax = pf.ops.attr.bbox_min_max(obj, global_coords=True)
    return np.array(bmin), np.array(bmax)


def _placed_rug(
    rng: pf.RNG,
    room_dimensions: pf.Vector,
    wall_clearance: float = 0.3,
) -> list[pf.MeshObject]:
    # a room too tight for the clearance shrinks the rug rather than inverting its range
    avail_x = max(0.5, room_dimensions.x - 2 * wall_clearance)
    avail_y = max(0.5, room_dimensions.y - 2 * wall_clearance)
    length = pf.random.uniform(rng, min(1.0, avail_x), avail_x)
    width = pf.random.uniform(rng, min(1.0, avail_y), avail_y)
    thickness = pf.random.uniform(rng, 0.01, 0.02)
    rug_result = rug.rug_rand(rng, dimensions=pf.Vector((length, width, thickness)))
    rug_result.mesh.item().name = rug.rug_rand.__name__
    # centred slack, so a rug filling the whole span gives an exactly zero-width range
    slack_x = (avail_x - length) / 2
    slack_y = (avail_y - width) / 2
    cx = pf.random.uniform(
        rng, room_dimensions.x / 2 - slack_x, room_dimensions.x / 2 + slack_x
    )
    cy = pf.random.uniform(
        rng, room_dimensions.y / 2 - slack_y, room_dimensions.y / 2 + slack_y
    )
    pf.ops.object.set_transform(rug_result.mesh, location=(cx, cy, 0.001))
    return [rug_result.mesh]


def _maybe_rug(
    rng: pf.RNG,
    room_dimensions: pf.Vector,
    rug_weight: float = 1.0,
) -> list[pf.MeshObject]:
    func = pf.control.choice(
        rng,
        [
            (_placed_rug, rug_weight),
            (lambda *_, **__: [], 1.0),
        ],
    )
    return func(rng, room_dimensions=room_dimensions)


def _snap_facing_carpet(
    rng: pf.RNG,
    child: MR,
    carpet: pf.MeshObject,
    parent_side: str,
) -> None:
    snap_to_plane(
        child=child.mesh,
        parent=carpet,
        parent_side=parent_side,
        child_side="front",
        margin=pf.random.uniform(rng, -0.1, 0.5),
        placement=pf.random.uniform(rng, 0.35, 0.65),
        constraint_axis=pf.Vector((0, 0, 1)),
    )


def _place_on_floor(rng: pf.RNG, child: MR, room_dimensions: pf.Vector) -> None:
    bmin, _ = pf.ops.attr.bbox_min_max(child.mesh, global_coords=False)
    child.mesh.item().location = (
        pf.random.uniform(rng, 0.2, 0.8) * room_dimensions.x,
        pf.random.uniform(rng, 0.4, 0.6) * room_dimensions.y,
        0.001 - bmin[2],
    )


def wall_sofa_setup_rand(
    rng: pf.RNG,
    wall_planes: list[pf.MeshObject],
    room_dimensions: pf.Vector,
    colliders: ccol.CollisionSet | None = None,
) -> WallSofaSetupResult:
    """Sofas snapped against the walls, plus an optional rug. Needs posed wall
    planes, so it is not registered as a standalone Scene."""
    if colliders is None:
        colliders = ccol.collision_set(wall_planes)
    n = pf.random.randint(rng, 0, 8)
    rngs = rng.spawn(n)
    sofas = [sofa.sofa_rand(rngs[i]) for i in range(n)]
    placed_sofas = []
    for i in range(n):
        sofa_obj = retry_place(
            rngs[i], sofas[i], colliders, snap_back_front, parents=wall_planes
        )
        placed_sofas.append(sofa_obj)
    sofas = placed_sofas
    sofa_objs, colliders = keep_non_colliding(sofas, colliders)
    logger.info(f"Placed {len(sofa_objs)} wall sofas out of {n} attempts")
    rug_objs = _maybe_rug(rng, room_dimensions)
    all_objects = [r.mesh for r in sofa_objs] + rug_objs
    return WallSofaSetupResult(sofa_objs, [], rug_objs, all_objects)


def centered_sofa_setup_rand(
    rng: pf.RNG,
    room_dimensions: pf.Vector | None = None,
    colliders: ccol.CollisionSet | None = None,
    wall_planes: list[pf.MeshObject] | None = None,
) -> CenteredSofaSetupResult:
    """Sofas ringing a central rug facing inward, with an optional center coffee
    table. Works without walls, so it is registered as a standalone Scene;
    `wall_planes` is accepted only for signature parity with the wall variant."""
    del wall_planes
    if room_dimensions is None:
        room_dimensions = pf.Vector((5.0, 5.0, 3.0))
    if colliders is None:
        colliders = ccol.collision_set([])
    rug_objs = _placed_rug(rng, room_dimensions=room_dimensions, wall_clearance=1.2)
    carpet = rug_objs[0]
    cmin, cmax = _world_vert_bbox(carpet)
    center = (cmin + cmax) / 2

    # one sofa per chosen rug side, snapped to that side facing inward
    side_names = ["right", "left", "front", "back"]
    n_sides = pf.random.randint(rng, 2, 5)
    sides = [
        side_names[int(i)]
        for i in rng.choice(len(side_names), size=n_sides, replace=False)
    ]
    n = len(sides)
    rngs = rng.spawn(n)
    sofas = [sofa.sofa_rand(rngs[i]) for i in range(n)]
    placed_sofas = []
    for i in range(n):
        sofa_obj = retry_place(
            rngs[i],
            sofas[i],
            colliders,
            _snap_facing_carpet,
            carpet=carpet,
            parent_side=sides[i],
        )
        placed_sofas.append(sofa_obj)
    sofas = placed_sofas
    sofas = keep_unobstructed(sofas, center, colliders)
    sofa_objs, colliders = keep_non_colliding(sofas, colliders)
    logger.info(f"Placed {len(sofa_objs)} carpet sofas out of {n} attempts")

    def _place_coffee(rng: pf.RNG) -> list[MeshResult]:
        child = table.coffee_table_rand(rng)
        child.mesh.item().location = (
            center[0] + pf.random.uniform(rng, -0.2, 0.2),
            center[1] + pf.random.uniform(rng, -0.2, 0.2),
            0.01,
        )
        return [child]

    coffee_fn = pf.control.choice(
        rng,
        [(_place_coffee, 1.0), (lambda *_, **__: [], 1.0)],
    )
    center_coffee = coffee_fn(rng)
    center_coffee, colliders = keep_non_colliding(center_coffee, colliders)

    # sofas already ringed the carpet above; drop the rug ~1/3 of the time
    out_rugs = pf.control.choice(rng, [(rug_objs, 2.0), ([], 1.0)])

    all_objects = [r.mesh for r in sofa_objs + center_coffee] + out_rugs
    return CenteredSofaSetupResult(sofa_objs, center_coffee, out_rugs, all_objects)


def _place_in_free_floorspace(
    rng: pf.RNG,
    child: MR,
    room_dimensions: pf.Vector,
    colliders: ccol.CollisionSet,
    clearance: float = 2.0,
    attempts: int = 7,
) -> MR | None:
    """Place `child` at a random floor location whose `clearance`x-footprint box
    (at the child's own height) clears all existing colliders. Returns the placed
    child, or None if no clear spot is found.
    """

    def footprint_clears(mesh: pf.MeshObject) -> bool:
        lo, hi = (
            np.array(b) for b in pf.ops.attr.bbox_min_max(mesh, global_coords=True)
        )
        ext = hi - lo
        transform = np.eye(4)
        transform[:3, 3] = (lo + hi) / 2
        return not ccol.box_intersection_test(
            colliders, transform, [clearance * ext[0], clearance * ext[1], ext[2]]
        )

    return retry_place(
        rng,
        child,
        colliders,
        _place_on_floor,
        attempts=attempts,
        accept_fn=footprint_clears,
        room_dimensions=room_dimensions,
    )


@pf.nodes.node_function
def _chair_row(
    chair_geo: pf.ProcNode[pf.MeshObject],
    count: t.SocketOrVal[int],
    dist: t.SocketOrVal[float],
    span: t.SocketOrVal[float],
    theta: t.SocketOrVal[float] = 0.0,
    seed: t.SocketOrVal[int] = 0,
    rot_jitter: t.SocketOrVal[float] = 0.0,
) -> pf.ProcNode[t.Instances]:
    """One row of `count` chairs centered over `span` along +Y at local +X=dist,
    each facing -X (toward the origin) with a small random z jitter, then the whole
    row rotated by `theta` about Z to reach any of the table's four sides."""
    count_f = count.astype(dtype=float)
    is_multi = pf.nodes.func.greater_than(a=count_f, b=1.0).astype(dtype=float)
    start = span * -0.5 * is_multi
    step = span / pf.nodes.math.maximum(count_f - 1.0, 1.0) * is_multi
    line = pf.nodes.geo.mesh_line(
        count=count,
        start_location=pf.nodes.math.combine_xyz(x=dist, y=start),
        offset=pf.nodes.math.combine_xyz(y=step),
    )
    jitter = pf.nodes.func.random_value(min=-1.0, max=1.0, seed=seed) * rot_jitter
    row = pf.nodes.geo.instance_on_points(
        points=line,
        instance=chair_geo,
        rotation=pf.nodes.math.combine_xyz(z=math.pi + jitter).astype(dtype=pf.Euler),
    )
    return pf.nodes.geo.transform(
        row, rotation=pf.nodes.math.combine_xyz(z=theta).astype(dtype=pf.Euler)
    )


def arrange_dining_chairs(
    chair_obj: pf.MeshObject,
    dining_table: pf.MeshObject,
    chair_spacing: float,
    tuck: float,
    edge_margin: float,
    rot_jitter: float,
    include_ends: bool = True,
) -> list[pf.MeshObject]:
    """Instance `chair_obj` around the sides of `dining_table`, facing inward, posed
    by the table's current transform, and realize the instances into one object per
    chair. Table and chair footprints are measured from bounding boxes inside the
    nodegraph. `chair_spacing` is the gap between adjacent chairs, `tuck` the meters
    the chair front is pulled in past the table edge (negative leaves a gap),
    `edge_margin` the clearance kept at each table corner so chairs never overhang
    the ends, `rot_jitter` the random per-chair z rotation (radians). `include_ends`
    toggles the chairs on the +/-Y ends."""
    # TODO replace the four per-side rows with one geonode pass: split the table bbox
    # into disconnected edge curves, resample each by length for spacing/count, then
    # instance chairs facing inward -- drops most of these args. Needs visual iteration.
    collection = pf.types.Collection([chair_obj], name="dining_chair")
    chair_geo = pf.nodes.geo.collection_info(collection, separate_children=True)

    table_info = pf.nodes.geo.object_info(dining_table)
    tbox = pf.nodes.geo.bound_box(table_info.geometry)
    tsize = pf.nodes.math.separate_xyz(tbox.max - tbox.min)
    # the bound_box min/max sockets ignore instances, so realize the chair first
    cbox = pf.nodes.geo.bound_box(pf.nodes.geo.realize_instances(chair_geo))
    csize = pf.nodes.math.separate_xyz(cbox.max - cbox.min)
    chair_depth, chair_width = csize.x, csize.y

    dist_x = tsize.x * 0.5 + chair_depth * 0.5 - tuck
    dist_y = tsize.y * 0.5 + chair_depth * 0.5 - tuck
    # subtract end margins and a half-chair per end so chair edges, not centers, fit
    avail_x = tsize.x - 2.0 * edge_margin
    avail_y = tsize.y - 2.0 * edge_margin
    pitch = chair_width + chair_spacing
    n_x = pf.nodes.math.maximum(
        pf.nodes.math.floor((avail_x + chair_spacing) / pitch), 1.0
    ).astype(dtype=int)
    n_y = pf.nodes.math.maximum(
        pf.nodes.math.floor((avail_y + chair_spacing) / pitch), 1.0
    ).astype(dtype=int)
    span_x = pf.nodes.math.maximum(avail_x - chair_width, 0.0)
    span_y = pf.nodes.math.maximum(avail_y - chair_width, 0.0)

    def make_row(count, dist, span, theta, seed):
        return _chair_row(
            chair_geo=chair_geo,
            count=count,
            dist=dist,
            span=span,
            theta=theta,
            seed=seed,
            rot_jitter=rot_jitter,
        )

    rows = [
        make_row(n_y, dist_x, span_y, 0.0, 1),
        make_row(n_y, dist_x, span_y, math.pi, 2),
    ]
    if include_ends:
        rows.append(make_row(n_x, dist_y, span_x, math.pi * 0.5, 3))
        rows.append(make_row(n_x, dist_y, span_x, math.pi * 1.5, 4))
    # chairs stand on the floor (z=0.001 like _place_on_floor), not at the table's z
    table_loc = pf.nodes.math.separate_xyz(table_info.location)
    chair_bottom = pf.nodes.math.separate_xyz(cbox.min).z
    posed = pf.nodes.geo.transform(
        pf.nodes.geo.join_geometry(rows),
        translation=pf.nodes.math.combine_xyz(
            x=table_loc.x, y=table_loc.y, z=0.001 - chair_bottom
        ),
        rotation=table_info.rotation,
    )
    return pf.nodes.to_aliases(posed)


def dining_setup_rand(
    rng: pf.RNG,
    chair_spacing: float | None = None,
    tuck: float | None = None,
    edge_margin: float | None = None,
    rot_jitter: float | None = None,
    include_ends: bool | None = None,
    dining_table: pf.MeshObject | None = None,
) -> DiningSetupResult:
    """A dining table with a single dining chair design instanced around all four
    sides facing inward. Builds the table, one chair, then arranges copies of the
    chair around it; chairs follow the table's pose. Pass `dining_table` to arrange
    chairs around an existing (already posed) table instead of sampling one."""
    rng, rng_dims, rng_table, rng_chair_dims, rng_chair = rng.spawn(5)
    if chair_spacing is None:
        chair_spacing = pf.random.uniform(rng, 0.0625, 0.1875)
    if tuck is None:
        tuck = pf.random.clip_gaussian(rng, 0.1, 0.12, -0.08, 0.3)
    if edge_margin is None:
        edge_margin = pf.random.uniform(rng, 0.08, 0.20)
    if rot_jitter is None:
        rot_jitter = pf.random.uniform(rng, 0.05, 0.10)
    if include_ends is None:
        include_ends = pf.control.choice(rng, [(True, 1.0), (False, 1.0)])

    if dining_table is None:
        dims = table.table_dimensions_rand(rng_dims)
        dining_table = table.dining_table_rand(rng_table, dimensions=dims).mesh

    chair_dims = chair.dining_chair_dimensions_rand(rng_chair_dims)
    chair_res = chair.chair_rand(rng_chair, dimensions=chair_dims)

    chairs = arrange_dining_chairs(
        chair_res.mesh,
        dining_table,
        chair_spacing,
        tuck,
        edge_margin,
        rot_jitter,
        include_ends,
    )

    all_objects = [dining_table, *chairs]
    return DiningSetupResult(
        dining_table=dining_table, chairs=chairs, all_objects=all_objects
    )


def _arrange_chairs_around(rng: pf.RNG, dining_table: pf.MeshObject) -> list:
    return dining_setup_rand(rng, dining_table=dining_table).chairs


def dining_table_setup_rand(
    rng: pf.RNG,
    wall_planes: list[pf.MeshObject],
    room_dimensions: pf.Vector,
    colliders: ccol.CollisionSet | None = None,
) -> DiningTableSetupResult:
    """Place a dining table (3/4 in clear floor space, 1/4 snapped to a wall), then
    arrange chairs around the posed table, plus an optional rug. Chairs are culled
    against `colliders` and dropped when a wall separates them from the table (a
    wall-snapped table's wall-side chairs land outside the room), retrying the
    arrangement with a fresh chair design and margins when too few survive."""
    if colliders is None:
        colliders = ccol.collision_set(wall_planes)
    rng_table, rng_place, rng_setup, rng_rug = rng.spawn(4)

    dims = table.table_dimensions_rand(rng_table)
    table_res = table.dining_table_rand(rng_table, dimensions=dims)
    dining_table = _BareMeshResult(mesh=table_res.mesh)

    def in_free_floorspace():
        return _place_in_free_floorspace(
            rng_place, dining_table, room_dimensions, colliders
        )

    def against_wall():
        return retry_place(
            rng_place,
            dining_table,
            colliders,
            _snap_to_wall,
            parents=wall_planes,
            margin=pf.random.uniform(rng_place, 0.03, 0.10),
        )

    placed = pf.control.choice(
        rng_place,
        [
            (in_free_floorspace, 3.0),
            (against_wall, 1.0),
        ],
    )()
    diningtable_objs = [placed] if placed is not None else []
    logger.info(f"Placed {len(diningtable_objs)} dining tables")

    chair_objs: list[MeshResult] = []
    if diningtable_objs:
        # colliders here excludes the table so the obstruction ray hits the wall behind it
        chair_meshes, colliders = place_surrounding(
            rng_setup, diningtable_objs[0].mesh, _arrange_chairs_around, colliders
        )
        chair_objs = [_BareMeshResult(mesh=c) for c in chair_meshes]
        logger.info(f"Kept {len(chair_objs)} dining chairs after obstruction/collision")

    colliders = ccol.collision_set(
        colliders.objs + [r.mesh for r in diningtable_objs], cache=colliders
    )
    rug_objs = _maybe_rug(rng_rug, room_dimensions)
    all_objects = [r.mesh for r in diningtable_objs + chair_objs] + rug_objs
    return DiningTableSetupResult(diningtable_objs, chair_objs, rug_objs, all_objects)


def sofa_setup_rand(
    rng: pf.RNG,
    wall_planes: list[pf.MeshObject],
    room_dimensions: pf.Vector,
    colliders: ccol.CollisionSet | None = None,
) -> SofaSetupResult:
    """A sofa grouping (choice of wall-snapped or rug-centered) plus side tables beside
    the sofas. `wall_planes` passes through to the wall-snapped variant. Lamps are
    placed by the caller from the returned sofas and side tables (see sofa_lamps_rand)."""
    if colliders is None:
        colliders = ccol.collision_set(wall_planes)
    rng_arr, rng_side = rng.spawn(2)

    arrangement_func = pf.control.choice(
        rng_arr,
        [
            (wall_sofa_setup_rand, 2.0),
            (centered_sofa_setup_rand, 3.0),
        ],
    )
    arrangement = arrangement_func(
        rng_arr,
        wall_planes=wall_planes,
        room_dimensions=room_dimensions,
        colliders=colliders,
    )
    sofa_meshes = [r.mesh for r in arrangement.sofas]
    coffee_tables = list(arrangement.coffee_tables)
    solid = sofa_meshes + [r.mesh for r in coffee_tables]
    colliders = ccol.collision_set(colliders.objs + solid, cache=colliders)

    n = min(pf.random.randint(rng_side, 0, 6), len(arrangement.sofas))
    rngs = rng_side.spawn(n)
    side_tables = [table.side_table_rand(rngs[i]) for i in range(n)]
    placed_side_tables = []
    for i in range(n):
        side_table = retry_place(
            rngs[i],
            side_tables[i],
            colliders,
            _snap_side_by_side,
            parents=sofa_meshes,
            attempts=12,
        )
        placed_side_tables.append(side_table)
    side_tables = placed_side_tables
    side_tables, colliders = keep_non_colliding(side_tables, colliders)
    logger.info(f"Placed {len(side_tables)} side tables out of {n} attempts")

    all_objects = arrangement.all_objects + [r.mesh for r in side_tables]
    return SofaSetupResult(
        sofas=arrangement.sofas,
        coffee_tables=coffee_tables,
        side_tables=side_tables,
        rugs=list(arrangement.rugs),
        all_objects=all_objects,
    )


def sofa_lamps_rand(
    rng: pf.RNG,
    sofa_meshes: list[pf.MeshObject],
    side_table_meshes: list[pf.MeshObject],
    colliders: ccol.CollisionSet,
) -> tuple[list[MeshResult], list[MeshResult], list[pf.LightObject], ccol.CollisionSet]:
    """Place floor lamps beside `sofa_meshes` and table lamps atop `side_table_meshes`,
    culling against `colliders`. Returns (floor_lamps, table_lamps, lights, colliders);
    each lamp's light is kept in `lights` ~2/3 of the time."""
    rng_lamp, rng_table_lamp = rng.spawn(2)

    n = min(pf.random.randint(rng_lamp, 0, 3), len(sofa_meshes))
    rngs = rng_lamp.spawn(n)
    floor_lamps = [
        lamp.lamp_rand(rngs[i], pf.random.uniform(rngs[i], 1.0, 2.0)) for i in range(n)
    ]
    placed_floor_lamps = []
    for i in range(n):
        floor_lamp = retry_place(
            rngs[i],
            floor_lamps[i],
            colliders,
            _snap_side_by_side,
            parents=sofa_meshes,
        )
        placed_floor_lamps.append(floor_lamp)
    floor_lamps, colliders = keep_non_colliding(placed_floor_lamps, colliders)
    logger.info(f"Placed {len(floor_lamps)} floor lamps out of {n} attempts")
    floor_lamp_lights = [r.light for r in floor_lamps if r.light is not None]
    lights: list[pf.LightObject] = []
    lights += pf.control.choice(rng_lamp, [(floor_lamp_lights, 2), ([], 1)])

    n = min(pf.random.randint(rng_table_lamp, 1, 3), len(side_table_meshes))
    rngs = rng_table_lamp.spawn(n)
    table_lamps = [lamp.desk_lamp_rand(rngs[i]) for i in range(n)]
    placed_table_lamps = []
    for i in range(n):
        table_lamp = retry_place(
            rngs[i],
            table_lamps[i],
            colliders,
            snap_on_top,
            parents=side_table_meshes,
        )
        placed_table_lamps.append(table_lamp)
    table_lamps, colliders = keep_non_colliding(placed_table_lamps, colliders)
    logger.info(f"Placed {len(table_lamps)} table lamps out of {n} attempts")
    table_lamp_lights = [r.light for r in table_lamps if r.light is not None]
    lights += pf.control.choice(rng_table_lamp, [(table_lamp_lights, 2), ([], 1)])

    return floor_lamps, table_lamps, lights, colliders
