# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

import logging
import math
from typing import NamedTuple, TypeVar

import numpy as np
import procfunc as pf
import shapely
from procfunc.nodes import types as t

from infinigen2.objects import bowl, chair, plant_pot, table, vase
from infinigen2.scenes.placement import collision as ccol
from infinigen2.scenes.placement.culling import keep_non_colliding, place_surrounding
from infinigen2.scenes.placement.distribute import propagate_modifiers_to_instances
from infinigen2.scenes.placement.retry import repeat_attempts
from infinigen2.scenes.setup_utils import (
    MeshResult,
    back_face_grounded,
    center_inside,
    jitter_object_rotation_rand,
    retry_place,
    snap_on_top,
    snap_to_wall,
)

__all__ = [
    "DiningSetupResult",
    "DiningTableSetupResult",
    "arrange_dining_chairs",
    "chairs_on_edge",
    "table_dining_setup_rand",
    "table_dining_setup_wall_rand",
    "table_dining_with_chairs_rand",
    "table_wall_setup_rand",
]

logger = logging.getLogger(__name__)
MR = TypeVar("MR", bound=MeshResult)


class DiningSetupResult(NamedTuple):
    dining_table: pf.MeshObject
    chairs: list[pf.MeshObject]
    all_objects: list[pf.MeshObject]


class DiningTableSetupResult(NamedTuple):
    all_objects: list[pf.MeshObject]
    dining_tables: list[MeshResult]
    dining_chairs: list[pf.MeshObject]
    storage_containers: list[pf.MeshObject]
    supports: list[pf.MeshObject]
    storages: list[pf.MeshObject]


def _place_on_floor(
    rng: pf.RNG,
    child: MR,
    bbox_min: pf.Vector,
    bbox_max: pf.Vector,
    clearance: float,
) -> None:
    bmin, bmax = pf.ops.attr.bbox_min_max(child.mesh, global_coords=False)
    half_x = (bmax[0] - bmin[0]) / 2
    half_y = (bmax[1] - bmin[1]) / 2
    centre_x = pf.random.uniform(
        rng, bbox_min.x + clearance + half_x, bbox_max.x - clearance - half_x
    )
    centre_y = pf.random.uniform(
        rng, bbox_min.y + clearance + half_y, bbox_max.y - clearance - half_y
    )
    child.mesh.item().location = (
        centre_x - (bmin[0] + bmax[0]) / 2,
        centre_y - (bmin[1] + bmax[1]) / 2,
        bbox_min.z + 0.001 - bmin[2],
    )


def _place_in_free_floorspace(
    rng: pf.RNG,
    child: MR,
    bbox_min: pf.Vector,
    bbox_max: pf.Vector,
    colliders: ccol.CollisionSet,
    region: shapely.Polygon | None,
    clearance: float,
) -> MR | None:
    def footprint_clears(mesh: pf.MeshObject) -> bool:
        if region is not None and not center_inside(region, mesh):
            return False
        lo, hi = (
            np.array(b) for b in pf.ops.attr.bbox_min_max(mesh, global_coords=True)
        )
        ext = hi - lo
        transform = np.eye(4)
        transform[:3, 3] = (lo + hi) / 2
        return not ccol.box_intersection_test(
            colliders,
            transform,
            [ext[0] + 2 * clearance, ext[1] + 2 * clearance, ext[2]],
        )

    return retry_place(
        rng,
        child,
        colliders,
        _place_on_floor,
        attempts=64,
        accept_fn=footprint_clears,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        clearance=clearance,
    )


@pf.nodes.node_function
def chairs_on_edge(
    chair_geo: pf.ProcNode[pf.MeshObject],
    edge: pf.ProcNode[pf.CurveObject],
    inward: t.SocketOrVal[pf.Vector],
    spacing: t.SocketOrVal[float],
    edge_offset: t.SocketOrVal[float],
    disorder: t.SocketOrVal[float],
    seed: t.SocketOrVal[int],
) -> pf.ProcNode[t.Instances]:
    points = pf.nodes.geo.curve_to_points_length(curve=edge, length=spacing)
    position_jitter = disorder * 0.30
    backward = pf.nodes.func.random_value(
        min=0.0,
        max=position_jitter,
        seed=pf.nodes.math.add(seed, 1).astype(dtype=int),
    )
    lateral = pf.nodes.func.random_value(
        min=position_jitter * -1.0,
        max=position_jitter,
        seed=pf.nodes.math.add(seed, 2).astype(dtype=int),
    )
    outward = inward * -1.0
    chair_centers = pf.nodes.geo.set_position(
        geometry=points.points,
        offset=outward * (edge_offset + backward) + points.tangent * lateral,
    )
    rotation = pf.nodes.func.axes_to_rotation(
        primary_axis_vector=inward,
        secondary_axis_vector=(0.0, 0.0, 1.0),
        primary_axis="X",
        secondary_axis="Z",
    )
    return pf.nodes.geo.instance_on_points(
        points=chair_centers,
        instance=chair_geo,
        rotation=rotation,
    )


def arrange_dining_chairs(
    chair_obj: pf.MeshObject,
    dining_table: pf.MeshObject,
    chair_spacing: float,
    tuck: float,
    edge_margin: float,
    include_ends: bool = True,
    disorder: float = 1.0,
    offset_seed: int = 0,
) -> list[pf.MeshObject]:
    """Instance one chair design along the table's bounding-box edge curves."""
    collection = pf.types.Collection([chair_obj], name="dining_chair")
    chair_geo = pf.nodes.geo.collection_info(collection, separate_children=True)
    table_info = pf.nodes.geo.object_info(dining_table)
    tbox = pf.nodes.geo.bound_box(table_info.geometry)
    tmin = pf.nodes.math.separate_xyz(tbox.min)
    tmax = pf.nodes.math.separate_xyz(tbox.max)
    cbox = pf.nodes.geo.bound_box(pf.nodes.geo.realize_instances(chair_geo))
    csize = pf.nodes.math.separate_xyz(cbox.max - cbox.min)
    chair_depth, chair_width = csize.x, csize.y
    trim = edge_margin + chair_width * 0.5
    pitch = chair_width + chair_spacing
    edge_offset = chair_depth * 0.5 - tuck

    right_edge = pf.nodes.geo.curve_line(
        start=pf.nodes.math.combine_xyz(x=tmax.x, y=tmin.y + trim),
        end=pf.nodes.math.combine_xyz(x=tmax.x, y=tmax.y - trim),
    )
    right = chairs_on_edge(
        chair_geo,
        right_edge,
        (-1.0, 0.0, 0.0),
        pitch,
        edge_offset,
        disorder,
        offset_seed + 1,
    )
    left_edge = pf.nodes.geo.curve_line(
        start=pf.nodes.math.combine_xyz(x=tmin.x, y=tmin.y + trim),
        end=pf.nodes.math.combine_xyz(x=tmin.x, y=tmax.y - trim),
    )
    left = chairs_on_edge(
        chair_geo,
        left_edge,
        (1.0, 0.0, 0.0),
        pitch,
        edge_offset,
        disorder,
        offset_seed + 2,
    )
    edge_instances = [right, left]
    if include_ends:
        top_edge = pf.nodes.geo.curve_line(
            start=pf.nodes.math.combine_xyz(x=tmin.x + trim, y=tmax.y),
            end=pf.nodes.math.combine_xyz(x=tmax.x - trim, y=tmax.y),
        )
        top = chairs_on_edge(
            chair_geo,
            top_edge,
            (0.0, -1.0, 0.0),
            pitch,
            edge_offset,
            disorder,
            offset_seed + 3,
        )
        bottom_edge = pf.nodes.geo.curve_line(
            start=pf.nodes.math.combine_xyz(x=tmin.x + trim, y=tmin.y),
            end=pf.nodes.math.combine_xyz(x=tmax.x - trim, y=tmin.y),
        )
        bottom = chairs_on_edge(
            chair_geo,
            bottom_edge,
            (0.0, 1.0, 0.0),
            pitch,
            edge_offset,
            disorder,
            offset_seed + 4,
        )
        edge_instances.extend([top, bottom])
    bottom_alignment = pf.nodes.math.combine_xyz(z=tbox.min.z - cbox.min.z)
    aligned = pf.nodes.geo.transform(
        pf.nodes.geo.join_geometry(edge_instances),
        translation=bottom_alignment,
    )
    posed = pf.nodes.geo.transform(
        aligned,
        translation=table_info.location,
        rotation=table_info.rotation,
    )
    chairs = pf.nodes.to_aliases(posed)
    propagate_modifiers_to_instances([chair_obj], chairs)
    return chairs


def table_dining_with_chairs_rand(
    rng: pf.RNG,
    chair_spacing: float | None = None,
    tuck: float | None = None,
    edge_margin: float | None = None,
    include_ends: bool | None = None,
    dining_table: pf.MeshObject | None = None,
    colliders: ccol.CollisionSet | None = None,
) -> DiningSetupResult:
    """A dining table with a single dining chair design instanced around all four
    sides facing inward. Builds the table, one chair, then arranges copies of the
    chair around it; chairs follow the table's pose. Pass `dining_table` to arrange
    chairs around an existing (already posed) table instead of sampling one."""
    if colliders is None:
        colliders = ccol.collision_set([])
    rng_table_choice, rng_table, rng_attempts, rng_jitter_objects = rng.spawn(4)

    if dining_table is None:
        table_fn = pf.control.choice(
            rng_table_choice,
            [
                (table.table_dining_rand, 1.0),
                (table.table_dining_circular_rand, 1.0),
            ],
        )
        dining_table = table_fn(rng_table).mesh
    table_min, table_max = pf.ops.attr.bbox_min_max(
        dining_table,
        global_coords=False,
    )
    table_dimensions = table_max - table_min

    def attempt(rng: pf.RNG) -> tuple[list[pf.MeshObject], float] | None:
        rng_params, rng_chair_dims, rng_chair, rng_disorder, rng_ends = rng.spawn(5)
        spacing = chair_spacing
        if spacing is None:
            spacing = pf.random.uniform(rng_params, 0.0625, 0.1875)
        chair_tuck = tuck
        if chair_tuck is None:
            chair_tuck = pf.random.clip_gaussian(rng_params, 0.1, 0.12, -0.08, 0.3)
        margin = edge_margin
        if margin is None:
            margin = pf.random.uniform(rng_params, 0.08, 0.20)
        ends = include_ends
        if ends is None:
            ends = pf.control.choice(rng_ends, [(True, 1.0), (False, 1.0)])

        seat_clearance = pf.random.uniform(rng_chair_dims, 0.27, 0.30)
        chair_dims = chair.chair_dining_dimensions_rand(
            rng_chair_dims,
            seat_elevation=table_dimensions[2] - seat_clearance,
        )
        chair_res = chair.chair_rand(rng_chair, dimensions=chair_dims)
        disorder = pf.random.uniform(rng_disorder, 0.0, 1.0) ** 2
        chairs = arrange_dining_chairs(
            chair_res.mesh,
            dining_table,
            spacing,
            chair_tuck,
            margin,
            ends,
            disorder=disorder,
            offset_seed=pf.random.randint(rng_params, 0, 2**20),
        )
        max_angle = (
            pf.random.uniform(rng_disorder, math.radians(30), math.radians(52.5))
            * disorder
        )
        arrangement_colliders = ccol.collision_set([dining_table, *chairs])
        if ccol.any_self_collision(arrangement_colliders):
            return None
        if any(ccol.intersection_test(colliders, chair_obj) for chair_obj in chairs):
            return None
        return chairs, max_angle

    arrangement = repeat_attempts(attempt, rng_attempts, attempts=12)
    if arrangement is None:
        raise RuntimeError("Could not generate a collision-free dining arrangement")
    chairs, max_angle = arrangement

    rotation_colliders = ccol.collision_set(
        [*colliders.objs, dining_table, *chairs], cache=colliders
    )
    chair_rngs = rng_jitter_objects.spawn(len(chairs))
    for chair_obj, chair_rng in zip(chairs, chair_rngs, strict=True):
        jitter_object_rotation_rand(chair_rng, chair_obj, max_angle, rotation_colliders)

    all_objects = [dining_table, *chairs]
    return DiningSetupResult(
        dining_table=dining_table, chairs=chairs, all_objects=all_objects
    )


def _table_dining_rectangular_rand(
    rng: pf.RNG, length: float | None, max_x: float, max_y: float
) -> table.TableResult:
    rng_dims, rng_table = rng.spawn(2)
    width, depth, height = table.table_dimensions_rand(rng_dims, depth=length)
    long_side = min(max(width, depth), max(max_x, max_y))
    short_side = min(min(width, depth), min(max_x, max_y))
    dimensions = (short_side, long_side, height)
    if max_x > max_y:
        dimensions = (long_side, short_side, height)
    return table.table_dining_rand(rng_table, dimensions=dimensions)


def _table_dining_circular_rand(
    rng: pf.RNG, length: float | None, max_x: float, max_y: float
) -> table.TableResult:
    rng_diameter, rng_table = rng.spawn(2)
    diameter = pf.random.clip_gaussian(rng_diameter, 1.15, 0.2, 0.95, 1.5)
    if length is not None:
        diameter = min(max(0.6 * length, 0.95), 1.5)
    diameter = min(diameter, max_x, max_y)
    return table.table_dining_circular_rand(rng_table, diameter=diameter)


def table_dining_setup_rand(
    rng: pf.RNG,
    wall_planes: list[pf.MeshObject] | None = None,
    bbox_min: pf.Vector | None = None,
    bbox_max: pf.Vector | None = None,
    colliders: ccol.CollisionSet | None = None,
    region: shapely.Polygon | None = None,
    table_length: float | None = None,
) -> DiningTableSetupResult:
    """Place a dining table in clear floor space and arrange chairs around it."""
    del wall_planes
    if bbox_min is None:
        bbox_min = pf.Vector((0.0, 0.0, 0.0))
    if bbox_max is None:
        bbox_max = pf.Vector((5.0, 15.0, 3.0))
    if colliders is None:
        colliders = ccol.collision_set([])
    rng_table_choice, rng_table, rng_place, rng_seat = rng.spawn(4)
    clearance = 0.45
    door_slack = 1.0
    max_x = max(bbox_max.x - bbox_min.x - 2 * clearance - door_slack, 0.7)
    max_y = max(bbox_max.y - bbox_min.y - 2 * clearance - door_slack, 0.7)
    table_fn = pf.control.choice(
        rng_table_choice,
        [(_table_dining_rectangular_rand, 1.0), (_table_dining_circular_rand, 1.0)],
    )
    dining_table = table_fn(rng_table, table_length, max_x, max_y)
    placed = _place_in_free_floorspace(
        rng_place,
        dining_table,
        bbox_min,
        bbox_max,
        colliders,
        region,
        clearance=clearance,
    )
    return _seated_table_rand(rng_seat, placed, colliders)


def _arrange_chairs(rng: pf.RNG, parent: pf.MeshObject) -> list[pf.MeshObject]:
    return table_dining_with_chairs_rand(
        rng, dining_table=parent, colliders=ccol.collision_set([])
    ).chairs


def _middle_decorations_rand(rng: pf.RNG, parent: pf.MeshObject) -> list[MeshResult]:
    rng_middle_choice, rng_middle_asset, rng_middle_place = rng.spawn(3)
    middle_func = pf.control.choice(
        rng_middle_choice,
        [
            (lambda _: [], 2.0),
            (lambda r: [vase.vase_rand(r)], 1.0),
            (lambda r: [bowl.bowl_rand(r)], 1.0),
            (lambda r: [plant_pot.plant_pot_small_rand(r)], 1.0),
        ],
    )
    decorations = middle_func(rng_middle_asset)
    for decoration in decorations:
        snap_on_top(rng_middle_place, decoration, parents=[parent])
    return decorations


def _seated_table_rand(
    rng: pf.RNG, placed: MeshResult | None, colliders: ccol.CollisionSet
) -> DiningTableSetupResult:
    """Arrange chairs and middle decorations around `placed`, culling chairs that
    collide; an unplaced table yields an empty setup."""
    rng_setup, rng_middle = rng.spawn(2)
    tables, _ = keep_non_colliding([placed], colliders)
    logger.info(f"Placed {len(tables)} dining tables")
    chair_objs: list[pf.MeshObject] = []
    middle_decorations: list[MeshResult] = []
    for placed_table in tables:
        # colliders here excludes the table so the obstruction ray hits the wall behind it
        chair_objs, _ = place_surrounding(
            rng_setup, placed_table.mesh, _arrange_chairs, colliders
        )
        logger.info(f"Kept {len(chair_objs)} dining chairs after obstruction/collision")
        middle_decorations = _middle_decorations_rand(rng_middle, placed_table.mesh)
    all_objects = (
        [r.mesh for r in tables] + chair_objs + [r.mesh for r in middle_decorations]
    )
    return DiningTableSetupResult(
        all_objects=all_objects,
        dining_tables=tables,
        dining_chairs=chair_objs,
        storage_containers=[],
        supports=[r.mesh for r in tables],
        storages=[],
    )


def _snap_any_side(
    rng: pf.RNG, child: MR, parents: list[pf.MeshObject], margin: float
) -> None:
    rng_side, rng_snap = rng.spawn(2)
    side = pf.control.choice(
        rng_side, [(s, 1.0) for s in ("front", "back", "left", "right")]
    )
    snap_to_wall(rng_snap, child, parents, margin=margin, child_side=side)


def table_wall_setup_rand(
    rng: pf.RNG,
    dining_table: MeshResult,
    wall_planes: list[pf.MeshObject],
    colliders: ccol.CollisionSet,
) -> DiningTableSetupResult:
    """Snap any side of `dining_table` against a wall, then arrange chairs on its
    open sides; place_surrounding drops the wall-side chairs."""
    rng_margin, rng_place, rng_seat = rng.spawn(3)
    margin = pf.random.uniform(rng_margin, 0.02, 0.1)
    dining_table.mesh.item().location.z = 0.001
    walls = ccol.collision_set(wall_planes)
    sides = ("front", "back", "left", "right")
    placed = retry_place(
        rng_place,
        dining_table,
        colliders,
        _snap_any_side,
        attempts=10,
        accept_fn=lambda mesh: any(
            back_face_grounded(mesh, walls, margin, side=s) for s in sides
        ),
        parents=wall_planes,
        margin=margin,
    )
    return _seated_table_rand(rng_seat, placed, colliders)


def table_dining_setup_wall_rand(
    rng: pf.RNG,
    wall_planes: list[pf.MeshObject],
    room_dimensions: pf.Vector,
    colliders: ccol.CollisionSet,
) -> DiningTableSetupResult:
    """A rectangular or circular dining table with any side against a wall."""
    del room_dimensions
    rng_table_choice, rng_table, rng_setup = rng.spawn(3)
    table_fn = pf.control.choice(
        rng_table_choice,
        [(table.table_dining_rand, 1.0), (table.table_dining_circular_rand, 1.0)],
    )
    dining_table = table_fn(rng_table)
    return table_wall_setup_rand(rng_setup, dining_table, wall_planes, colliders)
