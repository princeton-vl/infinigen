# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

"""Place floor, surface, and small decoration objects into furnished scenes.

Geometry-node Poisson-scatter of a collection of small objects onto a parent
surface: selecting upward-facing faces and instancing a collection at
random-density, min-distance points.

The room-facing entrypoint builds reusable decoration pools and scatters subsets
of them onto each target's real surface.
"""

import functools
import logging
import math
import re
from typing import Callable, NamedTuple, cast

import numpy as np
import procfunc as pf
from procfunc.nodes import types as t

from infinigen2.objects import bowl, lamp, plant_pot, random_primitives, vase
from infinigen2.scenes.placement import collision as ccol
from infinigen2.scenes.placement.culling import keep_non_colliding
from infinigen2.scenes.placement.distribute import propagate_modifiers_to_instances
from infinigen2.scenes.placement.snap import snap_to_plane
from infinigen2.scenes.setup_utils import (
    MeshResult,
    back_face_grounded,
    bbox_face_grounded,
    retry_place,
    snap_back_front,
    snap_on_top,
)

__all__ = [
    "DecorationObjectsResult",
    "MIN_PLACEABLE_AREA",
    "container_faces",
    "decoration_collection_primitives_and_real_rand",
    "decoration_collection_primitives_rand",
    "decorate_floor_objects_rand",
    "decorate_surface_objects_rand",
    "decoration_smallobj_rand",
    "objects_scatter_rand",
    "objects_scattered_on_surface",
    "placeable_faces",
    "scatter_small_objects_on_containers",
    "scatter_small_objects_on_support_tops",
    "support_top_faces",
]

logger = logging.getLogger(__name__)

_WALL_MARGIN_MIN = 0.03
_WALL_MARGIN_MAX = 0.10
_ADJACENT_MARGIN_MIN = 0.03
_ADJACENT_MARGIN_MAX = 0.15
_SURFACE_INSET = 0.2032
MIN_PLACEABLE_AREA = 0.005


def _smallobj_label(data_name: str) -> str:
    # strips the pool index and any blender copy suffix, leaving primitive_effect
    return re.sub(r"_\d+(\.\d+)?$", "", data_name)


def _scatter_region(
    parent: pf.ProcNode[pf.MeshObject],
    eligible: pf.ProcNode[bool],
    inset: t.SocketOrVal[float],
) -> pf.ProcNode[bool]:
    """Per-point selection: at least `inset` from the boundary of the upward face
    island(s), so instances don't overhang the surface edge into adjacent structure.
    Edges shared by two faces are interior, so a connected surface stays fully usable;
    only each island's outer rim insets."""
    region = pf.nodes.geo.separate_geometry(
        geometry=parent, selection=eligible, domain="FACE"
    ).selection
    interior = pf.nodes.func.greater_than(
        a=pf.nodes.geo.input_mesh_edge_neighbors(), b=1
    )
    cap = pf.nodes.geo.capture_attribute(
        region, domain="EDGE", interior=interior.astype(dtype=float)
    )
    wire = pf.nodes.geo.delete_geometry(
        cap.geometry, selection=True, domain="FACE", mode="ONLY_FACE"
    )
    boundary = pf.nodes.geo.delete_geometry(
        wire, selection=pf.nodes.func.greater_than(a=cap.interior, b=0.5), domain="EDGE"
    )
    dist = pf.nodes.geo.proximity(
        geometry=boundary,
        sample_position=pf.nodes.geo.input_position(),
        target_element="EDGES",
    ).distance
    return pf.nodes.func.greater_than(a=dist, b=inset)


@pf.nodes.node_function
def placeable_faces(
    parent: pf.ProcNode[pf.MeshObject],
    selection: t.SocketOrVal[bool],
) -> pf.ProcNode[pf.MeshObject]:
    """The selected faces, minus connected islands smaller than
    MIN_PLACEABLE_AREA, so small details never receive objects."""
    region = pf.nodes.geo.separate_geometry(
        geometry=parent, selection=selection, domain="FACE"
    ).selection
    island_area = pf.nodes.geo.accumulate_field(
        value=pf.nodes.geo.input_mesh_face_area(),
        group_id=pf.nodes.geo.input_mesh_island().island_index,
        domain="FACE",
    ).total
    too_small = pf.nodes.func.less_than(a=island_area, b=MIN_PLACEABLE_AREA)
    return pf.nodes.geo.delete_geometry(region, selection=too_small, domain="FACE")


@pf.nodes.node_function
def _smallobj_scatter_selected(
    parent: pf.ProcNode[pf.MeshObject],
    child: pf.ProcNode[pf.Collection],
    selection: t.SocketOrVal[bool],
    seed: t.SocketOrVal[int] = 0,
    density: t.SocketOrVal[float] = 20.0,
    distance_min: t.SocketOrVal[float] = 0.1,
    rotation_randomness: t.SocketOrVal[float] = 1.0,
    offset: t.SocketOrVal[pf.Vector] = (0, 0, 0.002),
    instance_index: t.SocketOrVal[int] = 0,
) -> pf.ProcNode[t.Instances]:
    points = pf.nodes.geo.distribute_points_on_faces_poisson(
        mesh=placeable_faces(parent, selection),
        seed=seed,
        density_factor=1.0,
        density_max=density,
        distance_min=distance_min,
    )
    z_rot = (
        pf.nodes.func.random_value(min=0.0, max=2 * math.pi, seed=seed)
        * rotation_randomness
    )
    instances = pf.nodes.geo.instance_on_points(
        points=points.points,
        instance=pf.nodes.geo.collection_info(child, separate_children=True),
        pick_instance=True,
        instance_index=instance_index,
        rotation=pf.nodes.math.combine_xyz(z=z_rot).astype(dtype=pf.Euler),
    )
    return pf.nodes.geo.transform(instances, translation=offset)


@pf.nodes.node_function
def _upward_faces(
    parent: pf.ProcNode[pf.MeshObject],
) -> pf.ProcNode[bool]:
    del parent
    return pf.nodes.func.greater_than(a=pf.nodes.geo.input_normal().z, b=0.98)


@pf.nodes.node_function
def support_top_faces(
    parent: pf.ProcNode[pf.MeshObject],
) -> pf.ProcNode[bool]:
    upward = pf.nodes.func.greater_than(a=pf.nodes.geo.input_normal().z, b=0.98)
    top_z = pf.nodes.geo.bound_box(parent).max.z
    near_top = pf.nodes.func.greater_equal(
        a=pf.nodes.geo.input_position().z, b=top_z - 0.005
    )
    return pf.nodes.func.boolean_and(a=upward, b=near_top)


@pf.nodes.node_function
def container_faces(
    parent: pf.ProcNode[pf.MeshObject],
) -> pf.ProcNode[bool]:
    upward = pf.nodes.func.greater_than(a=pf.nodes.geo.input_normal().z, b=0.98)
    top_z = pf.nodes.geo.bound_box(parent).max.z
    below_top = pf.nodes.func.less_than(
        a=pf.nodes.geo.input_position().z, b=top_z - 0.005
    )
    return pf.nodes.func.boolean_and(a=upward, b=below_top)


@pf.nodes.node_function
def _smallobj_scatter(
    parent: pf.ProcNode[pf.MeshObject],
    child: pf.ProcNode[pf.Collection],
    seed: t.SocketOrVal[int] = 0,
    density: t.SocketOrVal[float] = 20.0,
    distance_min: t.SocketOrVal[float] = 0.1,
    rotation_randomness: t.SocketOrVal[float] = 1.0,
    offset: t.SocketOrVal[pf.Vector] = (0, 0, 0.002),
    instance_index: t.SocketOrVal[int] = 0,
) -> pf.ProcNode[t.Instances]:
    return _smallobj_scatter_selected(
        parent=parent,
        child=child,
        selection=_upward_faces(parent),
        seed=seed,
        density=density,
        distance_min=distance_min,
        rotation_randomness=rotation_randomness,
        offset=offset,
        instance_index=instance_index,
    )


@pf.nodes.node_function
def _smallobj_scatter_support_tops(
    parent: pf.ProcNode[pf.MeshObject],
    child: pf.ProcNode[pf.Collection],
    seed: t.SocketOrVal[int] = 0,
    density: t.SocketOrVal[float] = 20.0,
    distance_min: t.SocketOrVal[float] = 0.1,
    rotation_randomness: t.SocketOrVal[float] = 1.0,
    offset: t.SocketOrVal[pf.Vector] = (0, 0, 0.002),
    instance_index: t.SocketOrVal[int] = 0,
) -> pf.ProcNode[t.Instances]:
    return _smallobj_scatter_selected(
        parent=parent,
        child=child,
        selection=support_top_faces(parent),
        seed=seed,
        density=density,
        distance_min=distance_min,
        rotation_randomness=rotation_randomness,
        offset=offset,
        instance_index=instance_index,
    )


@pf.nodes.node_function
def _smallobj_scatter_containers(
    parent: pf.ProcNode[pf.MeshObject],
    child: pf.ProcNode[pf.Collection],
    seed: t.SocketOrVal[int] = 0,
    density: t.SocketOrVal[float] = 20.0,
    distance_min: t.SocketOrVal[float] = 0.1,
    rotation_randomness: t.SocketOrVal[float] = 1.0,
    offset: t.SocketOrVal[pf.Vector] = (0, 0, 0.002),
    instance_index: t.SocketOrVal[int] = 0,
) -> pf.ProcNode[t.Instances]:
    return _smallobj_scatter_selected(
        parent=parent,
        child=child,
        selection=container_faces(parent),
        seed=seed,
        density=density,
        distance_min=distance_min,
        rotation_randomness=rotation_randomness,
        offset=offset,
        instance_index=instance_index,
    )


class _RowAxisResult(NamedTuple):
    geometry: pf.ProcNode[pf.MeshObject]
    center: pf.ProcNode[pf.Vector]
    axis: pf.ProcNode[pf.Vector]
    edge_len: pf.ProcNode[float]


@pf.nodes.node_function
def _row_axis_length(parent: pf.ProcNode[pf.MeshObject]) -> _RowAxisResult:
    """Capture each face's long-edge axis (unit) and mean length without sampling
    fixed corner indices: compare each loop edge with its neighbour, flip the longer
    ones into a common hemisphere so opposite sides do not cancel, then reduce per
    face with accumulate_field. Orientation-invariant (any wall N/E/S/W)."""
    pos = pf.nodes.geo.input_position()
    corner = pf.nodes.geo.input_index()
    nxt = pf.nodes.geo.offset_corner_in_face(corner_index=corner, offset=1)
    edge = pf.nodes.geo.field_at_index(value=pos, index=nxt, domain="CORNER") - pos
    edge_len_c = pf.nodes.math.vector_length(edge)
    next_len = pf.nodes.geo.field_at_index(value=edge_len_c, index=nxt, domain="CORNER")
    is_long = pf.nodes.func.greater_than(a=edge_len_c, b=next_len).astype(dtype=float)

    e = pf.nodes.math.separate_xyz(edge)
    flip = pf.nodes.math.sign(e.x + e.y * 1e-3 + e.z * 1e-6)
    face = pf.nodes.geo.face_of_corner(corner_index=corner).face_index
    sum_vec = pf.nodes.geo.accumulate_field(
        value=pf.nodes.math.vector_scale(vector=edge, scale=flip * is_long),
        group_id=face,
        domain="CORNER",
    )
    sum_len = pf.nodes.geo.accumulate_field(
        value=edge_len_c * is_long, group_id=face, domain="CORNER"
    )
    n_long = pf.nodes.geo.accumulate_field(
        value=is_long, group_id=face, domain="CORNER"
    )
    axis = pf.nodes.math.vector_normalize(sum_vec.total)
    edge_len = sum_len.total / pf.nodes.math.maximum(n_long.total, 1.0)
    cap = pf.nodes.geo.capture_attribute(
        parent, domain="FACE", center=pos, axis=axis, edge_len=edge_len
    )
    return _RowAxisResult(
        geometry=cap.geometry, center=cap.center, axis=cap.axis, edge_len=cap.edge_len
    )


@pf.nodes.node_function
def _smallobj_row(
    parent: pf.ProcNode[pf.MeshObject],
    child: pf.ProcNode[pf.Collection],
    selection: t.SocketOrVal[bool],
    obj_size: t.SocketOrVal[float],
    max_slots: t.SocketOrVal[int],
    seed: t.SocketOrVal[int] = 0,
    rotation_randomness: t.SocketOrVal[float] = 1.0,
    offset: t.SocketOrVal[pf.Vector] = (0, 0, 0.002),
    instance_index: t.SocketOrVal[int] = 0,
) -> pf.ProcNode[t.Instances]:
    """Lay an evenly-spaced, centered row of instances along the long edge of every
    selected face: floor(edge_len / obj_size) flush-ended slots, centered through the
    face centroid; max_slots sizes the per-face candidate line, culled to the count."""
    cap = _row_axis_length(parent)
    n_slots = pf.nodes.math.maximum(pf.nodes.math.floor(cap.edge_len / obj_size), 1.0)
    half_span = pf.nodes.math.maximum((cap.edge_len - obj_size) * 0.5, 0.0)

    facepts = pf.nodes.geo.mesh_to_points(
        mesh=cap.geometry, selection=selection, position=cap.center, mode="FACES"
    )
    facepts = pf.nodes.geo.capture_attribute(
        facepts, domain="POINT", rc=cap.center, ra=cap.axis, rn=n_slots, rhs=half_span
    )

    line = pf.nodes.geo.mesh_line(
        count=max_slots, start_location=(0, 0, 0), offset=(1, 0, 0)
    )
    line = pf.nodes.geo.capture_attribute(
        line, domain="POINT", ri=pf.nodes.geo.input_index().astype(dtype=float)
    )
    realized = pf.nodes.geo.realize_instances(
        pf.nodes.geo.instance_on_points(points=facepts.geometry, instance=line.geometry)
    )

    ri = line.ri
    rn, rhs = facepts.rn, facepts.rhs
    frac = ri / pf.nodes.math.maximum(rn - 1.0, 1.0)
    is_one = pf.nodes.func.less_than(a=rn, b=1.5)
    along = pf.nodes.func.switch(switch=is_one, a=(frac * 2.0 - 1.0) * rhs, b=0.0)
    target = facepts.rc + pf.nodes.math.vector_scale(vector=facepts.ra, scale=along)

    row = pf.nodes.geo.set_position(geometry=realized, position=target)
    row = pf.nodes.geo.delete_geometry(
        geometry=row,
        selection=pf.nodes.func.greater_equal(a=ri, b=rn),
        domain="POINT",
    )

    z_rot = (
        pf.nodes.func.random_value(min=0.0, max=2 * math.pi, seed=seed)
        * rotation_randomness
    )
    instances = pf.nodes.geo.instance_on_points(
        points=row,
        instance=pf.nodes.geo.collection_info(child, separate_children=True),
        pick_instance=True,
        instance_index=instance_index,
        rotation=pf.nodes.math.combine_xyz(z=z_rot).astype(dtype=pf.Euler),
    )
    return pf.nodes.geo.transform(instances, translation=offset)


def _clamped_extent(obj) -> tuple[float, float, float]:
    bmin, bmax = pf.ops.attr.bbox_min_max(obj, global_coords=False)
    return tuple(max(0.1, hi - lo) for lo, hi in zip(bmin, bmax, strict=True))


def _pick_child(
    rng: pf.RNG, pool: list[pf.MeshObject]
) -> tuple[pf.Collection, list[tuple[float, float, float]]] | None:
    selected = [obj for obj in pool if pf.random.uniform(rng, 0.0, 1.0) < 0.5]
    if not selected:
        return None
    child = pf.Collection(selected)
    extents = [_clamped_extent(o) for o in child]
    return child, extents


def _bake_and_filter(
    geometry: pf.ProcNode,
    parent: pf.MeshObject,
    templates: list[pf.MeshObject],
    colliders: ccol.CollisionSet,
) -> tuple[list[pf.MeshObject], ccol.CollisionSet]:
    """Realize instances into world space (the geonode ran in parent-local space)
    and drop any that collide with already-placed geometry."""
    instances = pf.nodes.to_aliases(geometry)
    propagate_modifiers_to_instances(templates, instances)
    for alias in instances:
        alias.item().matrix_world = (
            parent.item().matrix_world @ alias.item().matrix_world
        )
        alias.item().name = _smallobj_label(alias.item().data.name)
    kept, colliders = keep_non_colliding(instances, colliders, key=lambda o: o)
    logger.debug(
        "small objects on %s (z=%.2f): placed %d, kept %d, dropped %d by collision",
        parent.item().name,
        parent.item().matrix_world.translation.z,
        len(instances),
        len(kept),
        len(instances) - len(kept),
    )
    return kept, colliders


def _scatter_on_target(
    rng: pf.RNG,
    parent: pf.MeshObject,
    pool: list[pf.MeshObject],
    colliders: ccol.CollisionSet,
    density: float | None = None,
    spacing_factor: float | None = None,
    scatter_func: Callable = _smallobj_scatter,
    max_per_area: float | None = None,
) -> tuple[list[pf.MeshObject], ccol.CollisionSet]:
    """spacing_factor scales the Poisson min-distance by the object footprint:
    None randomizes it (sparse, objects never touch); 0 packs them (collision cull
    then thins overlaps), which suits cluttered furniture surfaces."""
    child_result = _pick_child(rng, pool)
    if child_result is None:
        return [], colliders
    child, extents = child_result
    sizes = [max(e[:2]) for e in extents]
    typ = sum(sizes) / len(sizes)
    if spacing_factor is None:
        spacing_factor = pf.random.uniform(rng, 0.7, 1.1)
    distance_min = typ * spacing_factor
    if density is None:
        density = pf.random.clip_gaussian(rng, 1.5, 0.5, 0.0, 2.5)
    if max_per_area is None:
        max_per_area = pf.random.uniform(rng, 8.0, 25.0)
    point_density = min(density / typ**2, max_per_area)

    geometry = scatter_func(
        parent=parent,
        child=child,
        seed=int(pf.random.randint(rng, 0, 2**31 - 1)),
        density=point_density,
        distance_min=distance_min,
        rotation_randomness=pf.random.uniform(rng, 0.0, 1.0),
        offset=(0, 0, 0.002),
        instance_index=pf.nodes.func.random_value(
            min=0,
            max=len(extents) - 1,
            seed=int(pf.random.randint(rng, 0, 2**31 - 1)),
            id=pf.nodes.geo.input_index(),
        ),
    )
    return _bake_and_filter(geometry, parent, list(child), colliders)


def _row_on_target(
    rng: pf.RNG,
    parent: pf.MeshObject,
    pool: list[pf.MeshObject],
    colliders: ccol.CollisionSet,
) -> tuple[list[pf.MeshObject], ccol.CollisionSet]:
    """Row placement: an evenly-spaced, centered line of objects along each upward
    face's long edge. Suited to narrow shelves where free scatter would overhang
    or clip. Spacing uses the largest chosen object (so nothing overlaps) inflated
    by a per-shelf fullness so shelves range from sparse to nearly packed."""
    child_result = _pick_child(rng, pool)
    if child_result is None:
        return [], colliders
    child, extents = child_result
    max_xy = max(max(e[:2]) for e in extents)
    fullness = pf.random.clip_gaussian(rng, 2.0, 0.7, 0.1, 4.0)
    obj_size = max_xy / fullness

    diag = sum(e * e for e in _clamped_extent(parent)) ** 0.5
    max_slots = int(diag / obj_size) + 1

    eligible = pf.nodes.func.greater_than(a=pf.nodes.geo.input_normal().z, b=0.98)
    geometry = _smallobj_row(
        parent=parent,
        child=child,
        selection=eligible,
        obj_size=obj_size,
        max_slots=max_slots,
        seed=int(pf.random.randint(rng, 0, 2**31 - 1)),
        rotation_randomness=pf.random.uniform(rng, 0.0, 1.0),
        offset=(0, 0, 0.002),
        instance_index=pf.nodes.func.random_value(
            min=0,
            max=len(extents) - 1,
            seed=int(pf.random.randint(rng, 0, 2**31 - 1)),
            id=pf.nodes.geo.input_index(),
        ),
    )
    return _bake_and_filter(geometry, parent, list(child), colliders)


def _mixed_on_target(
    rng: pf.RNG,
    parent: pf.MeshObject,
    pool: list[pf.MeshObject],
    colliders: ccol.CollisionSet,
) -> tuple[list[pf.MeshObject], ccol.CollisionSet]:
    """Per-shelf: free scatter 2/3 of the time, centered row 1/3."""
    rng_choice, rng_place = rng.spawn(2)
    on_target = pf.control.choice(
        rng_choice, [(_scatter_on_target, 2.0), (_row_on_target, 1.0)]
    )
    return on_target(rng_place, parent, pool, colliders)


def _decoration_primitive_rand(rng: pf.RNG) -> MeshResult:
    rng_size, rng_mesh = rng.spawn(2)
    target_size = pf.random.clip_gaussian(rng_size, 0.13, 0.105, 0.08, 0.45)
    return random_primitives.primitive_with_effect_rand(
        rng_mesh,
        target_size=target_size,
        max_subsurf_levels=1,
    )


def decoration_smallobj_rand(rng: pf.RNG) -> MeshResult:
    rng_choice, rng_object = rng.spawn(2)
    func = pf.control.choice(
        rng_choice,
        [
            (_decoration_primitive_rand, 4.0),
            (plant_pot.plant_pot_small_rand, 0.5),
            (bowl.bowl_rand, 0.5),
            (vase.cup_rand, 0.5),
        ],
    )
    return func(rng_object)


def _decoration_collection(results: list[MeshResult]) -> pf.Collection:
    meshes = []
    for i, result in enumerate(results):
        mesh = result.mesh
        pf.ops.mesh.transform_apply(mesh)
        bmin, _ = pf.ops.attr.bbox_min_max(mesh, global_coords=False)
        pf.ops.object.set_transform(mesh, location=(0, 0, -bmin[2]))
        pf.ops.mesh.transform_apply(mesh)
        # index keeps the stem unique, which propagate_modifiers_to_instances requires
        label = mesh.item().name.split(".")[0]
        mesh.item().data.name = f"{label}_{i:03d}"
        meshes.append(mesh)
    return pf.Collection(meshes)


def decoration_collection_primitives_rand(rng: pf.RNG) -> pf.Collection:
    """Draw reusable decoration primitives with origins seated at their bases."""
    rng_count, rng_meshes = rng.spawn(2)
    n_pool = int(pf.random.randint(rng_count, 8, 17))
    results = [_decoration_primitive_rand(r) for r in rng_meshes.spawn(n_pool)]
    return _decoration_collection(results)


def decoration_collection_primitives_and_real_rand(rng: pf.RNG) -> pf.Collection:
    """Draw reusable primitives and real objects seated at their bases."""
    rng_count, rng_meshes = rng.spawn(2)
    n_pool = int(pf.random.randint(rng_count, 8, 17))
    results = [decoration_smallobj_rand(r) for r in rng_meshes.spawn(n_pool)]
    return _decoration_collection(results)


def _place_on_targets(
    rng: pf.RNG,
    targets: list[pf.MeshObject],
    pool: pf.Collection,
    colliders: ccol.CollisionSet,
    fraction: float,
    on_target: Callable[
        [pf.RNG, pf.MeshObject, list[pf.MeshObject], ccol.CollisionSet],
        tuple[list[pf.MeshObject], ccol.CollisionSet],
    ],
    inter_object_collide: bool = True,
) -> tuple[list[pf.MeshObject], ccol.CollisionSet]:
    """inter_object_collide=False checks each target's placements against the
    original `colliders` and each other, not against other targets' placements."""
    instances: list[pf.MeshObject] = []
    try:
        colliders = _place_hidden_on_targets(
            rng,
            targets,
            list(pool),
            colliders,
            fraction,
            on_target,
            instances,
            inter_object_collide,
        )
    finally:
        for instance in instances:
            instance.item().hide_viewport = False
    return instances, colliders


def _place_hidden_on_targets(
    rng: pf.RNG,
    targets: list[pf.MeshObject],
    pool_list: list[pf.MeshObject],
    colliders: ccol.CollisionSet,
    fraction: float,
    on_target: Callable[
        [pf.RNG, pf.MeshObject, list[pf.MeshObject], ccol.CollisionSet],
        tuple[list[pf.MeshObject], ccol.CollisionSet],
    ],
    instances: list[pf.MeshObject],
    inter_object_collide: bool,
) -> ccol.CollisionSet:
    for rng_target, parent in zip(rng.spawn(len(targets)), targets, strict=True):
        rng_active, rng_place = rng_target.spawn(2)
        if pf.random.uniform(rng_active, 0.0, 1.0) >= fraction:
            continue
        placed, grown = on_target(rng_place, parent, pool_list, colliders)
        if inter_object_collide:
            colliders = grown
        for obj in placed:
            obj.item().hide_viewport = True
        instances.extend(placed)
    return colliders


def objects_scattered_on_surface(
    rng: pf.RNG,
    targets: list[pf.MeshObject],
    pool: pf.Collection,
    colliders: ccol.CollisionSet,
    skip_prob: float = 1 / 3,
    spacing_factor: float = 0.0,
) -> tuple[list[pf.MeshObject], ccol.CollisionSet]:
    """Poisson-scatter small objects on each target's real surface; each target
    skipped with prob skip_prob. spacing_factor=0 packs furniture surfaces densely.
    Returns placed + colliders."""
    on_target = functools.partial(_scatter_on_target, spacing_factor=spacing_factor)
    return _place_on_targets(rng, targets, pool, colliders, 1.0 - skip_prob, on_target)


def _objects_in_face_rows(
    rng: pf.RNG,
    targets: list[pf.MeshObject],
    pool: pf.Collection,
    colliders: ccol.CollisionSet,
    skip_prob: float = 1 / 3,
) -> tuple[list[pf.MeshObject], ccol.CollisionSet]:
    """Place small objects in a centered row along each target's long edge; each
    target skipped with prob skip_prob. For narrow surfaces (wall shelves) where
    free scatter overhangs or clips. Returns placed + colliders."""
    return _place_on_targets(
        rng, targets, pool, colliders, 1.0 - skip_prob, _row_on_target
    )


def objects_scatter_rand(
    rng: pf.RNG,
    targets: list[pf.MeshObject],
    pool: pf.Collection,
    colliders: ccol.CollisionSet,
    skip_prob: float = 1 / 3,
) -> tuple[list[pf.MeshObject], ccol.CollisionSet]:
    """Per target, free scatter 2/3 of the time and a centered row 1/3; each
    target skipped with prob skip_prob. Returns placed + colliders."""
    return _place_on_targets(
        rng, targets, pool, colliders, 1.0 - skip_prob, _mixed_on_target
    )


def scatter_small_objects_on_containers(
    rng: pf.RNG,
    targets: list[pf.MeshObject],
    colliders: ccol.CollisionSet,
    collection: pf.Collection | None = None,
    density: float | None = None,
    fraction: float | None = None,
    spacing_factor: float | None = None,
    inter_object_collide: bool = True,
) -> tuple[list[pf.MeshObject], ccol.CollisionSet]:
    """Scatter on recessed upward faces, drawing a collection when omitted."""
    rng_fraction, rng_place, rng_collection, rng_max_per_area = rng.spawn(4)
    if collection is None:
        collection = decoration_collection_primitives_rand(rng_collection)
    if fraction is None:
        fraction = pf.random.uniform(rng_fraction, 0.0, 1.0)
    max_per_area = pf.random.uniform(rng_max_per_area, 40.0, 80.0)
    on_target = functools.partial(
        _scatter_on_target,
        density=density,
        spacing_factor=spacing_factor,
        scatter_func=_smallobj_scatter_containers,
        max_per_area=max_per_area,
    )
    return _place_on_targets(
        rng_place,
        targets,
        collection,
        colliders,
        fraction,
        on_target,
        inter_object_collide,
    )


def scatter_small_objects_on_support_tops(
    rng: pf.RNG,
    targets: list[pf.MeshObject],
    colliders: ccol.CollisionSet,
    collection: pf.Collection | None = None,
    density: float | None = None,
    fraction: float | None = None,
    inter_object_collide: bool = True,
) -> tuple[list[pf.MeshObject], ccol.CollisionSet]:
    """Scatter on topmost upward faces, drawing a collection when omitted."""
    rng_fraction, rng_place, rng_collection = rng.spawn(3)
    if collection is None:
        collection = decoration_collection_primitives_rand(rng_collection)
    if fraction is None:
        fraction = pf.random.uniform(rng_fraction, 0.0, 1.0)
    on_target = functools.partial(
        _scatter_on_target,
        density=density,
        spacing_factor=None,
        scatter_func=_smallobj_scatter_support_tops,
    )
    return _place_on_targets(
        rng_place,
        targets,
        collection,
        colliders,
        fraction,
        on_target,
        inter_object_collide,
    )


class DecorationObjectsResult(NamedTuple):
    all_objects: list[pf.MeshObject]
    lights: list[pf.LightObject]
    colliders: ccol.CollisionSet


def _sample_surface_collection(
    rng: pf.RNG,
    count: int,
) -> list[MeshResult]:
    results = []
    for rng_object in rng.spawn(count):
        rng_choice, rng_asset = rng_object.spawn(2)
        func = pf.control.choice(
            rng_choice,
            [
                (lamp.lamp_desk_rand, 1.0),
                (plant_pot.plant_pot_small_rand, 1.0),
                (vase.vase_rand, 1.0),
            ],
        )
        result = func(rng_asset)
        result.mesh.item().name = func.__name__
        results.append(result)
    return results


def _sample_floor_collection(
    rng: pf.RNG,
    count: int,
) -> list[MeshResult]:
    results = []
    for rng_object in rng.spawn(count):
        rng_choice, rng_asset = rng_object.spawn(2)
        func = pf.control.choice(
            rng_choice,
            [
                (lamp.lamp_floor_rand, 1.0),
                (plant_pot.plant_pot_large_rand, 1.0),
            ],
        )
        result = func(rng_asset)
        result.mesh.item().name = func.__name__
        results.append(result)
    return results


def _place_on_floor(
    rng: pf.RNG,
    child: MeshResult,
    floor: pf.MeshObject,
) -> None:
    floor_min, floor_max = pf.ops.attr.bbox_min_max(floor, global_coords=True)
    child_min, child_max = pf.ops.attr.bbox_min_max(child.mesh, global_coords=False)
    half_x = (child_max[0] - child_min[0]) * 0.5
    half_y = (child_max[1] - child_min[1]) * 0.5
    x_min = floor_min[0] + half_x
    x_max = floor_max[0] - half_x
    y_min = floor_min[1] + half_y
    y_max = floor_max[1] - half_y
    x = (floor_min[0] + floor_max[0]) * 0.5
    y = (floor_min[1] + floor_max[1]) * 0.5
    if x_min < x_max:
        x = pf.random.uniform(rng, x_min, x_max)
    if y_min < y_max:
        y = pf.random.uniform(rng, y_min, y_max)
    z = floor_max[2] + 0.002 - child_min[2]
    yaw = pf.random.uniform(rng, 0.0, 2.0 * np.pi)
    pf.ops.object.set_transform(
        child.mesh,
        location=(x, y, z),
        rotation_euler=(0.0, 0.0, yaw),
    )


def _bottom_grounded(
    obj: pf.MeshObject,
    floor_colliders: ccol.CollisionSet,
) -> bool:
    return bbox_face_grounded(
        obj,
        floor_colliders,
        side="bottom",
        max_distance=0.01,
    )


def _floor_and_wall_grounded(
    obj: pf.MeshObject,
    floor_colliders: ccol.CollisionSet,
    wall_colliders: ccol.CollisionSet,
) -> bool:
    on_floor = _bottom_grounded(obj, floor_colliders)
    on_wall = back_face_grounded(
        obj,
        colliders=wall_colliders,
        margin=_WALL_MARGIN_MAX,
        eps=0.1,
    )
    return on_floor and on_wall


def _place_floor_against_wall(
    rng: pf.RNG,
    child: MeshResult,
    floor: pf.MeshObject,
    wall_planes: list[pf.MeshObject],
) -> None:
    _place_on_floor(rng, child, floor)
    snap_back_front(
        rng,
        child,
        wall_planes,
        placement=pf.random.uniform(rng, 0.0, 1.0),
        margin=pf.random.uniform(rng, _WALL_MARGIN_MIN, _WALL_MARGIN_MAX),
    )


def _place_floor_beside_storage(
    rng: pf.RNG,
    child: MeshResult,
    floor: pf.MeshObject,
    storage: list[pf.MeshObject],
) -> None:
    rng, rng_floor, rng_sides = rng.spawn(3)
    _place_on_floor(rng_floor, child, floor)
    child_side, parent_side = pf.control.choice(
        rng_sides,
        [(("left", "right"), 1.0), (("right", "left"), 1.0)],
    )
    snap_to_plane(
        child.mesh,
        parent=rng.choice(list(storage)),
        placement=pf.random.uniform(rng, 0.0, 1.0),
        margin=pf.random.uniform(rng, _ADJACENT_MARGIN_MIN, _ADJACENT_MARGIN_MAX),
        child_side=child_side,
        parent_side=parent_side,
    )


def _place_floor_decoration(
    rng: pf.RNG,
    child: MeshResult,
    floor: pf.MeshObject,
    wall_planes: list[pf.MeshObject],
    preferred_storage: list[pf.MeshObject],
) -> None:
    if not preferred_storage:
        _place_floor_against_wall(rng, child, floor, wall_planes)
        return
    rng_place_choice, rng_place = rng.spawn(2)
    place_fn = pf.control.choice(
        rng_place_choice,
        [
            (
                functools.partial(
                    _place_floor_beside_storage,
                    storage=preferred_storage,
                ),
                2.0,
            ),
            (
                functools.partial(
                    _place_floor_against_wall,
                    wall_planes=wall_planes,
                ),
                1.0,
            ),
        ],
    )
    place_fn(rng_place, child, floor)


@pf.tracer.grammar
def decorate_floor_objects_rand(
    rng: pf.RNG,
    objects: list[pf.MeshObject],
    colliders: ccol.CollisionSet,
    floor: pf.MeshObject,
    wall_planes: list[pf.MeshObject],
    storage: list[pf.MeshObject] | None = None,
    collection: list[MeshResult] | None = None,
) -> DecorationObjectsResult:
    rng_count, rng_collection, rng_place = rng.spawn(3)
    if collection is None:
        count = int(pf.random.randint(rng_count, 0, 5))
        collection = _sample_floor_collection(rng_collection, count)

    floor_colliders = ccol.collision_set([floor])
    wall_objects = cast(list[pf.Object], wall_planes)
    wall_colliders = ccol.collision_set(wall_objects)
    grounded = functools.partial(
        _floor_and_wall_grounded,
        floor_colliders=floor_colliders,
        wall_colliders=wall_colliders,
    )
    preferred_storage = [
        obj
        for obj in storage or []
        if back_face_grounded(
            obj,
            colliders=wall_colliders,
            margin=_WALL_MARGIN_MAX,
            eps=0.1,
        )
    ]
    kept_results: list[MeshResult] = []
    for rng_object, child in zip(
        rng_place.spawn(len(collection)), collection, strict=True
    ):
        result = retry_place(
            rng_object,
            child,
            colliders,
            _place_floor_decoration,
            attempts=5,
            accept_fn=grounded,
            floor=floor,
            wall_planes=wall_planes,
            preferred_storage=preferred_storage,
        )
        kept, colliders = keep_non_colliding([result], colliders)
        kept_results.extend(kept)
    meshes = [result.mesh for result in kept_results]
    lights = [result.light for result in kept_results if hasattr(result, "light")]
    return DecorationObjectsResult(
        all_objects=objects + meshes,
        lights=lights,
        colliders=colliders,
    )


def _place_surface_decoration(
    rng: pf.RNG,
    child: MeshResult,
    parents: list[pf.MeshObject],
) -> None:
    parent = rng.choice(list(parents))
    bbox_min, bbox_max = pf.ops.attr.bbox_min_max(parent, global_coords=True)
    dimensions = bbox_max - bbox_min

    x_inset_fraction = 0.5
    if dimensions[0] > 2.0 * _SURFACE_INSET:
        x_inset_fraction = _SURFACE_INSET / dimensions[0]
    y_inset_fraction = 0.5
    if dimensions[1] > 2.0 * _SURFACE_INSET:
        y_inset_fraction = _SURFACE_INSET / dimensions[1]
    xy_frac = (
        pf.random.uniform(rng, x_inset_fraction, 1.0 - x_inset_fraction),
        pf.random.uniform(rng, y_inset_fraction, 1.0 - y_inset_fraction),
    )
    snap_on_top(rng, child, [parent], xy_frac=xy_frac)


@pf.tracer.grammar
def decorate_surface_objects_rand(
    rng: pf.RNG,
    objects: list[pf.MeshObject],
    colliders: ccol.CollisionSet,
    support_tops: list[pf.MeshObject] | None = None,
    storages: list[pf.MeshObject] | None = None,
    collection: list[MeshResult] | None = None,
) -> DecorationObjectsResult:
    support_tops = [] if support_tops is None else support_tops
    storages = support_tops if storages is None else storages
    storage_items = {obj.item() for obj in storages}
    parents = [obj for obj in support_tops if obj.item() in storage_items]
    if not parents:
        return DecorationObjectsResult(objects, [], colliders)

    rng_count, rng_collection, rng_place = rng.spawn(3)
    if collection is None:
        count = int(pf.random.randint(rng_count, 0, 5))
        collection = _sample_surface_collection(rng_collection, count)

    support_colliders = ccol.collision_set(parents, cache=colliders)
    fully_supported = functools.partial(
        bbox_face_grounded,
        colliders=support_colliders,
        side="bottom",
        max_distance=0.005,
    )
    kept_results: list[MeshResult] = []
    for rng_object, child in zip(
        rng_place.spawn(len(collection)), collection, strict=True
    ):
        result = retry_place(
            rng_object,
            child,
            colliders,
            _place_surface_decoration,
            attempts=5,
            accept_fn=fully_supported,
            parents=parents,
        )
        kept, colliders = keep_non_colliding([result], colliders)
        kept_results.extend(kept)
    meshes = [result.mesh for result in kept_results]
    lights = [result.light for result in kept_results if hasattr(result, "light")]
    return DecorationObjectsResult(
        all_objects=objects + meshes,
        lights=lights,
        colliders=colliders,
    )
