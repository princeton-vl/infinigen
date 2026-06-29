"""Place collections of small primitive objects onto room surfaces.

Geometry-node Poisson-scatter of a collection of small objects onto a parent
surface: selecting upward-facing faces and instancing a collection at
random-density, min-distance points.

The room-facing entrypoint builds a shared pool of primitives and scatters
subsets of it onto each target's real surface.
"""

import functools
import logging
import math
from typing import Callable, NamedTuple

import procfunc as pf
from procfunc.nodes import types as t

from infinigen_v2.generators.objects import random_primitives
from infinigen_v2.generators.scenes import collision_collection as ccol
from infinigen_v2.generators.scenes.room.room_furniture import keep_non_colliding

logger = logging.getLogger(__name__)


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
def smallobj_scatter(
    parent: pf.ProcNode[pf.MeshObject],
    child: pf.ProcNode[pf.Collection],
    seed: t.SocketOrVal[int] = 0,
    density: t.SocketOrVal[float] = 20.0,
    distance_min: t.SocketOrVal[float] = 0.1,
    rotation_randomness: t.SocketOrVal[float] = 1.0,
    offset: t.SocketOrVal[pf.Vector] = (0, 0, 0.002),
    instance_index: t.SocketOrVal[int] = 0,
) -> pf.ProcNode[t.Instances]:
    eligible = pf.nodes.func.greater_than(a=pf.nodes.geo.input_normal().z, b=0.98)
    points = pf.nodes.geo.distribute_points_on_faces_poisson(
        mesh=parent,
        selection=eligible,
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


class RowAxisResult(NamedTuple):
    geometry: pf.ProcNode[pf.MeshObject]
    center: pf.ProcNode[pf.Vector]
    axis: pf.ProcNode[pf.Vector]
    edge_len: pf.ProcNode[float]


@pf.nodes.node_function
def _row_axis_length(parent: pf.ProcNode[pf.MeshObject]) -> RowAxisResult:
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
    return RowAxisResult(
        geometry=cap.geometry, center=cap.center, axis=cap.axis, edge_len=cap.edge_len
    )


@pf.nodes.node_function
def smallobj_row(
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
) -> tuple[pf.Collection, list[tuple[float, float, float]]]:
    k = int(pf.random.randint(rng, 1, len(pool) + 1))
    child = pf.Collection(
        [pool[i] for i in rng.choice(len(pool), size=k, replace=False)]
    )
    extents = [_clamped_extent(o) for o in child]
    return child, extents


def _bake_and_filter(
    geometry: pf.ProcNode,
    parent: pf.MeshObject,
    colliders: ccol.CollisionSet,
) -> tuple[list[pf.MeshObject], ccol.CollisionSet]:
    """Realize instances into world space (the geonode ran in parent-local space)
    and drop any that collide with already-placed geometry."""
    instances = pf.nodes.to_aliases(geometry)
    for alias in instances:
        alias.item().matrix_world = (
            parent.item().matrix_world @ alias.item().matrix_world
        )
        alias.item().name = alias.item().data.name
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
    spacing_factor: float | None = None,
) -> tuple[list[pf.MeshObject], ccol.CollisionSet]:
    """spacing_factor scales the Poisson min-distance by the object footprint:
    None randomizes it (sparse, objects never touch); 0 packs them (collision cull
    then thins overlaps), which suits cluttered furniture surfaces."""
    child, extents = _pick_child(rng, pool)
    sizes = [max(e[:2]) for e in extents]
    typ = sum(sizes) / len(sizes)
    if spacing_factor is None:
        spacing_factor = pf.random.uniform(rng, 0.7, 1.1)
    distance_min = typ * spacing_factor
    area_fill = pf.random.clip_gaussian(rng, 1.5, 0.5, 0.0, 2.5)
    density = min(area_fill / typ**2, 300.0)

    geometry = smallobj_scatter(
        parent=parent,
        child=child,
        seed=int(pf.random.randint(rng, 0, 2**31 - 1)),
        density=density,
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
    return _bake_and_filter(geometry, parent, colliders)


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
    child, extents = _pick_child(rng, pool)
    max_xy = max(max(e[:2]) for e in extents)
    fullness = pf.random.clip_gaussian(rng, 2.0, 0.7, 0.1, 4.0)
    obj_size = max_xy / fullness

    diag = sum(e * e for e in _clamped_extent(parent)) ** 0.5
    max_slots = int(diag / obj_size) + 1

    eligible = pf.nodes.func.greater_than(a=pf.nodes.geo.input_normal().z, b=0.98)
    geometry = smallobj_row(
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
    return _bake_and_filter(geometry, parent, colliders)


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


def small_objects_collection_distribution(rng: pf.RNG) -> pf.Collection:
    """One room-wide pool of small primitives to draw from per target. Each
    primitive's origin is dropped to its base so instances sit on the surface."""
    n_pool = int(pf.random.randint(rng, 8, 17))
    meshes = []
    for i, rng_mesh in enumerate(rng.spawn(n_pool)):
        mesh = random_primitives.primitives_distribution(
            rng_mesh,
            target_size=pf.random.clip_gaussian(rng_mesh, 0.13, 0.07, 0.08, 0.3),
        ).mesh
        pf.ops.mesh.transform_apply(mesh)
        bmin, _ = pf.ops.attr.bbox_min_max(mesh, global_coords=False)
        pf.ops.object.set_transform(mesh, location=(0, 0, -bmin[2]))
        pf.ops.mesh.transform_apply(mesh)
        # collection_info(separate_children) indexes by natural-name sort; give stable
        # draw-order names (warp realization reuses "mesh_single_vertex", which would tie).
        mesh.item().name = mesh.item().data.name = f"smallobj_{i:03d}"
        meshes.append(mesh)
    return pf.Collection(meshes)


def _place_on_targets(
    rng: pf.RNG,
    targets: list[pf.MeshObject],
    pool: pf.Collection,
    colliders: ccol.CollisionSet,
    skip_prob: float,
    on_target: Callable[
        [pf.RNG, pf.MeshObject, list[pf.MeshObject], ccol.CollisionSet],
        tuple[list[pf.MeshObject], ccol.CollisionSet],
    ],
) -> tuple[list[pf.MeshObject], ccol.CollisionSet]:
    pool_list = list(pool)
    instances: list[pf.MeshObject] = []
    for rng_target, parent in zip(rng.spawn(len(targets)), targets, strict=True):
        if pf.random.uniform(rng_target, 0.0, 1.0) < skip_prob:
            continue
        placed, colliders = on_target(rng_target, parent, pool_list, colliders)
        instances.extend(placed)
    return instances, colliders


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
    return _place_on_targets(rng, targets, pool, colliders, skip_prob, on_target)


def objects_in_face_rows(
    rng: pf.RNG,
    targets: list[pf.MeshObject],
    pool: pf.Collection,
    colliders: ccol.CollisionSet,
    skip_prob: float = 1 / 3,
) -> tuple[list[pf.MeshObject], ccol.CollisionSet]:
    """Place small objects in a centered row along each target's long edge; each
    target skipped with prob skip_prob. For narrow surfaces (wall shelves) where
    free scatter overhangs or clips. Returns placed + colliders."""
    return _place_on_targets(rng, targets, pool, colliders, skip_prob, _row_on_target)


def objects_scatter_distribution(
    rng: pf.RNG,
    targets: list[pf.MeshObject],
    pool: pf.Collection,
    colliders: ccol.CollisionSet,
    skip_prob: float = 1 / 3,
) -> tuple[list[pf.MeshObject], ccol.CollisionSet]:
    """Per target, free scatter 2/3 of the time and a centered row 1/3; each
    target skipped with prob skip_prob. Returns placed + colliders."""
    return _place_on_targets(rng, targets, pool, colliders, skip_prob, _mixed_on_target)
