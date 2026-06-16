"""Place collections of small primitive objects onto room surfaces.

Geometry-node Poisson-scatter of a collection of small objects onto a parent
surface: selecting upward-facing faces, eroding inward from the surface edge,
skipping surfaces whose region is smaller than min_surface, and instancing a
collection at random-density, min-distance points.

The room-facing entrypoint builds a shared pool of primitives and scatters
subsets of it onto each target's real surface.
"""

import procfunc as pf
from procfunc.nodes import types as t

from infinigen_v2.generators.objects import random_primitives
from infinigen_v2.generators.scenes import collision_collection as ccol
from infinigen_v2.generators.scenes.room.room_furniture import keep_non_colliding

_UP_NORMAL_Z = 0.98
_TWO_PI = 6.283185307179586

_OFFSET = (0, 0, 0.002)
_MIN_SURFACE = 0.2
_POOL_MIN, _POOL_MAX = 8, 16
_MAX_DISTINCT = 5
_SKIP_OBJECT_PROB = 1.0 / 3.0
_MIN_DIM = 0.1


def _scatter_region(
    parent: pf.ProcNode[pf.MeshObject],
    eligible: pf.ProcNode[bool],
    edge_margin: t.SocketOrVal[float],
    min_surface: t.SocketOrVal[float],
) -> pf.ProcNode[bool]:
    """Per-point selection: inside the upward region's edge-inset XY bbox, where
    that region's XY extent exceeds min_surface."""
    region = pf.nodes.geo.separate_geometry(
        geometry=parent, selection=eligible, domain="FACE"
    )
    bb = pf.nodes.geo.bound_box(region.selection)
    extent = pf.nodes.math.separate_xyz(bb.max - bb.min)
    big_enough = pf.nodes.func.boolean_and(
        a=pf.nodes.func.greater_than(a=extent.x, b=min_surface),
        b=pf.nodes.func.greater_than(a=extent.y, b=min_surface),
    )
    margin = pf.nodes.math.combine_xyz(x=edge_margin, y=edge_margin, z=0.0)
    lo = pf.nodes.math.separate_xyz(bb.min + margin)
    hi = pf.nodes.math.separate_xyz(bb.max - margin)
    p = pf.nodes.math.separate_xyz(pf.nodes.geo.input_position())
    inside = pf.nodes.func.boolean_and(
        a=pf.nodes.func.boolean_and(
            a=pf.nodes.func.greater_equal(a=p.x, b=lo.x),
            b=pf.nodes.func.less_equal(a=p.x, b=hi.x),
        ),
        b=pf.nodes.func.boolean_and(
            a=pf.nodes.func.greater_equal(a=p.y, b=lo.y),
            b=pf.nodes.func.less_equal(a=p.y, b=hi.y),
        ),
    )
    return pf.nodes.func.boolean_and(a=inside, b=big_enough)


@pf.nodes.node_function
def smallobj_scatter(
    parent: pf.ProcNode[pf.MeshObject],
    child: pf.ProcNode[pf.Collection],
    seed: t.SocketOrVal[int] = 0,
    density: t.SocketOrVal[float] = 20.0,
    distance_min: t.SocketOrVal[float] = 0.1,
    edge_margin: t.SocketOrVal[float] = 0.1,
    rotation_randomness: t.SocketOrVal[float] = 1.0,
    offset: t.SocketOrVal[pf.Vector] = (0, 0, 0.002),
    min_surface: t.SocketOrVal[float] = 0.2,
    instance_index: t.SocketOrVal[int] = 0,
) -> pf.ProcNode[t.Instances]:
    eligible = pf.nodes.func.greater_than(
        a=pf.nodes.geo.input_normal().z, b=_UP_NORMAL_Z
    )
    points = pf.nodes.geo.distribute_points_on_faces_poisson(
        mesh=parent,
        selection=eligible,
        seed=seed,
        density_factor=1.0,
        density_max=density,
        distance_min=distance_min,
    )
    z_rot = (
        pf.nodes.func.random_value(min=0.0, max=_TWO_PI, seed=seed)
        * rotation_randomness
    )
    instances = pf.nodes.geo.instance_on_points(
        points=points.points,
        instance=pf.nodes.geo.collection_info(child, separate_children=True),
        pick_instance=True,
        instance_index=instance_index,
        selection=_scatter_region(parent, eligible, edge_margin, min_surface),
        rotation=pf.nodes.math.combine_xyz(z=z_rot).astype(dtype=pf.Euler),
    )
    return pf.nodes.geo.transform(instances, translation=offset)


def _clamped_extent(obj) -> tuple[float, float, float]:
    bmin, bmax = pf.ops.attr.bbox_min_max(obj, global_coords=False)
    return tuple(max(_MIN_DIM, hi - lo) for lo, hi in zip(bmin, bmax, strict=True))


def _scatter_on_target(
    rng: pf.RNG,
    parent: pf.MeshObject,
    pool: list[pf.MeshObject],
    colliders: ccol.CollisionSet,
) -> tuple[list[pf.MeshObject], ccol.CollisionSet]:
    k = int(pf.random.randint(rng, 1, min(_MAX_DISTINCT, len(pool)) + 1))
    child = pf.Collection(
        [pool[i] for i in rng.choice(len(pool), size=k, replace=False)]
    )

    extents = [_clamped_extent(o) for o in child]
    avg_area = sum(e[0] * e[1] for e in extents) / len(extents)
    max_xy = max(max(e[:2]) for e in extents)
    distance_min = min(max_xy * pf.random.uniform(rng, 0.2, 0.5), 0.07)
    fullness = pf.random.clip_gaussian(rng, 0.5, 0.3, 0.1, 0.9)
    density = min(max(fullness / avg_area, 5.0), 300.0)

    geometry = smallobj_scatter(
        parent=parent,
        child=child,
        seed=int(pf.random.randint(rng, 0, 2**31 - 1)),
        density=density,
        distance_min=distance_min,
        edge_margin=pf.random.uniform(rng, 0.0, 0.06),
        rotation_randomness=pf.random.uniform(rng, 0.0, 1.0),
        offset=_OFFSET,
        min_surface=_MIN_SURFACE,
        instance_index=pf.nodes.func.random_value(
            min=0,
            max=len(extents) - 1,
            seed=int(pf.random.randint(rng, 0, 2**31 - 1)),
            id=pf.nodes.geo.input_index(),
        ),
    )
    instances = pf.nodes.to_aliases(geometry)
    for alias in instances:
        alias.item().matrix_world = (
            parent.item().matrix_world @ alias.item().matrix_world
        )
        alias.item().name = alias.item().data.name
    instances, colliders = keep_non_colliding(instances, colliders, key=lambda o: o)
    return instances, colliders


def small_objects_collection_distribution(rng: pf.RNG) -> pf.Collection:
    """One room-wide pool of small primitives to draw from per target. Each
    primitive's origin is dropped to its base so instances sit on the surface."""
    n_pool = int(pf.random.randint(rng, _POOL_MIN, _POOL_MAX + 1))
    meshes = []
    for rng_mesh in rng.spawn(n_pool):
        mesh = random_primitives.primitives_distribution(
            rng_mesh,
            target_size=pf.random.clip_gaussian(rng_mesh, 0.24, 0.14, 0.02, 0.8),
        ).mesh
        pf.ops.mesh.transform_apply(mesh)
        bmin, _ = pf.ops.attr.bbox_min_max(mesh, global_coords=False)
        pf.ops.object.set_transform(mesh, location=(0, 0, -bmin[2]))
        pf.ops.mesh.transform_apply(mesh)
        meshes.append(mesh)
    return pf.Collection(meshes)


def place_small_objects_on_targets(
    rng: pf.RNG,
    targets: list[pf.MeshObject],
    pool: pf.Collection,
    colliders: ccol.CollisionSet,
    skip_prob: float = _SKIP_OBJECT_PROB,
) -> tuple[list[pf.MeshObject], ccol.CollisionSet]:
    """Poisson-scatter small objects on each target's real surface; each target
    skipped with prob skip_prob. Returns placed + colliders."""
    pool_list = list(pool)
    instances: list[pf.MeshObject] = []
    for rng_target, parent in zip(rng.spawn(len(targets)), targets, strict=True):
        if pf.random.uniform(rng_target, 0.0, 1.0) < skip_prob:
            continue
        placed, colliders = _scatter_on_target(rng_target, parent, pool_list, colliders)
        instances.extend(placed)
    return instances, colliders
