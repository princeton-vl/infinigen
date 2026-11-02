# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Alexander Raistrick: original Infinigen asset placement (https://github.com/princeton-vl/infinigen/blob/224598d4a2ff95b4fc928e72c222598b7122c698/src/infinigen/core/placement/placement.py)
# - David Yan, Jia Deng: optimized implementation

from contextlib import nullcontext

import bpy
import mathutils
import numpy as np
from tqdm import tqdm

from infinigen.core import surface
from infinigen.core.nodes.node_wrangler import (
    Nodes,
    NodeWrangler,
    geometry_node_group_empty_new,
)
from infinigen.core.placement import detail, split_in_view
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.placement.placement import (
    get_placeholder_points,
    logger,
    parse_asset_name,
)
from infinigen.core.util import blender as butil
from infinigen.perf.core.placement.factory import _isolation


def placeholder_locs(
    terrain, overall_density, selection, distance_min=0, altitude=0.0, max_locs=None
):
    temp_vert = butil.spawn_vert("compute_placeholder_locations")
    geo = temp_vert.modifiers.new(name="GEOMETRY", type="NODES")
    if geo.node_group is None:
        group = geometry_node_group_empty_new()
        geo.node_group = group
    nw = NodeWrangler(geo)

    base_geo = nw.new_node(Nodes.ObjectInfo, [terrain]).outputs["Geometry"]

    points = nw.new_node(
        Nodes.DistributePointsOnFaces,
        attrs={"distribute_method": "POISSON"},
        input_kwargs={
            "Mesh": base_geo,
            "Selection": surface.eval_argument(nw, selection),
            "Seed": np.random.randint(1e5),
            "Density Max": overall_density,
            "Distance Min": distance_min,
        },
    )
    verts = nw.new_node(Nodes.PointsToVertices, input_kwargs={"Points": points})
    verts = nw.new_node(
        Nodes.SetPosition, input_kwargs={"Geometry": verts, "Offset": (0, 0, altitude)}
    )

    nw.new_node(Nodes.GroupOutput, input_kwargs={"Geometry": verts})

    # dump the point locations out as vertices
    butil.apply_modifiers(temp_vert, geo)
    locations = np.empty((len(temp_vert.data.vertices), 3), dtype=np.float64)
    temp_vert.data.vertices.foreach_get("co", locations.ravel())

    temp_mesh = temp_vert.data
    bpy.data.objects.remove(temp_vert, do_unlink=True)
    if temp_mesh.users == 0:
        bpy.data.meshes.remove(temp_mesh)

    np.random.shuffle(locations)

    return locations


def scatter_placeholders(locations, factory: AssetFactory):
    logger.info(f"Placing {len(locations)} placeholders for {factory}")
    objs = []
    # one batch isolation context for the whole loop, so finished placeholders
    # are parked out of the scratch view layer
    with _isolation(factory, batch=True):
        for i, loc in enumerate(tqdm(locations)):
            rot_z = np.random.uniform(0, 2 * np.pi)
            obj = factory.spawn_placeholder(i, loc, mathutils.Euler((0, 0, rot_z)))
            objs.append(obj)
    col = butil.group_in_collection(objs, "placeholders:" + repr(factory))
    factory.finalize_placeholders(objs)
    return col


def filter_populate_targets(
    placeholders: list[bpy.types.Object],
    cameras: list[bpy.types.Object],
    dist_cull: float,
    vis_cull: float,
    verbose: bool,
) -> list[tuple[bpy.types.Object, float, float]]:
    placeholder_iter = tqdm(placeholders) if verbose else placeholders

    placeholder_points = []
    for p in placeholder_iter:
        classname, *_ = parse_asset_name(p.name)
        if classname is None:
            raise ValueError(f"Could not parse {p.name=}, got {classname=}")
        placeholder_points.append(get_placeholder_points(p))

    if not placeholder_points:
        return []

    point_counts = np.fromiter(
        (len(points) for points in placeholder_points), dtype=np.int64
    )
    offsets = np.empty(len(point_counts) + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(point_counts, out=offsets[1:])
    mask, min_dists, min_vis_dists = split_in_view.compute_inview_distances(
        np.concatenate(placeholder_points),
        cameras,
        dist_max=dist_cull,
        vis_margin=vis_cull,
        verbose=False,
    )

    results = []
    for p, start, stop in zip(placeholders, offsets[:-1], offsets[1:]):
        placeholder_mask = mask[start:stop]
        dist = min_dists[start:stop].min()
        vis_dist = min_vis_dists[start:stop].min()

        if not placeholder_mask.any():
            logger.debug(
                f"{p.name=} culled, not in view of any camera. {dist=} {vis_dist=}"
            )
            continue

        results.append((p, dist, vis_dist))

    return results


def populate_collection(
    factory: AssetFactory,
    placeholder_col: bpy.types.Collection,
    cameras: list[bpy.types.Object] = None,
    asset_col_target=None,
    dist_cull=None,
    vis_cull=None,
    verbose=True,
    cache_system=None,
    **asset_kwargs,
):
    if asset_col_target is None:
        asset_col_target = butil.get_collection(f"unique_assets:{repr(factory)}")

    placeholders = [o for o in placeholder_col.objects if o.parent is None]

    if cameras is not None:
        logger.info(f"Checking visibility for {placeholder_col.name=}")
        targets = filter_populate_targets(
            placeholders, cameras, dist_cull, vis_cull, verbose
        )
    else:
        targets = [(p, detail.scatter_res_distance(), 0) for p in placeholders]

    print(
        f"Populating {len(targets)} placeholders for {factory=} out of {len(placeholders)} total"
    )

    all_objs = []
    updated_pholders = []

    if verbose:
        targets = tqdm(targets)

    # one batch isolation context for the whole loop, and none when there is
    # nothing to populate, since switching back to the main view layer is not
    # free in a large scene
    with _isolation(factory, batch=True) if len(targets) else nullcontext():
        for i, (p, dist, vis_dist) in enumerate(targets):
            classname, inst_seed, *_ = parse_asset_name(p.name)

            if cache_system:
                if (
                    sum(cache_system.n_placed.values()) < cache_system.max_fire_assets
                    and cache_system.n_placed[factory.__class__.__name__]
                    < cache_system.max_per_kind
                ):
                    i_list = cache_system.find_i_list(factory)
                    ind = np.random.choice(len(i_list))
                    i_chosen, full_sim_folder, sim_folder = i_list[ind]
                    obj = factory.spawn_asset(
                        int(i_chosen),
                        placeholder=p,
                        distance=dist,
                        vis_distance=vis_dist,
                    )
                    cache_system.link_fire(full_sim_folder, sim_folder, obj, factory)
                else:
                    break

                continue

            obj = factory.spawn_asset(
                i, placeholder=p, distance=dist, vis_distance=vis_dist, **asset_kwargs
            )

            if p is not obj:
                p.hide_render = True

            for o in butil.iter_object_tree(obj):
                butil.put_in_collection(o, asset_col_target)

            obj["dist"] = dist
            obj["vis_dist"] = vis_dist

            updated_pholders.append((inst_seed, p))
            all_objs.append((inst_seed, obj))

    asset_col_target.hide_viewport = False
    factory.finalize_assets([r for i, r in all_objs])
    asset_col_target.hide_viewport = True

    return all_objs, updated_pholders
