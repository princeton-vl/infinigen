# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors: Alexander Raistrick

import logging
import typing

import bpy
import gin
import numpy as np
from mathutils import Vector
from numpy.random import uniform

from infinigen.assets import weather
from infinigen.assets.materials import invisible_to_camera
from infinigen.assets.scatters import grass, pebbles
from infinigen.core import tags as t
from infinigen.core.constraints import constraint_language as cl
from infinigen.core.constraints import reasoning as r
from infinigen.core.constraints import usage_lookup
from infinigen.core.placement import density, split_in_view
from infinigen.core.util import blender as butil
from infinigen.core.util import pipeline
from infinigen.core.util.camera import points_inview
from infinigen.terrain import Terrain, hidden_in_viewport
from infinigen.terrain.utils import Mesh
from infinigen_examples.constraints import util as cu

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def within_bbox_2d(verts, bbox):
    return (
        (verts[:, 0] > bbox[0][0])
        & (verts[:, 0] < bbox[1][0])
        & (verts[:, 1] > bbox[0][1])
        & (verts[:, 1] < bbox[1][1])
    )


def create_outdoor_backdrop(
    terrain: Terrain,
    house_bbox: tuple,
    cameras: list[bpy.types.Object],
    p: pipeline.RandomStageExecutor,
    params: dict,
):
    all_vertices = []
    for name in terrain.terrain_objs:
        if name not in hidden_in_viewport:
            all_vertices.append(Mesh(obj=terrain.terrain_objs[name]).vertices)

    all_vertices = np.concatenate(all_vertices)
    all_mask = within_bbox_2d(all_vertices, house_bbox)

    if not all_mask.any():
        height = 0
    else:
        height = all_vertices[all_mask, 2].max()

    extra_zoff = uniform(0, 4)  # deliberately float above the terrain.
    height += extra_zoff

    for obj in terrain.terrain_objs.values():
        obj.location[2] -= height
        butil.apply_transform(obj, loc=True)

    main_terrain = bpy.data.objects["OpaqueTerrain"]
    verts = np.zeros(3 * len(main_terrain.data.vertices), float)
    main_terrain.data.vertices.foreach_get("co", verts)
    verts = verts.reshape(-1, 3)
    mask = within_bbox_2d(verts, house_bbox)

    with butil.ViewportMode(main_terrain, mode="EDIT"):
        bpy.ops.mesh.select_mode(use_extend=False, use_expand=False, type="FACE")
        bpy.ops.mesh.select_all(action="DESELECT")
    split_in_view.select_vertmask(main_terrain, mask)
    with butil.ViewportMode(main_terrain, mode="EDIT"):
        bpy.ops.mesh.select_more()
        bpy.ops.mesh.delete(type="VERT")

    p.run_stage("fancy_clouds", weather.kole_clouds.add_kole_clouds)

    terrain_inview, *_ = split_in_view.split_inview(
        main_terrain,
        cameras,
        verbose=True,
        outofview=False,
        vis_margin=2,
        dist_max=params["near_distance"],
        hide_render=True,
        suffix="inview",
    )

    def add_grass(target):
        select_max = params.get("grass_select_max", 0.5)
        selection = density.placement_mask(
            normal_dir=(0, 0, 1),
            scale=0.1,
            return_scalar=True,
            select_thresh=uniform(select_max / 2, select_max),
        )
        grass.apply(target, selection=selection)

    p.run_stage("grass", add_grass, terrain_inview)

    def add_rocks(target):
        selection = density.placement_mask(
            scale=0.15, select_thresh=0.5, normal_thresh=0.7, return_scalar=True
        )
        _, rock_col = pebbles.apply(target, selection=selection)
        return rock_col

    p.run_stage("rocks", add_rocks, terrain_inview)
    return height


def place_cam_overhead(cam: bpy.types.Object, bbox: tuple[np.array]):
    butil.spawn_point_cloud("place_cam_overhead", bbox)

    mins, maxs = bbox
    cam.location = (maxs + mins) / 2
    cam.rotation_euler = (0, 0, 0)
    for cam_dist in np.exp(np.linspace(-1.0, 5.5, 500)):
        cam.location[-1] = cam_dist
        bpy.context.view_layer.update()
        inview = points_inview(bbox, cam.children[0])
        if inview.all():
            for area in bpy.context.screen.areas:
                if area.type == "VIEW_3D":
                    area.spaces.active.region_3d.view_perspective = "CAMERA"
                    break
            return


def overhead_view(cam, room_name):
    room_name = room_name.split(".")[0]

    for o in bpy.data.objects:
        if ".exterior" in o.name:
            o.hide_viewport = True
            o.hide_render = True
        elif ".ceiling" in o.name:
            invisible_to_camera.apply(o)

    floor = bpy.data.objects[room_name + ".floor"]
    cam.location = floor.location + Vector((0, 0, 10))
    cam.rotation_euler = (0, 0, 0)


def hide_other_rooms(state, rooms_split, keep_rooms: list[str]):
    for col in rooms_split.values():
        for o in col.objects:
            if any(roomname.split(".")[0] in o.name for roomname in keep_rooms):
                continue
            o.hide_viewport = True
            o.hide_render = True

    hide_cutters = [
        o
        for k, os in state.objs.items()
        if t.Semantics.Cutter in os.tags
        and not any(
            rel.target_name == roomname
            for rel in os.relations
            for roomname in keep_rooms
        )
        for o in butil.iter_object_tree(os.obj)
    ]
    for o in hide_cutters:
        o.hide_render = True
        o.hide_viewport = True
    bpy.context.scene.render.film_transparent = True


def apply_greedy_restriction(
    stages: dict[str, r.Domain],
    filter_tags: set[str],
    var: t.Variable,
    scope_domain: r.Domain = None,
):
    filter_tags = t.to_tag_set(filter_tags, fac_context=usage_lookup._factory_lookup)
    for k, d in stages.items():
        if scope_domain is not None and not d.intersects(scope_domain):
            continue
        stages[k], match = r.domain_tag_substitute(
            d, var, r.Domain(filter_tags).with_tags(var), return_match=True
        )
        logger.info(
            f"{apply_greedy_restriction.__name__} restricting {k=} to {filter_tags=} for {var=}"
        )


@gin.configurable
def restrict_solving(
    stages: dict[str, r.Domain],
    problem: cl.Problem,
    # typically provided by gin
    restrict_parent_rooms: set[str] = None,
    restrict_parent_objs: set[str] = None,
    restrict_child_primary: set[str] = None,
    restrict_child_secondary: set[str] = None,
    solve_max_rooms: int = None,
    solve_max_parent_obj: int = None,
    consgraph_filters: typing.Iterable[str] = None,
):
    """Restricts solving to a subset of the full house or constraint graph.

    Parameters
    ----------
    stages : the original set of greedy stages
    problem : the original constraint specification
    restrict_parent_rooms : limit solving to only use these rooms as parent rooms
    restrict_parent_objs : limit solving to only use these objects as parent objects
    restrict_child_primary : limit solving to only place primary objects of these types (e.g, only place diningtables, no shelves etc)
    restrict_child_secondary : if specified, limit solving to only place secondary objects of these types (e.g only place mugs, no plates etc)
    solve_max_rooms : only place objects in at most this many rooms (e.g.. only 1 room has objects in it)
    solve_max_parent_obj : only place objects onto at most this many parent objects (e.g. only 1 shelf has objects on it)

    Returns
    -------
    stages : the modified set of greedy stages
    problem : the modified constraint specification
    limits : set of object-quantity-limits for solving

    """

    obj_domain = r.Domain({t.Semantics.Object})
    primary_obj_domain = r.Domain(
        {t.Semantics.Object}, [(-cl.AnyRelation(), obj_domain)]
    )
    secondary_obj_domain = r.Domain(
        {t.Semantics.Object}, [(cl.AnyRelation(), obj_domain)]
    )

    if restrict_parent_rooms is not None:
        apply_greedy_restriction(stages, restrict_parent_rooms, cu.variable_room)

    if restrict_parent_objs is not None:
        apply_greedy_restriction(stages, restrict_parent_objs, cu.variable_obj)

    if restrict_child_primary is not None:
        restrict_child_primary = t.to_tag_set(
            restrict_child_primary, fac_context=usage_lookup._factory_lookup
        )
        for k, d in stages.items():
            if d.intersects(primary_obj_domain):
                logger.info(
                    f"restrict_solving applying restrict_child_primary, limiting {k} to objects satisfying {restrict_child_primary}"
                )
                stages[k] = d.intersection(r.Domain(restrict_child_primary))

    if restrict_child_secondary is not None:
        restrict_child_secondary = t.to_tag_set(
            restrict_child_secondary, fac_context=usage_lookup._factory_lookup
        )
        for k, d in stages.items():
            if d.intersects(secondary_obj_domain):
                logger.info(
                    f"restrict_solving applying restrict_child_secondary, limiting {k} to objects satisfying {restrict_child_primary}"
                )
                stages[k] = d.intersection(r.Domain(restrict_child_secondary))

    quantity_limits = {
        cu.variable_room: solve_max_rooms,
        cu.variable_obj: solve_max_parent_obj,
    }

    if consgraph_filters is not None:
        if isinstance(consgraph_filters, str):
            consgraph_filters = [consgraph_filters]
        assert isinstance(consgraph_filters, typing.Iterable)
        old_counts = (len(problem.constraints), len(problem.score_terms))

        def filter(d):
            return {
                k: v for k, v in d.items() if any(fi in k for fi in consgraph_filters)
            }

        problem = cl.Problem(filter(problem.constraints), filter(problem.score_terms))

        new_counts = (len(problem.constraints), len(problem.score_terms))
        logger.info(
            f"restrict_solving filtered consgraph from {old_counts=} {new_counts=} using {consgraph_filters=}"
        )

    return stages, problem, quantity_limits
