# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Karhan Kayan

import logging

import gin
from shapely.geometry import MultiPolygon, Point, Polygon

import infinigen.core.constraints.constraint_language as cl
from infinigen.core import tags as t
from infinigen.core.constraints.constraint_language.util import (
    blender_objs_from_names,
    meshes_from_names,
    project_to_xy_poly,
)
from infinigen.core.constraints.evaluator.node_impl.trimesh_geometry import (
    any_touching,
    constrain_contact,
)
from infinigen.core.constraints.example_solver.geometry.stability import (
    coplanar,
    stable_against,
)
from infinigen.core.constraints.example_solver.state_def import State
from infinigen.core.util import blender as butil

logger = logging.getLogger(__name__)


def check_pre_move_validity(scene, a, parent_dict, dx, dy):
    """ """
    parent = parent_dict[a]
    a_mesh = meshes_from_names(scene, a)[0]
    parent_mesh = meshes_from_names(scene, parent)[0]
    blender_objs_from_names(a)[0]

    # move a mesh by dx, dy and check if the projection of a_mesh is contained in parent_mesh
    # a_mesh.apply_transform(trimesh.transformations.compose_matrix(translate=[dx,dy,0]))
    a_poly = project_to_xy_poly(a_mesh)
    parent_poly = project_to_xy_poly(parent_mesh)
    centroid = a_poly.centroid
    new_centroid = Point([centroid.x + dx, centroid.y + dy])
    # plot
    # fig, ax = plt.subplots()
    # if isinstance(parent_poly, Polygon):
    #     x, y = parent_poly.exterior.xy
    #     ax.fill(x, y, alpha=0.5, fc='red', ec='black', label='Polygon b')
    # elif isinstance(parent_poly, MultiPolygon):
    #     for sub_poly in parent_poly.geoms:
    #         x, y = sub_poly.exterior.xy
    #         ax.fill(x, y, alpha=0.5, fc='red', ec='black', label='Polygon b')
    # ax.plot(centroid.x, centroid.y, 'o', color='black', label='Random point')
    # plt.show()
    # scene.show()

    if isinstance(a_poly, Polygon):
        if not parent_poly.contains(new_centroid):
            # print("not contained")
            return False
    elif isinstance(a_poly, MultiPolygon):
        for sub_poly in a_poly.geoms:
            if not parent_poly.contains(new_centroid):
                # print("not contained")
                return False

    return True


def all_relations_valid(state, name):
    rels = state.objs[name].relations
    for i, relation_state in enumerate(rels):
        match relation_state.relation:
            case cl.StableAgainst(_child_tags, _parent_tags, _margin):
                res = stable_against(state, name, relation_state)
                if not res:
                    logger.debug(
                        f"{name} failed relation {i=}/{len(rels)} {relation_state.relation} on {relation_state.target_name}"
                    )
                    return False

            case cl.CoPlanar(_child_tags, _parent_tags, _margin):
                res = coplanar(state, name, relation_state)
                if not res:
                    logger.debug(
                        f"{name} failed relation {i=}/{len(rels)} {relation_state.relation} on {relation_state.target_name}"
                    )
                    return False
            case _:
                raise TypeError(f"Unhandled {relation_state.relation}")

    return True


@gin.configurable
def check_post_move_validity(
    state: State, name: str, disable_collision_checking=False, visualize=False
):
    scene = state.trimesh_scene
    objstate = state.objs[name]

    collision_objs = [
        os.obj.name
        for k, os in state.objs.items()
        if k != name and t.Semantics.NoCollision not in os.tags
    ]

    if len(collision_objs) == 0:
        return True

    if not all_relations_valid(state, name):
        if visualize:
            vis_obj = butil.copy(objstate.obj)
            vis_obj.name = f"validity_relations_fail_{name}"

        return False

    if disable_collision_checking:
        return True
    if t.Semantics.NoCollision in objstate.tags:
        return True

    touch = any_touching(
        scene, objstate.obj.name, collision_objs, bvh_cache=state.bvh_cache
    )
    if not constrain_contact(touch, should_touch=None, max_depth=0.0001):
        if visualize:
            vis_obj = butil.copy(objstate.obj)
            vis_obj.name = f"validity_contact_fail_{name}"

        contact_names = [
            [x for x in t.names if not x.startswith("_")] for t in touch.contacts
        ]
        logger.debug(
            f"validity failed - {name} touched {contact_names[0]} {len(contact_names)=}"
        )
        return False

    # supposed to go through the consgraph here
    return True
