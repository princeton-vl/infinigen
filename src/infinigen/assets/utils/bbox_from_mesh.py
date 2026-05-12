# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors: Alexander Raistrick

import bpy
import numpy as np

from infinigen.core.nodes import node_utils
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.util import blender as butil


@node_utils.to_nodegroup("nodegroup_cube_from_corners", singleton=True)
def nodegroup_cube_from_corners(nw: NodeWrangler):
    # Code generated using version 2.6.5 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketVector", "min_corner", (0.0000, 0.0000, 0.0000)),
            ("NodeSocketVector", "max_corner", (0.0000, 0.0000, 0.0000)),
        ],
    )

    subtract = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={
            0: group_input.outputs["max_corner"],
            1: group_input.outputs["min_corner"],
        },
        attrs={"operation": "SUBTRACT"},
    )

    cube = nw.new_node(
        Nodes.MeshCube, input_kwargs={"Size": subtract.outputs["Vector"]}
    )

    mix = nw.new_node(
        Nodes.Mix,
        input_kwargs={
            4: group_input.outputs["min_corner"],
            5: group_input.outputs["max_corner"],
        },
        attrs={"data_type": "VECTOR"},
    )

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": cube.outputs["Mesh"], "Translation": mix.outputs[1]},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput, input_kwargs={"Geometry": transform_geometry}
    )


def union_all_bbox(obj: bpy.types.Object):
    mins, maxs = None, None
    for oc in butil.iter_object_tree(obj):
        if not oc.type == "MESH":
            continue
        points = butil.apply_matrix_world(oc, np.array(oc.bound_box))
        pmins, pmaxs = points.min(axis=0), points.max(axis=0)
        mins = pmins if mins is None else np.minimum(pmins, mins)
        maxs = pmaxs if maxs is None else np.maximum(pmins, mins)

    return mins, maxs


def box_from_corners(min_corner, max_corner):
    bbox = butil.modify_mesh(
        butil.spawn_vert(),
        "NODES",
        apply=True,
        node_group=nodegroup_cube_from_corners(),
        ng_inputs=dict(min_corner=min_corner, max_corner=max_corner),
    )

    return bbox


def bbox_mesh_from_hipoly(gen: AssetFactory, inst_seed: int, use_pholder=False):
    objs = []
    objs.append(gen.spawn_placeholder(inst_seed, loc=(0, 0, 0), rot=(0, 0, 0)))
    if not use_pholder:
        objs.append(gen.spawn_asset(inst_seed, placeholder=objs[-1]))

    min_corner, max_corner = union_all_bbox(objs[-1])

    if (
        min_corner is None
        or max_corner is None
        or np.abs(min_corner - max_corner).sum() < 1e-5
    ):
        raise ValueError(
            f"{gen} spawned {objs[-1].name=} with total bbox {min_corner, max_corner}, invalid"
        )

    bbox = box_from_corners(min_corner, max_corner)

    cleanup = set()
    for o in objs:
        cleanup.update(butil.iter_object_tree(o))
    butil.delete(list(cleanup))

    bbox.name = (
        f"{gen.__class__.__name__}({gen.factory_seed}).bbox_placeholder({inst_seed})"
    )
    return bbox
