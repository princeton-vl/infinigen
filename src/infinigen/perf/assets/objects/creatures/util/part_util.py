# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Alexander Raistrick: original Infinigen creature part utilities (https://github.com/princeton-vl/infinigen/blob/224598d4a2ff95b4fc928e72c222598b7122c698/src/infinigen/assets/objects/creatures/util/part_util.py)
# - David Yan, Jia Deng: optimized implementation

import logging

import numpy as np

from infinigen.assets.objects.creatures.util.creature import (
    Part,
    infer_skeleton_from_mesh,
)
from infinigen.assets.utils.extract_nodegroup_parts import extract_nodegroup_geo
from infinigen.core.nodes.node_wrangler import ng_outputs
from infinigen.core.util import blender as butil
from infinigen.perf.assets.objects.creatures.util import mesh_join


def nodegroup_to_part(
    nodegroup_func, params, kwargs=None, base_obj=None, split_extras=False
):
    if base_obj is None:
        base_obj = butil.spawn_vert("temp")

    with butil.TemporaryObject(base_obj) as base_obj:
        if kwargs is not None:
            ng = nodegroup_func(**kwargs)
        else:
            ng = nodegroup_func()
        # a list, in the group's own order: a set of sockets iterates in
        # address order, which differs from run to run
        geo_outputs = [
            o
            for o in ng_outputs(ng).values()
            if o.bl_socket_idname == "NodeSocketGeometry"
        ]
        objs = {
            o.name: extract_nodegroup_geo(base_obj, ng, o.name, ng_params=params)
            for o in geo_outputs
        }

    skin_obj = objs.pop("Geometry", None)
    if skin_obj is None:
        skin_obj = butil.spawn_vert("nodegroup_to_part.no_geo_temp")

    attach_basemesh = objs.pop("Base Mesh", None)

    if "Skeleton Curve" in objs:
        skeleton_obj = objs.pop("Skeleton Curve")
        skeleton = np.array([v.co for v in skeleton_obj.data.vertices])
        if len(skeleton) == 0:
            raise ValueError(
                f"Skeleton export failed for {nodegroup_func}, {skeleton_obj}, got {skeleton.shape=}"
            )
        butil.delete(skeleton_obj)
    else:
        skeleton = infer_skeleton_from_mesh(skin_obj)

    # Handle any 'Extras' exported by the nodegroup
    for k, o in objs.items():
        if split_extras:
            for i, piece in enumerate(butil.split_object(o)):
                logging.debug(
                    f"Processing piece {i} for split_extras on {nodegroup_func}"
                )
                mesh_join.origin_to_geometry(piece, no_children=True)
                piece.name = f"{k}_{i}"
                piece.parent = skin_obj
        else:
            o.parent = skin_obj
            o.name = k

    return Part(
        skeleton, obj=skin_obj, attach_basemesh=attach_basemesh, joints=None, iks=None
    )
