# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Beining Han

import bpy
import numpy as np
from infinigen.core.util import blender as butil
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler,  geometry_node_group_empty_new
from infinigen.core.nodes import node_utils


def get_nodegroup_assets(func, params):
    bpy.ops.mesh.primitive_plane_add(
        size=1, enter_editmode=False, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
    obj = bpy.context.active_object

    with butil.TemporaryObject(obj) as base_obj:
        node_group_func = func(**params)
        geo_outputs = [o for o in node_group_func.outputs if o.bl_socket_idname == 'NodeSocketGeometry']
        results = {o.name: extract_nodegroup_geo(base_obj, node_group_func, o.name,
                                                 ng_params={}) for o in geo_outputs}

    return results

@node_utils.to_nodegroup('nodegroup_tagged_cube', singleton=False, type='GeometryNodeTree')
def nodegroup_tagged_cube(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler


    cube = nw.new_node(Nodes.MeshCube, input_kwargs={'Size': group_input.outputs["Size"]})








def blender_rotate(vec):
    if isinstance(vec, tuple):
        vec = list(vec)
    if isinstance(vec, list):
        vec = np.array(vec, dtype=np.float32)
    if len(vec.shape) == 1:
        vec = np.expand_dims(vec, axis=-1)
    if vec.shape[0] == 3:
        new_vec = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float32) @ vec
        return new_vec.squeeze()
    if vec.shape[0] == 4:
        new_vec = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=np.float32) @ vec
        return new_vec.squeeze()
