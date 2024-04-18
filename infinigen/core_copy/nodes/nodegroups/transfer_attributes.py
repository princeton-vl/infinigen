# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick


import bpy
import mathutils
import numpy as np
from numpy.random import uniform, normal
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.nodes import node_utils, node_info
from infinigen.core import surface
from infinigen.core.util import blender as butil


def uvs_to_attribute(obj, name='uv_map'):
    assert obj.type == 'MESH'
    
    n = len(obj.data.vertices)
    data = np.empty((n, 3), dtype=np.float32)

    for loop in obj.data.loops:
        u, v = obj.data.uv_layers.active.data[loop.index].uv
        data[loop.vertex_index] = u, v, 0

    attr = obj.data.attributes.new(name, type='FLOAT_VECTOR', domain='POINT')
    attr.data.foreach_set('vector', data.reshape(-1))

    return attr

def attribute_to_uvs(obj, attr_name):

    assert obj.type == 'MESH'

    obj.data.uv_layers.active = obj.data.uv_layers.new()
    
    n = len(obj.data.vertices)
    data = np.empty(n * 3, dtype=np.float32)
    obj.data.attributes[attr_name].data.foreach_get('vector', data)
    data = data.reshape((n, 3))

    for loop in obj.data.loops:
        u, v, _ = data[loop.vertex_index]
        obj.data.uv_layers.active.data[loop.index].uv = (u, v)

# list of supported data type:
# https://docs.blender.org/api/current/bpy.types.GeometryNodeCaptureAttribute.html

def transfer_all(source, target, attributes=None, uvs=False):
    assert source.type == 'MESH'
    assert target.type == 'MESH'

    if attributes is None:
        attributes = [a.name for a in source.data.attributes if not butil.blender_internal_attr(a)]

    if len(source.data.uv_layers) == 0:
        uvs = False

    if uvs:
        uv_att_name = uvs_to_attribute(source).name
        attributes.append(uv_att_name)

    dtypes = [source.data.attributes[n].data_type for n in attributes]
    domains = [source.data.attributes[n].domain for n in attributes]

    surface.add_geomod(source, transfer_att_node,
                       input_kwargs={'source': source,
                                     'target': target,
                                     'attribute_to_transfer_list': list(zip(attributes, dtypes))},
                       attributes=attributes, apply=True, domains=domains)

    surface.add_geomod(target, copy_geom_info,
                       input_kwargs={'source': source, 'target': target},
                       apply=True)

    if uvs:
        attribute_to_uvs(target, uv_att_name)


def copy_geom_info(nw, source, target):
    # simply copy the geom back to the target from source
    object_info = nw.new_node(Nodes.ObjectInfo, input_kwargs={'Object': source})
    group_output = nw.new_node(Nodes.GroupOutput, input_kwargs={'Geometry': object_info.outputs["Geometry"], })


def transfer_att_node(nw, source, target, attribute_to_transfer_list=[]):
    # create a geom node in the non-remeshed version of the mesh (i.e., source)
    object_info = nw.new_node(Nodes.ObjectInfo, input_kwargs={'Object': target})
    group_input = nw.new_node(Nodes.GroupInput,
                              expose_input=[('NodeSocketGeometry', 'Geometry', None), ])

    for att_name, att_type in attribute_to_transfer_list:
        nw.expose_input(att_name, attribute=att_name)

    position = nw.new_node(Nodes.InputPosition)

    group_output_sockets = {'Geometry': object_info.outputs["Geometry"]}

    for att_name, att_type in attribute_to_transfer_list:
        transfer_attribute = nw.new_node(
            Nodes.SampleNearestSurface,
            attrs={'data_type': att_type},
            input_kwargs={
                'Mesh': group_input.outputs["Geometry"],
                'Value': group_input.outputs[att_name],
                'Sample Position': position
            })

        group_output_sockets[att_name] = (transfer_attribute, 'Value')

    group_output = nw.new_node(Nodes.GroupOutput, input_kwargs=group_output_sockets)
