import numpy as np
from nodes import node_utils, node_info
from util import blender as butil

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

def transfer_all(source, target, attributes=None, uvs=False):
    assert source.type == 'MESH'
    assert target.type == 'MESH'

    if attributes is None:
        attributes = [a.name for a in source.data.attributes if a.data_type in node_info.DATATYPE_DIMS]

    if len(source.data.uv_layers) == 0:
        uvs = False

    if uvs:
        uv_att_name = uvs_to_attribute(source).name
        attributes.append(uv_att_name)

    dtypes = [source.data.attributes[n].data_type for n in attributes]
    if uvs:
        attribute_to_uvs(target, uv_att_name)

    # create a geom node in the non-remeshed version of the mesh (i.e., source)
    object_info = nw.new_node(Nodes.ObjectInfo, input_kwargs={'Object': target})
    group_input = nw.new_node(Nodes.GroupInput,

    for att_name, att_type in attribute_to_transfer_list:
        nw.expose_input(att_name, attribute=att_name)
    position = nw.new_node(Nodes.InputPosition)

    group_output_sockets = {'Geometry': object_info.outputs["Geometry"]}

    for att_name, att_type in attribute_to_transfer_list:
        transfer_attribute = nw.new_node(Nodes.TransferAttribute,

        group_output_sockets[att_name] = (transfer_attribute, 'Attribute')
