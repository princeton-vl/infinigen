# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick


import pdb
from pathlib import Path
import logging

import bpy
import numpy as np

from infinigen.assets.creatures.util.creature import Part, PartFactory, infer_skeleton_from_mesh
from infinigen.assets.creatures.util.geometry import nurbs
from infinigen.core.util import blender as butil

from infinigen.core.nodes.node_wrangler import NodeWrangler, Nodes

def extract_nodegroup_geo(target_obj, nodegroup, k, ng_params=None):

    assert k in nodegroup.outputs
    assert target_obj.type == 'MESH'

    vert = butil.spawn_vert('extract_nodegroup_geo.temp')

    butil.modify_mesh(vert, type='NODES', apply=False)
    mod = vert.modifiers[0]
    mod.node_group = bpy.data.node_groups.new('extract_nodegroup_geo', 'GeometryNodeTree')
    nw = NodeWrangler(mod.node_group)
    obj_inp = nw.new_node(Nodes.ObjectInfo, [target_obj])

    group_input_kwargs = {**ng_params}
    if 'Geometry' in nodegroup.inputs:
        group_input_kwargs['Geometry'] = obj_inp.outputs['Geometry']

    try:
        group = nw.new_node(nodegroup.name, input_kwargs=group_input_kwargs)
    except KeyError as e:
        print(f"Error while performing extract_nodegroup_geo for {nodegroup=} on {target_obj=}, output_key={k}")
        raise e

    geo = group.outputs[k]

    if k.endswith('Curve'): 
        # curves dont export from geonodes well, convert it to a mesh
        geo = nw.new_node(Nodes.CurveToMesh, [geo])

    output = nw.new_node(Nodes.GroupOutput, input_kwargs={'Geometry': geo})

    butil.apply_modifiers(vert)
    return vert

def nodegroup_to_part(nodegroup_func, params, kwargs=None, base_obj=None, split_extras=False):

    if base_obj is None:
        base_obj = butil.spawn_vert('temp')

    with butil.TemporaryObject(base_obj) as base_obj:
        if kwargs is not None:
            ng = nodegroup_func(**kwargs)
        else:
            ng = nodegroup_func()
        geo_outputs = [o for o in ng.outputs if o.bl_socket_idname == 'NodeSocketGeometry']
        objs = {o.name: extract_nodegroup_geo(base_obj, ng, o.name, ng_params=params) for o in geo_outputs}

    skin_obj = objs.pop('Geometry', None)
    if skin_obj is None:
        skin_obj = butil.spawn_vert('nodegroup_to_part.no_geo_temp')

    attach_basemesh = objs.pop('Base Mesh', None)

    if 'Skeleton Curve' in objs:
        skeleton_obj = objs.pop('Skeleton Curve')
        skeleton = np.array([v.co for v in skeleton_obj.data.vertices])
        if len(skeleton) == 0:
            raise ValueError(f"Skeleton export failed for {nodegroup_func}, {skeleton_obj}, got {skeleton.shape=}")
        butil.delete(skeleton_obj)
    else:
        skeleton = infer_skeleton_from_mesh(skin_obj)

    # Handle any 'Extras' exported by the nodegroup
    for k, o in objs.items():
        if split_extras:
            for i, piece in enumerate(butil.split_object(o)):
                logging.debug(f'Processing piece {i} for split_extras on {nodegroup_func}')
                with butil.SelectObjects(piece):
                    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY')
                piece.name = f'{k}_{i}'
                piece.parent = skin_obj
        else:
            o.parent = skin_obj
            o.name = k

    return Part(
        skeleton, 
        obj=skin_obj, 
        attach_basemesh=attach_basemesh,
        joints=None, iks=None
    )

def nurbs_to_part(handles, face_size=0.07):

    assert handles.shape[-1] == 3

    skeleton = handles.mean(axis=1)
    obj = nurbs.nurbs(handles, method='geomdl', face_size=face_size)

    # first and last ring are used to close the part, need not be included in skeleton
    skeleton = skeleton[1:-1] 
    skeleton_obj = butil.spawn_line('skeleton_subdiv_temp', skeleton)
    butil.modify_mesh(skeleton_obj, 'SUBSURF', levels=2, apply=True)

    mesh = skeleton_obj.data
    verts = [mesh.vertices[0].co]
    curr = 0
    while True:
        try:
            edge = next(e for e in mesh.edges if e.vertices[0] == curr)
        except StopIteration:
            break
        verts.append(mesh.vertices[curr].co)
        curr = edge.vertices[1]


    skeleton = np.array(verts)
    butil.delete(skeleton_obj)

    return Part(skeleton=skeleton, obj=obj)

def linear_combination(corners, weights):
    assert len(corners) == len(weights)
    first = corners[0]
    
    if not isinstance(first, dict):
        ret = sum(corners[i] * weights[i] for i in range(len(corners)))
        return ret

    results = dict()
    for k in first.keys():
        new_corners = [corner[k] for corner in corners]
        results[k] = linear_combination(new_corners, weights)
    return results

def rdict_comb(corners, weights):

    '''
    Take a linear combination of the dicts in `corners`, according to correspondng `weights`
    '''

    norm = sum(weights.values())
    for k in weights:
        weights[k] /= norm

    corners_list = []
    weights_list = []
    for k in weights:
        corners_list.append(corners[k])
        weights_list.append(weights[k])

    return linear_combination(corners_list, weights_list)

# def rdict_comb(corners, weights):

#     '''
#     Take a linear combination of the dicts in `corners`, according to correspondng `weights`
#     '''

#     res = {}
#     first = corners[next(iter(corners))]
#     for k in first.keys():
#         if isinstance(first[k], dict):
#             res[k] = rdict_comb({ke: corners[ke][k] for ke in weights}, weights)
#         else:
#             res[k] = sum(w * corners[ke][k] for ke, w in weights.items())

#             if isinstance(first[k], int):
#                 res[k] = int(res[k])
#     return res

def random_convex_coord(names, select=None, temp=1):

    '''
    corners: dict[dict[]]
    select: str | dict
    temp: float - like softmax, high temp = more even numbers, low temp = more 0s and 1s
    '''

    if isinstance(temp, (float, int)):
        temp = temp * np.ones(len(names))
    elif isinstance(temp, dict):
        temp = np.array([temp[n] for n in names])
    elif isinstance(temp, np.ndarray):
        pass
    else:
        raise ValueError(f'Unrecognized {temp=}')
    

    if isinstance(select, str):
        if not select in names:
            raise ValueError(f'Attempted to random_convex_comb({names=}, {select=}) but select is invalid')
        return {n: 1 if n == select else 0 for n in names}
    
    if isinstance(select, dict):
        if any(k not in names for k in select):
            raise ValueError(f'Attempted to random_convex_comb({names=}, {select.keys()=}) but select is invalid')
        weights = select
        norm = sum(weights.values())
        for k, v in weights.items():
            weights[k] = v / norm
        return weights

    
    vs = np.random.dirichlet(temp)
    weights = {k: vs[i] for i, k in enumerate(names)}
    return weights


