# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick


import pdb
import bpy

import numpy as np

from infinigen.assets.creatures.util.creature import Part, Joint, infer_skeleton_from_mesh
from infinigen.core.util import blender as butil

from infinigen.core.nodes.node_wrangler import NodeWrangler, Nodes, geometry_node_group_empty_new

def extract_nodegroup_geo(target_obj, nodegroup, k, ng_params=None):

    assert k in nodegroup.outputs
    assert target_obj.type == 'MESH'

    vert = butil.spawn_vert('extract_nodegroup_geo.temp')

    butil.modify_mesh(vert, type='NODES', apply=False)
    if vert.modifiers[0].node_group == None:
        group = geometry_node_group_empty_new()
        vert.modifiers[0].node_group = group
    ng = vert.modifiers[0].node_group
    nw = NodeWrangler(ng)
    obj_inp = nw.new_node(Nodes.ObjectInfo, [target_obj])

    group_input_kwargs = {**ng_params}
    if 'Geometry' in nodegroup.inputs:
        group_input_kwargs['Geometry'] = obj_inp.outputs['Geometry']
    group = nw.new_node(nodegroup.name, input_kwargs=group_input_kwargs)

    geo = group.outputs[k]

    if k.endswith('Curve'): 
        # curves dont export from geonodes well, convert it to a mesh
        geo = nw.new_node(Nodes.CurveToMesh, [geo])

    output = nw.new_node(Nodes.GroupOutput, input_kwargs={'Geometry': geo})

    butil.apply_modifiers(vert)
    bpy.data.node_groups.remove(ng)
    return vert

class GeonodePartFactory:

    def __init__(self, nodegroup_func, joints=None):
        self.nodegroup_func = nodegroup_func
        self.joints = joints

        self.species_params = self.params()

    def base_obj(self):
        # May be overridden
        return butil.spawn_vert('temp')

    def params(self):
        # Must be overridden
        raise NotImplementedError(f'{self.__class__} did not override abstract base method GeonodePartFactory.params')

    def _extract_geo_results(self):

        ng_params = self.species_params

        with butil.TemporaryObject(self.base_obj()) as base_obj:
            ng = self.nodegroup_func()
            geo_outputs = [o for o in ng.outputs if o.bl_socket_idname == 'NodeSocketGeometry']
            results = {o.name: extract_nodegroup_geo(base_obj, ng, o.name, ng_params=ng_params) for o in geo_outputs}
    
        return results

    def __call__(self):
    
        objs = self._extract_geo_results()

        skin_obj = objs.pop('Geometry')
        attach_basemesh = objs.pop('Base Mesh', None)

        if 'Skeleton Curve' in objs:
            skeleton_obj = objs.pop('Skeleton Curve')
            skeleton = np.array([v.co for v in skeleton_obj.data.vertices])
            if len(skeleton) == 0:
                raise ValueError(f"Skeleton export failed for {self}, {skeleton_obj}, got {skeleton.shape=}")
            butil.delete(skeleton_obj)
        else:
            skeleton = infer_skeleton_from_mesh(skin_obj)

        # Handle any 'Extras' exported by the nodegroup
        for k, o in objs.items():
            o.name = k
            o.mesh.name = k + '.mesh'
            o.parent = skin_obj

        return Part(
            skeleton, 
            obj=skin_obj, 
            attach_basemesh=attach_basemesh,
            joints=self.joints
        )
        



    