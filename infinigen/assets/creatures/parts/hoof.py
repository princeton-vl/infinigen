# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Hongyu Wen


import bpy 
import bmesh
import mathutils

import numpy as np
from math import sin, cos, pi, exp
from numpy.random import uniform as U, normal as N

from infinigen.assets.creatures.util.creature import PartFactory, Part
from infinigen.assets.creatures.util.genome import Joint, IKParams
from infinigen.assets.creatures.util import part_util
from infinigen.core.util import blender as butil

from infinigen.assets.creatures.util.nodegroups.curve import nodegroup_simple_tube, nodegroup_simple_tube_v2
from infinigen.assets.creatures.util.nodegroups.attach import nodegroup_surface_muscle
from infinigen.assets.creatures.util.part_util import nodegroup_to_part

from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.nodes import node_utils

from infinigen.assets.creatures.util.geometry import nurbs as nurbs_util
from infinigen.assets.utils.tag import tag_object, tag_nodegroup

def square(x):
    return x * x

def tri(x):
    return x ** 3

class Hoof():
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
         
    def scale(self, p):
        return 1 - 0.2 * p
    
    def transform(self, p):
        return -0.6 * p
    
    def down(self, p, theta):
        return 0.4 * p * cos(theta)
    
    def get_shape(self):
        points = []
        r = self.r
        N = lambda m, v: np.random.normal(m, v)
        for i in range(self.m):
            theta = 2 * pi * i / (self.m)
            nx = N(0, 0.01)
            ny = N(0, 0.01)
            if i >= self.m - r or i <= r:
                points.append((-0.2 * cos(theta) + nx, 0.05 * sin(theta) + ny))
            elif i >= self.m - 2 * r or i <= 2 * r:
                points.append((cos(theta) + nx, 0.2 * sin(theta) + ny))
            # elif i >= self.m - 4 * r or i <= 4 * r:
            #     points.append((cos(theta) + nx, 0.6 * sin(theta) + ny))
            else:
                points.append((cos(theta) + nx, sin(theta) + ny))
        return points
    
    def make_face(self, obj):
        bm = bmesh.new()
        for v in obj.data.vertices:
            x, y, z = obj.matrix_world @ v.co
            if z == 0:
                bm.verts.new((x, y, z))
        bm.faces.new(bm.verts)
        bm.normal_update()
        bm.from_mesh(obj.data)
        butil.delete(obj)

        me = bpy.data.meshes.new("face")
        bm.to_mesh(me)
        # add bmesh to scene
        ob = bpy.data.objects.new("face", me)
        bpy.context.scene.collection.objects.link(ob)
        return ob
            
    def generate(self):
        self.n = int(self.n)
        self.m = int(self.m)
        
        points = self.get_shape()
        ctrls = np.zeros((self.n, self.m, 3)) 
        for i in range(self.n):
            for j in range(self.m):
                p = i / (self.n - 1)
                theta = 2 * pi * j / (self.m)
                ctrls[i][j][0] = self.scale(p) * points[j][0] + self.transform(p)
                ctrls[i][j][1] = self.scale(p) * points[j][1] # + self.transform(p)
                ctrls[i][j][2] = p + self.down(p, theta)
                ctrls[i][j][0] *= self.sx
                ctrls[i][j][1] *= self.sy
                ctrls[i][j][2] *= self.sz

        method = 'blender' if False else 'geomdl'

        obj = nurbs_util.nurbs(ctrls, method, face_size=0.01)
        obj = self.make_face(obj)
        
        top_pos = mathutils.Vector(ctrls[-1].mean(axis=0))
        with butil.CursorLocation(top_pos), butil.SelectObjects(obj):
            bpy.ops.object.origin_set(type='ORIGIN_CURSOR')
        obj.location = (0,0,0)

        obj.rotation_euler.y -= np.pi / 2
        butil.apply_transform(obj, rot=True)
        tag_object(obj, 'hoof')

        return obj

class HoofClaw(PartFactory):

    param_templates = {}
    tags = ['head_detail', 'rigid']

    def sample_params(self, select=None, var=1):
        params = {
            'n': 20,
            'm': 20, 
            'sx': 0.1 * N(1, 0.05),
            'sy': 0.1 * N(1, 0.05),
            'sz': 0.08 * N(1, 0.05),
            'r': 0.5 + N(0, 1)
        }
        return params

    def make_part(self, params):
        obj = butil.spawn_vert('hoofclaw_parent_temp')
        
        hoof = Hoof(**params).generate()
        hoof.parent = obj
        hoof.name = 'HoofClaw'

        part = Part(skeleton=np.zeros((1, 3)), obj=obj, joints={}, iks={})
        tag_object(part.obj, 'hoof_claw')
        return part

@node_utils.to_nodegroup('nodegroup_hoof', singleton=False, type='GeometryNodeTree')
def nodegroup_hoof(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketVector', 'length_rad1_rad2', (1.4299999999999999, 0.10000000000000001, 0.10000000000000001)),
            ('NodeSocketVector', 'angles_deg', (-20.0, 16.0, 9.1999999999999993)),
            ('NodeSocketFloat', 'aspect', 1.0),
            ('NodeSocketVector', 'Upper Rad1 Rad2 Fullness', (0.22, 0.0, 0.0)),
            ('NodeSocketVector', 'Lower Rad1 Rad2 Fullness', (0.0, 0.0, 0.0)),
            ('NodeSocketVector', 'Height, Tilt1, Tilt2', (0.73999999999999999, 0.0, 0.0))])
    
    simple_tube_v2_001 = nw.new_node(nodegroup_simple_tube_v2().name,
        input_kwargs={'length_rad1_rad2': group_input.outputs["length_rad1_rad2"], 'angles_deg': group_input.outputs["angles_deg"], 'aspect': group_input.outputs["aspect"], 'fullness': 2.5})
    
    shoulder = nw.new_node(nodegroup_surface_muscle().name,
        input_kwargs={'Skin Mesh': simple_tube_v2_001.outputs["Geometry"], 'Skeleton Curve': simple_tube_v2_001.outputs["Skeleton Curve"], 'Coord 0': (0.0, 0.0, 0.0), 'Coord 1': (0.20000000000000001, 0.0, 0.0), 'Coord 2': (0.55000000000000004, 0.0, 0.0), 'StartRad, EndRad, Fullness': group_input.outputs["Lower Rad1 Rad2 Fullness"], 'ProfileHeight, StartTilt, EndTilt': group_input.outputs["Height, Tilt1, Tilt2"]},
        label='Shoulder')
    
    shoulder_1 = nw.new_node(nodegroup_surface_muscle().name,
        input_kwargs={'Skin Mesh': simple_tube_v2_001.outputs["Geometry"], 'Skeleton Curve': simple_tube_v2_001.outputs["Skeleton Curve"], 'Coord 0': (1.0, 0.0, 0.0), 'Coord 1': (0.20000000000000001, 0.0, 0.0), 'Coord 2': (0.80000000000000004, 0.0, 0.0), 'StartRad, EndRad, Fullness': group_input.outputs["Upper Rad1 Rad2 Fullness"], 'ProfileHeight, StartTilt, EndTilt': group_input.outputs["Height, Tilt1, Tilt2"]},
        label='Shoulder')
    
    join_geometry = nw.new_node(Nodes.JoinGeometry,
        input_kwargs={'Geometry': [shoulder, simple_tube_v2_001.outputs["Geometry"], shoulder_1]})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': join_geometry, 'Skeleton Curve': simple_tube_v2_001.outputs["Skeleton Curve"]})


class HoofAnkle(PartFactory):
    
    tags = ['foot_detail', 'rigid']
    ankle_scale = (0.8, 0.8, 0.8)

    def sample_params(self, var=1):
        ankle = {
            'length_rad1_rad2': (0.45 * N(1, 0.05), 0.07 * N(1, 0.05), 0.1 * N(1, 0.05)),
            'angles_deg': (-90.0 + N(0, 5), 40.0 + N(0, 5), N(0, 5)),
            'aspect': 1.0,
            'Upper Rad1 Rad2 Fullness': (0.2, 0.0, 4),
            'Lower Rad1 Rad2 Fullness': (0.15, 0.0, 4),
            'Height, Tilt1, Tilt2': (1, 0.0, 0.0)
        }
        return ankle

    def make_part(self, params):
        obj = butil.spawn_vert('hoof_parent_temp')

        part = nodegroup_to_part(nodegroup_hoof, params)
        ankle = part.obj
        with butil.SelectObjects(ankle):
            bpy.ops.object.shade_flat()
        butil.modify_mesh(ankle, 'SUBSURF', apply=True, levels=2)
        ankle.parent = obj
        ankle.name = "HoofAnkle"

        ankle.scale = self.ankle_scale
        butil.apply_transform(ankle, scale=True)
        tag_object(part.obj, 'hoof_ankle')
        
        part.iks = {1.0: IKParams('foot', rotation_weight=0.1, chain_parts=2, chain_length=-1)}

        return part