# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick


import bpy
import mathutils
import bmesh

import numpy as np

from infinigen.core.util import blender as butil

class MBallStructure:
    
    def __init__(self, name, resolution=0.1):
        
        self.name = name
        self.resolution = resolution
        self.root = butil.spawn_empty(name)
        
        assert self.name not in bpy.data.metaballs.keys()

        self.empty_elt((0, 0, 0), rot=mathutils.Quaternion(), scale=(1, 1, 1))

    def empty_elt(self, pos, rot, scale):

        mball = bpy.data.metaballs.new(self.name + '_mball')
        mball.resolution = self.resolution
        mball.render_resolution = self.resolution
        
        mball_obj = bpy.data.objects.new(self.name + '_element', mball)
        bpy.context.view_layer.active_layer_collection.collection.objects.link(mball_obj)
        
        mball_obj.parent = self.root
        mball_obj.location = pos
        mball_obj.rotation_euler = rot.to_euler()
        mball_obj.scale = scale 
        
        return mball_obj

    def apply_flags(self, ele, flags):
        
        ele.use_negative = flags.get('neg', False)
        ele.stiffness = flags.get('stiffness', 2)

    def ellipse(
        self, pos, rot, 
        length, rad, mode='scale',
        scale=(1, 1, 1), flags={}
    ):

        mball_obj = self.empty_elt(pos, rot, scale)

        ele = mball_obj.data.elements.new()
        ele.type = 'ELLIPSOID'

        if mode == 'sizes':
            ele.size_x = length 
            ele.size_y = rad
            ele.size_z = rad
            ele.radius = 1 # this seems to just scale everything up/down, no need
        elif mode == 'scale':
            mball_obj.scale.x *= length
            mball_obj.scale.y *= rad
            mball_obj.scale.z *= rad

            ele.radius = 1 # this seems to just scale everything up/down, no need
            ele.size_x = 1
            ele.size_y = 1
            ele.size_z = 1

        self.apply_flags(ele, flags)

        return mball_obj

    def capsule(self, pos, rot, length, rad, scale=(1, 1, 1), flags={}):
        
        mball_obj = self.empty_elt(pos, rot, scale)

        ele = mball_obj.data.elements.new()
        ele.type='CAPSULE'
        ele.size_x = length / 2
        ele.radius =  rad #/ 1.15 # blender always seems to overshoot what I ask for by 15%

        self.apply_flags(ele, flags)

        return mball_obj
    
    def ball(self, pos, rad, **kwargs):
        return self.capsule(pos, length=0, rad=rad, **kwargs)
    
    def to_object(self):

        bm = bmesh.new()
            
        mball_obj = self.root.children[0]
        
        if len(self.root.children) > 1:    
            first = self.root.children[1]
            mball_obj.location = first.location
            mball_obj.rotation_euler = first.rotation_euler
            
            # do resolution via scale, not using their settings
            for c in self.root.children:
                c.data.resolution = 1
            mball_obj.scale = np.full(3, self.resolution)

        depsgraph = bpy.context.evaluated_depsgraph_get()
        eval_mesh = mball_obj.evaluated_get(depsgraph).to_mesh()
        bm.from_mesh(eval_mesh)

        mesh = bpy.data.meshes.new(self.name)
        bm.to_mesh(mesh)

        obj = bpy.data.objects.new(self.name + '_mesh', object_data=mesh)
        bpy.context.scene.collection.objects.link(obj)

        if len(self.root.children) > 1:
            obj.location = first.location
            obj.rotation_euler = first.rotation_euler
            obj.scale = np.full(3, self.resolution)
            with butil.SelectObjects(obj):
                bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

        return obj
    
    def clean(self):
        for o in self.root.children:
            bpy.data.objects.remove(o)
        bpy.data.objects.remove(self.root)

def plusx_cylinder_unwrap(part):

    '''
    Rotate the part from +X to face -Z, cylinder project it, then rotate it back

    WARNING: The cylinder projection operation is VERY particular about the 'context' being right
    '''

    if len(part.data.vertices) == 0:
        return

    bpy.context.view_layer.objects.active = part
    butil.select(part)

    orig = part.rotation_euler.copy()

    # translate to pointing upwards
    part.rotation_euler = (0, np.pi/2, 0)
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)

    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')    
    bpy.ops.uv.cylinder_project(direction='ALIGN_TO_OBJECT', correct_aspect=True)   
    bpy.ops.object.mode_set(mode='OBJECT')                

    # undo the rotation we just applied into th emesh
    part.rotation_euler = (0, -np.pi/2, 0) 
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)

    # back to normal
    part.rotation_euler = orig