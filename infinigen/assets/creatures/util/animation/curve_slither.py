# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick


import logging

import bpy
import bpy_types
from mathutils import Vector

import numpy as np
from numpy.random import uniform as U, normal as N

import bpy
import mathutils
from numpy.random import uniform, normal, randint
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.nodes import node_utils
from infinigen.core.util.color import color_category
from infinigen.core import surface

import pdb
from infinigen.core.util import blender as butil

@node_utils.to_nodegroup('nodegroup_add_wiggles', singleton=True, type='GeometryNodeTree')
def nodegroup_add_wiggles(nw: NodeWrangler):
    # Code generated using version 2.6.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketGeometry', 'Geometry', None),
            ('NodeSocketFloat', 'Magnitude', 1.6800),
            ('NodeSocketFloat', 'MagRandom', 0.5000),
            ('NodeSocketVector', 'Up', (0.0000, 0.0000, 1.0000))])
    
    curve_tangent = nw.new_node(Nodes.CurveTangent)
    
    cross_product = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: curve_tangent, 1: group_input.outputs["Up"]},
        attrs={'operation': 'CROSS_PRODUCT'})
    
    index = nw.new_node(Nodes.Index)
    
    modulo = nw.new_node(Nodes.Math, input_kwargs={0: index, 1: 4.0000}, attrs={'operation': 'MODULO'})
    
    less_than = nw.new_node(Nodes.Math, input_kwargs={0: modulo, 1: 2.0000}, attrs={'operation': 'LESS_THAN'})
    
    map_range = nw.new_node(Nodes.MapRange, input_kwargs={'Value': less_than, 3: -1.0000})
    
    multiply = nw.new_node(Nodes.Math,
        input_kwargs={0: map_range.outputs["Result"], 1: group_input.outputs["Magnitude"]},
        attrs={'operation': 'MULTIPLY'})
    
    subtract = nw.new_node(Nodes.Math,
        input_kwargs={0: 1.0000, 1: group_input.outputs["MagRandom"]},
        attrs={'operation': 'SUBTRACT'})
    
    random_value = nw.new_node(Nodes.RandomValue, input_kwargs={2: subtract})
    
    multiply_1 = nw.new_node(Nodes.Math, input_kwargs={0: multiply, 1: random_value.outputs[1]}, attrs={'operation': 'MULTIPLY'})
    
    scale = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: cross_product.outputs["Vector"], 'Scale': multiply_1},
        attrs={'operation': 'SCALE'})
    
    set_position = nw.new_node(Nodes.SetPosition,
        input_kwargs={'Geometry': group_input.outputs["Geometry"], 'Offset': scale.outputs["Vector"]})
    
    group_output = nw.new_node(Nodes.GroupOutput, input_kwargs={'Geometry': set_position}, attrs={'is_active_output': True})

@node_utils.to_nodegroup('nodegroup_add_loopbacks', singleton=True, type='GeometryNodeTree')
def nodegroup_add_loopbacks(nw: NodeWrangler):
    # Code generated using version 2.6.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketGeometry', 'Geometry', None),
            ('NodeSocketVector', 'Vector', (0.0000, 0.0000, 0.0000)),
            ('NodeSocketFloat', 'Amount', 0.5800),
            ('NodeSocketFloat', 'Randomness', 0.0000)])
    
    index_1 = nw.new_node(Nodes.Index)
    
    add = nw.new_node(Nodes.Math, input_kwargs={0: index_1, 1: 1.0000})
    
    modulo = nw.new_node(Nodes.Math, input_kwargs={0: add, 1: 2.0000}, attrs={'operation': 'MODULO'})
    
    less_than = nw.new_node(Nodes.Math, input_kwargs={0: modulo}, attrs={'operation': 'LESS_THAN'})
    
    map_range_1 = nw.new_node(Nodes.MapRange, input_kwargs={'Value': less_than, 3: -1.0000})
    
    multiply = nw.new_node(Nodes.Math,
        input_kwargs={0: map_range_1.outputs["Result"], 1: group_input.outputs["Amount"]},
        attrs={'operation': 'MULTIPLY'})
    
    subtract = nw.new_node(Nodes.Math,
        input_kwargs={0: 1.0000, 1: group_input.outputs["Randomness"]},
        attrs={'operation': 'SUBTRACT'})
    
    random_value = nw.new_node(Nodes.RandomValue, input_kwargs={2: subtract, 'ID': index_1})
    
    multiply_1 = nw.new_node(Nodes.Math, input_kwargs={0: multiply, 1: random_value.outputs[1]}, attrs={'operation': 'MULTIPLY'})
    
    scale = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: group_input.outputs["Vector"], 'Scale': multiply_1},
        attrs={'operation': 'SCALE'})
    
    set_position_1 = nw.new_node(Nodes.SetPosition,
        input_kwargs={'Geometry': group_input.outputs["Geometry"], 'Offset': scale.outputs["Vector"]})
    
    group_output = nw.new_node(Nodes.GroupOutput, input_kwargs={'Geometry': set_position_1}, attrs={'is_active_output': True})

@node_utils.to_nodegroup('nodegroup_wiggles', singleton=True, type='GeometryNodeTree')
def nodegroup_wiggles(nw: NodeWrangler):
    # Code generated using version 2.6.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketGeometry', 'Geometry', None),
            ('NodeSocketFloatDistance', 'Wavelength', 2.3300),
            ('NodeSocketFloat', 'Magnitude', 1.6800),
            ('NodeSocketFloat', 'MagRandom', 1.0000),
            ('NodeSocketFloat', 'Loopyness', 0.5800),
            ('NodeSocketFloat', 'LoopRandom', 0.0000),
            ('NodeSocketFloat', 'AltitudeOffset', 0.00),
            ('NodeSocketVector', 'Up', (0.0000, 0.0000, 1.0000))])
    
    curve_tangent_1 = nw.new_node(Nodes.CurveTangent)
    
    capture_attribute = nw.new_node(Nodes.CaptureAttribute,
        input_kwargs={'Geometry': group_input.outputs["Geometry"], 1: curve_tangent_1},
        attrs={'data_type': 'FLOAT_VECTOR'})
    
    divide = nw.new_node(Nodes.Math,
        input_kwargs={0: group_input.outputs["Wavelength"], 1: 4.0000},
        attrs={'operation': 'DIVIDE'})
    
    resample_curve = nw.new_node(Nodes.ResampleCurve,
        input_kwargs={'Curve': capture_attribute.outputs["Geometry"], 'Length': divide},
        attrs={'mode': 'LENGTH'})
    
    addwiggles = nw.new_node(nodegroup_add_wiggles().name,
        input_kwargs={'Geometry': resample_curve, 'Magnitude': group_input.outputs["Magnitude"], 'MagRandom': group_input.outputs["MagRandom"], 'Up': group_input.outputs["Up"]})
    
    addloopbacks = nw.new_node(nodegroup_add_loopbacks().name,
        input_kwargs={'Geometry': addwiggles, 'Vector': capture_attribute.outputs["Attribute"], 'Amount': group_input.outputs["Loopyness"], 'Randomness': group_input.outputs["LoopRandom"]})
    
    curve_to_mesh = nw.new_node(Nodes.CurveToMesh, input_kwargs={'Curve': addloopbacks, 'Fill Caps': True})
    
    subdivision_surface = nw.new_node(Nodes.SubdivisionSurface, input_kwargs={'Mesh': curve_to_mesh, 'Level': 3})
    
    mesh_to_curve = nw.new_node(Nodes.MeshToCurve, input_kwargs={'Mesh': subdivision_surface})
    
    off = nw.new_node(Nodes.CombineXYZ, input_kwargs={'Z': group_input.outputs['AltitudeOffset']})
    result = nw.new_node(Nodes.SetPosition, input_kwargs={'Geometry': mesh_to_curve, 'Offset': off})

    group_output = nw.new_node(Nodes.GroupOutput, input_kwargs={'Geometry': result}, attrs={'is_active_output': True})


def add_curve_slithers(curve, snake_length):
    params = {
        'Wavelength': snake_length / U(2, 4),
        'Magnitude': snake_length * 0.05 * N(1, 0.2),
        'MagRandom': U(0, 0.7),
        'Loopyness': 0,
        'LoopRandom': 0,
        'AltitudeOffset': 0.02
    }
    butil.modify_mesh(curve, 'NODES', node_group=nodegroup_wiggles(), 
        ng_inputs=params, apply=False, show_viewport=True)
    with butil.SelectObjects(curve):
        bpy.ops.object.convert(target='MESH')
        bpy.ops.object.convert(target='CURVE')
    return curve

def slither_along_path(obj, curve, speed, zoff_pct=0.7, orig_len=None):

    if not curve.type == 'CURVE':
        with butil.SelectObjects(curve):
            bpy.ops.object.convert(target='CURVE')
            curve = bpy.context.active_object
    if curve.type != 'CURVE':
        message = f'slither_along_path failed, {curve.name=} had {curve.type=} but expected CURVE'
        if curve.type == 'MESH':
            message == f'. {len(curve.data.vertices)=}'
        logging.warning(message)
        return
    
    curve.data.twist_mode = 'Z_UP'

    xmax = max(v[0] for v in obj.bound_box)

    l = curve.data.splines[0].calc_length()

    zoff = zoff_pct * obj.dimensions[-1] / 2
    obj.location = (xmax,0,zoff)
    obj.keyframe_insert(data_path="location", frame=0)
    obj.location = (l, 0, zoff)
    obj.keyframe_insert(data_path="location", frame=bpy.context.scene.frame_end)

    for fc in obj.animation_data.action.fcurves:
        for k in fc.keyframe_points:
            k.interpolation = 'LINEAR'

    butil.modify_mesh(obj, 'CURVE', object=curve, apply=False, show_viewport=True)
    obj.rotation_euler = (0, 0, np.pi)

def snap_curve_to_floor(curve, bvh, step_height=1):

    s = curve.data.splines[0]
    for p in s.points:
        raystart = Vector(p.co[:3]) + Vector((0, 0, step_height))
        loc, *_ = bvh.ray_cast(raystart, Vector((0, 0, -1)))
        if loc is not None:
            p.co = (*loc, 1)