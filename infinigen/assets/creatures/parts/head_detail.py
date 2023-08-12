# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick


import bpy 

from platform import node
import numpy as np
from numpy.random import normal as N, uniform as U

from infinigen.assets.creatures.util.creature import PartFactory
from infinigen.assets.creatures.util.genome import Joint, IKParams
from infinigen.assets.creatures.util.part_util import nodegroup_to_part

from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.nodes import node_utils
from infinigen.assets.creatures.util.nodegroups.curve import nodegroup_polar_bezier, nodegroup_simple_tube_v2
from infinigen.assets.creatures.util.nodegroups.attach import nodegroup_surface_muscle
from infinigen.assets.creatures.util.nodegroups.geometry import nodegroup_solidify, nodegroup_symmetric_clone, nodegroup_taper
from infinigen.core.util.math import clip_gaussian
from infinigen.core.util import blender as butil
from infinigen.assets.utils.tag import tag_object, tag_nodegroup

@node_utils.to_nodegroup('nodegroup_cat_ear', singleton=False, type='GeometryNodeTree')
def nodegroup_cat_ear(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketVector', 'length_rad1_rad2', (0.0, 0.0, 0.0)),
            ('NodeSocketFloat', 'Depth', 0.0),
            ('NodeSocketFloatDistance', 'Thickness', 0.0),
            ('NodeSocketFloatDistance', 'Curl Deg', 0.0)])
    
    multiply = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: group_input.outputs["Curl Deg"], 1: (-1.0, 1.0, 1.0)},
        attrs={'operation': 'MULTIPLY'})
    
    separate_xyz = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': group_input.outputs["length_rad1_rad2"]})
    
    divide = nw.new_node(Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["X"], 1: 3.0},
        attrs={'operation': 'DIVIDE'})
    
    polarbezier = nw.new_node(nodegroup_polar_bezier().name,
        input_kwargs={'Origin': (-0.07, 0.0, 0.0), 'angles_deg': multiply.outputs["Vector"], 'Seg Lengths': divide})
    
    spline_parameter = nw.new_node(Nodes.SplineParameter)
    
    float_curve = nw.new_node(Nodes.FloatCurve,
        input_kwargs={'Value': spline_parameter.outputs["Factor"]})
    node_utils.assign_curve(float_curve.mapping.curves[0], [(0.0, 0.0), (0.3236, 0.98), (0.7462, 0.63), (1.0, 0.0)])
    
    set_curve_radius = nw.new_node(Nodes.SetCurveRadius,
        input_kwargs={'Curve': polarbezier.outputs["Curve"], 'Radius': float_curve})
    
    set_curve_tilt = nw.new_node(Nodes.SetCurveTilt,
        input_kwargs={'Curve': set_curve_radius})
    
    multiply_1 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: separate_xyz.outputs["Y"], 1: (-0.5, 0.0, 0.0)},
        attrs={'operation': 'MULTIPLY'})
    
    multiply_2 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: group_input.outputs["Depth"], 1: (0.0, -1.0, 0.0)},
        attrs={'operation': 'MULTIPLY'})
    
    multiply_3 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: separate_xyz.outputs["Y"], 1: (0.5, 0.0, 0.0)},
        attrs={'operation': 'MULTIPLY'})
    
    quadratic_bezier = nw.new_node(Nodes.QuadraticBezier,
        input_kwargs={'Start': multiply_1.outputs["Vector"], 'Middle': multiply_2.outputs["Vector"], 'End': multiply_3.outputs["Vector"]})
    
    curve_to_mesh = nw.new_node(Nodes.CurveToMesh,
        input_kwargs={'Curve': set_curve_tilt, 'Profile Curve': quadratic_bezier})
    
    solidify = nw.new_node(nodegroup_solidify().name,
        input_kwargs={'Mesh': curve_to_mesh, 'Distance': group_input.outputs["Thickness"]})
    
    merge_by_distance = nw.new_node(Nodes.MergeByDistance,
        input_kwargs={'Geometry': solidify, 'Distance': 0.005})
    
    subdivision_surface = nw.new_node(Nodes.SubdivisionSurface,
        input_kwargs={'Mesh': merge_by_distance})
    
    set_shade_smooth = nw.new_node(Nodes.SetShadeSmooth,
        input_kwargs={'Geometry': subdivision_surface, 'Shade Smooth': False})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': set_shade_smooth})

class CatEar(PartFactory):

    tags = ['head_detail']

    def sample_params(self):
        size = clip_gaussian(1, 0.1, 0.2, 5)
        return {
            'length_rad1_rad2': np.array((0.25, 0.1, 0.0)) * N(1, (0.1, 0.05, 0.05)),
            'Depth': 0.06 * N(1, 0.1),
            'Thickness': 0.01,
            'Curl Deg': 49.0 * N(1, 0.2)
        }
    
    def make_part(self, params):
        part = nodegroup_to_part(nodegroup_cat_ear, params)
        tag_object(part.obj, 'cat_ear')
        return part

@node_utils.to_nodegroup('nodegroup_cat_nose', singleton=False, type='GeometryNodeTree')
def nodegroup_cat_nose(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketFloatDistance', 'Nose Radius', 0.06),
            ('NodeSocketFloatDistance', 'Nostril Size', 0.025),
            ('NodeSocketFloatFactor', 'Crease', 0.008),
            ('NodeSocketVectorXYZ', 'Scale', (1.2, 1.0, 1.0))])
    
    cube = nw.new_node(Nodes.MeshCube,
        input_kwargs={'Size': group_input.outputs["Nose Radius"]})
    
    subdivision_surface = nw.new_node(Nodes.SubdivisionSurface,
        input_kwargs={'Mesh': cube, 'Level': 4, 'Edge Crease': group_input.outputs["Crease"]})
    
    transform = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': subdivision_surface, 'Scale': group_input.outputs["Scale"]})
    
    uv_sphere = nw.new_node(Nodes.MeshUVSphere,
        input_kwargs={'Radius': group_input.outputs["Nostril Size"]})
    
    transform_1 = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': uv_sphere, 'Translation': (0.04, 0.025, 0.015), 'Rotation': (0.5643, 0.0, 0.0), 'Scale': (1.0, 0.87, 0.31)})
    
    symmetric_clone = nw.new_node(nodegroup_symmetric_clone().name,
        input_kwargs={'Geometry': transform_1})
    
    difference = nw.new_node(Nodes.MeshBoolean,
        input_kwargs={'Mesh 1': transform, 'Mesh 2': symmetric_clone.outputs["Both"], 'Self Intersection': True})
    
    taper = nw.new_node(nodegroup_taper().name,
        input_kwargs={'Geometry': difference})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': taper})

class CatNose(PartFactory):

    tags = ['head_detail']

    def sample_params(self):
        size_mult = N(0.7, 0.05)
        return {
            'Nose Radius': 0.11 * size_mult, 
            'Nostril Size': 0.03 * size_mult * N(1, 0.1), 
            'Crease': 0.237 * N(1, 0.1)
        }

    def make_part(self, params):
        part = nodegroup_to_part(nodegroup_cat_nose, params)
        nose = part.obj
        nose.name = 'Nose'
        part.obj = butil.spawn_vert('nose_parent')
        nose.parent = part.obj
        tag_object(part.obj, 'cat_nose')
        return part

@node_utils.to_nodegroup('nodegroup_mandible', singleton=False, type='GeometryNodeTree')
def nodegroup_mandible(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketVector', 'length_rad1_rad2', (1.1, 0.1, 0.02)),
            ('NodeSocketVector', 'angles_deg', (-4.4, 58.22, 77.96)),
            ('NodeSocketFloat', 'aspect', 0.52)])
    
    simple_tube_v2 = nw.new_node(nodegroup_simple_tube_v2().name,
        input_kwargs={'length_rad1_rad2': group_input.outputs["length_rad1_rad2"], 'angles_deg': group_input.outputs["angles_deg"], 'aspect': group_input.outputs["aspect"]})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': simple_tube_v2.outputs["Geometry"], 'Skeleton Curve': simple_tube_v2.outputs["Skeleton Curve"], 'Endpoint': simple_tube_v2.outputs["Endpoint"]})

class InsectMandible(PartFactory):

    tags = ['head_detail', 'rigid', 'bald']

    def sample_params(self):
        return {
            'length_rad1_rad2': (1.1 * U(0.2, 1), 0.1 * N(1, 0.2), 0.02 * N(1, 0.1) ),
            'angles_deg': np.array((-4.4, 58.22, 77.96)) * N(1, 0.2, 3),
            'aspect': U(0.3, 1)
        }

    def make_part(self, params):
        part = nodegroup_to_part(nodegroup_mandible, params)
        part.joints = {
            0.4: Joint(rest=(0,0,0), bounds=np.zeros((2, 3)))
        }
        tag_object(part.obj, 'insect_mandible')
        return part