# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Hongyu Wen
# Acknowledgement: This file draws inspiration from https://www.youtube.com/watch?v=5BXvwqVyCQw by Artisans of Vaul


from re import M
import bpy 
import math

from platform import node
import numpy as np
from numpy.random import normal as N, uniform as U

from infinigen.assets.creatures.util.creature import PartFactory
from infinigen.assets.creatures.util.genome import Joint, IKParams
from infinigen.assets.creatures.util.part_util import nodegroup_to_part
from infinigen.assets.creatures.util import part_util

from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.nodes import node_utils
from infinigen.core.util import blender as butil
from infinigen.assets.utils.tag import tag_object, tag_nodegroup

@node_utils.to_nodegroup('nodegroup_noise', singleton=False, type='GeometryNodeTree')
def nodegroup_noise(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketGeometry', 'Geometry', None),
            ('NodeSocketFloat', 'Scale', 0.05),
            ('NodeSocketFloat', 'W', 0.0)])
    
    noise_texture = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'W': group_input.outputs["W"], 'Roughness': 0.0},
        attrs={'noise_dimensions': '4D'})
    
    subtract = nw.new_node(Nodes.Math,
        input_kwargs={0: noise_texture.outputs["Color"]},
        attrs={'operation': 'SUBTRACT'})
    
    multiply = nw.new_node(Nodes.Math,
        input_kwargs={0: subtract, 1: group_input.outputs["Scale"]},
        attrs={'operation': 'MULTIPLY'})
    
    set_position = nw.new_node(Nodes.SetPosition,
        input_kwargs={'Geometry': group_input.outputs["Geometry"], 'Offset': multiply})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': set_position})

@node_utils.to_nodegroup('nodegroup_ridge', singleton=False, type='GeometryNodeTree')
def nodegroup_ridge(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketFloat', 'thickness', 4.0),
            ('NodeSocketFloat', 'depth_of_ridge', 0.2),
            ('NodeSocketInt', 'number_of_ridge', 150),
            ('NodeSocketGeometry', 'geometry', None)])
    
    resample_curve = nw.new_node(Nodes.ResampleCurve,
        input_kwargs={'Curve': group_input.outputs["geometry"], 'Count': group_input.outputs["number_of_ridge"]})
    
    spline_parameter = nw.new_node(Nodes.SplineParameter)
    
    float_curve = nw.new_node(Nodes.FloatCurve,
        input_kwargs={'Value': spline_parameter.outputs["Factor"]})
    node_utils.assign_curve(float_curve.mapping.curves[0], [(0.0, 1.0), (0.2, 0.9), (0.3705, 0.7406), (0.55, 0.5938), (0.6886, 0.4188), (0.85, 0.1844), (1.0, 0.0)])
    
    modulo = nw.new_node(Nodes.Math,
        input_kwargs={0: spline_parameter.outputs["Index"], 1: 5.0},
        attrs={'operation': 'MODULO'})
    
    power = nw.new_node(Nodes.Math,
        input_kwargs={0: -1.0, 1: modulo},
        attrs={'operation': 'POWER'})
    
    multiply = nw.new_node(Nodes.Math,
        input_kwargs={0: group_input.outputs["depth_of_ridge"], 1: power},
        attrs={'operation': 'MULTIPLY'})
    
    add = nw.new_node(Nodes.Math,
        input_kwargs={0: 1.0, 1: multiply})
    
    multiply_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: float_curve, 1: add},
        attrs={'operation': 'MULTIPLY'})
    
    noise_texture = nw.new_node(Nodes.NoiseTexture)
    
    subtract = nw.new_node(Nodes.Math,
        input_kwargs={0: noise_texture.outputs["Color"]},
        attrs={'operation': 'SUBTRACT'})
    
    multiply_2 = nw.new_node(Nodes.Math,
        input_kwargs={0: subtract, 1: group_input.outputs["depth_of_ridge"]},
        attrs={'operation': 'MULTIPLY'})
    
    add_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: multiply_1, 1: multiply_2})
    
    multiply_3 = nw.new_node(Nodes.Math,
        input_kwargs={0: add_1, 1: group_input.outputs["thickness"]},
        attrs={'operation': 'MULTIPLY'})
    
    set_curve_radius = nw.new_node(Nodes.SetCurveRadius,
        input_kwargs={'Curve': resample_curve, 'Radius': multiply_3})
    
    noise = nw.new_node(nodegroup_noise().name,
        input_kwargs={'Geometry': set_curve_radius, 'Scale': 0.02},
        label='Noise')
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': noise})

@node_utils.to_nodegroup('nodegroup_horn', singleton=False, type='GeometryNodeTree')
def nodegroup_horn(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[
            ('NodeSocketFloat', 'length', 0.0),
            ('NodeSocketFloat', 'rad1', 0.0),
            ('NodeSocketFloat', 'rad2', 0.0),
            ('NodeSocketFloat', 'thickness', 4.0),
            ('NodeSocketFloat', 'density_of_ridge', 0.0),
            ('NodeSocketFloat', 'depth_of_ridge', 0.2),
            ('NodeSocketFloatDistance', 'height', 2.5),
            ('NodeSocketFloat', 'rotation_x', 0)])
    
    multiply = nw.new_node(Nodes.Math,
        input_kwargs={0: group_input.outputs["length"], 1: group_input.outputs["density_of_ridge"]},
        attrs={'operation': 'MULTIPLY'})
    
    add = nw.new_node(Nodes.Math,
        input_kwargs={0: group_input.outputs["rad1"], 1: group_input.outputs["rad2"]})
    
    # divide = nw.new_node(Nodes.Math,
    #     input_kwargs={0: add, 1: 2.0},
    #     attrs={'operation': 'DIVIDE'})
    
    divide_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: group_input.outputs["length"], 1: add},
        attrs={'operation': 'DIVIDE'})
    
    divide_2 = nw.new_node(Nodes.Math,
        input_kwargs={0: divide_1, 1: 3.1415},
        attrs={'operation': 'DIVIDE'})
    
    spiral = nw.new_node('GeometryNodeCurveSpiral',
        input_kwargs={'Resolution': 150, 'Rotations': divide_2, 'Start Radius': group_input.outputs["rad1"], 'End Radius': group_input.outputs["rad2"], 'Height': group_input.outputs["height"]})
    
    ridge = nw.new_node(nodegroup_ridge().name,
        input_kwargs={'thickness': group_input.outputs["thickness"], 'depth_of_ridge': group_input.outputs["depth_of_ridge"], 'number_of_ridge': multiply, 'geometry': spiral})
    
    curve_circle_2 = nw.new_node(Nodes.CurveCircle,
        input_kwargs={'Resolution': 10, 'Radius': 0.5})
    
    noise = nw.new_node(nodegroup_noise().name,
        input_kwargs={'Geometry': curve_circle_2.outputs["Curve"], 'Scale': 0.2},
        label='Noise')
    
    curve_to_mesh = nw.new_node(Nodes.CurveToMesh,
        input_kwargs={'Curve': ridge, 'Profile Curve': noise})
    
    multiply_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: group_input.outputs["rad1"], 1: -1.0},
        attrs={'operation': 'MULTIPLY'})
    
    combine_xyz = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'X': multiply_1})
    
    set_position = nw.new_node(Nodes.SetPosition,
        input_kwargs={'Geometry': curve_to_mesh, 'Offset': combine_xyz})
    
    transform_1 = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': set_position, 'Rotation': (-0.8, 0.0, 2.6)})

    combine_xyz_2 = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'X': group_input.outputs["rotation_x"]})
    
    transform_2 = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': transform_1, 'Rotation': combine_xyz_2})

    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': transform_2})  


class Horn(PartFactory):
    param_templates = {}
    tags = ['head_detail', 'rigid']

    def sample_params(self, select=None, var=1):
        N = lambda m, v: np.random.normal(m, v * var)
        U = lambda l, r: np.random.uniform(l, r)
        weights = part_util.random_convex_coord(self.param_templates.keys(), select=select)
        params = part_util.rdict_comb(self.param_templates, weights)

        for key in params['horn']:
            if key in params['range']:
                l, r = params['range'][key]
                noise = N(0, 0.02 * (r - l))
                params['horn'][key] += noise
        return params['horn']

    def make_part(self, params):
        part = nodegroup_to_part(nodegroup_horn, params)
        horn = part.obj

        # postprocess horn
        with butil.SelectObjects(horn):
            bpy.ops.object.shade_flat()
        horn.name = 'Horn'
        butil.modify_mesh(horn, 'SUBSURF', apply=True, levels=2)

        # swap the horn to be an extra so it doesnt get remeshed etc
        part.obj = butil.spawn_vert('horn_parent')        
        horn.parent = part.obj
        tag_object(part.obj, 'horn')

        return part

goat_horn = {
    'length': 0.5,
    'rad1': 0.18,
    'rad2': 0.3,
    'thickness': 0.15,
    'density_of_ridge': 250,
    'depth_of_ridge': 0.02,
    'height': 0.1,
    'rotation_x': 0,
}

gazelle_horn = {
    'length': 0.4,
    'rad1': 0.7,
    'rad2': 0.5,
    'thickness': 0.1,
    'density_of_ridge': 150,
    'depth_of_ridge': 0.1,
    'height': 0.1,
    'rotation_x': 0,
}

bull_horn = {
    'length': 0.1,
    'rad1': 0.5,
    'rad2': 0.1,
    'thickness': 0.1,
    'density_of_ridge': 150,
    'depth_of_ridge': 0.01,
    'height': -0.1,
    'rotation_x': -1
}

scales = {
    'length': [0.1, 0.6],
    'rad1': [0.1, 1],
    'rad2': [0.1, 1],
    'thickness': [0.05, 0.3],
    'density_of_ridge': [100, 300],
    'depth_of_ridge': [0.01, 0.1],
    'height': [-0.3, 0.3],
    'rotation_x': [-1, 1]
}

for k, v in scales.items():
    scales[k] = np.array(v)

Horn.param_templates['bull'] = {'horn': bull_horn, 'range': scales}
Horn.param_templates['gazelle'] = {'horn': gazelle_horn, 'range': scales}
Horn.param_templates['goat'] = {'horn': goat_horn, 'range': scales}
