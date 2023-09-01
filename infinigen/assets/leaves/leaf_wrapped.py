# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Yiming Zuo


import bpy
import mathutils
import numpy as np
from numpy.random import uniform, normal, randint
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.nodes import node_utils
from infinigen.core.util.color import color_category
from infinigen.core import surface

from infinigen.assets.leaves.leaf_maple import LeafFactoryMaple
from infinigen.assets.leaves.leaf_broadleaf import LeafFactoryBroadleaf
from infinigen.assets.leaves.leaf_ginko import LeafFactoryGinko
from infinigen.core.placement.factory import AssetFactory

def nodegroup_nodegroup_apply_wrap(nw: NodeWrangler, **kwargs):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketGeometry', 'Geometry', None)])
    
    angle = nw.new_node(Nodes.Value,
        label='angle')
    angle.outputs[0].default_value = kwargs['angle']
    
    radians = nw.new_node(Nodes.Math,
        input_kwargs={0: angle},
        attrs={'operation': 'RADIANS'})
    
    combine_xyz_2 = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'Z': radians})
    
    transform_2 = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': group_input.outputs["Geometry"], 'Rotation': combine_xyz_2})
    
    position_1 = nw.new_node(Nodes.InputPosition)
    
    separate_xyz = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': position_1})
    
    rotation = nw.new_node(Nodes.Value,
        label='rotation')
    rotation.outputs[0].default_value = kwargs['rotation']
    
    value = nw.new_node(Nodes.Value)
    value.outputs[0].default_value = 1.0
    
    end_radius = nw.new_node(Nodes.Value,
        label='end_radius')
    end_radius.outputs[0].default_value = kwargs['end_radius']
    
    spiral = nw.new_node('GeometryNodeCurveSpiral',
        input_kwargs={'Resolution': 1000, 'Rotations': rotation, 'Start Radius': value, 'End Radius': end_radius, 'Height': 0.0})
    
    curve_length = nw.new_node(Nodes.CurveLength,
        input_kwargs={'Curve': spiral})
    
    position = nw.new_node(Nodes.InputPosition)
    
    separate_xyz_1 = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': position})
    
    attribute_statistic = nw.new_node(Nodes.AttributeStatistic,
        input_kwargs={'Geometry': transform_2, 2: separate_xyz_1.outputs["Y"]})
    
    subtract = nw.new_node(Nodes.Math,
        input_kwargs={0: attribute_statistic.outputs["Max"], 1: attribute_statistic.outputs["Min"]},
        attrs={'operation': 'SUBTRACT'})
    
    divide = nw.new_node(Nodes.Math,
        input_kwargs={0: curve_length, 1: subtract},
        attrs={'operation': 'DIVIDE'})
    
    divide_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: value, 1: divide},
        attrs={'operation': 'DIVIDE'})
    
    divide_2 = nw.new_node(Nodes.Math,
        input_kwargs={0: end_radius, 1: divide},
        attrs={'operation': 'DIVIDE'})
    
    spiral_1 = nw.new_node('GeometryNodeCurveSpiral',
        input_kwargs={'Resolution': 1000, 'Rotations': rotation, 'Start Radius': divide_1, 'End Radius': divide_2, 'Height': 0.0})
    
    transform = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': spiral_1, 'Rotation': (0.0, 1.5708, 3.1416)})
    
    noise_texture = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Scale': 2.0})
    
    subtract_1 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: noise_texture.outputs["Color"], 1: (0.5, 0.5, 0.5)},
        attrs={'operation': 'SUBTRACT'})
    
    noise_level = nw.new_node(Nodes.Value,
        label='noise_level')
    noise_level.outputs[0].default_value = kwargs['noise_level']
    
    multiply = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: subtract_1.outputs["Vector"], 1: noise_level},
        attrs={'operation': 'MULTIPLY'})
    
    set_position_2 = nw.new_node(Nodes.SetPosition,
        input_kwargs={'Geometry': transform, 'Offset': multiply.outputs["Vector"]})
    
    map_range = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': separate_xyz_1.outputs["Y"], 1: attribute_statistic.outputs["Min"], 2: attribute_statistic.outputs["Max"]})
    
    sample_curve = nw.new_node(Nodes.SampleCurve,
        input_kwargs={'Curve': set_position_2, 'Factor': map_range.outputs["Result"]},
        attrs={'mode': 'FACTOR'})
    
    separate_xyz_2 = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': sample_curve.outputs["Position"]})
    
    combine_xyz = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'X': separate_xyz.outputs["X"], 'Y': separate_xyz_2.outputs["Y"], 'Z': separate_xyz_2.outputs["Z"]})
    
    normalize = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: sample_curve.outputs["Position"]},
        attrs={'operation': 'NORMALIZE'})
    
    multiply_1 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: separate_xyz.outputs["Z"], 1: normalize.outputs["Vector"]},
        attrs={'operation': 'MULTIPLY'})
    
    add = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: combine_xyz, 1: multiply_1.outputs["Vector"]})
    
    set_position = nw.new_node(Nodes.SetPosition,
        input_kwargs={'Geometry': transform_2, 'Position': add.outputs["Vector"]})
    
    subtract_2 = nw.new_node(Nodes.Math,
        input_kwargs={0: 0.0, 1: radians},
        attrs={'operation': 'SUBTRACT'})
    
    combine_xyz_3 = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'Z': subtract_2})
    
    transform_3 = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': set_position, 'Rotation': combine_xyz_3})
    
    combine_xyz_4 = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'Z': divide_1})
    
    transform_4 = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': transform_3, 'Translation': combine_xyz_4})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': transform_4})

class LeafFactoryWrapped(AssetFactory):

    def __init__(self, factory_seed, season='autumn', coarse=False):
        super().__init__(factory_seed, coarse=coarse)
        self.factory_list = [
            LeafFactoryMaple(factory_seed, season=season, coarse=coarse),
            LeafFactoryBroadleaf(factory_seed, season=season, coarse=coarse),
            LeafFactoryGinko(factory_seed, season=season, coarse=coarse),
        ]
    
    def create_asset(self, **params):

        fac_id = randint(len(self.factory_list))
        fac = self.factory_list[fac_id]

        wrap_params = {
            'angle': uniform(-70, 70),
            'rotation': uniform(0.2, 2.0),
            'end_radius': np.exp(uniform(-2.0, 2.0)),
            'noise_level': uniform(0.0, 0.5)
        }

        obj = fac.create_asset()
        surface.add_geomod(obj, nodegroup_nodegroup_apply_wrap, apply=False, input_kwargs=wrap_params)

        bpy.ops.object.convert(target='MESH')

        return obj