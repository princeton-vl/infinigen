# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Mingzhe Wang


import bpy

import numpy as np
from numpy.random import uniform, normal
import random

from infinigen.assets.creatures.util.genome import Joint, IKParams

from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.nodes import node_utils
from infinigen.core.surface import set_geomod_inputs
from infinigen.assets.materials.utils.surface_utils import clip, sample_range, sample_ratio

from infinigen.assets.creatures.util.nodegroups.curve import nodegroup_simple_tube_v2
from infinigen.assets.creatures.util.nodegroups.attach import nodegroup_attach_part
from infinigen.assets.creatures.util.creature import PartFactory, Part
from infinigen.assets.creatures.util.part_util import nodegroup_to_part
from infinigen.core.util import blender as butil
from infinigen.assets.utils.tag import tag_object, tag_nodegroup

@node_utils.to_nodegroup('nodegroup_mix2_values', singleton=True, type='GeometryNodeTree')
def nodegroup_mix2_values(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketFloat', 'Ratio', 0.5),
            ('NodeSocketFloat', 'Value1', 0.5),
            ('NodeSocketFloat', 'Value2', 0.5)])
    
    multiply = nw.new_node(Nodes.Math,
        input_kwargs={0: group_input.outputs["Value1"], 1: group_input.outputs["Ratio"]},
        attrs={'operation': 'MULTIPLY'})
    
    subtract = nw.new_node(Nodes.Math,
        input_kwargs={0: 1.0, 1: group_input.outputs["Ratio"]},
        attrs={'operation': 'SUBTRACT'})
    
    multiply_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: subtract, 1: group_input.outputs["Value2"]},
        attrs={'operation': 'MULTIPLY'})
    
    add = nw.new_node(Nodes.Math,
        input_kwargs={0: multiply, 1: multiply_1})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Value': add})

@node_utils.to_nodegroup('nodegroup_fish_fin', singleton=False, type='GeometryNodeTree')
def nodegroup_fish_fin(nw: NodeWrangler):
    # Code generated using version 2.5.1 of the node_transpiler

    grid = nw.new_node(Nodes.MeshGrid,
        input_kwargs={'Vertices X': 100, 'Vertices Y': 100})
    
    transform_3 = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': grid, 'Rotation': (1.5708, 0.0000, 0.0000)})
    
    position_3 = nw.new_node(Nodes.InputPosition)
    
    sep_z = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': position_3},
        label='sep_z')
    
    z_stats = nw.new_node(Nodes.AttributeStatistic,
        input_kwargs={'Geometry': transform_3, 2: sep_z.outputs["Z"]},
        label='z_stats')
    
    norm_z = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': sep_z.outputs["Z"], 1: z_stats.outputs["Min"], 2: z_stats.outputs["Max"]},
        label='norm_z')
    
    remap_z = nw.new_node(Nodes.FloatCurve,
        input_kwargs={'Value': norm_z.outputs["Result"]},
        label='remap_z')
    node_utils.assign_curve(remap_z.mapping.curves[0], [(0.1727, 0.9875), (0.5182, 0.2438), (1.0000, 0.0063)])
    
    capture_z_rigidity = nw.new_node(Nodes.CaptureAttribute,
        input_kwargs={'Geometry': transform_3, 2: remap_z},
        label='capture_z_rigidity')
    
    position = nw.new_node(Nodes.InputPosition)
    
    separate_xyz = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': position})
    
    greater_than = nw.new_node(Nodes.Compare,
        input_kwargs={0: separate_xyz.outputs["Y"]})
    
    op_and = nw.new_node(Nodes.BooleanMath,
        input_kwargs={1: greater_than})
    
    delete_geometry = nw.new_node(Nodes.DeleteGeometry,
        input_kwargs={'Geometry': capture_z_rigidity.outputs["Geometry"], 'Selection': op_and})
    
    capture_attribute = nw.new_node(Nodes.CaptureAttribute,
        input_kwargs={'Geometry': delete_geometry, 1: position},
        attrs={'data_type': 'FLOAT_VECTOR'})
    
    position_1 = nw.new_node(Nodes.InputPosition)
    
    add = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: position_1, 1: (0.5000, 0.0000, 0.5000)})
    
    separate_xyz_1 = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': add.outputs["Vector"]})
    
    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketVectorXYZ', 'FinScale', (1.0000, 1.0000, 0.5000)),
            ('NodeSocketFloat', 'RoundWeight', 1.0000),
            ('NodeSocketFloat', 'Freq', 69.1150),
            ('NodeSocketFloat', 'OffsetWeightZ', 1.0000),
            ('NodeSocketVector', 'PatternRotation', (4.0000, 0.0000, 2.0000)),
            ('NodeSocketFloat', 'OffsetWeightY', 1.0000),
            ('NodeSocketFloat', 'RoundingWeight', 0.0000),
            ('NodeSocketFloat', 'AffineX', 0.0000),
            ('NodeSocketFloat', 'AffineZ', 0.0000),
            ('NodeSocketFloat', 'Value', 0.5000),
            ('NodeSocketFloat', 'NoiseWeight', 0.0000),
            ('NodeSocketFloat', 'BumpX', 0.0000),
            ('NodeSocketFloat', 'BumpZ', 0.0000),
            ('NodeSocketFloat', 'NoiseRatioZ', 1.0000),
            ('NodeSocketFloat', 'NoiseRatioX', 1.0000)])
    
    add_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: separate_xyz_1.outputs["Z"], 1: group_input.outputs["NoiseWeight"]})
    
    multiply = nw.new_node(Nodes.Math,
        input_kwargs={0: add_1, 1: separate_xyz_1.outputs["X"]},
        attrs={'operation': 'MULTIPLY'})
    
    multiply_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: multiply, 1: group_input.outputs["AffineZ"]},
        attrs={'operation': 'MULTIPLY'})
    
    separate_xyz_2 = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': capture_attribute.outputs["Attribute"]})
    
    add_2 = nw.new_node(Nodes.Math,
        input_kwargs={0: separate_xyz_2.outputs["X"]})
    
    float_curve = nw.new_node(Nodes.FloatCurve,
        input_kwargs={'Value': add_2})
    node_utils.assign_curve(float_curve.mapping.curves[0], [(0.0068, 0.0000), (0.0455, 0.3812), (0.1091, 0.5419), (0.1955, 0.6437), (0.3205, 0.7300), (0.4955, 0.7719), (0.7545, 0.7350), (0.8705, 0.6562), (1.0000, 0.4413)])
    
    subtract = nw.new_node(Nodes.Math,
        input_kwargs={0: float_curve, 1: 0.7000},
        attrs={'operation': 'SUBTRACT'})
    
    multiply_2 = nw.new_node(Nodes.Math,
        input_kwargs={0: subtract, 1: group_input.outputs["RoundWeight"]},
        attrs={'operation': 'MULTIPLY'})
    
    add_3 = nw.new_node(Nodes.Math,
        input_kwargs={0: separate_xyz_1.outputs["X"], 1: group_input.outputs["Value"]})
    
    multiply_3 = nw.new_node(Nodes.Math,
        input_kwargs={0: add_3, 1: separate_xyz_1.outputs["Z"]},
        attrs={'operation': 'MULTIPLY'})
    
    multiply_4 = nw.new_node(Nodes.Math,
        input_kwargs={0: multiply_3, 1: group_input.outputs["AffineX"]},
        attrs={'operation': 'MULTIPLY'})
    
    add_4 = nw.new_node(Nodes.Math,
        input_kwargs={0: multiply_2, 1: multiply_4})
    
    combine_xyz = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'X': multiply_1, 'Z': add_4})
    
    set_position = nw.new_node(Nodes.SetPosition,
        input_kwargs={'Geometry': capture_attribute.outputs["Geometry"], 'Offset': combine_xyz})
    
    noise_texture = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Scale': 3.0000})
    
    subtract_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: noise_texture.outputs["Fac"]},
        attrs={'operation': 'SUBTRACT'})
    
    nodegroup_mix2_values_no_gc = nw.new_node(nodegroup_mix2_values().name,
        input_kwargs={'Ratio': group_input.outputs["NoiseRatioX"], 'Value1': separate_xyz_2.outputs["X"], 'Value2': subtract_1})
    
    add_5 = nw.new_node(Nodes.Math,
        input_kwargs={0: nodegroup_mix2_values_no_gc, 1: 10.0000})
    
    separate_xyz_3 = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': group_input.outputs["PatternRotation"]})
    
    multiply_5 = nw.new_node(Nodes.Math,
        input_kwargs={0: add_5, 1: 0.1000},
        attrs={'operation': 'MULTIPLY'})
    
    subtract_2 = nw.new_node(Nodes.Math,
        input_kwargs={0: separate_xyz_3.outputs["X"], 1: multiply_5},
        attrs={'operation': 'SUBTRACT'})
    
    add_6 = nw.new_node(Nodes.Math,
        input_kwargs={0: add_5, 1: subtract_2})
    
    power = nw.new_node(Nodes.Math,
        input_kwargs={0: add_6, 1: 2.0000},
        attrs={'operation': 'POWER'})
    
    nodegroup_mix2_values_no_gc_1 = nw.new_node(nodegroup_mix2_values().name,
        input_kwargs={'Ratio': group_input.outputs["NoiseRatioZ"], 'Value1': separate_xyz_2.outputs["Z"], 'Value2': subtract_1})
    
    add_7 = nw.new_node(Nodes.Math,
        input_kwargs={0: nodegroup_mix2_values_no_gc_1, 1: 1.0000})
    
    multiply_6 = nw.new_node(Nodes.Math,
        input_kwargs={0: add_7, 1: 0.1000},
        attrs={'operation': 'MULTIPLY'})
    
    subtract_3 = nw.new_node(Nodes.Math,
        input_kwargs={0: separate_xyz_3.outputs["Z"], 1: multiply_6},
        attrs={'operation': 'SUBTRACT'})
    
    add_8 = nw.new_node(Nodes.Math,
        input_kwargs={0: add_7, 1: subtract_3})
    
    multiply_7 = nw.new_node(Nodes.Math,
        input_kwargs={0: add_8},
        attrs={'operation': 'MULTIPLY'})
    
    power_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: multiply_7, 1: 2.0000},
        attrs={'operation': 'POWER'})
    
    add_9 = nw.new_node(Nodes.Math,
        input_kwargs={0: power, 1: power_1})
    
    sqrt = nw.new_node(Nodes.Math,
        input_kwargs={0: add_9},
        attrs={'operation': 'SQRT'})
    
    multiply_8 = nw.new_node(Nodes.Math,
        input_kwargs={0: sqrt, 1: group_input.outputs["Freq"]},
        attrs={'operation': 'MULTIPLY'})
    
    sine = nw.new_node(Nodes.Math,
        input_kwargs={0: multiply_8},
        attrs={'operation': 'SINE'})
    
    power_2 = nw.new_node(Nodes.Math,
        input_kwargs={0: sine, 1: 2.1000},
        attrs={'operation': 'POWER'})
    
    capture_attribute_1 = nw.new_node(Nodes.CaptureAttribute,
        input_kwargs={'Geometry': set_position, 2: power_2})
    
    multiply_9 = nw.new_node(Nodes.Math,
        input_kwargs={0: separate_xyz_2.outputs["X"], 1: group_input.outputs["BumpX"]},
        attrs={'operation': 'MULTIPLY'})
    
    multiply_10 = nw.new_node(Nodes.Math,
        input_kwargs={0: separate_xyz_2.outputs["Z"], 1: group_input.outputs["BumpZ"]},
        attrs={'operation': 'MULTIPLY'})
    
    add_10 = nw.new_node(Nodes.Math,
        input_kwargs={0: multiply_9, 1: multiply_10})
    
    subtract_4 = nw.new_node(Nodes.Math,
        input_kwargs={1: add_10},
        attrs={'operation': 'SUBTRACT'})
    
    capture_attribute_2 = nw.new_node(Nodes.CaptureAttribute,
        input_kwargs={'Geometry': capture_attribute_1.outputs["Geometry"], 2: subtract_4})
    
    noise_texture_1 = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Scale': 100.0000})
    
    subtract_5 = nw.new_node(Nodes.Math,
        input_kwargs={0: noise_texture_1.outputs["Fac"]},
        attrs={'operation': 'SUBTRACT'})
    
    normal = nw.new_node(Nodes.InputNormal)
    
    multiply_11 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: subtract_5, 1: normal},
        attrs={'operation': 'MULTIPLY'})
    
    value = nw.new_node(Nodes.Value)
    value.outputs[0].default_value = 0.0010
    
    multiply_12 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: multiply_11.outputs["Vector"], 1: value},
        attrs={'operation': 'MULTIPLY'})
    
    set_position_1 = nw.new_node(Nodes.SetPosition,
        input_kwargs={'Geometry': capture_attribute_2.outputs["Geometry"], 'Offset': multiply_12.outputs["Vector"]})
    
    subtract_6 = nw.new_node(Nodes.Math,
        input_kwargs={1: separate_xyz_2.outputs["X"]},
        attrs={'operation': 'SUBTRACT'})
    
    multiply_13 = nw.new_node(Nodes.Math,
        input_kwargs={0: sine, 1: subtract_6},
        attrs={'operation': 'MULTIPLY'})
    
    multiply_14 = nw.new_node(Nodes.Math,
        input_kwargs={0: group_input.outputs["OffsetWeightZ"], 1: -0.0200},
        attrs={'operation': 'MULTIPLY'})
    
    multiply_15 = nw.new_node(Nodes.Math,
        input_kwargs={0: multiply_13, 1: multiply_14},
        attrs={'operation': 'MULTIPLY'})
    
    sign = nw.new_node(Nodes.Math,
        input_kwargs={0: separate_xyz_2.outputs["Y"]},
        attrs={'operation': 'SIGN'})
    
    multiply_16 = nw.new_node(Nodes.Math,
        input_kwargs={0: power_2, 1: sign},
        attrs={'operation': 'MULTIPLY'})
    
    multiply_17 = nw.new_node(Nodes.Math,
        input_kwargs={0: group_input.outputs["OffsetWeightY"], 1: 0.0060},
        attrs={'operation': 'MULTIPLY'})
    
    multiply_18 = nw.new_node(Nodes.Math,
        input_kwargs={0: multiply_17, 1: subtract_4},
        attrs={'operation': 'MULTIPLY'})
    
    multiply_19 = nw.new_node(Nodes.Math,
        input_kwargs={0: multiply_16, 1: multiply_18},
        attrs={'operation': 'MULTIPLY'})
    
    multiply_20 = nw.new_node(Nodes.Math,
        input_kwargs={0: group_input.outputs["OffsetWeightZ"], 1: 0.0300},
        attrs={'operation': 'MULTIPLY'})
    
    multiply_21 = nw.new_node(Nodes.Math,
        input_kwargs={0: sine, 1: multiply_20},
        attrs={'operation': 'MULTIPLY'})
    
    add_11 = nw.new_node(Nodes.Math,
        input_kwargs={0: multiply_21, 1: 0.0000})
    
    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'X': multiply_15, 'Y': multiply_19, 'Z': add_11})
    
    set_position_2 = nw.new_node(Nodes.SetPosition,
        input_kwargs={'Geometry': set_position_1, 'Offset': combine_xyz_1})
    
    noise_texture_2 = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'W': -0.6000, 'Scale': 0.8000},
        attrs={'noise_dimensions': '4D'})
    
    subtract_7 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: noise_texture_2.outputs["Color"], 1: (0.5000, 0.5000, 0.5000)},
        attrs={'operation': 'SUBTRACT'})
    
    multiply_22 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: subtract_7.outputs["Vector"], 1: group_input.outputs["NoiseWeight"]},
        attrs={'operation': 'MULTIPLY'})
    
    set_position_3 = nw.new_node(Nodes.SetPosition,
        input_kwargs={'Geometry': set_position_2, 'Offset': multiply_22.outputs["Vector"]})
    
    position_2 = nw.new_node(Nodes.InputPosition)
    
    separate_xyz_4 = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': position_2})
    
    subtract_8 = nw.new_node(Nodes.Math,
        input_kwargs={0: separate_xyz_4.outputs["Z"]},
        attrs={'operation': 'SUBTRACT'})
    
    absolute = nw.new_node(Nodes.Math,
        input_kwargs={0: subtract_8},
        attrs={'operation': 'ABSOLUTE'})
    
    power_3 = nw.new_node(Nodes.Math,
        input_kwargs={0: absolute, 1: 1.0000},
        attrs={'operation': 'POWER'})
    
    multiply_23 = nw.new_node(Nodes.Math,
        input_kwargs={0: power_3, 1: 0.0000},
        attrs={'operation': 'MULTIPLY'})
    
    combine_xyz_2 = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'Z': multiply_23})
    
    set_position_4 = nw.new_node(Nodes.SetPosition,
        input_kwargs={'Geometry': set_position_3, 'Offset': combine_xyz_2})
    
    transform = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': set_position_4, 'Translation': (0.0000, 0.0000, 0.4000)})
    
    transform_1 = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': transform, 'Rotation': (0.0000, 0.0000, -1.5708), 'Scale': group_input.outputs["FinScale"]})
    
    transform_2 = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': transform_1, 'Rotation': (1.5708, 0.0000, 1.5708)})
    
    multiply_24 = nw.new_node(Nodes.Math,
        input_kwargs={0: capture_attribute_1.outputs[2], 1: capture_z_rigidity.outputs[2]},
        attrs={'operation': 'MULTIPLY'})
    
    multiply_25 = nw.new_node(Nodes.Math,
        input_kwargs={0: multiply_24, 1: 1.600},
        attrs={'operation': 'MULTIPLY'})
    
    add_final_rigidity = nw.new_node(Nodes.Math,
        input_kwargs={0: capture_z_rigidity.outputs[2], 1: multiply_25},
        label='add_final_rigidity',
        attrs={'use_clamp': True})
    
    store_cloth_pin = nw.new_node(Nodes.StoreNamedAttribute,
        input_kwargs={
            'Geometry': transform_2, 
            'Name': 'cloth_pin_rigidity', 
            'Value': add_final_rigidity
        },
        label='store_cloth_pin')
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': store_cloth_pin, 'Bump': capture_attribute_1.outputs[2], 'BumpMask': capture_attribute_2.outputs[2]})

class FishFin(PartFactory):

    tags = ['limb', 'fin']

    def __init__(self, *args, rig=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.rig = rig

    def sample_params(self):
        params = {
            'FinScale': np.array((1.0, 1.0, 0.5), dtype=np.float32),
            'RoundWeight': sample_range(0, 1),
            'Freq': sample_range(50, 100),
            'OffsetWeightZ': sample_range(0.1, 0.5),
            'PatternRotation': np.array((4.0 if random.random()<0.5 else -4, 0.0, 2.0), dtype=np.float32),
            'OffsetWeightY': sample_range(0.5, 1),
            'RoundingWeight': sample_range(0.02, 0.07),
            'AffineX': sample_range(0, 0.3),
            'AffineZ': sample_range(0, 1),
            'Value': 0.5,
            'Value': 0.5,
            'NoiseWeight': 0.0,
            'BumpX': 0.0,
            'BumpZ': 1.0,
            'NoiseRatioZ': 1.0,
            'NoiseRatioX': sample_range(0.9, 0.95)
        }
        return params

    def make_part(self, params):

        part = Part(skeleton=np.zeros((2, 3), dtype=float), obj=butil.spawn_vert('fin_parent'))

        fin = butil.spawn_vert('Fin')
        fin.parent = part.obj

        _, mod = butil.modify_mesh(fin, 'NODES', apply=False, return_mod=True, node_group=nodegroup_fish_fin())
        set_geomod_inputs(mod, params)
        
        id1 = mod.node_group.outputs['Bump'].identifier
        mod[f'{id1}_attribute_name'] = 'Bump'
        id2 = mod.node_group.outputs['BumpMask'].identifier
        mod[f'{id2}_attribute_name'] = 'BumpMask'

        butil.apply_modifiers(fin, mod)

        part.settings['rig_extras'] = self.rig
        tag_object(part.obj, 'fish_fin')
        return part

if __name__ == "__main__":
    fin = FishFin()
    import os
    fn = os.path.join(os.path.abspath(os.curdir), 'dev_scene_test_fin.blend')
    bpy.ops.wm.save_as_mainfile(filepath=fn)