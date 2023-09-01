# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Yiming Zuo


import bpy
import mathutils
import numpy as np
from numpy.random import uniform, normal, randint
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.nodes import node_utils
from infinigen.core.util.color import color_category, hsv2rgba
from infinigen.core import surface
from infinigen.assets.leaves.leaf_v2 import nodegroup_move_to_origin, nodegroup_apply_wave
from infinigen.assets.leaves.leaf_maple import nodegroup_leaf_shader

from infinigen.core.util.math import FixedSeed
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.util import blender as butil
from infinigen.assets.utils.tag import tag_object, tag_nodegroup

def deg2rad(deg):
    return deg / 180.0 * np.pi

@node_utils.to_nodegroup('nodegroup_ginko_stem', singleton=False, type='GeometryNodeTree')
def nodegroup_ginko_stem(nw: NodeWrangler, stem_curve_control_points=[(0.0, 0.4938), (0.3659, 0.4969), (0.7477, 0.4688), (1.0, 0.4969)]):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketVector', 'Coordinate', (0.0, 0.0, 0.0)),
            ('NodeSocketFloat', 'Length', 0.64),
            ('NodeSocketFloat', 'Value', 0.005)])
    
    add = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: group_input.outputs["Coordinate"], 1: (0.0, 0.03, 0.0)})
    
    separate_xyz = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': add.outputs["Vector"]})
    
    map_range_2 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': separate_xyz.outputs["Y"], 1: -1.0, 2: 0.0})
    
    float_curve_1 = nw.new_node(Nodes.FloatCurve,
        input_kwargs={'Value': map_range_2.outputs["Result"]})
    node_utils.assign_curve(float_curve_1.mapping.curves[0], stem_curve_control_points)
    
    map_range_3 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': float_curve_1, 3: -1.0})
    
    add_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: map_range_3.outputs["Result"], 1: separate_xyz.outputs["X"]})
    
    absolute = nw.new_node(Nodes.Math,
        input_kwargs={0: add_1},
        attrs={'operation': 'ABSOLUTE'})
    
    map_range = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': separate_xyz.outputs["Y"], 1: -1.72, 2: -0.35, 3: 0.03, 4: 0.008},
        attrs={'interpolation_type': 'SMOOTHSTEP'})
    
    subtract = nw.new_node(Nodes.Math,
        input_kwargs={0: absolute, 1: map_range.outputs["Result"]},
        attrs={'operation': 'SUBTRACT'})
    
    add_2 = nw.new_node(Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Y"], 1: group_input.outputs["Length"]})
    
    absolute_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: add_2},
        attrs={'operation': 'ABSOLUTE'})
    
    subtract_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: absolute_1, 1: group_input.outputs["Length"]},
        attrs={'operation': 'SUBTRACT'})
    
    smooth_max = nw.new_node(Nodes.Math,
        input_kwargs={0: subtract, 1: subtract_1, 2: 0.02},
        attrs={'operation': 'SMOOTH_MAX'})
    
    subtract_2 = nw.new_node(Nodes.Math,
        input_kwargs={0: smooth_max, 1: group_input.outputs["Value"]},
        attrs={'operation': 'SUBTRACT'})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Stem': subtract_2, 'Stem Raw': absolute})

@node_utils.to_nodegroup('nodegroup_ginko_vein', singleton=False, type='GeometryNodeTree')
def nodegroup_ginko_vein(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketVector', 'Vector', (0.0, 0.0, 0.0)),
            ('NodeSocketFloat', 'Scale Vein', 80.0),
            ('NodeSocketFloat', 'Scale Wave', 5.0)])
    
    subtract = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: group_input.outputs["Vector"], 1: (-0.18, 0.0, 0.0)},
        attrs={'operation': 'SUBTRACT'})
    
    noise_texture_1 = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Vector': subtract.outputs["Vector"]})
    
    gradient_texture_1 = nw.new_node(Nodes.GradientTexture,
        input_kwargs={'Vector': subtract.outputs["Vector"]},
        attrs={'gradient_type': 'RADIAL'}
        )
    
    pingpong = nw.new_node(Nodes.Math,
        input_kwargs={0: gradient_texture_1.outputs["Fac"]},
        attrs={'operation': 'PINGPONG'})
    
    length = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: subtract.outputs["Vector"]},
        attrs={'operation': 'LENGTH'})
    
    subtract_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: pingpong},
        attrs={'operation': 'SUBTRACT'})
    
    multiply = nw.new_node(Nodes.Math,
        input_kwargs={0: subtract_1, 1: -0.44},
        attrs={'operation': 'MULTIPLY'})
    
    multiply_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: length.outputs["Value"], 1: multiply},
        attrs={'operation': 'MULTIPLY'})
    
    add = nw.new_node(Nodes.Math,
        input_kwargs={0: pingpong, 1: multiply_1})
    
    multiply_add = nw.new_node(Nodes.Math,
        input_kwargs={0: noise_texture_1.outputs["Fac"], 1: 0.005, 2: add},
        attrs={'operation': 'MULTIPLY_ADD'})
    
    combine_xyz_2 = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'X': multiply_add})
    
    wave_texture_1 = nw.new_node(Nodes.WaveTexture,
        input_kwargs={'Vector': combine_xyz_2, 'Scale': group_input.outputs["Scale Vein"], 'Distortion': 0.6, 'Detail': 3.0, 'Detail Scale': 5.0, 'Detail Roughness': 1.0, 'Phase Offset': -4.62})
    
    multiply_2 = nw.new_node(Nodes.Math,
        input_kwargs={0: wave_texture_1.outputs["Color"], 1: length.outputs["Value"]},
        attrs={'operation': 'MULTIPLY'})
    
    map_range_1 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': multiply_2, 1: 0.15, 2: -0.32, 4: -0.02})
    
    multiply_add_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: noise_texture_1.outputs["Fac"], 1: 0.03, 2: add},
        attrs={'operation': 'MULTIPLY_ADD'})
    
    combine_xyz_3 = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'X': multiply_add_1})
    
    wave_texture_2 = nw.new_node(Nodes.WaveTexture,
        input_kwargs={'Vector': combine_xyz_3, 'Scale': group_input.outputs["Scale Wave"], 'Distortion': -0.42, 'Detail': 10.0, 'Detail Roughness': 1.0, 'Phase Offset': -4.62})
    
    multiply_3 = nw.new_node(Nodes.Math,
        input_kwargs={0: wave_texture_2.outputs["Fac"], 1: length.outputs["Value"]},
        attrs={'operation': 'MULTIPLY'})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Vein': map_range_1.outputs["Result"], 'Wave': multiply_3})

@node_utils.to_nodegroup('nodegroup_ginko_shape', singleton=False, type='GeometryNodeTree')
def nodegroup_ginko_shape(nw: NodeWrangler, shape_curve_control_points=[(0.0, 0.0), (0.523, 0.1156), (0.5805, 0.7469), (0.7742, 0.7719), (0.9461, 0.7531), (1.0, 0.0)]):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketVector', 'Coordinate', (0.0, 0.0, 0.0)),
        ('NodeSocketFloat', 'Multiplier', 1.980),
        ('NodeSocketFloat', 'Scale Margin', 6.6),
        ])
    
    multiply = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: group_input.outputs["Coordinate"], 1: (0.9, 1.0, 0.0)},
        attrs={'operation': 'MULTIPLY'})
    
    length = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: multiply.outputs["Vector"]},
        attrs={'operation': 'LENGTH'})
    
    gradient_texture = nw.new_node('ShaderNodeTexGradient',
        input_kwargs={'Vector': group_input.outputs["Coordinate"]})

    gradient_texture = nw.new_node(Nodes.GradientTexture,
        input_kwargs={'Vector': group_input.outputs["Coordinate"]},
        attrs={'gradient_type': 'RADIAL'})
    
    pingpong = nw.new_node(Nodes.Math,
        input_kwargs={0: gradient_texture.outputs["Fac"]},
        attrs={'operation': 'PINGPONG'})
    
    multiply_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: pingpong, 1: group_input.outputs["Multiplier"]},
        attrs={'operation': 'MULTIPLY'})
    
    noise_texture = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'W': gradient_texture.outputs["Fac"]},
        attrs={'noise_dimensions': '1D'})
    
    multiply_2 = nw.new_node(Nodes.Math,
        input_kwargs={0: noise_texture.outputs["Fac"], 1: 0.3},
        attrs={'operation': 'MULTIPLY'})
    
    add = nw.new_node(Nodes.Math,
        input_kwargs={0: multiply_1, 1: multiply_2})
    
    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'X': add})
    
    wave_texture = nw.new_node(Nodes.WaveTexture,
        input_kwargs={'Vector': combine_xyz_1, 'Scale': group_input.outputs["Scale Margin"], 'Distortion': 5.82, 'Detail': 1.52, 'Detail Roughness': 1.0})
    
    multiply_3 = nw.new_node(Nodes.Math,
        input_kwargs={0: wave_texture.outputs["Fac"], 1: 0.02},
        attrs={'operation': 'MULTIPLY'})
    
    float_curve = nw.new_node(Nodes.FloatCurve,
        input_kwargs={'Value': multiply_1})
    node_utils.assign_curve(float_curve.mapping.curves[0], shape_curve_control_points)
    
    add_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: multiply_3, 1: float_curve})
    
    subtract = nw.new_node(Nodes.Math,
        input_kwargs={0: length.outputs["Value"], 1: add_1},
        attrs={'operation': 'SUBTRACT'})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Value': subtract})

@node_utils.to_nodegroup('nodegroup_valid_area', singleton=False, type='GeometryNodeTree')
def nodegroup_valid_area(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketFloat', 'Value', 0.5)])
    
    sign = nw.new_node(Nodes.Math,
        input_kwargs={0: group_input.outputs["Value"]},
        attrs={'operation': 'SIGN'})
    
    map_range_4 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': sign, 1: -1.0, 3: 1.0, 4: 0.0})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Result': map_range_4.outputs["Result"]})

@node_utils.to_nodegroup('nodegroup_ginko', singleton=False, type='GeometryNodeTree')
def nodegroup_ginko(nw: NodeWrangler, stem_curve_control_points, shape_curve_control_points):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketGeometry', 'Mesh', None),
            ('NodeSocketFloat', 'Vein Length', 0.64),
            ('NodeSocketFloat', 'Vein Width', 0.005),
            ('NodeSocketFloatAngle', 'Angle', -1.7617),
            ('NodeSocketFloat', 'Displacenment', 0.5),
            ('NodeSocketFloat', 'Multiplier', 1.980),
            ('NodeSocketFloat', 'Scale Vein', 80.0),
            ('NodeSocketFloat', 'Scale Wave', 5.0),
            ('NodeSocketFloat', 'Scale Margin', 6.6),
            ('NodeSocketInt', 'Level', 9),
            ])
    
    subdivide_mesh = nw.new_node(Nodes.SubdivideMesh,
        input_kwargs={'Mesh': group_input.outputs["Mesh"], 'Level': group_input.outputs["Level"]})
    
    position = nw.new_node(Nodes.InputPosition)
    
    vector_rotate = nw.new_node(Nodes.VectorRotate,
        input_kwargs={'Vector': position, 'Angle': group_input.outputs["Angle"]},
        attrs={'rotation_type': 'Z_AXIS'})
    
    ginkoshape = nw.new_node(nodegroup_ginko_shape(shape_curve_control_points=shape_curve_control_points).name,
        input_kwargs={'Coordinate': vector_rotate, 'Multiplier': group_input.outputs["Multiplier"], 'Scale Margin': group_input.outputs["Scale Margin"]})
    
    validarea = nw.new_node(nodegroup_valid_area().name,
        input_kwargs={'Value': ginkoshape})
    
    ginkovein = nw.new_node(nodegroup_ginko_vein().name,
        input_kwargs={'Vector': vector_rotate, 'Scale Vein': group_input.outputs["Scale Vein"], 'Scale Wave': group_input.outputs["Scale Wave"]})
    
    multiply = nw.new_node(Nodes.Math,
        input_kwargs={0: validarea, 1: ginkovein.outputs["Vein"]},
        attrs={'operation': 'MULTIPLY'})
    
    map_range_4 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': ginkoshape, 1: -1.0, 2: 0.0, 3: -5.0, 4: 0.0},
        attrs={'clamp': False})
    
    multiply_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: multiply, 1: map_range_4.outputs["Result"]},
        attrs={'operation': 'MULTIPLY', 'use_clamp': True})
    
    clamp = nw.new_node(Nodes.Clamp,
        input_kwargs={'Value': multiply_1, 'Max': 0.01})
    
    capture_attribute_1 = nw.new_node(Nodes.CaptureAttribute,
        input_kwargs={'Geometry': subdivide_mesh, 2: clamp})
    
    capture_attribute = nw.new_node(Nodes.CaptureAttribute,
        input_kwargs={'Geometry': capture_attribute_1.outputs["Geometry"], 2: ginkoshape})
    
    ginkostem = nw.new_node(nodegroup_ginko_stem(stem_curve_control_points=stem_curve_control_points).name,
        input_kwargs={'Coordinate': position, 'Length': group_input.outputs["Vein Length"], 'Value': group_input.outputs["Vein Width"]})
    
    smooth_min = nw.new_node(Nodes.Math,
        input_kwargs={0: ginkoshape, 1: ginkostem.outputs["Stem"], 2: 0.1},
        attrs={'operation': 'SMOOTH_MIN'})
    
    multiply_2 = nw.new_node(Nodes.Math,
        input_kwargs={0: smooth_min, 1: -1.0},
        attrs={'operation': 'MULTIPLY'})
    
    stem_length = nw.new_node(Nodes.Compare,
        input_kwargs={0: multiply_2, 1: 0.0},
        label='stem length',
        attrs={'operation': 'LESS_THAN'})
    
    delete_geometry = nw.new_node(Nodes.DeleteGeom,
        input_kwargs={'Geometry': capture_attribute.outputs["Geometry"], 'Selection': stem_length})
    
    validarea_1 = nw.new_node(nodegroup_valid_area().name,
        input_kwargs={'Value': ginkostem.outputs["Stem"]})
    
    multiply_3 = nw.new_node(Nodes.Math,
        input_kwargs={0: validarea_1, 1: ginkostem.outputs["Stem Raw"]},
        attrs={'operation': 'MULTIPLY'})
    
    add = nw.new_node(Nodes.Math,
        input_kwargs={0: multiply_3, 1: clamp})
    
    multiply_4 = nw.new_node(Nodes.Math,
        input_kwargs={0: add, 1: group_input.outputs["Displacenment"]},
        attrs={'operation': 'MULTIPLY'})
    
    combine_xyz = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'Z': multiply_4})
    
    set_position = nw.new_node(Nodes.SetPosition,
        input_kwargs={'Geometry': delete_geometry, 'Offset': combine_xyz})
    
    validarea_2 = nw.new_node(nodegroup_valid_area().name,
        input_kwargs={'Value': ginkoshape})
    
    multiply_5 = nw.new_node(Nodes.Math,
        input_kwargs={0: validarea_2, 1: ginkovein.outputs["Wave"]},
        attrs={'operation': 'MULTIPLY'})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': set_position, 'Vein': capture_attribute_1.outputs[2], 'Shape': capture_attribute.outputs[2], 'Wave': multiply_5})

def shader_material(nw: NodeWrangler, **kwargs):
    # Code generated using version 2.4.3 of the node_transpiler

    attribute = nw.new_node(Nodes.Attribute,
        attrs={'attribute_name': 'vein'})
    
    map_range = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': attribute.outputs["Color"], 2: 0.12, 4: 6.26})
    
    attribute_1 = nw.new_node(Nodes.Attribute,
        attrs={'attribute_name': 'shape'})
    
    map_range_1 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': attribute_1.outputs["Color"], 1: -0.74, 2: 0.01, 3: 2.0, 4: 0.0})
    
    float_curve = nw.new_node(Nodes.FloatCurve,
        input_kwargs={'Value': map_range_1.outputs["Result"]})
    node_utils.assign_curve(float_curve.mapping.curves[0], [(0.0, 0.0), (0.3795, 0.6344), (1.0, 1.0)])
    
    separate_hsv = nw.new_node('ShaderNodeSeparateHSV',
        input_kwargs={'Color': kwargs['color_base']})
    
    subtract = nw.new_node(Nodes.Math,
        input_kwargs={0: separate_hsv.outputs["V"], 1: 0.2},
        attrs={'operation': 'SUBTRACT'})
    
    combine_hsv = nw.new_node(Nodes.CombineHSV,
        input_kwargs={'H': separate_hsv.outputs["H"], 'S': separate_hsv.outputs["S"], 'V': subtract})
    
    mix_1 = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': float_curve, 'Color1': kwargs['color_base'], 'Color2': combine_hsv})
    
    mix = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': map_range.outputs["Result"], 'Color1': mix_1, 'Color2': kwargs['color_vein']})
    
    group = nw.new_node(nodegroup_leaf_shader().name,
        input_kwargs={'Color': mix})
    
    material_output = nw.new_node(Nodes.MaterialOutput,
        input_kwargs={'Surface': group})

def geo_leaf_ginko(nw: NodeWrangler, **kwargs):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketGeometry', 'Geometry', None)])
    
    nodegroup = nw.new_node(nodegroup_ginko(stem_curve_control_points=kwargs['stem_curve_control_points'],
        shape_curve_control_points=kwargs['shape_curve_control_points']).name,
        input_kwargs={'Mesh': group_input.outputs["Geometry"], 
            'Vein Length': kwargs['vein_length'], 
            'Angle': deg2rad(kwargs['angle']),
            'Multiplier': kwargs['multiplier'],
            'Scale Vein': kwargs['scale_vein'],
            'Scale Wave': kwargs['scale_wave'],
            'Scale Margin': kwargs['scale_margin'],
            })
    
    map_range = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': nodegroup.outputs["Wave"], 4: 0.04})
    
    combine_xyz = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'Z': map_range.outputs["Result"]})
    
    set_position = nw.new_node(Nodes.SetPosition,
        input_kwargs={'Geometry': nodegroup.outputs["Geometry"], 'Offset': combine_xyz})
    
    position = nw.new_node(Nodes.InputPosition)
    
    separate_xyz = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': position})
    
    apply_wave = nw.new_node(nodegroup_apply_wave(y_wave_control_points=kwargs['y_wave_control_points'], x_wave_control_points=kwargs['x_wave_control_points']).name,
        input_kwargs={'Geometry': set_position, 'Wave Scale X': 0.0, 'Wave Scale Y': 1.0, 'X Modulated': separate_xyz.outputs["X"]})
    
    move_to_origin = nw.new_node(nodegroup_move_to_origin().name,
        input_kwargs={'Geometry': apply_wave})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': move_to_origin, 'Vein': nodegroup.outputs["Vein"], 'Shape': nodegroup.outputs["Shape"]})

class LeafFactoryGinko(AssetFactory):

    scale = 0.3

    def __init__(self, factory_seed, season='autumn', coarse=False):
        super(LeafFactoryGinko, self).__init__(factory_seed, coarse=coarse)

        with FixedSeed(factory_seed):
            self.genome = self.sample_geo_genome()

            t = uniform(0.0, 1.0)

            # self.blade_color = hsv2rgba([0.125 + 0.16 * factory_seed / 10, 0.95, 0.6])
            
            if season=='autumn':
                self.blade_color = [uniform(0.125, 0.2), 0.95, 0.6]
            elif season=='summer' or season=='spring':
                self.blade_color = [uniform(0.25, 0.3), 0.95, 0.6]
            elif season=='winter':
                self.blade_color = [uniform(0.125, 0.2), 0.95, 0.6]
            else:
                raise NotImplementedError

            self.color_randomness = 0.05
            
    @staticmethod
    def sample_geo_genome():
        return {
            'midrib_length': uniform(0.0, 0.8),
            'midrib_width': uniform(0.5, 1.0),
            'stem_length': uniform(0.7, 0.9),
            'vein_asymmetry': uniform(0.0, 1.0),
            'vein_angle': uniform(0.2, 2.0),
            'vein_density': uniform(5.0, 20.0),
            'subvein_scale': uniform(10.0, 20.0),
            'jigsaw_scale': uniform(5.0, 20.0),
            'jigsaw_depth': uniform(0.0, 2.0),
            'midrib_shape_control_points': [(0.0, 0.5), (0.25, uniform(0.48, 0.52)), (0.75, uniform(0.48, 0.52)), (1.0, 0.5)],
            'leaf_shape_control_points': [(0.0, 0.0), (uniform(0.2, 0.4), uniform(0.1, 0.4)), (uniform(0.6, 0.8), uniform(0.1, 0.4)), (1.0, 0.0)],
            'vein_shape_control_points': [(0.0, 0.0), (0.25, uniform(0.1, 0.4)), (0.75, uniform(0.6, 0.9)), (1.0, 1.0)],
        }

    def create_asset(self, **params):

        bpy.ops.mesh.primitive_plane_add(
            size=2, enter_editmode=False, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
        obj = bpy.context.active_object

        # add noise to the genotype output
        #hue_noise = np.random.randn() * 0
        #hsv_blade = self.hsv_blade + hue_noise
        #hsv_vein = self.hsv_vein + hue_noise

        phenome = self.genome.copy()

        phenome['y_wave_control_points'] = [(0.0, 0.5), (uniform(0.25, 0.75), uniform(0.50, 0.60)), (1.0, 0.5)]
        x_wave_val = np.random.uniform(0.50, 0.58)
        phenome['x_wave_control_points'] = [(0.0, 0.5), (0.4, x_wave_val), (0.5, 0.5), (0.6, x_wave_val), (1.0, 0.5)]

        phenome['stem_curve_control_points'] = [(0.0, 0.5), 
            (uniform(0.2, 0.3), uniform(0.45, 0.55)), 
            (uniform(0.7, 0.8), uniform(0.45, 0.55)), 
            (1.0, 0.5)]
        phenome['shape_curve_control_points'] = [(0.0, 0.0), (0.523, 0.1156), (0.5805, 0.7469), (0.7742, 0.7719), (0.9461, 0.7531), (1.0, 0.0)]
        phenome['vein_length'] = uniform(0.4, 0.5)
        phenome['angle'] = uniform(-110.0, -70.0)
        phenome['multiplier'] = uniform(1.90, 1.98)

        phenome['scale_vein'] = uniform(70.0, 90.0)
        phenome['scale_wave'] = uniform(4.0, 6.0)
        phenome['scale_margin'] = uniform(5.5, 7.5)

        material_kwargs = phenome.copy()
        material_kwargs['color_base'] = np.copy(self.blade_color) # (0.2346, 0.4735, 0.0273, 1.0), 
        material_kwargs['color_base'][0] += np.random.normal(0.0, 0.02)
        material_kwargs['color_base'][1] += np.random.normal(0.0, self.color_randomness)
        material_kwargs['color_base'][2] += np.random.normal(0.0, self.color_randomness)
        material_kwargs['color_base'] = hsv2rgba(material_kwargs['color_base'])

        material_kwargs['color_vein'] = hsv2rgba(np.copy(self.blade_color))

        surface.add_geomod(obj, geo_leaf_ginko, apply=False, attributes=['vein', 'shape'], input_kwargs=phenome)
        surface.add_material(obj, shader_material, reuse=False, input_kwargs=material_kwargs)

        bpy.ops.object.convert(target='MESH')

        obj = bpy.context.object
        obj.scale *= normal(1, 0.2) * self.scale
        butil.apply_transform(obj)
        tag_object(obj, 'leaf_ginko')

        return obj