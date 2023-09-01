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

from infinigen.core.util.math import FixedSeed
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.util import blender as butil
from infinigen.assets.utils.tag import tag_object, tag_nodegroup

@node_utils.to_nodegroup('nodegroup_surface_bump', singleton=False, type='GeometryNodeTree')
def nodegroup_surface_bump(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketGeometry', 'Geometry', None),
            ('NodeSocketFloat', 'Displacement', 0.0200),
            ('NodeSocketFloat', 'Scale', 50.0000),
            ('NodeSocketFloat', 'Seed', 0.0000)])
    
    normal = nw.new_node(Nodes.InputNormal)
    
    noise_texture = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'W': group_input.outputs["Seed"], 'Scale': group_input.outputs["Scale"]},
        attrs={'noise_dimensions': '4D'})
    
    subtract = nw.new_node(Nodes.Math, input_kwargs={0: noise_texture.outputs["Fac"]}, attrs={'operation': 'SUBTRACT'})
    
    multiply = nw.new_node(Nodes.Math,
        input_kwargs={0: subtract, 1: group_input.outputs["Displacement"]},
        attrs={'operation': 'MULTIPLY'})
    
    multiply_1 = nw.new_node(Nodes.VectorMath, input_kwargs={0: normal, 1: multiply}, attrs={'operation': 'MULTIPLY'})
    
    set_position = nw.new_node(Nodes.SetPosition,
        input_kwargs={'Geometry': group_input.outputs["Geometry"], 'Offset': multiply_1.outputs["Vector"]})
    
    group_output = nw.new_node(Nodes.GroupOutput, input_kwargs={'Geometry': set_position}, attrs={'is_active_output': True})

@node_utils.to_nodegroup('nodegroup_generate_anchor', singleton=False, type='GeometryNodeTree')
def nodegroup_generate_anchor(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketGeometry', 'Curve', None),
            ('NodeSocketFloat', 'curve parameter', 0.0000),
            ('NodeSocketFloat', 'trim_bottom', 0.2000),
            ('NodeSocketFloat', 'trim_top', 0.0000),
            ('NodeSocketInt', 'seed', 0),
            ('NodeSocketFloat', 'density', 0.5000),
            ('NodeSocketFloat', 'keep probablity', 0.0000)])
    
    divide = nw.new_node(Nodes.Math, input_kwargs={0: 1.0000, 1: group_input.outputs["density"]}, attrs={'operation': 'DIVIDE'})
    
    multiply = nw.new_node(Nodes.Math, input_kwargs={0: divide, 1: group_input.outputs["keep probablity"]}, attrs={'operation': 'MULTIPLY'})
    
    minimum = nw.new_node(Nodes.Math, input_kwargs={0: multiply}, attrs={'operation': 'MINIMUM'})
    
    curve_to_points_1 = nw.new_node(Nodes.CurveToPoints,
        input_kwargs={'Curve': group_input.outputs["Curve"], 'Length': minimum},
        attrs={'mode': 'LENGTH'})
    
    random_value_3 = nw.new_node(Nodes.RandomValue,
        input_kwargs={'Probability': group_input.outputs["keep probablity"], 'Seed': group_input.outputs["seed"]},
        attrs={'data_type': 'BOOLEAN'})
    
    greater_than = nw.new_node(Nodes.Compare,
        input_kwargs={0: group_input.outputs["curve parameter"], 1: group_input.outputs["trim_bottom"]})
    
    less_than = nw.new_node(Nodes.Compare,
        input_kwargs={0: group_input.outputs["curve parameter"], 1: group_input.outputs["trim_top"]},
        attrs={'operation': 'LESS_THAN'})
    
    op_and = nw.new_node(Nodes.BooleanMath, input_kwargs={0: greater_than, 1: less_than})
    
    op_and_1 = nw.new_node(Nodes.BooleanMath, input_kwargs={0: random_value_3.outputs[3], 1: op_and})
    
    op_not = nw.new_node(Nodes.BooleanMath, input_kwargs={0: op_and_1}, attrs={'operation': 'NOT'})
    
    delete_geometry = nw.new_node(Nodes.DeleteGeometry,
        input_kwargs={'Geometry': curve_to_points_1.outputs["Points"], 'Selection': op_not})
    
    group_output = nw.new_node(Nodes.GroupOutput, input_kwargs={'Points': delete_geometry}, attrs={'is_active_output': True})

@node_utils.to_nodegroup('nodegroup_create_instance', singleton=False, type='GeometryNodeTree')
def nodegroup_create_instance(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketGeometry', 'Points', None),
            ('NodeSocketGeometry', 'Instance', None),
            ('NodeSocketBool', 'Selection', True),
            ('NodeSocketBool', 'Pick Instance', False),
            ('NodeSocketVector', 'Tangent', (0.0000, 0.0000, 1.0000)),
            ('NodeSocketFloat', 'Rot x deg', 0.0000),
            ('NodeSocketFloat', 'Rot x range', 0.2000),
            ('NodeSocketFloat', 'Scale', 1.0000),
            ('NodeSocketInt', 'Seed', 0)])
    
    random_value_1 = nw.new_node(Nodes.RandomValue, input_kwargs={3: 6.2832, 'Seed': group_input.outputs["Seed"]})
    
    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ, input_kwargs={'Z': random_value_1.outputs[1]})
    
    align_euler_to_vector = nw.new_node(Nodes.AlignEulerToVector,
        input_kwargs={'Rotation': combine_xyz_1, 'Vector': group_input.outputs["Tangent"]},
        attrs={'axis': 'Y'})
    
    instance_on_points = nw.new_node(Nodes.InstanceOnPoints,
        input_kwargs={'Points': group_input.outputs["Points"], 'Selection': group_input.outputs["Selection"], 'Instance': group_input.outputs["Instance"], 'Pick Instance': group_input.outputs["Pick Instance"], 'Rotation': align_euler_to_vector, 'Scale': group_input.outputs["Scale"]})
    
    radians = nw.new_node(Nodes.Math, input_kwargs={0: group_input.outputs["Rot x deg"]}, attrs={'operation': 'RADIANS'})
    
    subtract = nw.new_node(Nodes.Math, input_kwargs={0: 1.0000, 1: group_input.outputs["Rot x range"]}, attrs={'operation': 'SUBTRACT'})
    
    multiply = nw.new_node(Nodes.Math, input_kwargs={0: radians, 1: subtract}, attrs={'operation': 'MULTIPLY'})
    
    add = nw.new_node(Nodes.Math, input_kwargs={0: 1.0000, 1: group_input.outputs["Rot x range"]})
    
    multiply_1 = nw.new_node(Nodes.Math, input_kwargs={0: radians, 1: add}, attrs={'operation': 'MULTIPLY'})
    
    random_value_2 = nw.new_node(Nodes.RandomValue, input_kwargs={2: multiply, 3: multiply_1, 'Seed': group_input.outputs["Seed"]})
    
    combine_xyz_2 = nw.new_node(Nodes.CombineXYZ, input_kwargs={'X': random_value_2.outputs[1]})
    
    rotate_instances = nw.new_node(Nodes.RotateInstances, input_kwargs={'Instances': instance_on_points, 'Rotation': combine_xyz_2})
    
    group_output = nw.new_node(Nodes.GroupOutput, input_kwargs={'Instances': rotate_instances}, attrs={'is_active_output': True})

def generate_branch(nw: NodeWrangler, **kwargs):
    # Code generated using version 2.6.4 of the node_transpiler

    curve_line = nw.new_node(Nodes.CurveLine)
    
    # group_input = nw.new_node(Nodes.GroupInput,
    #     expose_input=[('NodeSocketGeometry', 'Geometry', None),
    #         ('NodeSocketCollection', 'leaf collection', None),
    #         ('NodeSocketCollection', 'fruit collection', None),
    #         ('NodeSocketInt', 'resolution', 256),
    #         ('NodeSocketInt', 'seed', 0),
    #         ('NodeSocketFloat', 'main branch noise amount', 0.3000),
    #         ('NodeSocketFloat', 'main branch noise scale', 1.1000),
    #         ('NodeSocketFloatDistance', 'overall radius', 0.0200),
    #         ('NodeSocketFloat', 'twig density', 10.0000),
    #         ('NodeSocketFloat', 'twig rotation', 45.0000),
    #         ('NodeSocketFloat', 'twig scale', 5.0000),
    #         ('NodeSocketFloat', 'twig noise amount', 0.3000),
    #         ('NodeSocketFloat', 'leaf density', 15.0000),
    #         ('NodeSocketFloat', 'leaf scale', 0.3000),
    #         ('NodeSocketFloat', 'leaf rot', 45.0000),
    #         ('NodeSocketFloat', 'fruit density', 10.0000),
    #         ('NodeSocketFloat', 'fruit scale', 0.0500),
    #         ('NodeSocketFloat', 'fruit rot', 0.0000)])
    
    resample_curve = nw.new_node(Nodes.ResampleCurve, input_kwargs={'Curve': curve_line, 'Count': kwargs["resolution"]})
    
    spline_parameter = nw.new_node(Nodes.SplineParameter)
    
    combine_xyz = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'X': spline_parameter.outputs["Factor"], 'Y': kwargs["seed"]})
    
    noise_texture = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Vector': combine_xyz, 'Scale': kwargs["main branch noise scale"]},
        attrs={'noise_dimensions': '2D'})
    
    subtract = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: noise_texture.outputs["Color"], 1: (0.5000, 0.5000, 0.5000)},
        attrs={'operation': 'SUBTRACT'})
    
    map_range = nw.new_node(Nodes.MapRange, input_kwargs={'Value': spline_parameter.outputs["Factor"], 2: 0.2000})
    
    scale = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: subtract.outputs["Vector"], 'Scale': map_range.outputs["Result"]},
        attrs={'operation': 'SCALE'})
    
    scale_1 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: scale.outputs["Vector"], 'Scale': kwargs["main branch noise amount"]},
        attrs={'operation': 'SCALE'})
    
    set_position = nw.new_node(Nodes.SetPosition, input_kwargs={'Geometry': resample_curve, 'Offset': scale_1.outputs["Vector"]})
    
    capture_attribute = nw.new_node(Nodes.CaptureAttribute, input_kwargs={'Geometry': set_position, 2: spline_parameter.outputs["Factor"]})
    
    add = nw.new_node(Nodes.Math, input_kwargs={0: kwargs["seed"], 1: 13.0000})
    
    generateanchor = nw.new_node(nodegroup_generate_anchor().name,
        input_kwargs={'Curve': capture_attribute, 'curve parameter': capture_attribute.outputs[2], 'trim_top': 0.9000, 'seed': add, 'density': kwargs["fruit density"], 'keep probablity': 0.3000})
    
    collection_info_1 = nw.new_node(Nodes.CollectionInfo,
        input_kwargs={'Collection': kwargs["fruit collection"], 'Separate Children': True, 'Reset Children': True})
    
    createinstance = nw.new_node(nodegroup_create_instance().name,
        input_kwargs={'Points': generateanchor, 'Instance': collection_info_1, 'Pick Instance': True, 'Rot x deg': kwargs["fruit rot"], 'Scale': kwargs["fruit scale"], 'Seed': kwargs["seed"]})
    
    keep_probablity = nw.new_node(Nodes.Value, label='keep probablity')
    keep_probablity.outputs[0].default_value = 0.3000
    
    divide = nw.new_node(Nodes.Math,
        input_kwargs={0: kwargs["twig density"], 1: keep_probablity},
        attrs={'operation': 'DIVIDE'})
    
    curve_to_points = nw.new_node(Nodes.CurveToPoints, input_kwargs={'Curve': capture_attribute, 'Count': divide})
    
    curve_line_1 = nw.new_node(Nodes.CurveLine, input_kwargs={'End': (0.0000, 0.0000, 0.1000)})
    
    divide_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: kwargs["resolution"], 1: 2.0000},
        attrs={'operation': 'DIVIDE'})
    
    resample_curve_2 = nw.new_node(Nodes.ResampleCurve, input_kwargs={'Curve': curve_line_1, 'Count': divide_1})
    
    spline_parameter_1 = nw.new_node(Nodes.SplineParameter)
    
    capture_attribute_1 = nw.new_node(Nodes.CaptureAttribute,
        input_kwargs={'Geometry': resample_curve_2, 2: spline_parameter_1.outputs["Factor"]})
    
    add_1 = nw.new_node(Nodes.Math, input_kwargs={0: kwargs["seed"], 1: 37.0000})
    
    random_value = nw.new_node(Nodes.RandomValue,
        input_kwargs={'Probability': keep_probablity, 'Seed': add_1},
        attrs={'data_type': 'BOOLEAN'})
    
    index = nw.new_node(Nodes.Index)
    
    multiply = nw.new_node(Nodes.Math, input_kwargs={0: divide, 1: 0.0500}, attrs={'operation': 'MULTIPLY'})
    
    greater_equal = nw.new_node(Nodes.Compare,
        input_kwargs={2: index, 3: multiply},
        attrs={'data_type': 'INT', 'operation': 'GREATER_EQUAL'})
    
    multiply_1 = nw.new_node(Nodes.Math, input_kwargs={0: divide, 1: 0.9000}, attrs={'operation': 'MULTIPLY'})
    
    less_equal = nw.new_node(Nodes.Compare,
        input_kwargs={2: index, 3: multiply_1},
        attrs={'data_type': 'INT', 'operation': 'LESS_EQUAL'})
    
    op_and = nw.new_node(Nodes.BooleanMath, input_kwargs={0: greater_equal, 1: less_equal})
    
    op_and_1 = nw.new_node(Nodes.BooleanMath, input_kwargs={0: random_value.outputs[3], 1: op_and})
    
    multiply_2 = nw.new_node(Nodes.Math,
        input_kwargs={0: kwargs["twig rotation"], 1: -1.0000},
        attrs={'operation': 'MULTIPLY'})
    
    map_range_2 = nw.new_node(Nodes.MapRange, input_kwargs={'Value': capture_attribute.outputs[2], 3: 1.0000, 4: 0.1000})
    
    multiply_3 = nw.new_node(Nodes.Math,
        input_kwargs={0: map_range_2.outputs["Result"], 1: kwargs["twig scale"]},
        attrs={'operation': 'MULTIPLY'})
    
    createinstance_1 = nw.new_node(nodegroup_create_instance().name,
        input_kwargs={'Points': curve_to_points.outputs["Points"], 'Instance': capture_attribute_1.outputs["Geometry"], 'Selection': op_and_1, 'Tangent': curve_to_points.outputs["Tangent"], 'Rot x deg': multiply_2, 'Scale': multiply_3, 'Seed': kwargs["seed"]})
    
    realize_instances = nw.new_node(Nodes.RealizeInstances, input_kwargs={'Geometry': createinstance_1})
    
    position = nw.new_node(Nodes.InputPosition)
    
    noise_texture_1 = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Vector': position, 'W': kwargs["seed"], 'Scale': 1.5000},
        attrs={'noise_dimensions': '4D'})
    
    subtract_1 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: noise_texture_1.outputs["Color"], 1: (0.5000, 0.5000, 0.5000)},
        attrs={'operation': 'SUBTRACT'})
    
    map_range_3 = nw.new_node(Nodes.MapRange, input_kwargs={'Value': capture_attribute_1.outputs[2], 2: 0.2000})
    
    scale_2 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: subtract_1.outputs["Vector"], 'Scale': map_range_3.outputs["Result"]},
        attrs={'operation': 'SCALE'})
    
    scale_3 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: scale_2.outputs["Vector"], 'Scale': kwargs["twig noise amount"]},
        attrs={'operation': 'SCALE'})
    
    set_position_1 = nw.new_node(Nodes.SetPosition, input_kwargs={'Geometry': realize_instances, 'Offset': scale_3.outputs["Vector"]})
    
    curve_tangent = nw.new_node(Nodes.CurveTangent)
    
    capture_attribute_2 = nw.new_node(Nodes.CaptureAttribute,
        input_kwargs={'Geometry': set_position_1, 1: curve_tangent},
        attrs={'data_type': 'FLOAT_VECTOR'})
    
    add_2 = nw.new_node(Nodes.Math, input_kwargs={0: kwargs["seed"], 1: 17.0000})
    
    generateanchor_1 = nw.new_node(nodegroup_generate_anchor().name,
        input_kwargs={'Curve': capture_attribute_2.outputs["Geometry"], 'curve parameter': capture_attribute_1.outputs[2], 'trim_top': 1.0000, 'seed': add_2, 'density': kwargs["leaf density"], 'keep probablity': 0.3000})
    
    collection_info = nw.new_node(Nodes.CollectionInfo,
        input_kwargs={'Collection': kwargs["leaf collection"], 'Separate Children': True, 'Reset Children': True})
    
    createinstance_2 = nw.new_node(nodegroup_create_instance().name,
        input_kwargs={'Points': generateanchor_1, 'Instance': collection_info, 'Pick Instance': True, 'Tangent': capture_attribute_2.outputs["Attribute"], 'Rot x deg': kwargs["leaf rot"], 'Scale': kwargs["leaf scale"], 'Seed': kwargs["seed"]})
    
    map_range_1 = nw.new_node(Nodes.MapRange, input_kwargs={'Value': capture_attribute.outputs[2], 3: 1.0000, 4: 0.4000})
    
    multiply_4 = nw.new_node(Nodes.Math,
        input_kwargs={0: map_range_1.outputs["Result"], 1: kwargs["overall radius"]},
        attrs={'operation': 'MULTIPLY'})
    
    set_curve_radius = nw.new_node(Nodes.SetCurveRadius, input_kwargs={'Curve': capture_attribute, 'Radius': multiply_4})
    
    multiply_5 = nw.new_node(Nodes.Math,
        input_kwargs={0: kwargs["resolution"], 1: kwargs["overall radius"]},
        attrs={'operation': 'MULTIPLY'})
    
    multiply_6 = nw.new_node(Nodes.Math, input_kwargs={0: multiply_5, 1: 6.2832}, attrs={'operation': 'MULTIPLY'})
    
    curve_circle = nw.new_node(Nodes.CurveCircle, input_kwargs={'Resolution': multiply_6})
    
    curve_to_mesh = nw.new_node(Nodes.CurveToMesh,
        input_kwargs={'Curve': set_curve_radius, 'Profile Curve': curve_circle.outputs["Curve"], 'Fill Caps': True})
    
    map_range_4 = nw.new_node(Nodes.MapRange, input_kwargs={'Value': capture_attribute_1.outputs[2], 3: 0.8000, 4: 0.1000})
    
    multiply_7 = nw.new_node(Nodes.Math,
        input_kwargs={0: map_range_4.outputs["Result"], 1: map_range_1.outputs["Result"]},
        attrs={'operation': 'MULTIPLY'})
    
    multiply_8 = nw.new_node(Nodes.Math,
        input_kwargs={0: multiply_7, 1: kwargs["overall radius"]},
        attrs={'operation': 'MULTIPLY'})
    
    set_curve_radius_1 = nw.new_node(Nodes.SetCurveRadius,
        input_kwargs={'Curve': capture_attribute_2.outputs["Geometry"], 'Radius': multiply_8})
    
    divide_2 = nw.new_node(Nodes.Math, input_kwargs={0: multiply_6, 1: 2.0000}, attrs={'operation': 'DIVIDE'})
    
    curve_circle_1 = nw.new_node(Nodes.CurveCircle, input_kwargs={'Resolution': divide_2})
    
    curve_to_mesh_1 = nw.new_node(Nodes.CurveToMesh,
        input_kwargs={'Curve': set_curve_radius_1, 'Profile Curve': curve_circle_1.outputs["Curve"], 'Fill Caps': True})
    
    join_geometry = nw.new_node(Nodes.JoinGeometry, input_kwargs={'Geometry': [curve_to_mesh, curve_to_mesh_1]})
    
    set_material = nw.new_node(Nodes.SetMaterial,
        input_kwargs={'Geometry': join_geometry, 'Material': kwargs['material']})
    
    surfacebump = nw.new_node(nodegroup_surface_bump().name, input_kwargs={'Geometry': set_material, 'Displacement': 0.0050})
    
    join_geometry_1 = nw.new_node(Nodes.JoinGeometry, input_kwargs={'Geometry': [createinstance, createinstance_2, surfacebump]})
    
    transform = nw.new_node(Nodes.Transform, input_kwargs={'Geometry': join_geometry_1, 'Rotation': (-1.5708, 0.0000, 0.0000)})
    
    group_output = nw.new_node(Nodes.GroupOutput, input_kwargs={'Geometry': transform}, attrs={'is_active_output': True})

class BranchFactory(AssetFactory):
    def __init__(self, factory_seed, twig_col, fruit_col, coarse=False):
        super().__init__(factory_seed, coarse=coarse)

        self.avg_fruit_dim = np.cbrt(np.mean([np.prod(list(o.dimensions)) for o in fruit_col.objects]))

        with FixedSeed(factory_seed):
            self.branch_params = self.sample_branch_params()
        
        self.branch_params['leaf collection'] = twig_col
        self.branch_params['fruit collection'] = fruit_col
        self.branch_params['material'] = twig_col.objects[0].active_material

    def sample_branch_params(self):
        return {
            'resolution': 256,
            'main branch noise amount': uniform(0.2, 0.4),
            'main branch noise scale': uniform(0.9, 1.3),
            'overall radius': uniform(0.015, 0.025),
            'twig density': uniform(5, 15),
            'twig rotation': uniform(30, 60),
            'twig scale': uniform(3, 7),
            'twig noise amount': uniform(0.2, 0.4),
            'leaf density': uniform(5, 25),
            'leaf scale': uniform(0.25, 0.35),
            'leaf rot': uniform(30, 60),
            'fruit scale': uniform(0.15, 0.25),
            'fruit rot': 0.0,
            'fruit density': np.clip(uniform(1, 5) / self.avg_fruit_dim, 0.01, 50)
        }
            
    def create_asset(self, **params):

        bpy.ops.mesh.primitive_plane_add(
            size=2, enter_editmode=False, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
        obj = bpy.context.active_object

        phenome = self.branch_params.copy()
        phenome['seed'] = randint(10000000)

        surface.add_geomod(obj, generate_branch, input_kwargs=phenome)

        return obj