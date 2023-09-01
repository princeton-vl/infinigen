# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Yiming Zuo


import bpy
import mathutils
from numpy.random import uniform, normal, randint
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.nodes import node_utils
from infinigen.core.util.color import color_category
from infinigen.core import surface

from infinigen.assets.creatures.insects.utils.shader_utils import nodegroup_add_noise

@node_utils.to_nodegroup('nodegroup_dragonfly_wing', singleton=False, type='GeometryNodeTree')
def nodegroup_dragonfly_wing(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    resolution = nw.new_node(Nodes.Integer,
        label='resolution',
        attrs={'integer': 32})
    resolution.integer = 32
    
    pivot1 = nw.new_node(Nodes.Vector,
        label='pivot1')
    pivot1.vector = (1.84, -0.28, 0.0)
    
    add = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: pivot1})
    
    reroute = nw.new_node(Nodes.Reroute,
        input_kwargs={'Input': add.outputs["Vector"]})
    
    quadratic_bezier = nw.new_node(Nodes.QuadraticBezier,
        input_kwargs={'Resolution': resolution, 'Start': (0.0, 0.0, 0.0), 'Middle': (1.2, -0.16, 0.0), 'End': reroute})
    
    pivot2 = nw.new_node(Nodes.Vector,
        label='pivot2')
    pivot2.vector = (3.98, -0.78, 0.0)
    
    quadratic_bezier_1 = nw.new_node(Nodes.QuadraticBezier,
        input_kwargs={'Resolution': resolution, 'Start': reroute, 'Middle': (3.98, -0.32, 0.0), 'End': pivot2})
    
    pivot3 = nw.new_node(Nodes.Vector,
        label='pivot3')
    pivot3.vector = (2.54, -1.14, 0.0)
    
    quadratic_bezier_2 = nw.new_node(Nodes.QuadraticBezier,
        input_kwargs={'Resolution': resolution, 'Start': pivot2, 'Middle': (4.0, -1.1, 0.0), 'End': pivot3})
    
    pivot4 = nw.new_node(Nodes.Vector,
        label='pivot4')
    pivot4.vector = (-0.06, -0.74, 0.0)
    
    quadratic_bezier_3 = nw.new_node(Nodes.QuadraticBezier,
        input_kwargs={'Resolution': resolution, 'Start': pivot3, 'Middle': (0.28, -1.34, 0.0), 'End': pivot4})
    
    pivot5 = nw.new_node(Nodes.Vector,
        label='pivot5')
    pivot5.vector = (0.0, -0.14, 0.0)
    
    bezier_segment = nw.new_node(Nodes.CurveBezierSegment,
        input_kwargs={'Resolution': resolution, 'Start': pivot4, 'Start Handle': (0.16, -0.44, 0.0), 'End Handle': (-0.24, -0.34, 0.0), 'End': pivot5})
    
    resample_curve = nw.new_node(Nodes.ResampleCurve,
        input_kwargs={'Curve': bezier_segment, 'Count': resolution})
    
    quadratic_bezier_4 = nw.new_node(Nodes.QuadraticBezier,
        input_kwargs={'Resolution': resolution, 'Start': pivot5, 'Middle': (-0.18, -0.04, 0.0), 'End': (0.0, 0.0, 0.0)})
    
    join_geometry = nw.new_node(Nodes.JoinGeometry,
        input_kwargs={'Geometry': [quadratic_bezier, quadratic_bezier_1, quadratic_bezier_2, quadratic_bezier_3, resample_curve, quadratic_bezier_4]})
    
    curve_to_mesh = nw.new_node(Nodes.CurveToMesh,
        input_kwargs={'Curve': join_geometry})
    
    merge_by_distance = nw.new_node(Nodes.MergeByDistance,
        input_kwargs={'Geometry': curve_to_mesh})
    
    mesh_to_curve = nw.new_node(Nodes.MeshToCurve,
        input_kwargs={'Mesh': merge_by_distance})
    
    fill_curve = nw.new_node(Nodes.FillCurve,
        input_kwargs={'Curve': mesh_to_curve})
    
    subdivide_mesh = nw.new_node(Nodes.SubdivideMesh,
        input_kwargs={'Mesh': fill_curve})
    
    curve_to_mesh_1 = nw.new_node(Nodes.CurveToMesh,
        input_kwargs={'Curve': quadratic_bezier_2})
    
    geometry_proximity = nw.new_node(Nodes.Proximity,
        input_kwargs={'Target': curve_to_mesh_1},
        attrs={'target_element': 'EDGES'})
    
    store_named_attribute = nw.new_node(Nodes.StoreNamedAttribute,
        input_kwargs={'Geometry': subdivide_mesh, 'Name': 'distance to edge', 'Value': geometry_proximity.outputs["Distance"]})
    
    spline_parameter = nw.new_node(Nodes.SplineParameter)
    
    capture_attribute = nw.new_node(Nodes.CaptureAttribute,
        input_kwargs={'Geometry': quadratic_bezier_1, 2: spline_parameter.outputs["Factor"]})
    
    curve_to_mesh_2 = nw.new_node(Nodes.CurveToMesh,
        input_kwargs={'Curve': capture_attribute.outputs["Geometry"]})
    
    less_than = nw.new_node(Nodes.Compare,
        input_kwargs={0: capture_attribute.outputs[2], 1: 0.65},
        attrs={'operation': 'LESS_THAN'})
    
    greater_than = nw.new_node(Nodes.Compare,
        input_kwargs={0: capture_attribute.outputs[2], 1: 0.84})
    
    op_or = nw.new_node(Nodes.BooleanMath,
        input_kwargs={0: less_than, 1: greater_than},
        attrs={'operation': 'OR'})
    
    delete_geometry = nw.new_node(Nodes.DeleteGeometry,
        input_kwargs={'Geometry': curve_to_mesh_2, 'Selection': op_or})
    
    geometry_proximity_1 = nw.new_node(Nodes.Proximity,
        input_kwargs={'Target': delete_geometry},
        attrs={'target_element': 'EDGES'})
    
    store_named_attribute_2 = nw.new_node(Nodes.StoreNamedAttribute,
        input_kwargs={'Geometry': store_named_attribute, 'Name': 'stripes coordinate', 'Value': geometry_proximity_1.outputs["Distance"]})
    
    position = nw.new_node(Nodes.InputPosition)
    
    store_named_attribute_1 = nw.new_node(Nodes.StoreNamedAttribute,
        input_kwargs={'Geometry': store_named_attribute_2, 'Name': 'pos', 'Value': position},
        attrs={'data_type': 'FLOAT_VECTOR'})
    
    set_material = nw.new_node(Nodes.SetMaterial,
        input_kwargs={'Geometry': store_named_attribute_1, 'Material': surface.shaderfunc_to_material(shader_wing_shader)})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': set_material})

def shader_wing_shader(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    attribute_2 = nw.new_node(Nodes.Attribute,
        attrs={'attribute_name': 'stripes coordinate'})
    
    map_range_1 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': attribute_2.outputs["Fac"], 1: 0.04, 2: 0.54})
    
    attribute_1 = nw.new_node(Nodes.Attribute,
        attrs={'attribute_name': 'pos'})
    
    vector_rotate = nw.new_node(Nodes.VectorRotate,
        input_kwargs={'Vector': attribute_1.outputs["Vector"], 'Angle': 0.1047})
    
    value = nw.new_node(Nodes.Value)
    value.outputs[0].default_value = 0.08
    
    group = nw.new_node(nodegroup_add_noise().name,
        input_kwargs={'Vector': vector_rotate, 'amount': value})
    
    voronoi_texture_2 = nw.new_node(Nodes.VoronoiTexture,
        input_kwargs={'Vector': group, 'Scale': 12.0, 'Randomness': 0.7},
        attrs={'voronoi_dimensions': '2D', 'feature': 'DISTANCE_TO_EDGE'})
    
    multiply = nw.new_node(Nodes.Math,
        input_kwargs={0: voronoi_texture_2.outputs["Distance"], 1: 2.34},
        attrs={'operation': 'MULTIPLY'})
    
    separate_xyz = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': group})
    
    voronoi_texture = nw.new_node(Nodes.VoronoiTexture,
        input_kwargs={'W': separate_xyz.outputs["Y"], 'Scale': 14.96, 'Randomness': 0.5},
        attrs={'voronoi_dimensions': '1D', 'feature': 'DISTANCE_TO_EDGE'})
    
    value_2 = nw.new_node(Nodes.Value)
    value_2.outputs[0].default_value = -0.18
    
    less_than = nw.new_node(Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Y"], 1: value_2},
        attrs={'operation': 'LESS_THAN'})
    
    maximum = nw.new_node(Nodes.Math,
        input_kwargs={0: voronoi_texture.outputs["Distance"], 1: less_than},
        attrs={'operation': 'MAXIMUM'})
    
    multiply_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: maximum, 1: 0.56},
        attrs={'operation': 'MULTIPLY'})
    
    vector_rotate_1 = nw.new_node(Nodes.VectorRotate,
        input_kwargs={'Vector': attribute_1.outputs["Vector"], 'Angle': 0.2485})
    
    value_1 = nw.new_node(Nodes.Value)
    value_1.outputs[0].default_value = 0.08
    
    group_1 = nw.new_node(nodegroup_add_noise().name,
        input_kwargs={'Vector': vector_rotate_1, 'amount': value_1})
    
    separate_xyz_1 = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': group_1})
    
    subtract = nw.new_node(Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Y"], 1: value_2},
        attrs={'operation': 'SUBTRACT'})
    
    multiply_2 = nw.new_node(Nodes.Math,
        input_kwargs={0: subtract, 1: -0.74},
        attrs={'operation': 'MULTIPLY'})
    
    power = nw.new_node(Nodes.Math,
        input_kwargs={0: multiply_2, 1: 2.22},
        attrs={'operation': 'POWER'})
    
    add = nw.new_node(Nodes.Math,
        input_kwargs={0: separate_xyz_1.outputs["Y"], 1: power})
    
    voronoi_texture_1 = nw.new_node(Nodes.VoronoiTexture,
        input_kwargs={'W': add, 'Scale': 10.02},
        attrs={'voronoi_dimensions': '1D', 'feature': 'DISTANCE_TO_EDGE'})
    
    greater_than = nw.new_node(Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Y"], 1: value_2},
        attrs={'operation': 'GREATER_THAN'})
    
    maximum_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: voronoi_texture_1.outputs["Distance"], 1: greater_than},
        attrs={'operation': 'MAXIMUM'})
    
    less_than_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: add, 1: -0.48},
        attrs={'operation': 'LESS_THAN'})
    
    maximum_2 = nw.new_node(Nodes.Math,
        input_kwargs={0: maximum_1, 1: less_than_1},
        attrs={'operation': 'MAXIMUM'})
    
    multiply_3 = nw.new_node(Nodes.Math,
        input_kwargs={0: maximum_2, 1: 3.0},
        attrs={'operation': 'MULTIPLY'})
    
    minimum = nw.new_node(Nodes.Math,
        input_kwargs={0: multiply_1, 1: multiply_3},
        attrs={'operation': 'MINIMUM'})
    
    minimum_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: multiply, 1: minimum},
        attrs={'operation': 'MINIMUM'})
    
    attribute = nw.new_node(Nodes.Attribute,
        attrs={'attribute_name': 'distance to edge'})
    
    map_range = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': attribute.outputs["Color"], 3: 0.1, 4: 0.0})
    
    maximum_3 = nw.new_node(Nodes.Math,
        input_kwargs={0: minimum_1, 1: map_range.outputs["Result"]},
        attrs={'operation': 'MAXIMUM'})
    
    minimum_2 = nw.new_node(Nodes.Math,
        input_kwargs={0: map_range_1.outputs["Result"], 1: maximum_3},
        attrs={'operation': 'MINIMUM'})
    
    colorramp = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': minimum_2})
    colorramp.color_ramp.elements[0].position = 0.0
    colorramp.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    colorramp.color_ramp.elements[1].position = 0.1136
    colorramp.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
    
    reroute = nw.new_node(Nodes.Reroute,
        input_kwargs={'Input': colorramp.outputs["Color"]})
    
    transparent_bsdf_1 = nw.new_node(Nodes.TransparentBSDF,
        input_kwargs={'Color': reroute})
    
    principled_bsdf = nw.new_node(Nodes.PrincipledBSDF,
        input_kwargs={'Base Color': reroute})
    
    mix_shader = nw.new_node(Nodes.MixShader,
        input_kwargs={'Fac': 0.1, 1: transparent_bsdf_1, 2: principled_bsdf})
    
    material_output = nw.new_node(Nodes.MaterialOutput,
        input_kwargs={'Surface': mix_shader})