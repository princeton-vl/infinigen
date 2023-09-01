# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Yiming Zuo


from random import randint
import bpy
import mathutils
from numpy.random import uniform, normal
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.nodes import node_utils
from infinigen.core.util.color import color_category
from infinigen.core import surface

from infinigen.core.util.math import FixedSeed
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.util import blender as butil
from infinigen.assets.utils.tag import tag_object, tag_nodegroup

######## code for creating pine needles ########

def shader_needle(nw):
    # Code generated using version 2.3.1 of the node_transpiler

    velvet_bsdf = nw.new_node('ShaderNodeBsdfVelvet',
        input_kwargs={'Color': (0.016, 0.2241, 0.0252, 1.0)})
    
    glossy_bsdf = nw.new_node('ShaderNodeBsdfGlossy',
        input_kwargs={'Color': (0.5771, 0.8, 0.5713, 1.0), 'Roughness': 0.4})
    
    mix_shader = nw.new_node(Nodes.MixShader,
        input_kwargs={'Fac': 0.3, 1: velvet_bsdf, 2: glossy_bsdf})
    
    translucent_bsdf = nw.new_node(Nodes.TranslucentBSDF,
        input_kwargs={'Color': (0.0116, 0.4409, 0.0262, 1.0)})
    
    mix_shader_1 = nw.new_node(Nodes.MixShader,
        input_kwargs={'Fac': 0.1, 1: mix_shader, 2: translucent_bsdf})
    
    material_output = nw.new_node(Nodes.MaterialOutput,
        input_kwargs={'Surface': mix_shader_1})

def geometry_needle(nw):
    # Code generated using version 2.3.1 of the node_transpiler

    cone = nw.new_node('GeometryNodeMeshCone',
        input_kwargs={'Vertices': 4, 'Radius Top': 0.01, 'Radius Bottom': 0.02, 'Depth': 1.0})
    
    set_material = nw.new_node(Nodes.SetMaterial,
        input_kwargs={'Geometry': cone.outputs["Mesh"], 'Material': surface.shaderfunc_to_material(shader_needle)})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': set_material})

def apply_needle(obj, selection=None, **kwargs):
    surface.add_geomod(obj, geometry_needle, selection=selection, attributes=[])

def make_needle(name='Needle'):
    if bpy.context.scene.objects.get(name):
        return bpy.context.scene.objects.get(name)
    
    else:
        bpy.ops.mesh.primitive_plane_add(
                size=2, enter_editmode=False, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
        needle = bpy.context.active_object
        needle.name = name
        apply_needle(needle)

        bpy.ops.object.convert(target='MESH')

        return needle

######## code for creating pine needles ########

######## code for creating pine twigs ########

@node_utils.to_nodegroup('nodegroup_instance_needle', singleton=True, type='GeometryNodeTree')
def nodegroup_instance_needle(nw):
    # Code generated using version 2.3.2 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketGeometry', 'Curve', None),
            ('NodeSocketFloatFactor', 'Needle Density', 0.9),
            ('NodeSocketInt', 'Seed', 0),
            ('NodeSocketGeometry', 'Instance', None),
            ('NodeSocketFloat', 'X Angle Mean', 0.5),
            ('NodeSocketFloat', 'X Angle Range', 0.0)])
    
    spline_parameter_1 = nw.new_node('GeometryNodeSplineParameter')
    
    greater_than = nw.new_node(Nodes.Compare,
        input_kwargs={0: spline_parameter_1.outputs["Factor"], 1: 0.1})
    
    random_value_3 = nw.new_node(Nodes.RandomValue,
        input_kwargs={'Probability': group_input.outputs["Needle Density"], 'Seed': group_input.outputs["Seed"]},
        attrs={'data_type': 'BOOLEAN'})
    
    op_and = nw.new_node(Nodes.BooleanMath,
        input_kwargs={0: greater_than, 1: random_value_3.outputs[3]})
    
    curve_tangent = nw.new_node('GeometryNodeInputTangent')
    
    align_euler_to_vector = nw.new_node(Nodes.AlignEulerToVector,
        input_kwargs={'Vector': curve_tangent},
        attrs={'axis': 'Y'}
        )
    
    random_value = nw.new_node(Nodes.RandomValue,
        input_kwargs={2: 0.6, 'Seed': group_input.outputs["Seed"]})
    
    combine_xyz = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'X': 0.8, 'Y': 0.8, 'Z': random_value.outputs[1]})
    
    value_1 = nw.new_node(Nodes.Value)
    value_1.outputs[0].default_value = 0.3
    
    multiply = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: combine_xyz, 1: value_1},
        attrs={'operation': 'MULTIPLY'})
    
    instance_on_points = nw.new_node(Nodes.InstanceOnPoints,
        input_kwargs={'Points': group_input.outputs["Curve"], 'Selection': op_and, 'Instance': group_input.outputs["Instance"], 'Rotation': align_euler_to_vector, 'Scale': multiply.outputs["Vector"]})
    
    add = nw.new_node(Nodes.Math,
        input_kwargs={0: group_input.outputs["X Angle Mean"], 1: group_input.outputs["X Angle Range"]})
    
    subtract = nw.new_node(Nodes.Math,
        input_kwargs={0: group_input.outputs["X Angle Mean"], 1: group_input.outputs["X Angle Range"]},
        attrs={'operation': 'SUBTRACT'})
    
    random_value_2 = nw.new_node(Nodes.RandomValue,
        input_kwargs={2: add, 3: subtract, 'Seed': group_input.outputs["Seed"]})
    
    radians = nw.new_node(Nodes.Math,
        input_kwargs={0: random_value_2.outputs[1]},
        attrs={'operation': 'RADIANS'})
    
    random_value_1 = nw.new_node(Nodes.RandomValue,
        input_kwargs={3: 360.0, 'Seed': group_input.outputs["Seed"]})
    
    radians_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: random_value_1.outputs[1]},
        attrs={'operation': 'RADIANS'})
    
    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'X': radians, 'Y': radians_1})
    
    rotate_instances = nw.new_node('GeometryNodeRotateInstances',
        input_kwargs={'Instances': instance_on_points, 'Rotation': combine_xyz_1})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Instances': rotate_instances})

@node_utils.to_nodegroup('nodegroup_needle5', singleton=True, type='GeometryNodeTree')
def nodegroup_needle5(nw):
    # Code generated using version 2.3.2 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketGeometry', 'Curve', None),
            ('NodeSocketGeometry', 'Instance', None),
            ('NodeSocketFloat', 'X Angle Mean', 0.5),
            ('NodeSocketFloat', 'X Angle Range', 0.0),
            ('NodeSocketFloatFactor', 'Needle Density', 0.9),
            ('NodeSocketInt', 'Seed', 0)])
    
    instanceneedle = nw.new_node(nodegroup_instance_needle().name,
        input_kwargs={'Curve': group_input.outputs["Curve"], 'Needle Density': group_input.outputs["Needle Density"], 'Seed': group_input.outputs["Seed"], 'Instance': group_input.outputs["Instance"], 'X Angle Mean': group_input.outputs["X Angle Mean"], 'X Angle Range': group_input.outputs["X Angle Range"]})
    
    add = nw.new_node(Nodes.Math,
        input_kwargs={0: group_input.outputs["Seed"], 1: 1.0})
    
    instanceneedle_1 = nw.new_node(nodegroup_instance_needle().name,
        input_kwargs={'Curve': group_input.outputs["Curve"], 'Needle Density': group_input.outputs["Needle Density"], 'Seed': add, 'X Angle Mean': group_input.outputs["X Angle Mean"], 'X Angle Range': group_input.outputs["X Angle Range"]})
    
    add_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: group_input.outputs["Seed"], 1: 2.0})
    
    instanceneedle_2 = nw.new_node(nodegroup_instance_needle().name,
        input_kwargs={'Curve': group_input.outputs["Curve"], 'Needle Density': group_input.outputs["Needle Density"], 'Seed': add_1, 'Instance': group_input.outputs["Instance"], 'X Angle Mean': group_input.outputs["X Angle Mean"], 'X Angle Range': group_input.outputs["X Angle Range"]})
    
    add_2 = nw.new_node(Nodes.Math,
        input_kwargs={0: group_input.outputs["Seed"], 1: 3.0})
    
    instanceneedle_3 = nw.new_node(nodegroup_instance_needle().name,
        input_kwargs={'Curve': group_input.outputs["Curve"], 'Needle Density': group_input.outputs["Needle Density"], 'Seed': add_2, 'Instance': group_input.outputs["Instance"], 'X Angle Mean': group_input.outputs["X Angle Mean"], 'X Angle Range': group_input.outputs["X Angle Range"]})
    
    add_3 = nw.new_node(Nodes.Math,
        input_kwargs={0: group_input.outputs["Seed"], 1: 4.0})
    
    instanceneedle_4 = nw.new_node(nodegroup_instance_needle().name,
        input_kwargs={'Curve': group_input.outputs["Curve"], 'Needle Density': group_input.outputs["Needle Density"], 'Seed': add_3, 'Instance': group_input.outputs["Instance"], 'X Angle Mean': group_input.outputs["X Angle Mean"], 'X Angle Range': group_input.outputs["X Angle Range"]})
    
    join_geometry = nw.new_node(Nodes.JoinGeometry,
        input_kwargs={'Geometry': [instanceneedle, instanceneedle_1, instanceneedle_2, instanceneedle_3, instanceneedle_4]})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Instances': join_geometry})

def shader_twig(nw):
    # Code generated using version 2.3.2 of the node_transpiler

    principled_bsdf = nw.new_node(Nodes.PrincipledBSDF,
        input_kwargs={'Base Color': (0.08, 0.0329, 0.0414, 1.0), 'Specular': 0.0527, 'Roughness': 0.4491})
    
    material_output = nw.new_node(Nodes.MaterialOutput,
        input_kwargs={'Surface': principled_bsdf})

@node_utils.to_nodegroup('nodegroup_pine_twig', singleton=False, type='GeometryNodeTree')
def nodegroup_pine_twig(nw):
    # Code generated using version 2.3.2 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketIntUnsigned', 'Resolution', 20),
            ('NodeSocketFloat', 'Middle Y', 0.0),
            ('NodeSocketFloat', 'Middle Z', 0.0),
            ('NodeSocketFloatFactor', 'Needle Density', 0.9),
            ('NodeSocketGeometry', 'Instance', None),
            ('NodeSocketFloat', 'X Angle Mean', 0.5),
            ('NodeSocketFloat', 'X Angle Range', 0.0),
            ('NodeSocketInt', 'Seed', 0)])
    
    divide = nw.new_node(Nodes.Math,
        input_kwargs={0: group_input.outputs["Resolution"], 1: 30.0},
        attrs={'operation': 'DIVIDE'})
    
    divide_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: divide, 1: 2.0},
        attrs={'operation': 'DIVIDE'})
    
    combine_xyz = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'X': group_input.outputs["Middle Y"], 'Y': divide_1, 'Z': group_input.outputs["Middle Z"]})
    
    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'Y': divide})
    
    quadratic_bezier = nw.new_node(Nodes.QuadraticBezier,
        input_kwargs={'Resolution': group_input.outputs["Resolution"], 'Start': (0.0, 0.0, 0.0), 'Middle': combine_xyz, 'End': combine_xyz_1})
    
    noise_texture = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'W': -1.7},
        attrs={'noise_dimensions': '4D'})
    
    value = nw.new_node(Nodes.Value)
    value.outputs[0].default_value = 0.5
    
    subtract = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: noise_texture.outputs["Color"], 1: value},
        attrs={'operation': 'SUBTRACT'})
    
    spline_parameter = nw.new_node('GeometryNodeSplineParameter')
    
    multiply = nw.new_node(Nodes.Math,
        input_kwargs={0: spline_parameter.outputs["Factor"], 1: 0.1},
        attrs={'operation': 'MULTIPLY'})
    
    multiply_1 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: subtract.outputs["Vector"], 1: multiply},
        attrs={'operation': 'MULTIPLY'})
    
    set_position = nw.new_node(Nodes.SetPosition,
        input_kwargs={'Geometry': quadratic_bezier, 'Offset': multiply_1.outputs["Vector"]})
    
    map_range = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': spline_parameter.outputs["Factor"], 3: 1.0, 4: 0.0})
    
    power = nw.new_node(Nodes.Math,
        input_kwargs={0: 2.0, 1: map_range.outputs["Result"]},
        attrs={'operation': 'POWER'})
    
    set_curve_radius = nw.new_node(Nodes.SetCurveRadius,
        input_kwargs={'Curve': set_position, 'Radius': power})
    
    curve_circle = nw.new_node(Nodes.CurveCircle,
        input_kwargs={'Resolution': 16, 'Radius': 0.01})
    
    curve_to_mesh = nw.new_node(Nodes.CurveToMesh,
        input_kwargs={'Curve': set_curve_radius, 'Profile Curve': curve_circle.outputs["Curve"], 'Fill Caps': True})
    
    set_material = nw.new_node(Nodes.SetMaterial,
        input_kwargs={'Geometry': curve_to_mesh, 'Material': surface.shaderfunc_to_material(shader_twig)})
    
    needle5 = nw.new_node(nodegroup_needle5().name,
        input_kwargs={'Curve': set_position, 'Instance': group_input.outputs["Instance"], 'X Angle Mean': group_input.outputs["X Angle Mean"], 'X Angle Range': group_input.outputs["X Angle Range"], 'Needle Density': group_input.outputs["Needle Density"], 'Seed': group_input.outputs["Seed"]})
    
    join_geometry = nw.new_node(Nodes.JoinGeometry,
        input_kwargs={'Geometry': [set_material, needle5]})
    
    realize_instances = nw.new_node(Nodes.RealizeInstances,
        input_kwargs={'Geometry': join_geometry})
    
    set_shade_smooth = nw.new_node(Nodes.SetShadeSmooth,
        input_kwargs={'Geometry': realize_instances, 'Shade Smooth': False})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': set_shade_smooth})

def shader_twig(nw):
    # Code generated using version 2.3.2 of the node_transpiler

    principled_bsdf = nw.new_node(Nodes.PrincipledBSDF,
        input_kwargs={'Base Color': (0.08, 0.0329, 0.0414, 1.0), 'Specular': 0.0527, 'Roughness': 0.4491})
    
    material_output = nw.new_node(Nodes.MaterialOutput,
        input_kwargs={'Surface': principled_bsdf})

def geometry_node_pine_twig(nw, needle_name='Needle', length=30, middle_y=0.0, middle_z=0.0, seed=0, x_angle_mean=-50.0, x_angle_range=10.0):
    # Code generated using version 2.3.2 of the node_transpiler

    object_info = nw.new_node(Nodes.ObjectInfo,
        input_kwargs={'Object': bpy.data.objects[needle_name]})
    
    pine_needle = nw.new_node(nodegroup_pine_twig().name,
        input_kwargs={'Resolution': length, 'Middle Y': middle_y, 'Middle Z': middle_z, 'Instance': object_info.outputs["Geometry"], 
        'X Angle Mean': x_angle_mean, 'X Angle Range': x_angle_range, 'Seed': seed})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': pine_needle})

def apply_twig(obj, selection=None, **kwargs):
    surface.add_geomod(obj, geometry_node_pine_twig, selection=selection, attributes=[], input_kwargs=kwargs)
    surface.add_material(obj, shader_twig, selection=selection)

def make_pine_twig(**kwargs):
    bpy.ops.mesh.primitive_plane_add(
            size=2, enter_editmode=False, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
    twig = bpy.context.active_object
    twig.name = "Twig"
    apply_twig(twig, **kwargs)

    # bpy.ops.object.convert(target='MESH')

    return twig

class LeafFactoryPine(AssetFactory):
    
    scale = 0.7

    def __init__(self, factory_seed, season='autumn', coarse=False):
        super(LeafFactoryPine, self).__init__(factory_seed, coarse=coarse)
        self.needle = make_needle('Needle')
        self.needle.hide_viewport = True
        self.needle.hide_render = True

    def create_asset(self, **params):

        # with FixedSeed(self.factory_seed):
        seed = randint(0, 1e6)
        middle_y = normal(0.0, 0.1)
        middle_z = normal(0.0, 0.1)
        length = randint(25, 35)
        x_angle_mean = uniform(-40, -60)

        obj = make_pine_twig(
            needle_name='Needle', 
            length=length, 
            middle_y=middle_y, 
            middle_z=middle_z, 
            seed=seed, 
            x_angle_mean=x_angle_mean, 
            x_angle_range=10.0,
            )

        bpy.ops.object.convert(target='MESH')

        obj = bpy.context.object
        obj.scale *= normal(1, 0.05) * self.scale
        butil.apply_transform(obj)
        tag_object(obj, 'leaf_pine')

        return obj




