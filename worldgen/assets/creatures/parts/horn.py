import bpy 

from platform import node
import numpy as np
from numpy.random import normal as N, uniform as U

from assets.creatures.creature import PartFactory
from assets.creatures.genome import Joint, IKParams
from assets.creatures.util.part_util import nodegroup_to_part

from nodes.node_wrangler import Nodes, NodeWrangler
from nodes import node_utils
from util import blender as butil

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
            ('NodeSocketFloat', 'thickness', 4.0),
            ('NodeSocketFloat', 'density_of_ridge', 0.0),
            ('NodeSocketFloat', 'depth_of_ridge', 0.2),
    
    multiply = nw.new_node(Nodes.Math,
        attrs={'operation': 'MULTIPLY'})
    
    add = nw.new_node(Nodes.Math,
    
    
    divide_1 = nw.new_node(Nodes.Math,
        attrs={'operation': 'DIVIDE'})
    
    divide_2 = nw.new_node(Nodes.Math,
        input_kwargs={0: divide_1, 1: 3.1415},
        attrs={'operation': 'DIVIDE'})
    
    spiral = nw.new_node('GeometryNodeCurveSpiral',
    
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
        attrs={'operation': 'MULTIPLY'})
    
    combine_xyz = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'X': multiply_1})
    
    set_position = nw.new_node(Nodes.SetPosition,
        input_kwargs={'Geometry': curve_to_mesh, 'Offset': combine_xyz})
    
        input_kwargs={'Geometry': set_position, 'Rotation': (-0.8, 0.0, 2.6)})

    group_output = nw.new_node(Nodes.GroupOutput,


class Horn(PartFactory):
    tags = ['head_detail', 'rigid']


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

