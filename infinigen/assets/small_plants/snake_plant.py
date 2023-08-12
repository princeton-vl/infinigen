# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Beining Han
# Acknowledgements: This file draws inspiration from https://blenderartists.org/t/extrude-face-along-curve-with-geometry-nodes/1432653/3

import bpy
import mathutils
from numpy.random import uniform, normal, randint
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.nodes import node_utils
from infinigen.core import surface
from infinigen.assets.materials import snake_plant
from infinigen.core.placement.factory import AssetFactory
import numpy as np
from infinigen.core.util import blender as butil
from infinigen.assets.utils.tag import tag_object, tag_nodegroup

@node_utils.to_nodegroup('nodegroup_pedal_thickness', singleton=False, type='GeometryNodeTree')
def nodegroup_pedal_thickness(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
                              expose_input=[('NodeSocketFloat', 'Value', 1.0)])

    map_range_3 = nw.new_node(Nodes.MapRange,
                              input_kwargs={'Value':group_input.outputs["Value"], 3: 0.2, 4: 0.04})

    thickness = nw.new_node(Nodes.Value)
    thickness.outputs[0].default_value = uniform(0.1, 0.35)

    multiply = nw.new_node(Nodes.Math,
                           input_kwargs={0: map_range_3.outputs["Result"], 1: thickness},
                           attrs={'operation': 'MULTIPLY'})

    group_output = nw.new_node(Nodes.GroupOutput,
                               input_kwargs={'Value': multiply})


@node_utils.to_nodegroup('nodegroup_z_pedal_rotation', singleton=False, type='GeometryNodeTree')
def nodegroup_z_pedal_rotation(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    position_1 = nw.new_node(Nodes.InputPosition)

    group_input = nw.new_node(Nodes.GroupInput,
                              expose_input=[('NodeSocketFloat', 'Value', 1.0)])

    
    float_curve = nw.new_node(Nodes.FloatCurve,
                              input_kwargs={'Value': group_input.outputs["Value"]})
    node_utils.assign_curve(float_curve.mapping.curves[0],
                            [(0.0, 0.0), (0.25, 0.25 + uniform(-0.1, 0.1)),
                             (0.50, 0.5 + uniform(-0.15, 0.15)), 
                             (0.75, 0.5 + uniform(0.25, 0.25)),
                             (1.0, 1.0)])

    multiply = nw.new_node(Nodes.Math,
                           input_kwargs={0: float_curve, 1: uniform(0.8, 2.0)},
                           attrs={'operation': 'MULTIPLY'})

    vector_rotate_1 = nw.new_node(Nodes.VectorRotate,
                                  input_kwargs={'Vector': position_1, 'Angle': multiply},
                                  attrs={'rotation_type': 'Z_AXIS'})

    group_output = nw.new_node(Nodes.GroupOutput,
                               input_kwargs={'Vector': vector_rotate_1})


@node_utils.to_nodegroup('nodegroup_x_pedal_rotation', singleton=False, type='GeometryNodeTree')
def nodegroup_x_pedal_rotation(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    position_1 = nw.new_node(Nodes.InputPosition)

    spline_parameter_1 = nw.new_node(Nodes.SplineParameter)

    multiply = nw.new_node(Nodes.Math,
                           input_kwargs={0: 0.5, 1: spline_parameter_1.outputs["Factor"]},
                           attrs={'operation': 'MULTIPLY'})

    vector_rotate = nw.new_node(Nodes.VectorRotate,
                                input_kwargs={'Vector': position_1, 'Angle': multiply},
                                attrs={'rotation_type': 'X_AXIS'})

    group_output = nw.new_node(Nodes.GroupOutput,
                               input_kwargs={'Vector': vector_rotate})


@node_utils.to_nodegroup('nodegroup_setup', singleton=False, type='GeometryNodeTree')
def nodegroup_setup(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    quadratic_bezier = nw.new_node(Nodes.QuadraticBezier,
                                   input_kwargs={'Resolution': 25, 'Start': (0.0, 0.0, 0.0), 'Middle': (0.0, 0.0, 1.0),
                                                 'End': (uniform(-0.2, 0.2), uniform(0.2, 0.2), 2.0)})

    x_pedal_rotation = nw.new_node(nodegroup_x_pedal_rotation().name)

    set_position =  nw.new_node(Nodes.SetPosition, input_kwargs={'Geometry': quadratic_bezier, 'Offset': x_pedal_rotation})

    spline_parameter = nw.new_node(Nodes.SplineParameter)

    capture_attribute_1 = nw.new_node(Nodes.CaptureAttribute,
                                      input_kwargs={'Geometry': set_position,
                                                    2: spline_parameter.outputs["Factor"]})

    group_output = nw.new_node(Nodes.GroupOutput,
                               input_kwargs={'Spline': capture_attribute_1.outputs[2],
                                             'Geometry': capture_attribute_1.outputs["Geometry"]})


@node_utils.to_nodegroup('nodegroup_edge_extrusion', singleton=False, type='GeometryNodeTree')
def nodegroup_edge_extrusion(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
                              expose_input=[('NodeSocketFloat', 'Value', 1.0),
                                            ('NodeSocketGeometry', 'Geometry', None)])

    init_width = uniform(0.15, 0.3)

    normal = nw.new_node(Nodes.InputNormal)

    capture_attribute = nw.new_node(Nodes.CaptureAttribute,
                                    input_kwargs={'Geometry': group_input.outputs['Geometry'], 1: normal},
                                    attrs={'data_type': 'FLOAT_VECTOR'})

    float_curve = nw.new_node(Nodes.FloatCurve,
                              input_kwargs={'Value': group_input.outputs["Value"]})
    node_utils.assign_curve(float_curve.mapping.curves[0],
                            [(0.0, init_width), (0.25, init_width + uniform(0.0, 0.1)),
                             (0.50, init_width + uniform(0.02, 0.18)), (0.75, init_width + uniform(0.02, 0.1)),
                             (1.0, 0.0)])

    combine_xyz = nw.new_node(Nodes.CombineXYZ,
                              input_kwargs={'X': float_curve})

    set_position_1 = nw.new_node(Nodes.SetPosition,
                                 input_kwargs={'Geometry': capture_attribute.outputs["Geometry"], 
                                               'Offset': combine_xyz})

    curve_to_mesh = nw.new_node(Nodes.CurveToMesh,
                                input_kwargs={'Curve': set_position_1})

    extrude_mesh = nw.new_node(Nodes.ExtrudeMesh,
                               input_kwargs={'Mesh': curve_to_mesh, 'Offset': capture_attribute.outputs["Attribute"],
                                             'Offset Scale': float_curve},
                               attrs={'mode': 'EDGES'})

    group_output = nw.new_node(Nodes.GroupOutput,
                               input_kwargs={'Mesh': extrude_mesh.outputs["Mesh"]})


@node_utils.to_nodegroup('nodegroup_face_extrusion', singleton=False, type='GeometryNodeTree')
def nodegroup_face_extrusion(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
                              expose_input=[('NodeSocketGeometry', 'Geometry', None),
                                            ('NodeSocketFloat', 'Value', 1.0)])

    z_pedal_rotation = nw.new_node(nodegroup_z_pedal_rotation().name,
                                   input_kwargs={'Value': group_input.outputs["Value"]})

    set_position_2 = nw.new_node(Nodes.SetPosition,
                                 input_kwargs={'Geometry': group_input.outputs["Geometry"], 'Offset': z_pedal_rotation})

    pedal_thickness = nw.new_node(nodegroup_pedal_thickness().name,
                                  input_kwargs={'Value': group_input.outputs["Value"]})

    extrude_mesh_2 = nw.new_node(Nodes.ExtrudeMesh,
                                 input_kwargs={'Mesh': set_position_2, 'Offset Scale': pedal_thickness,
                                               'Individual': False})

    group_output = nw.new_node(Nodes.GroupOutput,
                               input_kwargs={'Geometry': extrude_mesh_2})


@node_utils.to_nodegroup('nodegroup_single_pedal', singleton=False, type='GeometryNodeTree')
def nodegroup_single_pedal_nodes(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    setup = nw.new_node(nodegroup_setup().name)

    edge_extrusion = nw.new_node(nodegroup_edge_extrusion().name,
                                 input_kwargs={'Value': setup.outputs["Spline"],
                                               'Geometry': setup.outputs["Geometry"]})

    face_extrusion = nw.new_node(nodegroup_face_extrusion().name,
                                 input_kwargs={'Geometry': edge_extrusion, 'Value': setup.outputs["Spline"]})

    subdivision_surface = nw.new_node(Nodes.SubdivisionSurface,
                                      input_kwargs={'Mesh': face_extrusion, 'Level': 2})

    set_shade_smooth = nw.new_node(Nodes.SetShadeSmooth,
                                   input_kwargs={'Geometry': subdivision_surface})

    group_output = nw.new_node(Nodes.GroupOutput, input_kwargs={'Geometry': set_shade_smooth})


def check_vicinity(param, pedal_params):
    for p in pedal_params:
        r1 = max(param[0] * np.sin(param[1]), 0.2)
        r2 = max(p[0] * np.sin(p[1]), 0.2)
        dist = np.linalg.norm([param[2] - p[2], param[3] - p[3]])
        if r1 + r2 > dist:
            return True
    return False


def geometry_snake_plant_nodes(nw: NodeWrangler, **kwargs):
    num_pedals = kwargs['num_pedals']
    pedals = []
    pedal_params = []
    c = 0
    while c < 50 and len(pedal_params) < num_pedals:
        c += 1
        scale = uniform(0.7, 1.0)
        x_rotation = normal(0, 0.15)
        x, y = uniform(-0.7, 0.7), uniform(-0.7, 0.7)
        param = (scale, x_rotation, x, y)
        if check_vicinity(param, pedal_params):
            continue
        else:
            pedal_params.append(param)

    for param in pedal_params:
        scale = param[0]
        z_rotation = uniform(0, 6.28)
        x_rotation = param[1]
        z2_rotation = uniform(0, 6.28)
        x, y = param[2], param[3]
        pedal = nw.new_node(nodegroup_single_pedal_nodes().name)
        s_transform = nw.new_node(Nodes.Transform,
                                  input_kwargs={'Geometry': pedal, 'Scale': (scale, scale, scale),
                                                'Rotation': (0., 0., z_rotation)})
        x_transform = nw.new_node(Nodes.Transform,
                                  input_kwargs={'Geometry': s_transform, 'Rotation': (x_rotation, 0., 0.)})
        z_transform = nw.new_node(Nodes.Transform,
                                  input_kwargs={'Geometry': x_transform, 'Rotation': (0., 0., z2_rotation),
                                                'Translation': (x, y, 0)})
        pedals.append(z_transform)
    pedals = nw.new_node(Nodes.JoinGeometry,  input_kwargs={'Geometry': pedals})

    set_material = nw.new_node(Nodes.SetMaterial,
                               input_kwargs={'Geometry': pedals,
                                             'Material': surface.shaderfunc_to_material(snake_plant.shader_snake_plant)})

    group_output = nw.new_node(Nodes.GroupOutput,
                               input_kwargs={'Geometry': set_material})


class SnakePlantFactory(AssetFactory):
    def __init__(self, factory_seed, coarse=False):
        super(SnakePlantFactory, self).__init__(factory_seed, coarse=coarse)

    def create_asset(self, **params):
        bpy.ops.mesh.primitive_plane_add(size=1, enter_editmode=False, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
        obj = bpy.context.active_object

        pedal_num = randint(4, 8)
        params["num_pedals"] = pedal_num

        surface.add_geomod(obj, geometry_snake_plant_nodes, apply=True, input_kwargs=params)

        # convert to appropriate units - TODO replace this
        butil.apply_modifiers(obj)
        obj.scale = (0.2, 0.2, 0.2)
        butil.apply_transform(obj, scale=True)        

        tag_object(obj, 'snake_plant')
        return obj


if __name__ == '__main__':
    grass = SnakePlantFactory(0)
    obj = grass.create_asset()