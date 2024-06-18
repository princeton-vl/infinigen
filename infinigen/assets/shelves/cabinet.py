# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Beining Han

from numpy.random import uniform, normal, randint
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.nodes import node_utils
from infinigen.core import surface
from infinigen.core.placement.factory import AssetFactory
import numpy as np
from infinigen.core.util import blender as butil

import bpy
from infinigen.assets.shelves.utils import nodegroup_tagged_cube, blender_rotate
from infinigen.assets.shelves.large_shelf import LargeShelfBaseFactory, LargeShelfFactory, LargeShelfIkeaFactory


@node_utils.to_nodegroup('nodegroup_node_group', singleton=False, type='GeometryNodeTree')
def nodegroup_node_group(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler

    cube = nw.new_node(Nodes.MeshCube, input_kwargs={'Size': (0.0120, 0.00060, 0.0400)})



    cube_1 = nw.new_node(Nodes.MeshCube, input_kwargs={'Size': (0.0200, 0.0006, 0.0120)})

    transform_1 = nw.new_node(Nodes.Transform,
                              input_kwargs={'Geometry': cube_1, 'Translation': (0.0080, 0.0000, 0.0000)})

    join_geometry_1 = nw.new_node(Nodes.JoinGeometry, input_kwargs={'Geometry': [cube, transform, transform_1]})


    multiply = nw.new_node(Nodes.Math, input_kwargs={0: group_input.outputs["door_width"]},
                           attrs={'operation': 'MULTIPLY'})

    subtract = nw.new_node(Nodes.Math, input_kwargs={0: multiply, 1: 0.0181}, attrs={'operation': 'SUBTRACT'})



    group_output = nw.new_node(Nodes.GroupOutput, input_kwargs={'Geometry': transform_2},
                               attrs={'is_active_output': True})


@node_utils.to_nodegroup('nodegroup_knob_handle', singleton=False, type='GeometryNodeTree')
def nodegroup_knob_handle(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler


    add = nw.new_node(Nodes.Math,

    add_1 = nw.new_node(Nodes.Math, input_kwargs={0: add, 1: group_input.outputs["length"]})

    cylinder = nw.new_node('GeometryNodeMeshCylinder',


    multiply = nw.new_node(Nodes.Math, input_kwargs={0: subtract, 1: -0.5000}, attrs={'operation': 'MULTIPLY'})

    add_2 = nw.new_node(Nodes.Math, input_kwargs={0: multiply, 1: -0.005})

    multiply_1 = nw.new_node(Nodes.Math, input_kwargs={0: add_1}, attrs={'operation': 'MULTIPLY'})



    group_output = nw.new_node(Nodes.GroupOutput, input_kwargs={'Geometry': transform_6},
                               attrs={'is_active_output': True})


@node_utils.to_nodegroup('nodegroup_mid_board', singleton=False, type='GeometryNodeTree')
def nodegroup_mid_board(nw: NodeWrangler, **kwargs):
    # Code generated using version 2.6.4 of the node_transpiler


    add = nw.new_node(Nodes.Math, input_kwargs={0: group_input.outputs["width"], 1: -0.0001})

    add_1 = nw.new_node(Nodes.Math, input_kwargs={0: group_input.outputs["thickness"], 1: 0.0000})


    multiply_k = nw.new_node(Nodes.Math, input_kwargs={0: add_1, 1: 0.5000}, attrs={'operation': 'MULTIPLY'})

    add_k = nw.new_node(Nodes.Math, input_kwargs={0: multiply_k, 1: 0.004})

    add_2 = nw.new_node(Nodes.Math, input_kwargs={0: multiply, 1: -0.0001})

    combine_xyz_3 = nw.new_node(Nodes.CombineXYZ, input_kwargs={'X': add, 'Y': add_1, 'Z': add_2})

    cube = nw.new_node(Nodes.MeshCube,
                       input_kwargs={'Size': combine_xyz_3, 'Vertices X': 5, 'Vertices Y': 5, 'Vertices Z': 5})

    multiply_1 = nw.new_node(Nodes.Math, input_kwargs={0: multiply}, attrs={'operation': 'MULTIPLY'})

    combine_xyz_4 = nw.new_node(Nodes.CombineXYZ, input_kwargs={'Y': add_k, 'Z': multiply_1})

    transform_4 = nw.new_node(Nodes.Transform, input_kwargs={'Geometry': cube, 'Translation': combine_xyz_4})

    set_material = nw.new_node(Nodes.SetMaterial,

    combine_xyz_7 = nw.new_node(Nodes.CombineXYZ, input_kwargs={'X': add, 'Y': add_1, 'Z': add_2})

    cube_1 = nw.new_node(Nodes.MeshCube,

    multiply_2 = nw.new_node(Nodes.Math, input_kwargs={0: multiply, 1: 1.5000}, attrs={'operation': 'MULTIPLY'})

    combine_xyz_8 = nw.new_node(Nodes.CombineXYZ, input_kwargs={'Y': add_k, 'Z': multiply_2})

    transform_7 = nw.new_node(Nodes.Transform, input_kwargs={'Geometry': cube_1, 'Translation': combine_xyz_8})

    set_material_1 = nw.new_node(Nodes.SetMaterial,

    join_geometry_1 = nw.new_node(Nodes.JoinGeometry, input_kwargs={'Geometry': [set_material, set_material_1]})

    realize_instances = nw.new_node(Nodes.RealizeInstances, input_kwargs={'Geometry': join_geometry_1})

    group_output = nw.new_node(Nodes.GroupOutput,
                               input_kwargs={'Geometry': realize_instances, 'mid_height': multiply},
                               attrs={'is_active_output': True})


@node_utils.to_nodegroup('nodegroup_mid_board_001', singleton=False, type='GeometryNodeTree')
def nodegroup_mid_board_001(nw: NodeWrangler, **kwargs):
    # Code generated using version 2.6.4 of the node_transpiler


    add = nw.new_node(Nodes.Math, input_kwargs={0: group_input.outputs["width"], 1: -0.0001})

    add_1 = nw.new_node(Nodes.Math, input_kwargs={0: group_input.outputs["thickness"], 1: 0.0000})

    multiply_k = nw.new_node(Nodes.Math, input_kwargs={0: add_1, 1: 0.5000}, attrs={'operation': 'MULTIPLY'})

    add_k = nw.new_node(Nodes.Math, input_kwargs={0: multiply_k, 1: 0.004})

    multiply = nw.new_node(Nodes.Math, input_kwargs={0: group_input.outputs["height"], 1: 1.0000},
                           attrs={'operation': 'MULTIPLY'})

    add_2 = nw.new_node(Nodes.Math, input_kwargs={0: multiply, 1: -0.0001})

    combine_xyz_3 = nw.new_node(Nodes.CombineXYZ, input_kwargs={'X': add, 'Y': add_1, 'Z': add_2})

    cube = nw.new_node(Nodes.MeshCube,
                       input_kwargs={'Size': combine_xyz_3, 'Vertices X': 5, 'Vertices Y': 5, 'Vertices Z': 5})

    multiply_1 = nw.new_node(Nodes.Math, input_kwargs={0: multiply}, attrs={'operation': 'MULTIPLY'})

    combine_xyz_4 = nw.new_node(Nodes.CombineXYZ, input_kwargs={'Y': add_k, 'Z': multiply_1})

    transform_4 = nw.new_node(Nodes.Transform, input_kwargs={'Geometry': cube, 'Translation': combine_xyz_4})

    set_material = nw.new_node(Nodes.SetMaterial,

    join_geometry_1 = nw.new_node(Nodes.JoinGeometry, input_kwargs={'Geometry': set_material})

    realize_instances = nw.new_node(Nodes.RealizeInstances, input_kwargs={'Geometry': join_geometry_1})

    group_output = nw.new_node(Nodes.GroupOutput,
                               input_kwargs={'Geometry': realize_instances, 'mid_height': multiply},
                               attrs={'is_active_output': True})


@node_utils.to_nodegroup('nodegroup_double_rampled_edge', singleton=False, type='GeometryNodeTree')
def nodegroup_double_rampled_edge(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler


    add = nw.new_node(Nodes.Math, input_kwargs={0: group_input.outputs["height"], 1: 0.0000})

    combine_xyz_10 = nw.new_node(Nodes.CombineXYZ, input_kwargs={'Z': add})

    curve_line = nw.new_node(Nodes.CurveLine, input_kwargs={'End': combine_xyz_10})

    curve_circle = nw.new_node(Nodes.CurveCircle, input_kwargs={'Resolution': 3, 'Radius': 0.0100})

    endpoint_selection = nw.new_node(Nodes.EndpointSelection, input_kwargs={'End Size': 0})

    add_1 = nw.new_node(Nodes.Math, input_kwargs={0: group_input.outputs["width"], 1: 0.0000})

    add_2 = nw.new_node(Nodes.Math, input_kwargs={0: group_input.outputs["ramp_angle"], 1: 0.0000})

    tangent = nw.new_node(Nodes.Math, input_kwargs={0: add_2}, attrs={'operation': 'TANGENT'})

    add_3 = nw.new_node(Nodes.Math, input_kwargs={0: group_input.outputs["thickness_2"], 1: 0.0000})

    multiply = nw.new_node(Nodes.Math, input_kwargs={0: tangent, 1: add_3}, attrs={'operation': 'MULTIPLY'})

    multiply_1 = nw.new_node(Nodes.Math, input_kwargs={0: 2.0000, 1: multiply}, attrs={'operation': 'MULTIPLY'})

    subtract = nw.new_node(Nodes.Math, input_kwargs={0: add_1, 1: multiply_1}, attrs={'operation': 'SUBTRACT'})

    multiply_2 = nw.new_node(Nodes.Math, input_kwargs={0: subtract}, attrs={'operation': 'MULTIPLY'})


    add_4 = nw.new_node(Nodes.Math, input_kwargs={0: group_input.outputs["thickness_1"], 1: 0.0000})

    combine_xyz_7 = nw.new_node(Nodes.CombineXYZ, input_kwargs={'X': multiply_3, 'Y': add_4})


    endpoint_selection_1 = nw.new_node(Nodes.EndpointSelection, input_kwargs={'Start Size': 0})

    add_5 = nw.new_node(Nodes.Math, input_kwargs={0: add_4, 1: add_3})

    combine_xyz_8 = nw.new_node(Nodes.CombineXYZ, input_kwargs={'X': multiply_3, 'Y': add_5})


    index = nw.new_node(Nodes.Index)

    less_than = nw.new_node(Nodes.Math, input_kwargs={0: index, 1: 1.0100}, attrs={'operation': 'LESS_THAN'})


    op_and = nw.new_node(Nodes.BooleanMath, input_kwargs={0: less_than, 1: greater_than})

    multiply_4 = nw.new_node(Nodes.Math, input_kwargs={0: add_1}, attrs={'operation': 'MULTIPLY'})


    combine_xyz_9 = nw.new_node(Nodes.CombineXYZ, input_kwargs={'X': multiply_5, 'Y': add_4})



    combine_xyz = nw.new_node(Nodes.CombineXYZ, input_kwargs={'X': add_1, 'Y': add_4, 'Z': add})

    cube = nw.new_node(Nodes.MeshCube, input_kwargs={'Size': combine_xyz})

    multiply_6 = nw.new_node(Nodes.Math, input_kwargs={0: add_4}, attrs={'operation': 'MULTIPLY'})

    combine_xyz_2 = nw.new_node(Nodes.CombineXYZ, input_kwargs={'Y': multiply_6})

    transform = nw.new_node(Nodes.Transform, input_kwargs={'Geometry': cube, 'Translation': combine_xyz_2})

    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ, input_kwargs={'X': subtract, 'Y': add_3, 'Z': add})

    cube_1 = nw.new_node(Nodes.MeshCube, input_kwargs={'Size': combine_xyz_1})

    multiply_7 = nw.new_node(Nodes.Math, input_kwargs={0: add_3}, attrs={'operation': 'MULTIPLY'})

    add_6 = nw.new_node(Nodes.Math, input_kwargs={0: add_4, 1: multiply_7})

    combine_xyz_3 = nw.new_node(Nodes.CombineXYZ, input_kwargs={'Y': add_6})

    transform_1 = nw.new_node(Nodes.Transform, input_kwargs={'Geometry': cube_1, 'Translation': combine_xyz_3})

    join_geometry = nw.new_node(Nodes.JoinGeometry, input_kwargs={'Geometry': [transform, transform_1]})

    multiply_8 = nw.new_node(Nodes.Math, input_kwargs={0: add}, attrs={'operation': 'MULTIPLY'})

    combine_xyz_11 = nw.new_node(Nodes.CombineXYZ, input_kwargs={'Z': multiply_8})


    combine_xyz_12 = nw.new_node(Nodes.CombineXYZ, input_kwargs={'Z': add})

    curve_line_1 = nw.new_node(Nodes.CurveLine, input_kwargs={'End': combine_xyz_12})

    transform_2 = nw.new_node(Nodes.Transform,
                              input_kwargs={'Geometry': set_position_2, 'Scale': (-1.0000, 1.0000, 1.0000)})


    join_geometry_1 = nw.new_node(Nodes.JoinGeometry,
                                  input_kwargs={'Geometry': [curve_to_mesh, transform_4, curve_to_mesh_1]})

    merge_by_distance = nw.new_node(Nodes.MergeByDistance,
                                    input_kwargs={'Geometry': join_geometry_1, 'Distance': 0.0001})

    realize_instances = nw.new_node(Nodes.RealizeInstances, input_kwargs={'Geometry': merge_by_distance})

    subdivide_mesh = nw.new_node(Nodes.SubdivideMesh, input_kwargs={'Mesh': realize_instances, 'Level': 4})

    group_output = nw.new_node(Nodes.GroupOutput, input_kwargs={'Geometry': subdivide_mesh},
                               attrs={'is_active_output': True})


@node_utils.to_nodegroup('nodegroup_ramped_edge', singleton=False, type='GeometryNodeTree')
def nodegroup_ramped_edge(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler


    add = nw.new_node(Nodes.Math, input_kwargs={0: group_input.outputs["height"], 1: 0.0000})

    combine_xyz_10 = nw.new_node(Nodes.CombineXYZ, input_kwargs={'Z': add})

    curve_line = nw.new_node(Nodes.CurveLine, input_kwargs={'End': combine_xyz_10})

    curve_circle = nw.new_node(Nodes.CurveCircle, input_kwargs={'Resolution': 3, 'Radius': 0.0100})

    endpoint_selection = nw.new_node(Nodes.EndpointSelection, input_kwargs={'End Size': 0})

    add_1 = nw.new_node(Nodes.Math, input_kwargs={0: group_input.outputs["width"], 1: 0.0000})

    multiply = nw.new_node(Nodes.Math, input_kwargs={0: add_1}, attrs={'operation': 'MULTIPLY'})

    add_2 = nw.new_node(Nodes.Math, input_kwargs={0: group_input.outputs["ramp_angle"], 1: 0.0000})

    tangent = nw.new_node(Nodes.Math, input_kwargs={0: add_2}, attrs={'operation': 'TANGENT'})

    add_3 = nw.new_node(Nodes.Math, input_kwargs={0: group_input.outputs["thickness_2"], 1: 0.0000})

    multiply_1 = nw.new_node(Nodes.Math, input_kwargs={0: tangent, 1: add_3}, attrs={'operation': 'MULTIPLY'})

    subtract = nw.new_node(Nodes.Math, input_kwargs={0: add_1, 1: multiply_1}, attrs={'operation': 'SUBTRACT'})


    add_4 = nw.new_node(Nodes.Math, input_kwargs={0: group_input.outputs["thickness_1"], 1: 0.0000})

    combine_xyz_7 = nw.new_node(Nodes.CombineXYZ, input_kwargs={'X': subtract_1, 'Y': add_4})


    endpoint_selection_1 = nw.new_node(Nodes.EndpointSelection, input_kwargs={'Start Size': 0})

    add_5 = nw.new_node(Nodes.Math, input_kwargs={0: add_4, 1: add_3})

    combine_xyz_8 = nw.new_node(Nodes.CombineXYZ, input_kwargs={'X': subtract_1, 'Y': add_5})


    index = nw.new_node(Nodes.Index)

    less_than = nw.new_node(Nodes.Math, input_kwargs={0: index, 1: 1.0100}, attrs={'operation': 'LESS_THAN'})


    op_and = nw.new_node(Nodes.BooleanMath, input_kwargs={0: less_than, 1: greater_than})


    combine_xyz_9 = nw.new_node(Nodes.CombineXYZ, input_kwargs={'X': multiply_2, 'Y': add_4})



    combine_xyz = nw.new_node(Nodes.CombineXYZ, input_kwargs={'X': add_1, 'Y': add_4, 'Z': add})

    cube = nw.new_node(Nodes.MeshCube, input_kwargs={'Size': combine_xyz})

    multiply_3 = nw.new_node(Nodes.Math, input_kwargs={0: add_4}, attrs={'operation': 'MULTIPLY'})

    combine_xyz_2 = nw.new_node(Nodes.CombineXYZ, input_kwargs={'Y': multiply_3})

    transform = nw.new_node(Nodes.Transform, input_kwargs={'Geometry': cube, 'Translation': combine_xyz_2})

    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ, input_kwargs={'X': subtract, 'Y': add_3, 'Z': add})

    cube_1 = nw.new_node(Nodes.MeshCube, input_kwargs={'Size': combine_xyz_1})

    multiply_4 = nw.new_node(Nodes.Math, input_kwargs={0: multiply_1}, attrs={'operation': 'MULTIPLY'})

    multiply_5 = nw.new_node(Nodes.Math, input_kwargs={0: add_3}, attrs={'operation': 'MULTIPLY'})

    add_6 = nw.new_node(Nodes.Math, input_kwargs={0: add_4, 1: multiply_5})

    combine_xyz_3 = nw.new_node(Nodes.CombineXYZ, input_kwargs={'X': multiply_4, 'Y': add_6})

    transform_1 = nw.new_node(Nodes.Transform, input_kwargs={'Geometry': cube_1, 'Translation': combine_xyz_3})

    join_geometry = nw.new_node(Nodes.JoinGeometry, input_kwargs={'Geometry': [transform, transform_1]})

    multiply_6 = nw.new_node(Nodes.Math, input_kwargs={0: add}, attrs={'operation': 'MULTIPLY'})

    combine_xyz_11 = nw.new_node(Nodes.CombineXYZ, input_kwargs={'Z': multiply_6})


    join_geometry_1 = nw.new_node(Nodes.JoinGeometry, input_kwargs={'Geometry': [curve_to_mesh, transform_4]})

    merge_by_distance = nw.new_node(Nodes.MergeByDistance,
                                    input_kwargs={'Geometry': join_geometry_1, 'Distance': 0.0001})

    realize_instances = nw.new_node(Nodes.RealizeInstances, input_kwargs={'Geometry': merge_by_distance})

    subdivide_mesh = nw.new_node(Nodes.SubdivideMesh, input_kwargs={'Mesh': realize_instances, 'Level': 4})

    multiply_7 = nw.new_node(Nodes.Math, input_kwargs={0: add_1, 1: -0.5000}, attrs={'operation': 'MULTIPLY'})

    combine_xyz_4 = nw.new_node(Nodes.CombineXYZ, input_kwargs={'X': multiply_7})


    group_output = nw.new_node(Nodes.GroupOutput, input_kwargs={'Geometry': transform_2},
                               attrs={'is_active_output': True})


@node_utils.to_nodegroup('nodegroup_panel_edge_frame', singleton=False, type='GeometryNodeTree')
def nodegroup_panel_edge_frame(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler


                               attrs={'operation': 'MULTIPLY_ADD'})




    add = nw.new_node(Nodes.Math, input_kwargs={0: multiply_1, 1: -0.0001})

    add_1 = nw.new_node(Nodes.Math, input_kwargs={0: group_input.outputs["door_height"], 1: 0.0001})

    combine_xyz_2 = nw.new_node(Nodes.CombineXYZ, input_kwargs={'X': add, 'Z': add_1})


    add_2 = nw.new_node(Nodes.Math, input_kwargs={0: multiply, 1: 0.0001})

    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ, input_kwargs={'X': add_2})


    combine_xyz = nw.new_node(Nodes.CombineXYZ, input_kwargs={'X': multiply_add})



    join_geometry_1 = nw.new_node(Nodes.JoinGeometry,
                                  input_kwargs={'Geometry': [transform_3, transform_2, transform_1, transform]})

                               attrs={'is_active_output': True})


def geometry_door_nodes(nw: NodeWrangler, **kwargs):
    # Code generated using version 2.6.4 of the node_transpiler

    door_height = nw.new_node(Nodes.Value, label='door_height')
    door_height.outputs[0].default_value = kwargs['door_height']

    door_edge_thickness_2 = nw.new_node(Nodes.Value, label='door_edge_thickness_2')
    door_edge_thickness_2.outputs[0].default_value = kwargs['edge_thickness_2']

    door_edge_width = nw.new_node(Nodes.Value, label='door_edge_width')
    door_edge_width.outputs[0].default_value = kwargs['edge_width']

    door_edge_thickness_1 = nw.new_node(Nodes.Value, label='door_edge_thickness_1')
    door_edge_thickness_1.outputs[0].default_value = kwargs['edge_thickness_1']

    door_edge_ramp_angle = nw.new_node(Nodes.Value, label='door_edge_ramp_angle')
    door_edge_ramp_angle.outputs[0].default_value = kwargs['edge_ramp_angle']


    door_width = nw.new_node(Nodes.Value, label='door_width')
    door_width.outputs[0].default_value = kwargs['door_width']


    add = nw.new_node(Nodes.Math, input_kwargs={0: panel_edge_frame.outputs["Value"], 1: 0.0001})

    mid_board_thickness = nw.new_node(Nodes.Value, label='mid_board_thickness')
    mid_board_thickness.outputs[0].default_value = kwargs['board_thickness']

    if kwargs['has_mid_ramp']:
    else:

    combine_xyz_5 = nw.new_node(Nodes.CombineXYZ,
                                input_kwargs={'X': add, 'Y': -0.0001, 'Z': mid_board.outputs["mid_height"]})

    frame = [panel_edge_frame.outputs["Geometry"]]
    if kwargs['has_mid_ramp']:
        frame.append(transform_5)

    join_geometry_1 = nw.new_node(Nodes.JoinGeometry, input_kwargs={'Geometry': frame})

    set_material_2 = nw.new_node(Nodes.SetMaterial,

    knob_raduis = nw.new_node(Nodes.Value, label='knob_raduis')
    knob_raduis.outputs[0].default_value = kwargs['knob_R']

    know_length = nw.new_node(Nodes.Value, label='know_length')
    know_length.outputs[0].default_value = kwargs['knob_length']

    multiply = nw.new_node(Nodes.Math, input_kwargs={0: door_height}, attrs={'operation': 'MULTIPLY'})


    set_material_3 = nw.new_node(Nodes.SetMaterial,

    attach_gadgets = []

    for h in kwargs['attach_height']:
        attach_height = nw.new_node(Nodes.Value, label='attach_height')
        attach_height.outputs[0].default_value = h

        attach = nw.new_node(nodegroup_node_group().name,
                             input_kwargs={'attach_height': attach_height, 'door_width': door_width})

        set_material_1 = nw.new_node(Nodes.SetMaterial,
        attach_gadgets.append(set_material_1)

    geos = [set_material_2, set_material_3, mid_board.outputs["Geometry"]] + attach_gadgets


    combine_xyz = nw.new_node(Nodes.CombineXYZ, input_kwargs={'X': multiply})


    realize_instances_1 = nw.new_node(Nodes.RealizeInstances, input_kwargs={'Geometry': transform})

    triangulate = nw.new_node('GeometryNodeTriangulate', input_kwargs={'Mesh': realize_instances_1})


    transform_2 = nw.new_node(Nodes.Transform,
                              input_kwargs={'Geometry': transform_1, 'Rotation': (0.0000, 0.0000, -1.5708)})

    group_output = nw.new_node(Nodes.GroupOutput, input_kwargs={'Geometry': transform_2},
                               attrs={'is_active_output': True})


def geometry_cabinet_nodes(nw: NodeWrangler, **kwargs):
    # Code generated using version 2.6.4 of the node_transpiler
    right_door_info = nw.new_node(Nodes.ObjectInfo, input_kwargs={'Object': kwargs['door'][0]})
    left_door_info = nw.new_node(Nodes.ObjectInfo, input_kwargs={'Object': kwargs['door'][1]})
    shelf_info = nw.new_node(Nodes.ObjectInfo, input_kwargs={'Object': kwargs['shelf']})

    doors = []
    doors.append(transform_r)
    if len(kwargs['door_hinge_pos']) > 1:
        doors.append(transform_l)

    attaches = []
    for pos in kwargs['attach_pos']:
        cube = nw.new_node(Nodes.MeshCube, input_kwargs={'Size': (0.0006, 0.0200, 0.04500)})

        combine_xyz = nw.new_node(Nodes.CombineXYZ, input_kwargs={'Y': -0.0100})

        transform = nw.new_node(Nodes.Transform, input_kwargs={'Geometry': cube, 'Translation': combine_xyz})

        cube_1 = nw.new_node(Nodes.MeshCube, input_kwargs={'Size': (0.0005, 0.0340, 0.0200)})

        join_geometry = nw.new_node(Nodes.JoinGeometry, input_kwargs={'Geometry': [transform, cube_1]})


        transform_2 = nw.new_node(Nodes.Transform,
                                  input_kwargs={'Geometry': transform_1, 'Rotation': (0.0000, 0.0000, -1.5708)})


        attaches.append(transform_3)

    join_geometry_a = nw.new_node(Nodes.JoinGeometry, input_kwargs={'Geometry': attaches})



    group_output = nw.new_node(Nodes.GroupOutput, input_kwargs={'Geometry': join_geometry},
                               attrs={'is_active_output': True})


class CabinetDoorBaseFactory(AssetFactory):
    def __init__(self, factory_seed, params={}, coarse=False):
        super(CabinetDoorBaseFactory, self).__init__(factory_seed, coarse=coarse)
        self.params = {}

    def get_asset_params(self, i=0):
        params = self.params.copy()
        if params.get('door_height', None) is None:
            params['door_height'] = uniform(0.7, 2.2)
        if params.get('door_width', None) is None:
            params['door_width'] = uniform(0.3, 0.4)
        if params.get('edge_thickness_1', None) is None:
            params['edge_thickness_1'] = uniform(0.01, 0.02)
        if params.get('edge_width', None) is None:
            params['edge_width'] = uniform(0.03, 0.05)
        if params.get('edge_thickness_2', None) is None:
            params['edge_thickness_2'] = uniform(0.005, 0.01)
        if params.get('edge_ramp_angle', None) is None:
            params['edge_ramp_angle'] = uniform(0.6, 0.8)
        params['board_thickness'] = params['edge_thickness_1'] - 0.005
        if params.get('knob_R', None) is None:
            params['knob_R'] = uniform(0.003, 0.006)
        if params.get('knob_length', None) is None:
            params['knob_length'] = uniform(0.018, 0.035)
        if params.get('attach_height', None) is None:
            gap = uniform(0.05, 0.15)
            params['attach_height'] = [gap, params['door_height'] - gap]
        if params.get('has_mid_ramp', None) is None:
            params['has_mid_ramp'] = np.random.choice([True, False], p=[0.6, 0.4])
        if params.get('door_left_hinge', None) is None:
            params['door_left_hinge'] = False

        if params.get('frame_material', None) is None:
            params['frame_material'] = np.random.choice(['white', 'black_wood', 'wood'], p=[0.5, 0.2, 0.3])
        if params.get('board_material', None) is None:
            if params['has_mid_ramp']:
                lower_mat = np.random.choice([params['frame_material'], 'glass'], p=[0.7, 0.3])
                upper_mat = np.random.choice([lower_mat, 'glass'], p=[0.6, 0.4])
                params['board_material'] = [lower_mat, upper_mat]
            else:
                params['board_material'] = [params['frame_material']]

        params = self.get_material_func(params)
        return params

    def get_material_func(self, params, randomness=True):
        materials = []
        if not isinstance(params['board_material'], list):
            params['board_material'] = [params['board_material']]
        for mat in params['board_material']:
        params['board_material'] = materials
        return params

    def create_asset(self, i=0, **params):
        obj = bpy.context.active_object

        obj_params = self.get_asset_params(i)
        surface.add_geomod(obj, geometry_door_nodes, apply=True, attributes=[], input_kwargs=obj_params)

        if params.get('ret_params', False):
            return obj, obj_params

        return obj


class CabinetDoorIkeaFactory(CabinetDoorBaseFactory):
    def __init__(self, factory_seed, params={}, coarse=False):
        super(CabinetDoorIkeaFactory, self).__init__(factory_seed, coarse=coarse)
        self.params = {
            'edge_thickness_1': 0.012,
            'edge_thickness_2': 0.008,
            'board_thickness': 0.006,
            'edge_width': 0.02,
            'edge_ramp_angle': 0.5,
            'knob_R': 0.004,
            'knob_length': 0.03,
            'has_mid_ramp': False,
            'attach_height': 0.08
        }

    def get_asset_params(self, i=0):
        params = self.params.copy()
        if params.get('door_height', None) is None:
            params['door_height'] = uniform(0.7, 2.2)
        if params.get('door_width', None) is None:
            params['door_width'] = uniform(0.3, 0.4)
        if params.get('door_left_hinge', None) is None:
            params['door_left_hinge'] = False

        params['attach_height'] = [params['door_height'] - params['attach_height'], params['attach_height']]
        params = self.get_material_func(params)
        return params


class CabinetBaseFactory(AssetFactory):
    def __init__(self, factory_seed, params={}, coarse=False):
        super(CabinetBaseFactory, self).__init__(factory_seed, coarse=coarse)
        self.shelf_params = {}
        self.door_params = {}
        self.mat_params = {}
        self.shelf_fac = LargeShelfBaseFactory(factory_seed)
        self.door_fac = CabinetDoorBaseFactory(factory_seed)

    def sample_params(self):
        # Update fac params
        pass


    def get_shelf_params(self, i=0):
        params = self.shelf_params.copy()
        if params.get('shelf_cell_width', None) is None:
        if params.get('shelf_cell_height', None) is None:
            num_v_cells = randint(3, 7)
            shelf_cell_height = []
            for i in range(num_v_cells):
                shelf_cell_height.append(0.3 * np.clip(normal(1., 0.06), 0.75, 1.25))
            params['shelf_cell_height'] = shelf_cell_height
        if params.get('frame_material', None) is None:
            params['frame_material'] = self.mat_params['frame_material']

        return params

    def get_door_params(self, i=0):
        params = self.door_params.copy()

        # get door params
        shelf_width = self.shelf_params['shelf_width'] + self.shelf_params['side_board_thickness'] * 2
        if params.get('door_width', None) is None:
            if shelf_width < 0.55:
                params['door_width'] = shelf_width
                params['num_door'] = 1
            else:
                params['door_width'] = shelf_width / 2. - 0.0005
                params['num_door'] = 2
        if params.get('door_height', None) is None:
            params['door_height'] = (self.shelf_params['division_board_z_translation'][-1] -
                params['door_height'] = (self.shelf_params['division_board_z_translation'][3] -
                                         self.shelf_params['division_board_z_translation'][0] +
                                         self.shelf_params['division_board_thickness'])
        if params.get('frame_material', None) is None:
            params['frame_material'] = self.mat_params['frame_material']

        return params

    def get_cabinet_params(self, i=0):
        params = dict()

        shelf_width = self.shelf_params['shelf_width'] + self.shelf_params['side_board_thickness'] * 2
        if self.door_params['num_door'] == 1:
            params['door_hinge_pos'] = [(self.shelf_params['shelf_depth'] / 2. + 0.0025, -shelf_width / 2.,
            params['door_open_angle'] = 0
        elif self.door_params['num_door'] == 2:
            params['door_hinge_pos'] = [(self.shelf_params['shelf_depth'] / 2. + 0.008, -shelf_width / 2.,
            params['door_open_angle'] = 0
        else:
            raise NotImplementedError

        return params

    def get_cabinet_components(self, i):
        # update material params
        self.sample_params()

        # create shelf
        shelf_params = self.get_shelf_params(i=i)
        self.shelf_fac.params = shelf_params
        shelf, shelf_params = self.shelf_fac.create_asset(i=i, ret_params=True)
        shelf.name = 'cabinet_frame'
        self.shelf_params = shelf_params

        # create doors
        door_params = self.get_door_params(i=i)
        self.door_fac.params = door_params
        self.door_fac.params['door_left_hinge'] = False
        right_door, door_obj_params = self.door_fac.create_asset(i=i, ret_params=True)
        right_door.name = 'cabinet_right_door'
        self.door_fac.params = door_obj_params
        self.door_fac.params['door_left_hinge'] = True
        left_door, _ = self.door_fac.create_asset(i=i, ret_params=True)
        left_door.name = 'cabinet_left_door'
        self.door_params = door_obj_params

        return shelf, right_door, left_door

    def create_asset(self, i=0, **params):
        obj = bpy.context.active_object

        shelf, right_door, left_door = self.get_cabinet_components(i=i)

        # create cabinet
        cabinet_params = self.get_cabinet_params(i=i)
        surface.add_geomod(obj, geometry_cabinet_nodes, attributes=[], input_kwargs={
            'door': [right_door, left_door],
            'shelf': shelf,
            'door_hinge_pos': cabinet_params['door_hinge_pos'],
            'door_open_angle': cabinet_params['door_open_angle'],
            'attach_pos': cabinet_params['attach_pos']
        })
        butil.delete([shelf, left_door, right_door])
        return obj


class CabinetFactory(CabinetBaseFactory):
    def sample_params(self):
        params = dict()

        params['bottom_board_height'] = 0.083
        params['shelf_depth'] = params['Dimensions'][0] - 0.01
        num_h = int((params['Dimensions'][2] - 0.083) / 0.3)
        params['shelf_cell_height'] = [(params['Dimensions'][2] - 0.083) / num_h for _ in range(num_h)]
        params['shelf_cell_width'] = [params['Dimensions'][1]]
        self.shelf_params = self.shelf_fac.sample_params()

