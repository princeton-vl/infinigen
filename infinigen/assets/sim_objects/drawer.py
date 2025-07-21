import bpy
import gin
from numpy.random import uniform, normal, randint

from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.nodes import node_utils
from infinigen.core import surface
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.util import blender as butil
from infinigen.core.util.paths import blueprint_path_completion
from infinigen.core.sim.exporters import factory
from infinigen.core.util.random import weighted_sample
from infinigen.assets.composition import material_assignments



@node_utils.to_nodegroup('nodegroup_handle', singleton=False, type='GeometryNodeTree')
def nodegroup_handle(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketInt', 'Handle Type', 0),
            ('NodeSocketMaterial', 'Handle Material', None)])
    
    equal = nw.new_node(Nodes.Compare,
        input_kwargs={2: group_input.outputs["Handle Type"]},
        attrs={'operation': 'EQUAL', 'data_type': 'INT'})
    
    equal_1 = nw.new_node(Nodes.Compare,
        input_kwargs={2: group_input.outputs["Handle Type"], 3: 1},
        attrs={'operation': 'EQUAL', 'data_type': 'INT'})
    
    arc = nw.new_node('GeometryNodeCurveArc', input_kwargs={'Radius': 0.0500, 'Sweep Angle': 3.1416})
    
    curve_circle = nw.new_node(Nodes.CurveCircle, input_kwargs={'Radius': 0.0100})
    
    curve_to_mesh = nw.new_node(Nodes.CurveToMesh,
        input_kwargs={'Curve': arc.outputs["Curve"], 'Profile Curve': curve_circle.outputs["Curve"]})
    
    transform_geometry_5 = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': curve_to_mesh, 'Rotation': (0.0000, 0.0000, -1.5708), 'Scale': (2.0000, 1.0000, 1.5000)})
    
    cylinder = nw.new_node('GeometryNodeMeshCylinder', input_kwargs={'Vertices': 16, 'Radius': 0.0100, 'Depth': 0.0500})
    
    transform_geometry = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': cylinder.outputs["Mesh"], 'Rotation': (0.0000, 1.5708, 0.0000)})
    
    reroute_1 = nw.new_node(Nodes.Reroute, input_kwargs={'Input': transform_geometry})
    
    transform_geometry_1 = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': transform_geometry, 'Translation': (0.0000, 0.1000, 0.0000)})
    
    cylinder_1 = nw.new_node('GeometryNodeMeshCylinder', input_kwargs={'Vertices': 16, 'Radius': 0.0100, 'Depth': 0.2000})
    
    transform_geometry_2 = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': cylinder_1.outputs["Mesh"], 'Translation': (0.0250, 0.0500, 0.0000), 'Rotation': (1.5708, 0.0000, 0.0000)})
    
    join_geometry = nw.new_node(Nodes.JoinGeometry, input_kwargs={'Geometry': [reroute_1, transform_geometry_1, transform_geometry_2]})
    
    transform_geometry_3 = nw.new_node(Nodes.Transform, input_kwargs={'Geometry': join_geometry, 'Translation': (0.0250, -0.0500, 0.0000)})
    
    switch_1 = nw.new_node(Nodes.Switch,
        input_kwargs={'Switch': equal_1, 'False': transform_geometry_5, 'True': transform_geometry_3})
    
    uv_sphere = nw.new_node(Nodes.MeshUVSphere, input_kwargs={'Segments': 12, 'Rings': 8, 'Radius': 0.0200})
    
    transform_geometry_4 = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': uv_sphere.outputs["Mesh"], 'Translation': (0.0200, 0.0000, 0.0000)})
    
    switch = nw.new_node(Nodes.Switch, input_kwargs={'Switch': equal, 'False': switch_1, 'True': transform_geometry_4})
    
    reroute = nw.new_node(Nodes.Reroute, input_kwargs={'Input': group_input.outputs["Handle Material"]})
    
    set_material = nw.new_node(Nodes.SetMaterial, input_kwargs={'Geometry': switch, 'Material': reroute})
    
    group_output = nw.new_node(Nodes.GroupOutput, input_kwargs={'Geometry': set_material}, attrs={'is_active_output': True})

@node_utils.to_nodegroup('nodegroup_drawers', singleton=False, type='GeometryNodeTree')
def nodegroup_drawers(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketVector', 'Size', (1.0000, 1.0000, 1.0000)),
            ('NodeSocketFloat', 'Thickness', 0.0000),
            ('NodeSocketFloat', 'Base Offset', 0.0000),
            ('NodeSocketInt', 'Num Rows', 0),
            ('NodeSocketInt', 'Num Columns', 0),
            ('NodeSocketFloat', 'Drawer X Thickness', 0.0000),
            ('NodeSocketFloat', 'Drawer Y Thickness', 0.0000),
            ('NodeSocketInt', 'Handle Type', 0),
            ('NodeSocketMaterial', 'Handle Material', None),
            ('NodeSocketMaterial', 'Drawer Material', None)])
    
    handle = nw.new_node(nodegroup_handle().name,
        input_kwargs={'Handle Type': group_input.outputs["Handle Type"], 'Handle Material': group_input.outputs["Handle Material"]})
    
    reroute_11 = nw.new_node(Nodes.Reroute, input_kwargs={'Input': handle})
    
    separate_xyz = nw.new_node(Nodes.SeparateXYZ, input_kwargs={'Vector': group_input.outputs["Size"]})
    
    reroute = nw.new_node(Nodes.Reroute, input_kwargs={'Input': group_input.outputs["Thickness"]})
    
    reroute_1 = nw.new_node(Nodes.Reroute, input_kwargs={'Input': reroute})
    
    subtract = nw.new_node(Nodes.Math, input_kwargs={0: separate_xyz.outputs["X"], 1: reroute_1}, attrs={'operation': 'SUBTRACT'})
    
    divide = nw.new_node(Nodes.Math, input_kwargs={0: subtract, 1: 2.0000}, attrs={'operation': 'DIVIDE'})
    
    combine_xyz_2 = nw.new_node(Nodes.CombineXYZ, input_kwargs={'X': divide})
    
    transform_geometry_1 = nw.new_node(Nodes.Transform, input_kwargs={'Geometry': reroute_11, 'Translation': combine_xyz_2})
    
    reroute_16 = nw.new_node(Nodes.Reroute, input_kwargs={'Input': transform_geometry_1})
    
    reroute_13 = nw.new_node(Nodes.Reroute, input_kwargs={'Input': subtract})
    
    reroute_14 = nw.new_node(Nodes.Reroute, input_kwargs={'Input': reroute_13})
    
    multiply = nw.new_node(Nodes.Math,
        input_kwargs={0: group_input.outputs["Thickness"], 1: 2.0000},
        attrs={'operation': 'MULTIPLY'})
    
    subtract_1 = nw.new_node(Nodes.Math, input_kwargs={0: separate_xyz.outputs["Y"], 1: multiply}, attrs={'operation': 'SUBTRACT'})
    
    subtract_2 = nw.new_node(Nodes.Math,
        input_kwargs={0: group_input.outputs["Num Columns"], 1: 1.0000},
        attrs={'operation': 'SUBTRACT'})
    
    multiply_1 = nw.new_node(Nodes.Math, input_kwargs={0: subtract_2, 1: reroute_1}, attrs={'operation': 'MULTIPLY'})
    
    subtract_3 = nw.new_node(Nodes.Math, input_kwargs={0: subtract_1, 1: multiply_1}, attrs={'operation': 'SUBTRACT'})
    
    reroute_6 = nw.new_node(Nodes.Reroute, input_kwargs={'Input': group_input.outputs["Num Columns"]})
    
    reroute_7 = nw.new_node(Nodes.Reroute, input_kwargs={'Input': reroute_6})
    
    divide_1 = nw.new_node(Nodes.Math, input_kwargs={0: subtract_3, 1: reroute_7}, attrs={'operation': 'DIVIDE'})
    
    value = nw.new_node(Nodes.Value)
    value.outputs[0].default_value = 0.0030
    
    subtract_4 = nw.new_node(Nodes.Math, input_kwargs={0: divide_1, 1: value}, attrs={'operation': 'SUBTRACT'})
    
    reroute_15 = nw.new_node(Nodes.Reroute, input_kwargs={'Input': subtract_4})
    
    reroute_3 = nw.new_node(Nodes.Reroute, input_kwargs={'Input': group_input.outputs["Base Offset"]})
    
    reroute_4 = nw.new_node(Nodes.Reroute, input_kwargs={'Input': reroute_3})
    
    subtract_5 = nw.new_node(Nodes.Math, input_kwargs={0: separate_xyz.outputs["Z"], 1: reroute_4}, attrs={'operation': 'SUBTRACT'})
    
    reroute_12 = nw.new_node(Nodes.Reroute, input_kwargs={'Input': reroute_1})
    
    subtract_6 = nw.new_node(Nodes.Math, input_kwargs={0: subtract_5, 1: reroute_12}, attrs={'operation': 'SUBTRACT'})
    
    multiply_2 = nw.new_node(Nodes.Math,
        input_kwargs={0: group_input.outputs["Thickness"], 1: group_input.outputs["Num Rows"]},
        attrs={'operation': 'MULTIPLY'})
    
    reroute_8 = nw.new_node(Nodes.Reroute, input_kwargs={'Input': multiply_2})
    
    reroute_9 = nw.new_node(Nodes.Reroute, input_kwargs={'Input': reroute_8})
    
    subtract_7 = nw.new_node(Nodes.Math, input_kwargs={0: subtract_6, 1: reroute_9}, attrs={'operation': 'SUBTRACT'})
    
    reroute_5 = nw.new_node(Nodes.Reroute, input_kwargs={'Input': group_input.outputs["Num Rows"]})
    
    divide_2 = nw.new_node(Nodes.Math, input_kwargs={0: subtract_7, 1: reroute_5}, attrs={'operation': 'DIVIDE'})
    
    value_1 = nw.new_node(Nodes.Value)
    value_1.outputs[0].default_value = 0.0030
    
    subtract_8 = nw.new_node(Nodes.Math, input_kwargs={0: divide_2, 1: value_1}, attrs={'operation': 'SUBTRACT'})
    
    combine_xyz = nw.new_node(Nodes.CombineXYZ, input_kwargs={'X': reroute_14, 'Y': reroute_15, 'Z': subtract_8})
    
    reroute_17 = nw.new_node(Nodes.Reroute, input_kwargs={'Input': combine_xyz})
    
    cube = nw.new_node(Nodes.MeshCube, input_kwargs={'Size': reroute_17})
    
    cube_1 = nw.new_node(Nodes.MeshCube, input_kwargs={'Size': combine_xyz})
    
    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'X': group_input.outputs["Drawer X Thickness"], 'Y': group_input.outputs["Drawer Y Thickness"], 'Z': 1.0000})
    
    reroute_10 = nw.new_node(Nodes.Reroute, input_kwargs={'Input': combine_xyz_1})
    
    transform_geometry = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': cube_1.outputs["Mesh"], 'Translation': (0.0000, 0.0000, 0.0300), 'Scale': reroute_10})
    
    difference = nw.new_node(Nodes.MeshBoolean,
        input_kwargs={'Mesh 1': cube.outputs["Mesh"], 'Mesh 2': transform_geometry},
        attrs={'solver': 'EXACT'})
    
    reroute_2 = nw.new_node(Nodes.Reroute, input_kwargs={'Input': group_input.outputs["Drawer Material"]})
    
    set_material = nw.new_node(Nodes.SetMaterial, input_kwargs={'Geometry': difference.outputs["Mesh"], 'Material': reroute_2})
    
    join_geometry = nw.new_node(Nodes.JoinGeometry, input_kwargs={'Geometry': [reroute_16, set_material]})
    
    group_output = nw.new_node(Nodes.GroupOutput, input_kwargs={'Mesh': join_geometry}, attrs={'is_active_output': True})

@node_utils.to_nodegroup('nodegroup_drawer_base', singleton=False, type='GeometryNodeTree')
def nodegroup_drawer_base(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketVector', 'Size', (0.0000, 0.0000, 0.0000)),
            ('NodeSocketFloat', 'Thickness', 0.0000),
            ('NodeSocketFloat', 'Bottom Offset', 0.5000),
            ('NodeSocketInt', 'Num Rows', 0),
            ('NodeSocketInt', 'Num Columns', 0),
            ('NodeSocketMaterial', 'Base Material', None)])
    
    reroute = nw.new_node(Nodes.Reroute, input_kwargs={'Input': group_input.outputs["Thickness"]})
    
    reroute_1 = nw.new_node(Nodes.Reroute, input_kwargs={'Input': reroute})
    
    reroute_12 = nw.new_node(Nodes.Reroute, input_kwargs={'Input': reroute_1})
    
    separate_xyz = nw.new_node(Nodes.SeparateXYZ, input_kwargs={'Vector': group_input.outputs["Size"]})
    
    reroute_9 = nw.new_node(Nodes.Reroute, input_kwargs={'Input': separate_xyz.outputs["Y"]})
    
    multiply = nw.new_node(Nodes.Math,
        input_kwargs={0: group_input.outputs["Thickness"], 1: 2.0000},
        attrs={'operation': 'MULTIPLY'})
    
    subtract = nw.new_node(Nodes.Math, input_kwargs={0: separate_xyz.outputs["Z"], 1: multiply}, attrs={'operation': 'SUBTRACT'})
    
    combine_xyz_3 = nw.new_node(Nodes.CombineXYZ, input_kwargs={'X': reroute_12, 'Y': reroute_9, 'Z': subtract})
    
    cube_1 = nw.new_node(Nodes.MeshCube, input_kwargs={'Size': combine_xyz_3})
    
    multiply_1 = nw.new_node(Nodes.Math, input_kwargs={0: separate_xyz.outputs["X"], 1: -0.5000}, attrs={'operation': 'MULTIPLY'})
    
    divide = nw.new_node(Nodes.Math, input_kwargs={0: group_input.outputs["Thickness"], 1: 2.0000}, attrs={'operation': 'DIVIDE'})
    
    reroute_10 = nw.new_node(Nodes.Reroute, input_kwargs={'Input': divide})
    
    reroute_11 = nw.new_node(Nodes.Reroute, input_kwargs={'Input': reroute_10})
    
    add = nw.new_node(Nodes.Math, input_kwargs={0: multiply_1, 1: reroute_11})
    
    combine_xyz_4 = nw.new_node(Nodes.CombineXYZ, input_kwargs={'X': add})
    
    transform_geometry = nw.new_node(Nodes.Transform, input_kwargs={'Geometry': cube_1.outputs["Mesh"], 'Translation': combine_xyz_4})
    
    reroute_16 = nw.new_node(Nodes.Reroute, input_kwargs={'Input': transform_geometry})
    
    reroute_5 = nw.new_node(Nodes.Reroute, input_kwargs={'Input': group_input.outputs["Num Columns"]})
    
    add_1 = nw.new_node(Nodes.Math, input_kwargs={0: reroute_5, 1: 1.0000})
    
    divide_1 = nw.new_node(Nodes.Math, input_kwargs={0: separate_xyz.outputs["Y"], 1: 2.0000}, attrs={'operation': 'DIVIDE'})
    
    multiply_2 = nw.new_node(Nodes.Math, input_kwargs={0: divide, 1: -1.0000}, attrs={'operation': 'MULTIPLY'})
    
    add_2 = nw.new_node(Nodes.Math, input_kwargs={0: divide_1, 1: multiply_2})
    
    combine_xyz_5 = nw.new_node(Nodes.CombineXYZ, input_kwargs={'Y': add_2})
    
    divide_2 = nw.new_node(Nodes.Math, input_kwargs={0: separate_xyz.outputs["Y"], 1: -2.0000}, attrs={'operation': 'DIVIDE'})
    
    add_3 = nw.new_node(Nodes.Math, input_kwargs={0: divide_2, 1: reroute_11})
    
    combine_xyz_6 = nw.new_node(Nodes.CombineXYZ, input_kwargs={'Y': add_3})
    
    mesh_line_1 = nw.new_node(Nodes.MeshLine,
        input_kwargs={'Count': add_1, 'Start Location': combine_xyz_5, 'Offset': combine_xyz_6},
        attrs={'mode': 'END_POINTS'})
    
    transform_geometry_1 = nw.new_node(Nodes.Transform, input_kwargs={'Geometry': mesh_line_1})
    
    reroute_7 = nw.new_node(Nodes.Reroute, input_kwargs={'Input': separate_xyz.outputs["X"]})
    
    reroute_8 = nw.new_node(Nodes.Reroute, input_kwargs={'Input': reroute_7})
    
    multiply_3 = nw.new_node(Nodes.Math,
        input_kwargs={0: group_input.outputs["Thickness"], 1: 2.0000},
        attrs={'operation': 'MULTIPLY'})
    
    subtract_1 = nw.new_node(Nodes.Math, input_kwargs={0: separate_xyz.outputs["Z"], 1: multiply_3}, attrs={'operation': 'SUBTRACT'})
    
    combine_xyz_7 = nw.new_node(Nodes.CombineXYZ, input_kwargs={'X': reroute_8, 'Y': reroute_12, 'Z': subtract_1})
    
    cube_2 = nw.new_node(Nodes.MeshCube, input_kwargs={'Size': combine_xyz_7})
    
    transform_geometry_2 = nw.new_node(Nodes.Transform, input_kwargs={'Geometry': cube_2.outputs["Mesh"]})
    
    reroute_15 = nw.new_node(Nodes.Reroute, input_kwargs={'Input': transform_geometry_2})
    
    instance_on_points_1 = nw.new_node(Nodes.InstanceOnPoints, input_kwargs={'Points': transform_geometry_1, 'Instance': reroute_15})
    
    realize_instances_1 = nw.new_node(Nodes.RealizeInstances, input_kwargs={'Geometry': instance_on_points_1})
    
    reroute_17 = nw.new_node(Nodes.Reroute, input_kwargs={'Input': realize_instances_1})
    
    reroute_3 = nw.new_node(Nodes.Reroute, input_kwargs={'Input': group_input.outputs["Num Rows"]})
    
    reroute_4 = nw.new_node(Nodes.Reroute, input_kwargs={'Input': reroute_3})
    
    add_4 = nw.new_node(Nodes.Math, input_kwargs={0: reroute_4, 1: 1.0000})
    
    divide_3 = nw.new_node(Nodes.Math, input_kwargs={0: separate_xyz.outputs["Z"], 1: 2.0000}, attrs={'operation': 'DIVIDE'})
    
    add_5 = nw.new_node(Nodes.Math, input_kwargs={0: multiply_2, 1: divide_3})
    
    multiply_4 = nw.new_node(Nodes.Math, input_kwargs={0: add_5, 1: -1.0000}, attrs={'operation': 'MULTIPLY'})
    
    reroute_2 = nw.new_node(Nodes.Reroute, input_kwargs={'Input': group_input.outputs["Bottom Offset"]})
    
    add_6 = nw.new_node(Nodes.Math, input_kwargs={0: multiply_4, 1: reroute_2})
    
    combine_xyz = nw.new_node(Nodes.CombineXYZ, input_kwargs={'Z': add_6})
    
    reroute_13 = nw.new_node(Nodes.Reroute, input_kwargs={'Input': add_5})
    
    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ, input_kwargs={'Z': reroute_13})
    
    mesh_line = nw.new_node(Nodes.MeshLine,
        input_kwargs={'Count': add_4, 'Start Location': combine_xyz, 'Offset': combine_xyz_1},
        attrs={'mode': 'END_POINTS'})
    
    combine_xyz_2 = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'X': separate_xyz.outputs["X"], 'Y': separate_xyz.outputs["Y"], 'Z': reroute_1})
    
    cube = nw.new_node(Nodes.MeshCube, input_kwargs={'Size': combine_xyz_2})
    
    reroute_14 = nw.new_node(Nodes.Reroute, input_kwargs={'Input': cube.outputs["Mesh"]})
    
    instance_on_points = nw.new_node(Nodes.InstanceOnPoints, input_kwargs={'Points': mesh_line, 'Instance': reroute_14})
    
    realize_instances = nw.new_node(Nodes.RealizeInstances, input_kwargs={'Geometry': instance_on_points})
    
    join_geometry = nw.new_node(Nodes.JoinGeometry, input_kwargs={'Geometry': [reroute_16, reroute_17, realize_instances]})
    
    reroute_6 = nw.new_node(Nodes.Reroute, input_kwargs={'Input': group_input.outputs["Base Material"]})
    
    set_material = nw.new_node(Nodes.SetMaterial, input_kwargs={'Geometry': join_geometry, 'Material': reroute_6})
    
    group_output = nw.new_node(Nodes.GroupOutput, input_kwargs={'Mesh': set_material}, attrs={'is_active_output': True})

@node_utils.to_nodegroup('nodegroup_add_jointed_geometry_metadata', singleton=False, type='GeometryNodeTree')
def nodegroup_add_jointed_geometry_metadata(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketGeometry', 'Geometry', None),
            ('NodeSocketString', 'Label', '')])
    
    store_named_attribute = nw.new_node(Nodes.StoreNamedAttribute,
        input_kwargs={'Geometry': group_input.outputs["Geometry"], 'Name': group_input.outputs["Label"], 'Value': 1},
        attrs={'data_type': 'INT'})
    
    group_output = nw.new_node(Nodes.GroupOutput, input_kwargs={'Geometry': store_named_attribute}, attrs={'is_active_output': True})

@node_utils.to_nodegroup('nodegroup_sliding_joint', singleton=False, type='GeometryNodeTree')
def nodegroup_sliding_joint(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketString', 'Joint ID (do not set)', ''),
            ('NodeSocketString', 'Joint Label', ''),
            ('NodeSocketString', 'Parent Label', ''),
            ('NodeSocketGeometry', 'Parent', None),
            ('NodeSocketString', 'Child Label', ''),
            ('NodeSocketGeometry', 'Child', None),
            ('NodeSocketVector', 'Position', (0.0000, 0.0000, 0.0000)),
            ('NodeSocketVector', 'Axis', (0.0000, 0.0000, 1.0000)),
            ('NodeSocketFloat', 'Value', 0.0000),
            ('NodeSocketFloat', 'Min', 0.0000),
            ('NodeSocketFloat', 'Max', 0.0000),
            ('NodeSocketBool', 'Show Center of Parent', False),
            ('NodeSocketBool', 'Show Center of Child', False),
            ('NodeSocketBool', 'Show Joint', False)])
    
    named_attribute_4 = nw.new_node(Nodes.NamedAttribute, input_kwargs={'Name': 'part_id'}, attrs={'data_type': 'INT'})
    
    integer = nw.new_node(Nodes.Integer)
    integer.integer = 0
    
    switch_2 = nw.new_node(Nodes.Switch,
        input_kwargs={'Switch': named_attribute_4.outputs["Exists"], 'False': integer, 'True': named_attribute_4.outputs["Attribute"]},
        attrs={'input_type': 'INT'})
    
    store_named_attribute_1 = nw.new_node(Nodes.StoreNamedAttribute,
        input_kwargs={'Geometry': group_input.outputs["Parent"], 'Name': 'part_id', 'Value': switch_2},
        attrs={'data_type': 'INT'})
    
    named_attribute_1 = nw.new_node(Nodes.NamedAttribute, input_kwargs={'Name': 'part_id'}, attrs={'data_type': 'INT'})
    
    attribute_statistic = nw.new_node(Nodes.AttributeStatistic,
        input_kwargs={'Geometry': store_named_attribute_1, 'Attribute': named_attribute_1.outputs["Attribute"]})
    
    equal = nw.new_node(Nodes.Compare,
        input_kwargs={2: named_attribute_1.outputs["Attribute"], 3: attribute_statistic.outputs["Min"]},
        attrs={'operation': 'EQUAL', 'data_type': 'INT'})
    
    separate_geometry_2 = nw.new_node(Nodes.SeparateGeometry, input_kwargs={'Geometry': store_named_attribute_1, 'Selection': equal})
    
    join_geometry_2 = nw.new_node(Nodes.JoinGeometry,
        input_kwargs={'Geometry': [separate_geometry_2.outputs["Selection"], separate_geometry_2.outputs["Inverted"]]})
    
    cone = nw.new_node('GeometryNodeMeshCone', input_kwargs={'Radius Bottom': 0.0500, 'Depth': 0.2000})
    
    transform_geometry_3 = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': cone.outputs["Mesh"], 'Translation': (0.0000, 0.0000, -0.0500)})
    
    named_attribute_3 = nw.new_node(Nodes.NamedAttribute, input_kwargs={'Name': 'part_id'}, attrs={'data_type': 'INT'})
    
    integer_1 = nw.new_node(Nodes.Integer)
    integer_1.integer = 1
    
    add = nw.new_node(Nodes.Math, input_kwargs={0: named_attribute_3.outputs["Attribute"], 1: 1.0000})
    
    switch_3 = nw.new_node(Nodes.Switch,
        input_kwargs={'Switch': named_attribute_3.outputs["Exists"], 'False': integer_1, 'True': add},
        attrs={'input_type': 'INT'})
    
    store_named_attribute = nw.new_node(Nodes.StoreNamedAttribute,
        input_kwargs={'Geometry': group_input.outputs["Child"], 'Name': 'part_id', 'Value': switch_3},
        attrs={'data_type': 'INT'})
    
    named_attribute_2 = nw.new_node(Nodes.NamedAttribute, input_kwargs={'Name': 'part_id'}, attrs={'data_type': 'INT'})
    
    attribute_statistic_1 = nw.new_node(Nodes.AttributeStatistic,
        input_kwargs={'Geometry': store_named_attribute, 'Attribute': named_attribute_2.outputs["Attribute"]})
    
    equal_1 = nw.new_node(Nodes.Compare,
        input_kwargs={2: named_attribute_2.outputs["Attribute"], 3: attribute_statistic_1.outputs["Min"]},
        attrs={'operation': 'EQUAL', 'data_type': 'INT'})
    
    separate_geometry_3 = nw.new_node(Nodes.SeparateGeometry, input_kwargs={'Geometry': store_named_attribute, 'Selection': equal_1})
    
    reroute = nw.new_node(Nodes.Reroute, input_kwargs={'Input': separate_geometry_3.outputs["Selection"]})
    
    bounding_box_3 = nw.new_node(Nodes.BoundingBox, input_kwargs={'Geometry': reroute})
    
    position_7 = nw.new_node(Nodes.InputPosition)
    
    attribute_statistic_9 = nw.new_node(Nodes.AttributeStatistic,
        input_kwargs={'Geometry': bounding_box_3.outputs["Bounding Box"], 'Attribute': position_7},
        attrs={'data_type': 'FLOAT_VECTOR'})
    
    add_1 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: group_input.outputs["Position"], 1: attribute_statistic_9.outputs["Mean"]})
    
    named_attribute_5 = nw.new_node(Nodes.NamedAttribute, input_kwargs={'Name': 'body_transform'}, attrs={'data_type': 'FLOAT4X4'})
    
    transpose_matrix = nw.new_node('FunctionNodeTransposeMatrix', input_kwargs={'Matrix': named_attribute_5.outputs["Attribute"]})
    
    transform_direction = nw.new_node('FunctionNodeTransformDirection',
        input_kwargs={'Direction': group_input.outputs["Axis"], 'Transform': transpose_matrix})
    
    attribute_statistic_5 = nw.new_node(Nodes.AttributeStatistic,
        input_kwargs={'Geometry': reroute, 'Attribute': transform_direction},
        attrs={'data_type': 'FLOAT_VECTOR'})
    
    normalize = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: attribute_statistic_5.outputs["Mean"]},
        attrs={'operation': 'NORMALIZE'})
    
    align_rotation_to_vector_1 = nw.new_node('FunctionNodeAlignRotationToVector', input_kwargs={'Vector': normalize.outputs["Vector"]})
    
    transform_geometry_2 = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': transform_geometry_3, 'Translation': add_1.outputs["Vector"], 'Rotation': align_rotation_to_vector_1})
    
    switch_7 = nw.new_node(Nodes.Switch, input_kwargs={'Switch': group_input.outputs["Show Joint"], 'True': transform_geometry_2})
    
    store_named_attribute_15 = nw.new_node(Nodes.StoreNamedAttribute,
        input_kwargs={'Geometry': switch_7, 'Name': 'part_id', 'Value': 999999999},
        attrs={'data_type': 'INT'})
    
    uv_sphere_1 = nw.new_node(Nodes.MeshUVSphere, input_kwargs={'Segments': 10, 'Rings': 10, 'Radius': 0.0500})
    
    transform_geometry_1 = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': uv_sphere_1.outputs["Mesh"], 'Translation': attribute_statistic_9.outputs["Mean"]})
    
    switch_6 = nw.new_node(Nodes.Switch,
        input_kwargs={'Switch': group_input.outputs["Show Center of Child"], 'True': transform_geometry_1})
    
    store_named_attribute_14 = nw.new_node(Nodes.StoreNamedAttribute,
        input_kwargs={'Geometry': switch_6, 'Name': 'part_id', 'Value': 999999999},
        attrs={'data_type': 'INT'})
    
    uv_sphere = nw.new_node(Nodes.MeshUVSphere, input_kwargs={'Segments': 10, 'Rings': 10, 'Radius': 0.0500})
    
    bounding_box = nw.new_node(Nodes.BoundingBox, input_kwargs={'Geometry': separate_geometry_2.outputs["Selection"]})
    
    position = nw.new_node(Nodes.InputPosition)
    
    attribute_statistic_2 = nw.new_node(Nodes.AttributeStatistic,
        input_kwargs={'Geometry': bounding_box.outputs["Bounding Box"], 'Attribute': position},
        attrs={'data_type': 'FLOAT_VECTOR'})
    
    transform_geometry = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': uv_sphere.outputs["Mesh"], 'Translation': attribute_statistic_2.outputs["Mean"]})
    
    switch_4 = nw.new_node(Nodes.Switch,
        input_kwargs={'Switch': group_input.outputs["Show Center of Parent"], 'True': transform_geometry})
    
    store_named_attribute_13 = nw.new_node(Nodes.StoreNamedAttribute,
        input_kwargs={'Geometry': switch_4, 'Name': 'part_id', 'Value': 999999999},
        attrs={'data_type': 'INT'})
    
    named_attribute_11 = nw.new_node(Nodes.NamedAttribute, input_kwargs={'Name': 'is_jointed'}, attrs={'data_type': 'BOOLEAN'})
    
    attribute_statistic_7 = nw.new_node(Nodes.AttributeStatistic,
        input_kwargs={'Geometry': separate_geometry_3.outputs["Selection"], 'Attribute': named_attribute_11.outputs["Attribute"]})
    
    greater_than = nw.new_node(Nodes.Compare, input_kwargs={2: attribute_statistic_7.outputs["Sum"]}, attrs={'data_type': 'INT'})
    
    combine_matrix = nw.new_node('FunctionNodeCombineMatrix')
    
    named_attribute_10 = nw.new_node(Nodes.NamedAttribute, input_kwargs={'Name': 'body_transform'}, attrs={'data_type': 'FLOAT4X4'})
    
    switch_1 = nw.new_node(Nodes.Switch,
        input_kwargs={'Switch': greater_than, 'False': combine_matrix, 'True': named_attribute_10.outputs["Attribute"]},
        attrs={'input_type': 'MATRIX'})
    
    store_named_attribute_2 = nw.new_node(Nodes.StoreNamedAttribute,
        input_kwargs={'Geometry': store_named_attribute, 'Name': 'body_transform', 'Value': switch_1},
        attrs={'data_type': 'FLOAT4X4'})
    
    named_attribute_7 = nw.new_node(Nodes.NamedAttribute, input_kwargs={'Name': 'is_jointed'}, attrs={'data_type': 'BOOLEAN'})
    
    attribute_statistic_4 = nw.new_node(Nodes.AttributeStatistic,
        input_kwargs={'Geometry': separate_geometry_3.outputs["Selection"], 'Attribute': named_attribute_7.outputs["Attribute"]})
    
    equal_2 = nw.new_node(Nodes.Compare,
        input_kwargs={2: attribute_statistic_4.outputs["Sum"]},
        attrs={'operation': 'EQUAL', 'data_type': 'INT'})
    
    position_4 = nw.new_node(Nodes.InputPosition)
    
    position_1 = nw.new_node(Nodes.InputPosition)
    
    add_2 = nw.new_node(Nodes.VectorMath, input_kwargs={0: position_1, 1: attribute_statistic_2.outputs["Mean"]})
    
    switch = nw.new_node(Nodes.Switch,
        input_kwargs={'Switch': equal_2, 'False': position_4, 'True': add_2.outputs["Vector"]},
        attrs={'input_type': 'VECTOR'})
    
    set_position = nw.new_node(Nodes.SetPosition, input_kwargs={'Geometry': store_named_attribute_2, 'Position': switch})
    
    store_named_attribute_3 = nw.new_node(Nodes.StoreNamedAttribute,
        input_kwargs={'Geometry': set_position, 'Name': 'is_jointed', 'Value': True},
        attrs={'data_type': 'BOOLEAN'})
    
    equal_3 = nw.new_node(Nodes.Compare,
        input_kwargs={0: group_input.outputs["Min"], 'Epsilon': 0.0000},
        attrs={'operation': 'EQUAL'})
    
    equal_4 = nw.new_node(Nodes.Compare,
        input_kwargs={0: group_input.outputs["Max"], 'Epsilon': 0.0000},
        attrs={'operation': 'EQUAL'})
    
    op_and = nw.new_node(Nodes.BooleanMath, input_kwargs={0: equal_3, 1: equal_4})
    
    clamp = nw.new_node(Nodes.Clamp,
        input_kwargs={'Value': group_input.outputs["Value"], 'Min': group_input.outputs["Min"], 'Max': group_input.outputs["Max"]})
    
    switch_5 = nw.new_node(Nodes.Switch,
        input_kwargs={'Switch': op_and, 'False': clamp, 'True': group_input.outputs["Value"]},
        attrs={'input_type': 'FLOAT'})
    
    scale = nw.new_node(Nodes.VectorMath, input_kwargs={0: transform_direction, 'Scale': switch_5}, attrs={'operation': 'SCALE'})
    
    position_5 = nw.new_node(Nodes.InputPosition)
    
    add_3 = nw.new_node(Nodes.VectorMath, input_kwargs={0: scale.outputs["Vector"], 1: position_5})
    
    set_position_2 = nw.new_node(Nodes.SetPosition,
        input_kwargs={'Geometry': store_named_attribute_3, 'Position': add_3.outputs["Vector"]})
    
    string = nw.new_node('FunctionNodeInputString', attrs={'string': 'pos'})
    
    reroute_2 = nw.new_node(Nodes.Reroute, input_kwargs={'Input': group_input.outputs["Joint ID (do not set)"]})
    
    join_strings = nw.new_node('GeometryNodeStringJoin', input_kwargs={'Delimiter': '_', 'Strings': [string, reroute_2]})
    
    reroute_1 = nw.new_node(Nodes.Reroute, input_kwargs={'Input': group_input.outputs["Position"]})
    
    store_named_attribute_5 = nw.new_node(Nodes.StoreNamedAttribute,
        input_kwargs={'Geometry': set_position_2, 'Name': join_strings, 'Value': reroute_1},
        attrs={'data_type': 'FLOAT_VECTOR'})
    
    string_1 = nw.new_node('FunctionNodeInputString', attrs={'string': 'axis'})
    
    join_strings_1 = nw.new_node('GeometryNodeStringJoin', input_kwargs={'Delimiter': '_', 'Strings': [string_1, reroute_2]})
    
    store_named_attribute_6 = nw.new_node(Nodes.StoreNamedAttribute,
        input_kwargs={'Geometry': store_named_attribute_5, 'Name': join_strings_1, 'Value': group_input.outputs["Axis"]},
        attrs={'data_type': 'FLOAT_VECTOR'})
    
    string_2 = nw.new_node('FunctionNodeInputString', attrs={'string': 'min'})
    
    join_strings_2 = nw.new_node('GeometryNodeStringJoin', input_kwargs={'Delimiter': '_', 'Strings': [string_2, reroute_2]})
    
    store_named_attribute_8 = nw.new_node(Nodes.StoreNamedAttribute,
        input_kwargs={'Geometry': store_named_attribute_6, 'Name': join_strings_2, 'Value': group_input.outputs["Min"]})
    
    string_3 = nw.new_node('FunctionNodeInputString', attrs={'string': 'max'})
    
    join_strings_3 = nw.new_node('GeometryNodeStringJoin', input_kwargs={'Delimiter': '_', 'Strings': [string_3, reroute_2]})
    
    store_named_attribute_7 = nw.new_node(Nodes.StoreNamedAttribute,
        input_kwargs={'Geometry': store_named_attribute_8, 'Name': join_strings_3, 'Value': group_input.outputs["Max"]})
    
    join_geometry = nw.new_node(Nodes.JoinGeometry,
        input_kwargs={'Geometry': [join_geometry_2, store_named_attribute_15, store_named_attribute_14, store_named_attribute_13, store_named_attribute_7]})
    
    store_named_attribute_9 = nw.new_node(Nodes.StoreNamedAttribute,
        input_kwargs={'Geometry': set_position_2, 'Name': join_strings, 'Value': reroute_1},
        attrs={'data_type': 'FLOAT_VECTOR'})
    
    store_named_attribute_10 = nw.new_node(Nodes.StoreNamedAttribute,
        input_kwargs={'Geometry': store_named_attribute_9, 'Name': join_strings_1, 'Value': group_input.outputs["Axis"]},
        attrs={'data_type': 'FLOAT_VECTOR'})
    
    store_named_attribute_12 = nw.new_node(Nodes.StoreNamedAttribute,
        input_kwargs={'Geometry': store_named_attribute_10, 'Name': join_strings_2, 'Value': group_input.outputs["Min"]})
    
    store_named_attribute_11 = nw.new_node(Nodes.StoreNamedAttribute,
        input_kwargs={'Geometry': store_named_attribute_12, 'Name': join_strings_3, 'Value': group_input.outputs["Max"]})
    
    join_geometry_1 = nw.new_node(Nodes.JoinGeometry,
        input_kwargs={'Geometry': [store_named_attribute_15, store_named_attribute_14, store_named_attribute_13, store_named_attribute_11]})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': join_geometry, 'Parent': join_geometry_2, 'Child': join_geometry_1},
        attrs={'is_active_output': True})

@node_utils.to_nodegroup('nodegroup_duplicate_joints_on_parent', singleton=False, type='GeometryNodeTree')
def nodegroup_duplicate_joints_on_parent(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketString', 'Duplicate ID (do not set)', ''),
            ('NodeSocketGeometry', 'Parent', None),
            ('NodeSocketGeometry', 'Child', None),
            ('NodeSocketGeometry', 'Points', None)])
    
    instance_on_points = nw.new_node(Nodes.InstanceOnPoints,
        input_kwargs={'Points': group_input.outputs["Points"], 'Instance': group_input.outputs["Child"]})
    
    index = nw.new_node(Nodes.Index)
    
    add = nw.new_node(Nodes.Math, input_kwargs={0: index, 1: 1.0000})
    
    store_named_attribute = nw.new_node(Nodes.StoreNamedAttribute,
        input_kwargs={'Geometry': instance_on_points, 'Name': group_input.outputs["Duplicate ID (do not set)"], 'Value': add},
        attrs={'domain': 'INSTANCE', 'data_type': 'INT'})
    
    realize_instances = nw.new_node(Nodes.RealizeInstances, input_kwargs={'Geometry': store_named_attribute})
    
    join_geometry = nw.new_node(Nodes.JoinGeometry, input_kwargs={'Geometry': [group_input.outputs["Parent"], realize_instances]})
    
    group_output = nw.new_node(Nodes.GroupOutput, input_kwargs={'Geometry': join_geometry}, attrs={'is_active_output': True})

@node_utils.to_nodegroup('geometry_nodes', singleton=False, type='GeometryNodeTree')
def geometry_nodes(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketVector', 'Size', (0.0000, 0.0000, 0.0000)),
            ('NodeSocketFloat', 'Thickness', 0.0000),
            ('NodeSocketFloat', 'Bottom Offset', 0.5000),
            ('NodeSocketInt', 'Num Rows', 0),
            ('NodeSocketInt', 'Num Columns', 0),
            ('NodeSocketFloat', 'Drawer X Cut Scale', 0.0000),
            ('NodeSocketFloat', 'Drawer Y Cut Scale', 0.0000),
            ('NodeSocketInt', 'Handle Type', 0),
            ('NodeSocketMaterial', 'Handle Material', None),
            ('NodeSocketMaterial', 'Drawer Material', None),
            ('NodeSocketMaterial', 'Base Material', None)])
    
    drawer_base = nw.new_node(nodegroup_drawer_base().name,
        input_kwargs={'Size': group_input.outputs["Size"], 'Thickness': group_input.outputs["Thickness"], 'Bottom Offset': group_input.outputs["Bottom Offset"], 'Num Rows': group_input.outputs["Num Rows"], 'Num Columns': group_input.outputs["Num Columns"], 'Base Material': group_input.outputs["Base Material"]})
    
    add_jointed_geometry_metadata = nw.new_node(nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={'Geometry': drawer_base, 'Label': 'drawer_base'})
    
    drawers = nw.new_node(nodegroup_drawers().name,
        input_kwargs={'Size': group_input.outputs["Size"], 'Thickness': group_input.outputs["Thickness"], 'Base Offset': group_input.outputs["Bottom Offset"], 'Num Rows': group_input.outputs["Num Rows"], 'Num Columns': group_input.outputs["Num Columns"], 'Drawer X Thickness': group_input.outputs["Drawer X Cut Scale"], 'Drawer Y Thickness': group_input.outputs["Drawer Y Cut Scale"], 'Handle Type': group_input.outputs["Handle Type"], 'Handle Material': group_input.outputs["Handle Material"], 'Drawer Material': group_input.outputs["Drawer Material"]})
    
    add_jointed_geometry_metadata_1 = nw.new_node(nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={'Geometry': drawers, 'Label': 'drawer_door'})
    
    separate_xyz = nw.new_node(Nodes.SeparateXYZ, input_kwargs={'Vector': group_input.outputs["Size"]})
    
    reroute = nw.new_node(Nodes.Reroute, input_kwargs={'Input': group_input.outputs["Thickness"]})
    
    reroute_1 = nw.new_node(Nodes.Reroute, input_kwargs={'Input': reroute})
    
    subtract = nw.new_node(Nodes.Math, input_kwargs={0: separate_xyz.outputs["X"], 1: reroute_1}, attrs={'operation': 'SUBTRACT'})
    
    sliding_joint = nw.new_node(nodegroup_sliding_joint().name,
        input_kwargs={'Joint Label': 'drawer_slider', 'Parent': add_jointed_geometry_metadata, 'Child': add_jointed_geometry_metadata_1, 'Axis': (1.0000, 0.0000, 0.0000), 'Max': subtract})
    
    reroute_6 = nw.new_node(Nodes.Reroute, input_kwargs={'Input': group_input.outputs["Bottom Offset"]})
    
    reroute_7 = nw.new_node(Nodes.Reroute, input_kwargs={'Input': reroute_6})
    
    subtract_1 = nw.new_node(Nodes.Math, input_kwargs={0: separate_xyz.outputs["Z"], 1: reroute_7}, attrs={'operation': 'SUBTRACT'})
    
    multiply = nw.new_node(Nodes.Math, input_kwargs={0: reroute_1, 1: 2.0000}, attrs={'operation': 'MULTIPLY'})
    
    subtract_2 = nw.new_node(Nodes.Math, input_kwargs={0: subtract_1, 1: multiply}, attrs={'operation': 'SUBTRACT'})
    
    reroute_9 = nw.new_node(Nodes.Reroute, input_kwargs={'Input': reroute_1})
    
    reroute_17 = nw.new_node(Nodes.Reroute, input_kwargs={'Input': reroute_9})
    
    add = nw.new_node(Nodes.Math, input_kwargs={0: subtract_2, 1: reroute_17})
    
    subtract_3 = nw.new_node(Nodes.Math,
        input_kwargs={0: group_input.outputs["Num Rows"], 1: 1.0000},
        attrs={'operation': 'SUBTRACT'})
    
    reroute_2 = nw.new_node(Nodes.Reroute, input_kwargs={'Input': group_input.outputs["Num Rows"]})
    
    reroute_3 = nw.new_node(Nodes.Reroute, input_kwargs={'Input': reroute_2})
    
    divide = nw.new_node(Nodes.Math, input_kwargs={0: subtract_3, 1: reroute_3}, attrs={'operation': 'DIVIDE'})
    
    reroute_13 = nw.new_node(Nodes.Reroute, input_kwargs={'Input': divide})
    
    multiply_1 = nw.new_node(Nodes.Math, input_kwargs={0: add, 1: reroute_13}, attrs={'operation': 'MULTIPLY'})
    
    multiply_2 = nw.new_node(Nodes.Math,
        input_kwargs={0: group_input.outputs["Thickness"], 1: 2.0000},
        attrs={'operation': 'MULTIPLY'})
    
    subtract_4 = nw.new_node(Nodes.Math, input_kwargs={0: separate_xyz.outputs["Y"], 1: multiply_2}, attrs={'operation': 'SUBTRACT'})
    
    add_1 = nw.new_node(Nodes.Math, input_kwargs={0: subtract_4, 1: reroute_9})
    
    subtract_5 = nw.new_node(Nodes.Math,
        input_kwargs={0: group_input.outputs["Num Columns"], 1: 1.0000},
        attrs={'operation': 'SUBTRACT'})
    
    reroute_4 = nw.new_node(Nodes.Reroute, input_kwargs={'Input': group_input.outputs["Num Columns"]})
    
    reroute_5 = nw.new_node(Nodes.Reroute, input_kwargs={'Input': reroute_4})
    
    divide_1 = nw.new_node(Nodes.Math, input_kwargs={0: subtract_5, 1: reroute_5}, attrs={'operation': 'DIVIDE'})
    
    reroute_14 = nw.new_node(Nodes.Reroute, input_kwargs={'Input': divide_1})
    
    multiply_3 = nw.new_node(Nodes.Math, input_kwargs={0: add_1, 1: reroute_14}, attrs={'operation': 'MULTIPLY'})
    
    reroute_18 = nw.new_node(Nodes.Reroute, input_kwargs={'Input': multiply_3})
    
    reroute_10 = nw.new_node(Nodes.Reroute, input_kwargs={'Input': reroute_3})
    
    reroute_11 = nw.new_node(Nodes.Reroute, input_kwargs={'Input': reroute_5})
    
    grid = nw.new_node(Nodes.MeshGrid,
        input_kwargs={'Size X': multiply_1, 'Size Y': reroute_18, 'Vertices X': reroute_10, 'Vertices Y': reroute_11})
    
    divide_2 = nw.new_node(Nodes.Math, input_kwargs={0: group_input.outputs["Thickness"], 1: 2.0000}, attrs={'operation': 'DIVIDE'})
    
    reroute_8 = nw.new_node(Nodes.Reroute, input_kwargs={'Input': divide_2})
    
    divide_3 = nw.new_node(Nodes.Math,
        input_kwargs={0: group_input.outputs["Bottom Offset"], 1: 2.0000},
        attrs={'operation': 'DIVIDE'})
    
    subtract_6 = nw.new_node(Nodes.Math, input_kwargs={0: divide_3, 1: divide_2}, attrs={'operation': 'SUBTRACT'})
    
    combine_xyz = nw.new_node(Nodes.CombineXYZ, input_kwargs={'X': reroute_8, 'Z': subtract_6})
    
    reroute_12 = nw.new_node(Nodes.Reroute, input_kwargs={'Input': combine_xyz})
    
    transform_geometry = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': grid.outputs["Mesh"], 'Translation': reroute_12, 'Rotation': (0.0000, 1.5708, 0.0000)})
    
    duplicate_joints_on_parent = nw.new_node(nodegroup_duplicate_joints_on_parent().name,
        input_kwargs={'Parent': sliding_joint.outputs["Parent"], 'Child': sliding_joint.outputs["Child"], 'Points': transform_geometry})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': duplicate_joints_on_parent},
        attrs={'is_active_output': True})




class DrawerFactory(AssetFactory):

    def __init__(self, factory_seed=None, coarse=False):
        super().__init__(factory_seed=factory_seed, coarse=False)

    @classmethod
    @gin.configurable(module='DrawerFactory')
    def sample_joint_parameters(
        cls,
        drawer_slider_stiffness_min: float = 0.0,
        drawer_slider_stiffness_max: float = 0.0,
        drawer_slider_damping_min: float = 200.0,
        drawer_slider_damping_max: float = 300.0,
    ):
        return {
            "drawer_slider": {
                "stiffness": uniform(
                    drawer_slider_stiffness_min,
                    drawer_slider_stiffness_max
                ),
                "damping": uniform(
                    drawer_slider_damping_min,
                    drawer_slider_damping_max
                ),
            },
        }


    def sample_parameters(self):
        # add code here to randomly sample from parameters
        body_material = weighted_sample(material_assignments.shelf_board)()()

        if uniform() < 0.5:
            drawer_material = weighted_sample(material_assignments.shelf_board)()()
        else:
            drawer_material = body_material

        handle_material = weighted_sample(material_assignments.hard_materials)()()

        return {
            "Size": (uniform(0.4, 0.8),
                     uniform(0.7, 1.5),
                     uniform(0.6, 1.3)),
            "Thickness": uniform(0.02, 0.08),
            "Bottom Offset": uniform(0.0, 0.3),
            "Num Rows": randint(1, 4),
            "Num Columns": randint(1, 4),
            "Drawer X Cut Scale": uniform(0.6, 0.97),
            "Drawer Y Cut Scale": uniform(0.6, 0.97),
            "Handle Type": 0,
            "Handle Material": body_material,
            "Drawer Material": drawer_material,
            "Base Material": handle_material
        }


    def create_asset(self,
                     asset_params=None,
                     **kwargs):
        obj = butil.spawn_vert()
        butil.modify_mesh(
            obj,
            "NODES",
            apply=False,
            node_group=geometry_nodes(),
            ng_inputs=self.sample_parameters()
        )

        return obj
    