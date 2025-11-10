import gin
from numpy.random import randint, uniform

from infinigen.assets.composition import material_assignments
from infinigen.assets.utils.joints import (
    nodegroup_add_jointed_geometry_metadata,
    nodegroup_duplicate_joints_on_parent,
    nodegroup_sliding_joint,
)
from infinigen.core.nodes import node_utils
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.util import blender as butil
from infinigen.core.util.random import weighted_sample


@node_utils.to_nodegroup("nodegroup_handle", singleton=False, type="GeometryNodeTree")
def nodegroup_handle(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketInt", "Handle Type", 0),
            ("NodeSocketMaterial", "Handle Material", None),
        ],
    )

    reroute = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Handle Type"]}
    )

    reroute_1 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute})

    equal = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: reroute_1},
        attrs={"operation": "EQUAL", "data_type": "INT"},
    )

    equal_1 = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: group_input.outputs["Handle Type"], 3: 1},
        attrs={"operation": "EQUAL", "data_type": "INT"},
    )

    arc = nw.new_node(
        "GeometryNodeCurveArc", input_kwargs={"Radius": 0.0500, "Sweep Angle": 3.1416}
    )

    curve_circle = nw.new_node(Nodes.CurveCircle, input_kwargs={"Radius": 0.0100})

    curve_to_mesh = nw.new_node(
        Nodes.CurveToMesh,
        input_kwargs={
            "Curve": arc.outputs["Curve"],
            "Profile Curve": curve_circle.outputs["Curve"],
        },
    )

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": curve_to_mesh,
            "Rotation": (0.0000, 0.0000, -1.5708),
            "Scale": (2.0000, 1.0000, 1.5000),
        },
    )

    cylinder = nw.new_node(
        "GeometryNodeMeshCylinder",
        input_kwargs={"Vertices": 16, "Radius": 0.0100, "Depth": 0.0500},
    )

    transform_geometry_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cylinder.outputs["Mesh"],
            "Rotation": (0.0000, 1.5708, 0.0000),
        },
    )

    transform_geometry_2 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": transform_geometry_1,
            "Translation": (0.0000, 0.1000, 0.0000),
        },
    )

    cylinder_1 = nw.new_node(
        "GeometryNodeMeshCylinder",
        input_kwargs={"Vertices": 16, "Radius": 0.0100, "Depth": 0.2000},
    )

    transform_geometry_3 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cylinder_1.outputs["Mesh"],
            "Translation": (0.0250, 0.0500, 0.0000),
            "Rotation": (1.5708, 0.0000, 0.0000),
        },
    )

    reroute_3 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": transform_geometry_1})

    reroute_4 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_3})

    join_geometry = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={
            "Geometry": [transform_geometry_2, transform_geometry_3, reroute_4]
        },
    )

    transform_geometry_4 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": join_geometry,
            "Translation": (0.0250, -0.0500, 0.0000),
        },
    )

    switch = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": equal_1,
            "False": transform_geometry,
            "True": transform_geometry_4,
        },
    )

    uv_sphere = nw.new_node(
        Nodes.MeshUVSphere, input_kwargs={"Segments": 12, "Rings": 8, "Radius": 0.0200}
    )

    transform_geometry_5 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": uv_sphere.outputs["Mesh"],
            "Translation": (0.0200, 0.0000, 0.0000),
        },
    )

    switch_1 = nw.new_node(
        Nodes.Switch,
        input_kwargs={"Switch": equal, "False": switch, "True": transform_geometry_5},
    )

    reroute_2 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Handle Material"]}
    )

    set_material = nw.new_node(
        Nodes.SetMaterial, input_kwargs={"Geometry": switch_1, "Material": reroute_2}
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": set_material},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_drawer_door", singleton=False, type="GeometryNodeTree"
)
def nodegroup_drawer_door(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketVector", "Size", (1.0000, 1.0000, 1.0000)),
            ("NodeSocketFloat", "Thickness", 0.0000),
            ("NodeSocketFloat", "Base Offset", 0.0000),
            ("NodeSocketInt", "Num Rows", 0),
            ("NodeSocketInt", "Num Columns", 0),
            ("NodeSocketFloat", "Drawer X Thickness", 0.0000),
            ("NodeSocketFloat", "Drawer Y Thickness", 0.0000),
            ("NodeSocketInt", "Handle Type", 0),
            ("NodeSocketMaterial", "Handle Material", None),
            ("NodeSocketMaterial", "Drawer Material", None),
        ],
    )

    separate_xyz = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": group_input.outputs["Size"]}
    )

    reroute = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Thickness"]}
    )

    reroute_1 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute})

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["X"], 1: reroute_1},
        attrs={"operation": "SUBTRACT"},
    )

    reroute_13 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": subtract})

    reroute_14 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_13})

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Thickness"], 1: 2.0000},
        attrs={"operation": "MULTIPLY"},
    )

    subtract_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Y"], 1: multiply},
        attrs={"operation": "SUBTRACT"},
    )

    subtract_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Num Columns"], 1: 1.0000},
        attrs={"operation": "SUBTRACT"},
    )

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: subtract_2, 1: reroute_1},
        attrs={"operation": "MULTIPLY"},
    )

    subtract_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: subtract_1, 1: multiply_1},
        attrs={"operation": "SUBTRACT"},
    )

    reroute_5 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Num Columns"]}
    )

    divide = nw.new_node(
        Nodes.Math,
        input_kwargs={0: subtract_3, 1: reroute_5},
        attrs={"operation": "DIVIDE"},
    )

    value = nw.new_node(Nodes.Value)
    value.outputs[0].default_value = 0.0030

    subtract_4 = nw.new_node(
        Nodes.Math, input_kwargs={0: divide, 1: value}, attrs={"operation": "SUBTRACT"}
    )

    reroute_16 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": subtract_4})

    reroute_17 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_16})

    reroute_2 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Base Offset"]}
    )

    subtract_5 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Z"], 1: reroute_2},
        attrs={"operation": "SUBTRACT"},
    )

    reroute_12 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_1})

    subtract_6 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: subtract_5, 1: reroute_12},
        attrs={"operation": "SUBTRACT"},
    )

    multiply_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input.outputs["Thickness"],
            1: group_input.outputs["Num Rows"],
        },
        attrs={"operation": "MULTIPLY"},
    )

    reroute_8 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": multiply_2})

    reroute_9 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_8})

    subtract_7 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: subtract_6, 1: reroute_9},
        attrs={"operation": "SUBTRACT"},
    )

    reroute_3 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Num Rows"]}
    )

    reroute_4 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_3})

    divide_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: subtract_7, 1: reroute_4},
        attrs={"operation": "DIVIDE"},
    )

    value_1 = nw.new_node(Nodes.Value)
    value_1.outputs[0].default_value = 0.0030

    subtract_8 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: divide_1, 1: value_1},
        attrs={"operation": "SUBTRACT"},
    )

    combine_xyz_1 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={"X": reroute_14, "Y": reroute_17, "Z": subtract_8},
    )

    reroute_18 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": combine_xyz_1})

    cube = nw.new_node(Nodes.MeshCube, input_kwargs={"Size": reroute_18})

    cube_1 = nw.new_node(Nodes.MeshCube, input_kwargs={"Size": combine_xyz_1})

    combine_xyz_2 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": group_input.outputs["Drawer X Thickness"],
            "Y": group_input.outputs["Drawer Y Thickness"],
            "Z": 1.0000,
        },
    )

    reroute_10 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": combine_xyz_2})

    transform_geometry_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cube_1.outputs["Mesh"],
            "Translation": (0.0000, 0.0000, 0.0300),
            "Scale": reroute_10,
        },
    )

    difference = nw.new_node(
        Nodes.MeshBoolean,
        input_kwargs={"Mesh 1": cube.outputs["Mesh"], "Mesh 2": transform_geometry_1},
        attrs={"solver": "EXACT"},
    )

    reroute_6 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Drawer Material"]}
    )

    reroute_7 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_6})

    set_material = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={"Geometry": difference.outputs["Mesh"], "Material": reroute_7},
    )

    handle = nw.new_node(
        nodegroup_handle().name,
        input_kwargs={
            "Handle Type": group_input.outputs["Handle Type"],
            "Handle Material": group_input.outputs["Handle Material"],
        },
    )

    reroute_11 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": handle})

    divide_2 = nw.new_node(
        Nodes.Math, input_kwargs={0: subtract, 1: 2.0000}, attrs={"operation": "DIVIDE"}
    )

    combine_xyz = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": divide_2})

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": reroute_11, "Translation": combine_xyz},
    )

    reroute_15 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": transform_geometry})

    join_geometry = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [set_material, reroute_15]}
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Mesh": join_geometry},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_add_jointed_geometry_metadata_001",
    singleton=False,
    type="GeometryNodeTree",
)
def nodegroup_add_jointed_geometry_metadata_001(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketGeometry", "Geometry", None),
            ("NodeSocketString", "Label", ""),
        ],
    )

    store_named_attribute = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": group_input.outputs["Geometry"],
            "Name": group_input.outputs["Label"],
            "Value": 1,
        },
        attrs={"data_type": "INT"},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": store_named_attribute},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_drawer_base", singleton=False, type="GeometryNodeTree"
)
def nodegroup_drawer_base(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketVector", "Size", (0.0000, 0.0000, 0.0000)),
            ("NodeSocketFloat", "Thickness", 0.0000),
            ("NodeSocketFloat", "Bottom Offset", 0.5000),
            ("NodeSocketInt", "Num Rows", 0),
            ("NodeSocketInt", "Num Columns", 0),
            ("NodeSocketMaterial", "Base Material", None),
        ],
    )

    reroute_4 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Num Rows"]}
    )

    reroute_5 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_4})

    add = nw.new_node(Nodes.Math, input_kwargs={0: reroute_5, 1: 1.0000})

    divide = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Thickness"], 1: 2.0000},
        attrs={"operation": "DIVIDE"},
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: divide, 1: -1.0000},
        attrs={"operation": "MULTIPLY"},
    )

    separate_xyz = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": group_input.outputs["Size"]}
    )

    divide_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Z"], 1: 2.0000},
        attrs={"operation": "DIVIDE"},
    )

    add_1 = nw.new_node(Nodes.Math, input_kwargs={0: multiply, 1: divide_1})

    multiply_1 = nw.new_node(
        Nodes.Math, input_kwargs={0: add_1, 1: -1.0000}, attrs={"operation": "MULTIPLY"}
    )

    reroute_2 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Bottom Offset"]}
    )

    reroute_3 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_2})

    add_2 = nw.new_node(Nodes.Math, input_kwargs={0: multiply_1, 1: reroute_3})

    combine_xyz_5 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": add_2})

    reroute_15 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": add_1})

    combine_xyz_6 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": reroute_15})

    mesh_line_1 = nw.new_node(
        Nodes.MeshLine,
        input_kwargs={
            "Count": add,
            "Start Location": combine_xyz_5,
            "Offset": combine_xyz_6,
        },
        attrs={"mode": "END_POINTS"},
    )

    reroute = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Thickness"]}
    )

    reroute_1 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute})

    combine_xyz_7 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": separate_xyz.outputs["X"],
            "Y": separate_xyz.outputs["Y"],
            "Z": reroute_1,
        },
    )

    cube_2 = nw.new_node(Nodes.MeshCube, input_kwargs={"Size": combine_xyz_7})

    reroute_16 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": cube_2.outputs["Mesh"]}
    )

    instance_on_points_1 = nw.new_node(
        Nodes.InstanceOnPoints,
        input_kwargs={"Points": mesh_line_1, "Instance": reroute_16},
    )

    realize_instances_1 = nw.new_node(
        Nodes.RealizeInstances, input_kwargs={"Geometry": instance_on_points_1}
    )

    reroute_14 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_1})

    reroute_10 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": separate_xyz.outputs["Y"]}
    )

    reroute_11 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_10})

    multiply_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Thickness"], 1: 2.0000},
        attrs={"operation": "MULTIPLY"},
    )

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Z"], 1: multiply_2},
        attrs={"operation": "SUBTRACT"},
    )

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": reroute_14, "Y": reroute_11, "Z": subtract}
    )

    cube = nw.new_node(Nodes.MeshCube, input_kwargs={"Size": combine_xyz})

    multiply_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["X"], 1: -0.5000},
        attrs={"operation": "MULTIPLY"},
    )

    reroute_12 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": divide})

    reroute_13 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_12})

    add_3 = nw.new_node(Nodes.Math, input_kwargs={0: multiply_3, 1: reroute_13})

    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": add_3})

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": cube.outputs["Mesh"], "Translation": combine_xyz_1},
    )

    reroute_18 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": transform_geometry})

    reroute_6 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Num Columns"]}
    )

    add_4 = nw.new_node(Nodes.Math, input_kwargs={0: reroute_6, 1: 1.0000})

    divide_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Y"], 1: 2.0000},
        attrs={"operation": "DIVIDE"},
    )

    add_5 = nw.new_node(Nodes.Math, input_kwargs={0: divide_2, 1: multiply})

    combine_xyz_2 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Y": add_5})

    divide_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Y"], 1: -2.0000},
        attrs={"operation": "DIVIDE"},
    )

    add_6 = nw.new_node(Nodes.Math, input_kwargs={0: divide_3, 1: reroute_13})

    combine_xyz_3 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Y": add_6})

    mesh_line = nw.new_node(
        Nodes.MeshLine,
        input_kwargs={
            "Count": add_4,
            "Start Location": combine_xyz_2,
            "Offset": combine_xyz_3,
        },
        attrs={"mode": "END_POINTS"},
    )

    transform_geometry_1 = nw.new_node(
        Nodes.Transform, input_kwargs={"Geometry": mesh_line}
    )

    reroute_8 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": separate_xyz.outputs["X"]}
    )

    reroute_9 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_8})

    multiply_4 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Thickness"], 1: 2.0000},
        attrs={"operation": "MULTIPLY"},
    )

    subtract_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Z"], 1: multiply_4},
        attrs={"operation": "SUBTRACT"},
    )

    combine_xyz_4 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={"X": reroute_9, "Y": reroute_14, "Z": subtract_1},
    )

    cube_1 = nw.new_node(Nodes.MeshCube, input_kwargs={"Size": combine_xyz_4})

    transform_geometry_2 = nw.new_node(
        Nodes.Transform, input_kwargs={"Geometry": cube_1.outputs["Mesh"]}
    )

    reroute_17 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": transform_geometry_2}
    )

    instance_on_points = nw.new_node(
        Nodes.InstanceOnPoints,
        input_kwargs={"Points": transform_geometry_1, "Instance": reroute_17},
    )

    realize_instances = nw.new_node(
        Nodes.RealizeInstances, input_kwargs={"Geometry": instance_on_points}
    )

    reroute_19 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": realize_instances})

    join_geometry = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [realize_instances_1, reroute_18, reroute_19]},
    )

    reroute_7 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Base Material"]}
    )

    set_material = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={"Geometry": join_geometry, "Material": reroute_7},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Mesh": set_material},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup("geometry_nodes", singleton=False, type="GeometryNodeTree")
def geometry_nodes(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketVector", "Size", (0.0000, 0.0000, 0.0000)),
            ("NodeSocketFloat", "Thickness", 0.0000),
            ("NodeSocketFloat", "Bottom Offset", 0.5000),
            ("NodeSocketInt", "Num Rows", 0),
            ("NodeSocketInt", "Num Columns", 0),
            ("NodeSocketFloat", "Drawer X Cut Scale", 0.0000),
            ("NodeSocketFloat", "Drawer Y Cut Scale", 0.0000),
            ("NodeSocketInt", "Handle Type", 0),
            ("NodeSocketMaterial", "Handle Material", None),
            ("NodeSocketMaterial", "Drawer Material", None),
            ("NodeSocketMaterial", "Base Material", None),
        ],
    )

    reroute_7 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Size"]}
    )

    reroute_8 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Thickness"]}
    )

    reroute_9 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Num Rows"]}
    )

    base = nw.new_node(
        nodegroup_drawer_base().name,
        input_kwargs={
            "Size": reroute_7,
            "Thickness": reroute_8,
            "Bottom Offset": group_input.outputs["Bottom Offset"],
            "Num Rows": reroute_9,
            "Num Columns": group_input.outputs["Num Columns"],
            "Base Material": group_input.outputs["Base Material"],
        },
        label="Base",
    )

    add_jointed_geometry_metadata = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": base, "Label": "drawer_base"},
    )

    drawer_door = nw.new_node(
        nodegroup_drawer_door().name,
        input_kwargs={
            "Size": group_input.outputs["Size"],
            "Thickness": group_input.outputs["Thickness"],
            "Base Offset": group_input.outputs["Bottom Offset"],
            "Num Rows": group_input.outputs["Num Rows"],
            "Num Columns": group_input.outputs["Num Columns"],
            "Drawer X Thickness": group_input.outputs["Drawer X Cut Scale"],
            "Drawer Y Thickness": group_input.outputs["Drawer Y Cut Scale"],
            "Handle Type": group_input.outputs["Handle Type"],
            "Handle Material": group_input.outputs["Handle Material"],
            "Drawer Material": group_input.outputs["Drawer Material"],
        },
        label="drawer_door",
    )

    add_jointed_geometry_metadata_001 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata_001().name,
        input_kwargs={"Geometry": drawer_door, "Label": "drawer_door"},
    )

    separate_xyz = nw.new_node(Nodes.SeparateXYZ, input_kwargs={"Vector": reroute_7})

    reroute = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Thickness"]}
    )

    reroute_1 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute})

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["X"], 1: reroute_1},
        attrs={"operation": "SUBTRACT"},
    )

    sliding_joint = nw.new_node(
        nodegroup_sliding_joint().name,
        input_kwargs={
            "Joint Label": "drawer_slider",
            "Parent": add_jointed_geometry_metadata,
            "Child": add_jointed_geometry_metadata_001,
            "Axis": (1.0000, 0.0000, 0.0000),
            "Max": subtract,
        },
    )

    reroute_6 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Bottom Offset"]}
    )

    subtract_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Z"], 1: reroute_6},
        attrs={"operation": "SUBTRACT"},
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_1, 1: 2.0000},
        attrs={"operation": "MULTIPLY"},
    )

    subtract_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: subtract_1, 1: multiply},
        attrs={"operation": "SUBTRACT"},
    )

    reroute_10 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_1})

    reroute_18 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_10})

    add = nw.new_node(Nodes.Math, input_kwargs={0: subtract_2, 1: reroute_18})

    subtract_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_9, 1: 1.0000},
        attrs={"operation": "SUBTRACT"},
    )

    reroute_15 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": subtract_3})

    reroute_17 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_15})

    reroute_2 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Num Rows"]}
    )

    reroute_3 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_2})

    divide = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_17, 1: reroute_3},
        attrs={"operation": "DIVIDE"},
    )

    reroute_22 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": divide})

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: add, 1: reroute_22},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_8, 1: 2.0000},
        attrs={"operation": "MULTIPLY"},
    )

    subtract_4 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Y"], 1: multiply_2},
        attrs={"operation": "SUBTRACT"},
    )

    add_1 = nw.new_node(Nodes.Math, input_kwargs={0: subtract_4, 1: reroute_10})

    subtract_5 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Num Columns"], 1: 1.0000},
        attrs={"operation": "SUBTRACT"},
    )

    reroute_4 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Num Columns"]}
    )

    reroute_5 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_4})

    divide_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: subtract_5, 1: reroute_5},
        attrs={"operation": "DIVIDE"},
    )

    reroute_16 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": divide_1})

    multiply_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: add_1, 1: reroute_16},
        attrs={"operation": "MULTIPLY"},
    )

    reroute_23 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": multiply_3})

    reroute_24 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_23})

    reroute_11 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_3})

    reroute_12 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_5})

    grid = nw.new_node(
        Nodes.MeshGrid,
        input_kwargs={
            "Size X": multiply_1,
            "Size Y": reroute_24,
            "Vertices X": reroute_11,
            "Vertices Y": reroute_12,
        },
    )

    divide_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_8, 1: 2.0000},
        attrs={"operation": "DIVIDE"},
    )

    reroute_13 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": divide_2})

    reroute_14 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_13})

    divide_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Bottom Offset"], 1: 2.0000},
        attrs={"operation": "DIVIDE"},
    )

    subtract_6 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: divide_3, 1: divide_2},
        attrs={"operation": "SUBTRACT"},
    )

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": reroute_14, "Z": subtract_6}
    )

    reroute_21 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": combine_xyz})

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": grid.outputs["Mesh"],
            "Translation": reroute_21,
            "Rotation": (0.0000, 1.5708, 0.0000),
        },
    )

    duplicate_joints_on_parent = nw.new_node(
        nodegroup_duplicate_joints_on_parent().name,
        input_kwargs={
            "Parent": sliding_joint.outputs["Parent"],
            "Child": sliding_joint.outputs["Child"],
            "Points": transform_geometry,
        },
    )

    reroute_25 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": duplicate_joints_on_parent}
    )

    bounding_box = nw.new_node(
        Nodes.BoundingBox, input_kwargs={"Geometry": duplicate_joints_on_parent}
    )

    separate_xyz_1 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": bounding_box.outputs["Min"]}
    )

    combine_xyz_1 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"Z": separate_xyz_1.outputs["Z"]}
    )

    subtract_7 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={1: combine_xyz_1},
        attrs={"operation": "SUBTRACT"},
    )

    transform_geometry_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": reroute_25,
            "Translation": subtract_7.outputs["Vector"],
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": transform_geometry_1},
        attrs={"is_active_output": True},
    )


class DrawerFactory(AssetFactory):
    def __init__(self, factory_seed=None, coarse=False):
        super().__init__(factory_seed=factory_seed, coarse=False)

    @classmethod
    @gin.configurable(module="DrawerFactory")
    def sample_joint_parameters(
        cls,
        drawer_slider_stiffness_min: float = 0.0,
        drawer_slider_stiffness_max: float = 0.0,
        drawer_slider_damping_min: float = 100.0,
        drawer_slider_damping_max: float = 200.0,
    ):
        return {
            "drawer_slider": {
                "stiffness": uniform(
                    drawer_slider_stiffness_min, drawer_slider_stiffness_max
                ),
                "damping": uniform(
                    drawer_slider_damping_min, drawer_slider_damping_max
                ),
                "friction": 1000.0,
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
            "Size": (uniform(0.4, 0.8), uniform(0.7, 1.5), uniform(0.6, 1.3)),
            "Thickness": uniform(0.02, 0.08),
            "Bottom Offset": uniform(0.0, 0.3),
            "Num Rows": randint(1, 4),
            "Num Columns": randint(1, 4),
            "Drawer X Cut Scale": uniform(0.6, 0.97),
            "Drawer Y Cut Scale": uniform(0.6, 0.97),
            "Handle Type": 0,
            "Handle Material": body_material,
            "Drawer Material": drawer_material,
            "Base Material": handle_material,
        }

    def create_asset(self, asset_params=None, **kwargs):
        obj = butil.spawn_vert()
        butil.modify_mesh(
            obj,
            "NODES",
            apply=False,
            node_group=geometry_nodes(),
            ng_inputs=self.sample_parameters(),
        )

        return obj
