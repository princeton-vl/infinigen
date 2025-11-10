from numpy.random import randint, uniform

from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
@node_utils.to_nodegroup(
)

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "Length", 0.0000),
            ("NodeSocketFloat", "Width", 0.0000),
            ("NodeSocketFloat", "Height", 0.0000),
            ("NodeSocketFloat", "joint_radius", 0.0000),
            ("NodeSocketFloat", "handle_head_length_ratio", 0.0000),
            ("NodeSocketFloat", "Handle_curvature", 0.0000),
            ("NodeSocketFloat", "handle_thickness", 0.0000),
            ("NodeSocketFloat", "handle shape", 0.0000),
            ("NodeSocketInt", "handle_patterned", 0),
            ("NodeSocketMaterial", "handle_material", None),
            ("NodeSocketMaterial", "metal_material", None),
            ("NodeSocketFloat", "Value", 0.1000),
        ],
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input.outputs["Length"],
            1: group_input.outputs["handle_head_length_ratio"],
        },
        attrs={"operation": "MULTIPLY"},
    )

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply, 1: -1.0000},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_2 = nw.new_node(
        Nodes.Math,
        attrs={"operation": "MULTIPLY"},
    )

    equal = nw.new_node(
        Nodes.Compare,
        attrs={"operation": "EQUAL", "data_type": "INT"},
    )

    equal_1 = nw.new_node(
        Nodes.Compare,
        attrs={"operation": "EQUAL", "data_type": "INT"},
    )



    switch_1 = nw.new_node(
        Nodes.Switch,
        input_kwargs={"Switch": equal_1, "False": value_1, "True": value_2},
        attrs={"input_type": "FLOAT"},
    )


    switch = nw.new_node(
        Nodes.Switch,
        input_kwargs={"Switch": equal, "False": switch_1, "True": value},
        attrs={"input_type": "FLOAT"},
    )

    multiply_3 = nw.new_node(
        Nodes.Math,
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz = nw.new_node(
    )

    multiply_4 = nw.new_node(
    )

    combine_xyz_1 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": multiply_1, "Y": multiply_4}
    )

    quadratic_b_zier = nw.new_node(
        Nodes.QuadraticBezier,
    )

    )

    trim_curve = nw.new_node(
        Nodes.TrimCurve, input_kwargs={"Curve": quadratic_b_zier, 2: add}
    )

    multiply_5 = nw.new_node(
    )

    curve_circle = nw.new_node(Nodes.CurveCircle, input_kwargs={"Radius": multiply_5})

    divide = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input.outputs["joint_radius"],
            1: group_input.outputs["Height"],
        },
        attrs={"operation": "DIVIDE"},
    )

    multiply_6 = nw.new_node(
        Nodes.Math, input_kwargs={0: divide, 1: 1.5000}, attrs={"operation": "MULTIPLY"}
    )

    minimum = nw.new_node(
        Nodes.Math,
        attrs={"operation": "MINIMUM"},
    )

    combine_xyz_5 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": minimum, "Y": 1.0000, "Z": 1.0000}
    )

    transform_geometry_2 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": curve_circle.outputs["Curve"],
            "Scale": combine_xyz_5,
        },
    )

    curve_to_mesh = nw.new_node(
        Nodes.CurveToMesh,
        input_kwargs={
            "Curve": trim_curve,
            "Fill Caps": True,
        },
    )

    set_material = nw.new_node(
        Nodes.SetMaterial,
    )

    trim_curve_1 = nw.new_node(
        Nodes.TrimCurve, input_kwargs={"Curve": quadratic_b_zier, 3: add}
    )

    multiply_7 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply_5, 1: 0.8000},
        attrs={"operation": "MULTIPLY"},
    )

    curve_circle_1 = nw.new_node(
        Nodes.CurveCircle, input_kwargs={"Resolution": 4, "Radius": multiply_7}
    )

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": curve_circle_1.outputs["Curve"],
            "Rotation": (0.0000, 0.0000, 0.7854),
        },
    )

    transform_geometry_3 = nw.new_node(
        Nodes.Transform,
    )

    curve_to_mesh_1 = nw.new_node(
        Nodes.CurveToMesh,
        input_kwargs={
            "Curve": trim_curve_1,
            "Profile Curve": transform_geometry_3,
            "Fill Caps": True,
        },
    )

    set_material_1 = nw.new_node(
        Nodes.SetMaterial,
    )

    join_geometry = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [set_material, set_material_1]}
    )

    divide_1 = nw.new_node(
        Nodes.Math,
        attrs={"operation": "DIVIDE"},
    )

    combine_xyz_4 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Y": divide_1})

    transform_geometry_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": join_geometry,
            "Translation": combine_xyz_4,
            "Rotation": (0.0000, 0.0000, -0.1000),
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": transform_geometry_1},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
)
    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "Length", 0.5000),
            ("NodeSocketInt", "jagged_edge", 0),
            ("NodeSocketFloat", "Width", 0.5000),
            ("NodeSocketFloat", "Height", 0.0000),
            ("NodeSocketInt", "head_type", 0),
            ("NodeSocketFloat", "head_width", 0.0000),
            ("NodeSocketFloat", "handle_head_length_ratio", 0.0000),
            ("NodeSocketFloat", "joint_radius", 0.0000),
            ("NodeSocketMaterial", "Material", None),
            ("NodeSocketFloat", "head curvature", 0.0150),
            ("NodeSocketFloat", "number of jagged edges", 30.0000),
        ],
    )

    equal = nw.new_node(
        Nodes.Compare,
        attrs={"operation": "EQUAL", "data_type": "INT"},
    )

    subtract = nw.new_node(
        Nodes.Math,
        attrs={"operation": "SUBTRACT"},
    )

    multiply = nw.new_node(
        Nodes.Math,
        attrs={"operation": "MULTIPLY"},
    )


    quadratic_b_zier_4 = nw.new_node(
        Nodes.QuadraticBezier,
        input_kwargs={
            "Start": (0.0000, 0.0000, 0.0000),
            "Middle": (0.0010, 0.0000, 0.0000),
        },
    )

    )

    )

    multiply_1 = nw.new_node(
        attrs={"operation": "MULTIPLY"},
    )

        Nodes.VectorMath,
    )

        Nodes.QuadraticBezier,
        input_kwargs={
        },
    )

    )

    )

    multiply_3 = nw.new_node(
        Nodes.VectorMath,
        attrs={"operation": "MULTIPLY"},
    )

    separate_xyz = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": add.outputs["Vector"]}
    )

    multiply_4 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["X"], 1: 0.7713},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_5 = nw.new_node(
        Nodes.Math,
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_11 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": multiply_4, "Y": multiply_5}
    )

    quadratic_b_zier_6 = nw.new_node(
        Nodes.QuadraticBezier,
        input_kwargs={
            "Middle": combine_xyz_11,
        },
    )

    )

        Nodes.QuadraticBezier,
        input_kwargs={
        },
    )

    )

    join_geometry_4 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={
            "Geometry": [
            ]
        },
    )

    )

    fill_curve_1 = nw.new_node(
    )


    multiply_6 = nw.new_node(
        Nodes.Math,
        attrs={"operation": "MULTIPLY"},
    )

    extrude_mesh_3 = nw.new_node(
        Nodes.ExtrudeMesh,
        input_kwargs={
            "Mesh": fill_curve_1,
            "Offset Scale": multiply_6,
            "Individual": False,
        },
    )

    cube = nw.new_node(Nodes.MeshCube, input_kwargs={"Size": (0.0500, 0.0150, 0.3000)})

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cube.outputs["Mesh"],
            "Translation": (0.0000, 0.0075, 0.0000),
            "Rotation": (3.1416, 1.5708, 0.0000),
        },
    )

    combine_xyz_14 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": 0.5530})

    transform_geometry_5 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": transform_geometry, "Rotation": combine_xyz_14},
    )

    difference = nw.new_node(
        Nodes.MeshBoolean,
        input_kwargs={
        },
    )

        Nodes.JoinGeometry,
    )

    )

        Nodes.Math,
        attrs={"operation": "MULTIPLY"},
    )


    transform_geometry_4 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Translation": combine_xyz_9,
            "Rotation": (0.0000, 0.0000, -0.1000),
        },
    )

    )

        Nodes.VectorMath,
        attrs={"operation": "MULTIPLY"},
    )

    )

        Nodes.QuadraticBezier,
        input_kwargs={
        },
    )

    )

    equal_1 = nw.new_node(
        Nodes.Compare,
        attrs={"operation": "EQUAL", "data_type": "INT"},
    )

    quadratic_b_zier = nw.new_node(
        Nodes.QuadraticBezier,
        input_kwargs={
            "Start": (0.0000, 0.0000, 0.0000),
            "Middle": (0.0010, 0.0000, 0.0000),
        },
    )

    curve_to_points = nw.new_node(
        Nodes.CurveToPoints,
        input_kwargs={"Curve": quadratic_b_zier, "Length": 0.0050},
        attrs={"mode": "LENGTH"},
    )

    curve_line_1 = nw.new_node(
        Nodes.CurveLine,
        input_kwargs={"Start": combine_xyz_2, "End": (0.0000, 0.0000, 0.0000)},
    )

    curve_to_points_8 = nw.new_node(
    )


    modulo = nw.new_node(
        Nodes.Math, input_kwargs={0: index_1, 1: 2.0000}, attrs={"operation": "MODULO"}
    )

    set_position_1 = nw.new_node(
        Nodes.SetPosition,
        input_kwargs={
            "Geometry": curve_to_points_8.outputs["Points"],
            "Selection": modulo,
            "Offset": (0.0000, -0.0010, 0.0000),
        },
    )

    points_to_curves_2 = nw.new_node(
        "GeometryNodePointsToCurves", input_kwargs={"Points": set_position_1}
    )

    curve_to_points_9 = nw.new_node(
        Nodes.CurveToPoints,
    )

    switch_3 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": equal_1,
            "False": curve_to_points.outputs["Points"],
            "True": curve_to_points_9.outputs["Points"],
        },
    )

    )


    )

    )

    )

    add_2 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: add_1.outputs["Vector"], 1: multiply_10.outputs["Vector"]},
    )

    divide = nw.new_node(
        Nodes.VectorMath,
        attrs={"operation": "DIVIDE"},
    )

    cross_product = nw.new_node(
        Nodes.VectorMath,
        attrs={"operation": "CROSS_PRODUCT"},
    )

    normalize = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: cross_product.outputs["Vector"]},
        attrs={"operation": "NORMALIZE"},
    )

    )

    combine_xyz_3 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": add_3, "Y": add_3, "Z": add_3}
    )

        Nodes.VectorMath,
        attrs={"operation": "MULTIPLY"},
    )

    divide_1 = nw.new_node(
        Nodes.VectorMath,
        attrs={"operation": "DIVIDE"},
    )

    add_4 = nw.new_node(
        Nodes.VectorMath,
    )

    quadratic_b_zier_2 = nw.new_node(
        Nodes.QuadraticBezier,
        input_kwargs={
            "Middle": add_4.outputs["Vector"],
            "End": add_1.outputs["Vector"],
        },
    )

    )

        Nodes.JoinGeometry,
        input_kwargs={
            "Geometry": [
            ]
        },
    )

    )

    fill_curve = nw.new_node(
    )


        Nodes.Math,
        attrs={"operation": "MULTIPLY"},
    )

    extrude_mesh_1 = nw.new_node(
        Nodes.ExtrudeMesh,
    )

        Nodes.Math,
        attrs={"operation": "MULTIPLY"},
    )


    transform_geometry_3 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Translation": combine_xyz_4,
            "Rotation": (0.0000, 0.0000, -0.1000),
        },
    )

    switch = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": equal,
            "False": transform_geometry_4,
        },
    )

    set_material = nw.new_node(
    )

    curve_line_2 = nw.new_node(
        Nodes.CurveLine,
        input_kwargs={"Start": combine_xyz_2, "End": (0.0000, 0.0000, 0.0000)},
    )

    curve_to_points_11 = nw.new_node(
    )


    modulo_1 = nw.new_node(
        Nodes.Math, input_kwargs={0: index_2, 1: 2.0000}, attrs={"operation": "MODULO"}
    )

    set_position_2 = nw.new_node(
        Nodes.SetPosition,
        input_kwargs={
            "Geometry": curve_to_points_11.outputs["Points"],
            "Selection": modulo_1,
            "Offset": (0.0000, 0.0010, 0.0000),
        },
    )

    points_to_curves_3 = nw.new_node(
        "GeometryNodePointsToCurves", input_kwargs={"Points": set_position_2}
    )

    )

        Nodes.JoinGeometry,
        input_kwargs={
            "Geometry": [
            ]
        },
    )

    )

    fill_curve_2 = nw.new_node(
        Nodes.FillCurve,
        attrs={"mode": "NGONS"},
    )

        Nodes.Math,
        attrs={"operation": "MULTIPLY"},
    )

    extrude_mesh_4 = nw.new_node(
        Nodes.ExtrudeMesh,
    )

    transform_geometry_7 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Translation": combine_xyz_4,
            "Rotation": (0.0000, 0.0000, -0.1000),
        },
    )

    set_material_1 = nw.new_node(
        Nodes.SetMaterial,
    )

    group_output_1 = nw.new_node(
        Nodes.GroupOutput,
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
)
    group_input_1 = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "Height", 0.0000),
            ("NodeSocketFloat", "joint_radius", 0.0000),
            ("NodeSocketMaterial", "Material", None),
            ("NodeSocketFloat", "Value", 0.5000),
        ],
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input_1.outputs["Value"], 1: 0.4000},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_1 = nw.new_node(
        Nodes.Math,
        attrs={"operation": "MULTIPLY"},
    )

    maximum = nw.new_node(
        Nodes.Math,
        attrs={"operation": "MAXIMUM"},
    )

    multiply_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input_1.outputs["Height"]},
        attrs={"operation": "MULTIPLY"},
    )

    cylinder = nw.new_node(
        "GeometryNodeMeshCylinder",
        input_kwargs={"Radius": maximum, "Depth": multiply_2},
    )

    multiply_3 = nw.new_node(
        Nodes.Math, input_kwargs={0: multiply_2}, attrs={"operation": "MULTIPLY"}
    )

    combine_xyz = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": multiply_3})

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": cylinder.outputs["Mesh"], "Translation": combine_xyz},
    )

    set_material = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": transform_geometry,
            "Material": group_input_1.outputs["Material"],
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": set_material},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup("geometry_nodes", singleton=False, type="GeometryNodeTree")
    group_input_1 = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "Length", 0.0000),
            ("NodeSocketFloat", "Width", 0.0000),
            ("NodeSocketFloat", "Height", 0.0100),
            ("NodeSocketInt", "head_type", 0),
            ("NodeSocketFloat", "head_width", 0.7500),
            ("NodeSocketFloat", "handle_curvature", 0.0000),
            ("NodeSocketFloat", "handle_thickness", 1.2000),
            ("NodeSocketFloat", "joint_radius", 0.0100),
            ("NodeSocketMaterial", "head_and_joint_material", None),
            ("NodeSocketMaterial", "handle_material", None),
            ("NodeSocketFloat", "Value", 0.0000),
            ("NodeSocketFloat", "head curvature", 0.0150),
            ("NodeSocketInt", "jagged_head", 0),
            ("NodeSocketInt", "number of jagged edges", 30),
            ("NodeSocketInt", "handle_roundness", 0),
            ("NodeSocketFloat", "plastic handle cover length", 0.1000),
            ("NodeSocketFloat", "cut length ratio", 0.7300),
            ("NodeSocketFloat", "pincer length ratio", 0.7500),
        ],
    )

    joint_details = nw.new_node(
        input_kwargs={
            "Height": group_input_1.outputs["Height"],
            "joint_radius": group_input_1.outputs["joint_radius"],
            "Material": group_input_1.outputs["head_and_joint_material"],
            "Value": group_input_1.outputs["handle_thickness"],
        },
        label="joint_details",
    )

    equal = nw.new_node(
        Nodes.Compare,
        attrs={"operation": "EQUAL", "data_type": "INT"},
    )

    multiply = nw.new_node(
        Nodes.Math,
        attrs={"operation": "MULTIPLY"},
    )

    equal_1 = nw.new_node(
        Nodes.Compare,
        attrs={"operation": "EQUAL", "data_type": "INT"},
    )

    multiply_1 = nw.new_node(
        Nodes.Math,
        attrs={"operation": "MULTIPLY"},
    )


    plier_head = nw.new_node(
        input_kwargs={
            "handle_head_length_ratio": add,
        },
        label="Plier_head",
    )

    plier_handle = nw.new_node(
        input_kwargs={
            "handle_head_length_ratio": add,
            "handle_patterned": 1,
        },
        label="Plier_handle",
    )

    join_geometry_1 = nw.new_node(
        Nodes.JoinGeometry,
    )


    )

    transform_geometry = nw.new_node(
        Nodes.Transform,
    )

    equal_2 = nw.new_node(
        Nodes.Compare,
        attrs={"operation": "EQUAL", "data_type": "INT"},
    )

    equal_3 = nw.new_node(
        Nodes.Compare,
        attrs={"operation": "EQUAL", "data_type": "INT"},
    )


    transform_geometry_2 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Scale": (1.0000, -1.0000, 1.0000),
        },
    )

    transform_geometry_3 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": plier_head.outputs["jagged_opposite"],
            "Scale": (1.0000, -1.0000, 1.0000),
        },
    )

    switch = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": op_and,
            "True": transform_geometry_3,
        },
    )


    join_geometry = nw.new_node(
        Nodes.JoinGeometry,
    )

        input_kwargs={
            "Geometry": join_geometry,
            "Label": "the plier arm attached to the hinge",
        },
    )

    bounding_box = nw.new_node(
        Nodes.BoundingBox, input_kwargs={"Geometry": join_geometry_1}
    )

    add_1 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: bounding_box.outputs["Min"], 1: bounding_box.outputs["Max"]},
    )

    scale = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: add_1.outputs["Vector"], "Scale": -0.5000},
        attrs={"operation": "SCALE"},
    )

    hinge_joint = nw.new_node(
        nodegroup_hinge_joint().name,
        input_kwargs={
            "Position": scale.outputs["Vector"],
        },
    )

    divide = nw.new_node(
        Nodes.Math,
        attrs={"operation": "DIVIDE"},
    )

    combine_xyz = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": divide})

    transform_geometry_4 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": hinge_joint.outputs["Geometry"],
            "Translation": combine_xyz,
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": transform_geometry_4},
        attrs={"is_active_output": True},
    )

class PlierFactory(AssetFactory):
    @gin.configurable(module="PlierFactory")
                "stiffness": uniform(
                ),
            },
        }

        handle_mat = sample_handle_mat()  # np.random.choice([plastic.Plastic, metal.MetalBasic, plastic.BlackPlastic], p=[0.7, 0.2, 0.1])()
            "handle_material": handle_mat,
            "cut length ratio": uniform(0.65, 0.83),
            "pincer length ratio": uniform(0.6, 0.75),
        print(
            f"=======================================================\nDictionary of configs: {return_dict}\n\n"
        )
    def create_asset(self, asset_params=None, **kwargs):
            ng_inputs=self.sample_parameters(),
        return obj
