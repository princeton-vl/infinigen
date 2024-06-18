
from infinigen.assets.materials import metal

            "RackAmount": rack_h_amount,

        butil.modify_mesh(obj, 'NODES', node_group=nodegroup_dishwasher_geometry(), ng_inputs=self.params, apply=True)

    def finalize_assets(self, assets):
        if self.scratch:
            self.scratch.apply(assets)
        if self.edge_wear:
            self.edge_wear.apply(assets)

    curve_line = nw.new_node(Nodes.CurveLine,
                             input_kwargs={'Start': (0.0000, -1.0000, 0.0000), 'End': (0.0000, 1.0000, 0.0000)})

    group_input = nw.new_node(Nodes.GroupInput, expose_input=[('NodeSocketFloatDistance', 'Depth', 2.0000),
        ('NodeSocketFloatDistance', 'Width', 2.0000), ('NodeSocketFloatDistance', 'Radius', 0.0200),
        ('NodeSocketInt', 'Amount', 5), ('NodeSocketFloat', 'Height', 0.5000)])

    combine_xyz_4 = nw.new_node(Nodes.CombineXYZ,
                                input_kwargs={'Y': -1.0000, 'Z': group_input.outputs["Height"]})

    curve_line_1 = nw.new_node(Nodes.CurveLine,
                               input_kwargs={'Start': (0.0000, -1.0000, 0.0000), 'End': combine_xyz_4})

    geometry_to_instance_1 = nw.new_node('GeometryNodeGeometryToInstance',
                                         input_kwargs={'Geometry': curve_line_1})

    multiply = nw.new_node(Nodes.Math, input_kwargs={0: group_input.outputs["Amount"], 1: 2.0000},
                           attrs={'operation': 'MULTIPLY'})

                                       input_kwargs={'Geometry': geometry_to_instance_1, 'Amount': multiply},
                                       attrs={'domain': 'INSTANCE'})

    divide = nw.new_node(Nodes.Math, input_kwargs={0: 1.0000, 1: group_input.outputs["Amount"]},
                         attrs={'operation': 'DIVIDE'})

                             input_kwargs={0: duplicate_elements_2.outputs["Duplicate Index"], 1: divide},
                             attrs={'operation': 'MULTIPLY'})


    set_position_2 = nw.new_node(Nodes.SetPosition, input_kwargs={
        'Geometry': duplicate_elements_2.outputs["Geometry"],
        'Offset': combine_xyz_3
    })


    geometry_to_instance = nw.new_node('GeometryNodeGeometryToInstance',
                                       input_kwargs={'Geometry': join_geometry_1})

                                     input_kwargs={'Geometry': geometry_to_instance, 'Amount': multiply},
                                     attrs={'domain': 'INSTANCE'})

    subtract = nw.new_node(Nodes.Math, input_kwargs={
        0: duplicate_elements.outputs["Duplicate Index"],
        1: group_input.outputs["Amount"]
    }, attrs={'operation': 'SUBTRACT'})



    set_position = nw.new_node(Nodes.SetPosition, input_kwargs={
        'Geometry': duplicate_elements.outputs["Geometry"],
        'Offset': combine_xyz
    })

    transform_1 = nw.new_node(Nodes.Transform,
                              input_kwargs={'Geometry': set_position, 'Rotation': (0.0000, 0.0000, 1.5708)})

                                       input_kwargs={'Geometry': geometry_to_instance, 'Amount': multiply},
                                       attrs={'domain': 'INSTANCE'})

    subtract_1 = nw.new_node(Nodes.Math, input_kwargs={
        0: duplicate_elements_1.outputs["Duplicate Index"],
        1: group_input.outputs["Amount"]
    }, attrs={'operation': 'SUBTRACT'})

    multiply_3 = nw.new_node(Nodes.Math, input_kwargs={0: subtract_1, 1: divide},
                             attrs={'operation': 'MULTIPLY'})


    set_position_1 = nw.new_node(Nodes.SetPosition, input_kwargs={
        'Geometry': duplicate_elements_1.outputs["Geometry"],
        'Offset': combine_xyz_1
    })


    multiply_4 = nw.new_node(Nodes.Math, input_kwargs={0: group_input.outputs["Height"], 1: 0.8000},
                             attrs={'operation': 'MULTIPLY'})


    transform_2 = nw.new_node(Nodes.Transform,
                              input_kwargs={'Geometry': quadrilateral_1, 'Translation': combine_xyz_5})

    join_geometry = nw.new_node(Nodes.JoinGeometry, input_kwargs={
        'Geometry': [quadrilateral, transform_1, set_position_1, transform_2]
    })


    curve_to_mesh = nw.new_node(Nodes.CurveToMesh, input_kwargs={
        'Curve': join_geometry,
        'Profile Curve': curve_circle.outputs["Curve"],
        'Fill Caps': True
    })

    multiply_5 = nw.new_node(Nodes.Math, input_kwargs={0: group_input.outputs["Depth"]},
                             attrs={'operation': 'MULTIPLY'})

    multiply_6 = nw.new_node(Nodes.Math, input_kwargs={0: group_input.outputs["Width"]},
                             attrs={'operation': 'MULTIPLY'})


    transform = nw.new_node(Nodes.Transform, input_kwargs={
        'Geometry': curve_to_mesh,
        'Rotation': (0.0000, 0.0000, 1.5708),
        'Scale': combine_xyz_2
    })

    group_output = nw.new_node(Nodes.GroupOutput, input_kwargs={'Mesh': transform},
                               attrs={'is_active_output': True})

    group_input = nw.new_node(Nodes.GroupInput, expose_input=[
        ('NodeSocketVectorTranslation', 'Translation', (1.5000, 0.0000, 0.0000)),
        ('NodeSocketString', 'String', 'BrandName'), ('NodeSocketFloatDistance', 'Size', 0.0500),
        ('NodeSocketFloat', 'Offset Scale', 0.0020)])

    string_to_curves = nw.new_node('GeometryNodeStringToCurves', input_kwargs={
        'String': group_input.outputs["String"],
        'Size': group_input.outputs["Size"]
    }, attrs={'align_y': 'BOTTOM_BASELINE', 'align_x': 'CENTER'})

    fill_curve = nw.new_node(Nodes.FillCurve,
                             input_kwargs={'Curve': string_to_curves.outputs["Curve Instances"]})

    extrude_mesh = nw.new_node(Nodes.ExtrudeMesh, input_kwargs={
        'Mesh': fill_curve,
        'Offset Scale': group_input.outputs["Offset Scale"]
    })

    transform_1 = nw.new_node(Nodes.Transform, input_kwargs={
        'Geometry': extrude_mesh.outputs["Mesh"],
        'Translation': group_input.outputs["Translation"],
        'Rotation': (1.5708, 0.0000, 1.5708)
    })

    group_output = nw.new_node(Nodes.GroupOutput, input_kwargs={'Geometry': transform_1},
                               attrs={'is_active_output': True})

    group_input = nw.new_node(Nodes.GroupInput, expose_input=[('NodeSocketFloat', 'width', 0.0000),
        ('NodeSocketFloat', 'length', 0.0000), ('NodeSocketFloat', 'thickness', 0.0200)])


    store_named_attribute = nw.new_node(Nodes.StoreNamedAttribute, input_kwargs={
        'Geometry': cube.outputs["Mesh"],
        'Name': 'uv_map',
        3: cube.outputs["UV Map"]
    }, attrs={'domain': 'CORNER', 'data_type': 'FLOAT_VECTOR'})


    store_named_attribute_1 = nw.new_node(Nodes.StoreNamedAttribute, input_kwargs={
        'Geometry': cube_1.outputs["Mesh"],
        'Name': 'uv_map',
        3: cube_1.outputs["UV Map"]
    }, attrs={'domain': 'CORNER', 'data_type': 'FLOAT_VECTOR'})


    transform = nw.new_node(Nodes.Transform,
                            input_kwargs={'Geometry': store_named_attribute_1, 'Translation': combine_xyz})

    join_geometry_1 = nw.new_node(Nodes.JoinGeometry,
                                  input_kwargs={'Geometry': [store_named_attribute, transform]})

    multiply = nw.new_node(Nodes.Math, input_kwargs={0: group_input.outputs["width"]},
                           attrs={'operation': 'MULTIPLY'})


    transform_2 = nw.new_node(Nodes.Transform,
                              input_kwargs={'Geometry': join_geometry_1, 'Translation': combine_xyz_3})

    add = nw.new_node(Nodes.Math,
                      input_kwargs={0: group_input.outputs["length"], 1: group_input.outputs["width"]})

    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ, input_kwargs={
        'X': group_input.outputs["width"],
        'Y': add,
        'Z': group_input.outputs["thickness"]
    })


    store_named_attribute_2 = nw.new_node(Nodes.StoreNamedAttribute, input_kwargs={
        'Geometry': cube_2.outputs["Mesh"],
        'Name': 'uv_map',
        3: cube_2.outputs["UV Map"]
    }, attrs={'domain': 'CORNER', 'data_type': 'FLOAT_VECTOR'})

    multiply_1 = nw.new_node(Nodes.Math, input_kwargs={0: group_input.outputs["length"]},
                             attrs={'operation': 'MULTIPLY'})

    multiply_2 = nw.new_node(Nodes.Math, input_kwargs={0: group_input.outputs["thickness"]},
                             attrs={'operation': 'MULTIPLY'})



    transform_1 = nw.new_node(Nodes.Transform,
                              input_kwargs={'Geometry': store_named_attribute_2, 'Translation': combine_xyz_2})


    group_output = nw.new_node(Nodes.GroupOutput, input_kwargs={'Geometry': join_geometry},
                               attrs={'is_active_output': True})

    group_input = nw.new_node(Nodes.GroupInput, expose_input=[('NodeSocketGeometry', 'Geometry', None),
        ('NodeSocketVector', 'Vector', (0.0000, 0.0000, 0.0000)), ('NodeSocketFloat', 'MarginX', 0.5000),
        ('NodeSocketFloat', 'MarginY', 0.0000), ('NodeSocketFloat', 'MarginZ', 0.0000)])


                           input_kwargs={0: group_input.outputs["Vector"], 1: bounding_box.outputs["Min"]},
                           attrs={'operation': 'SUBTRACT'})


                               input_kwargs={0: separate_xyz.outputs["X"], 1: group_input.outputs["MarginX"]},
                               attrs={'operation': 'GREATER_THAN', 'use_clamp': True})

                             input_kwargs={0: bounding_box.outputs["Max"], 1: group_input.outputs["Vector"]},
                             attrs={'operation': 'SUBTRACT'})


    greater_than_1 = nw.new_node(Nodes.Math, input_kwargs={
        0: separate_xyz_1.outputs["X"],
        1: group_input.outputs["MarginX"]
    }, attrs={'operation': 'GREATER_THAN', 'use_clamp': True})


                                 input_kwargs={0: separate_xyz.outputs["Y"], 1: group_input.outputs["MarginY"]},
                                 attrs={'operation': 'GREATER_THAN'})

    greater_than_3 = nw.new_node(Nodes.Math, input_kwargs={
        0: separate_xyz_1.outputs["Y"],
        1: group_input.outputs["MarginY"]
    }, attrs={'operation': 'GREATER_THAN', 'use_clamp': True})



                                 input_kwargs={0: separate_xyz.outputs["Z"], 1: group_input.outputs["MarginZ"]},
                                 attrs={'operation': 'GREATER_THAN', 'use_clamp': True})

    greater_than_5 = nw.new_node(Nodes.Math, input_kwargs={
        0: separate_xyz_1.outputs["Z"],
        1: group_input.outputs["MarginZ"]
    }, attrs={'operation': 'GREATER_THAN', 'use_clamp': True})




    group_output = nw.new_node(Nodes.GroupOutput, input_kwargs={'In': op_and_4, 'Out': op_not},
                               attrs={'is_active_output': True})


    cube = nw.new_node(Nodes.MeshCube, input_kwargs={
        'Size': group_input.outputs["Size"],
        'Vertices X': group_input.outputs["Resolution"],
        'Vertices Y': group_input.outputs["Resolution"],
        'Vertices Z': group_input.outputs["Resolution"]
    })

    store_named_attribute_1 = nw.new_node(Nodes.StoreNamedAttribute, input_kwargs={
        'Geometry': cube.outputs["Mesh"],
        'Name': 'uv_map',
        3: cube.outputs["UV Map"]
    }, attrs={'domain': 'CORNER', 'data_type': 'FLOAT_VECTOR'})

                                        input_kwargs={'Geometry': store_named_attribute_1, 'Name': 'uv_map'},
                                        attrs={'domain': 'CORNER', 'data_type': 'FLOAT_VECTOR'})

    multiply_add = nw.new_node(Nodes.VectorMath, input_kwargs={
        0: group_input.outputs["Size"],
        1: (0.5000, 0.5000, 0.5000),
        2: group_input.outputs["Pos"]
    }, attrs={'operation': 'MULTIPLY_ADD'})

    transform = nw.new_node(Nodes.Transform, input_kwargs={
        'Geometry': store_named_attribute,
        'Translation': multiply_add.outputs["Vector"]
    })

    group_output = nw.new_node(Nodes.GroupOutput, input_kwargs={'Geometry': transform},
                               attrs={'is_active_output': True})



    multiply = nw.new_node(Nodes.Math, input_kwargs={0: group_input.outputs["Thickness"], 1: 2.0000},
                           attrs={'operation': 'MULTIPLY'})

    subtract = nw.new_node(Nodes.Math, input_kwargs={0: separate_xyz.outputs["Y"], 1: multiply},
                           attrs={'operation': 'SUBTRACT'})

    subtract_1 = nw.new_node(Nodes.Math, input_kwargs={0: separate_xyz.outputs["Z"], 1: multiply},
                             attrs={'operation': 'SUBTRACT'})

    combine_xyz_4 = nw.new_node(Nodes.CombineXYZ, input_kwargs={
        'X': group_input.outputs["Thickness"],
        'Y': subtract,
        'Z': subtract_1
    })


    store_named_attribute_1 = nw.new_node(Nodes.StoreNamedAttribute, input_kwargs={
        'Geometry': cube_2.outputs["Mesh"],
        'Name': 'uv_map',
        3: cube_2.outputs["UV Map"]
    }, attrs={'domain': 'CORNER', 'data_type': 'FLOAT_VECTOR'})

    multiply_1 = nw.new_node(Nodes.Math, input_kwargs={0: group_input.outputs["Thickness"]},
                             attrs={'operation': 'MULTIPLY'})



    scale = nw.new_node(Nodes.VectorMath, input_kwargs={0: group_input.outputs["Size"], 'Scale': 0.5000},
                        attrs={'operation': 'SCALE'})


    add_1 = nw.new_node(Nodes.Math,
                        input_kwargs={0: separate_xyz_2.outputs["Y"], 1: separate_xyz_1.outputs["Y"]})

                             input_kwargs={0: separate_xyz_2.outputs["Z"], 1: separate_xyz_1.outputs["Z"]},
                             attrs={'operation': 'SUBTRACT'})


    transform_2 = nw.new_node(Nodes.Transform,
                              input_kwargs={'Geometry': store_named_attribute_1, 'Translation': combine_xyz_5})


    subtract_3 = nw.new_node(Nodes.Math, input_kwargs={0: separate_xyz.outputs["Y"], 1: multiply},
                             attrs={'operation': 'SUBTRACT'})

    combine_xyz_2 = nw.new_node(Nodes.CombineXYZ, input_kwargs={
        'X': separate_xyz.outputs["X"],
        'Y': subtract_3,
        'Z': group_input.outputs["Thickness"]
    })


    store_named_attribute_4 = nw.new_node(Nodes.StoreNamedAttribute, input_kwargs={
        'Geometry': cube_1.outputs["Mesh"],
        'Name': 'uv_map',
        3: cube_1.outputs["UV Map"]
    }, attrs={'domain': 'CORNER', 'data_type': 'FLOAT_VECTOR'})

    add_2 = nw.new_node(Nodes.Math,
                        input_kwargs={0: separate_xyz_2.outputs["X"], 1: separate_xyz_1.outputs["X"]})

    add_3 = nw.new_node(Nodes.Math,
                        input_kwargs={0: separate_xyz_2.outputs["Y"], 1: separate_xyz_1.outputs["Y"]})

    subtract_4 = nw.new_node(Nodes.Math, input_kwargs={0: separate_xyz.outputs["Z"], 1: multiply_1},
                             attrs={'operation': 'SUBTRACT'})


    transform_1 = nw.new_node(Nodes.Transform,
                              input_kwargs={'Geometry': store_named_attribute_4, 'Translation': combine_xyz_3})


    subtract_5 = nw.new_node(Nodes.Math, input_kwargs={0: separate_xyz.outputs["Y"], 1: multiply},
                             attrs={'operation': 'SUBTRACT'})

    combine_xyz = nw.new_node(Nodes.CombineXYZ, input_kwargs={
        'X': separate_xyz.outputs["X"],
        'Y': subtract_5,
        'Z': group_input.outputs["Thickness"]
    })


    store_named_attribute = nw.new_node(Nodes.StoreNamedAttribute, input_kwargs={
        'Geometry': cube.outputs["Mesh"],
        'Name': 'uv_map',
        3: cube.outputs["UV Map"]
    }, attrs={'domain': 'CORNER', 'data_type': 'FLOAT_VECTOR'})

    add_4 = nw.new_node(Nodes.Math,
                        input_kwargs={0: separate_xyz_2.outputs["X"], 1: separate_xyz_1.outputs["X"]})

    add_5 = nw.new_node(Nodes.Math,
                        input_kwargs={0: separate_xyz_2.outputs["Y"], 1: separate_xyz_1.outputs["Y"]})



    transform = nw.new_node(Nodes.Transform,
                            input_kwargs={'Geometry': store_named_attribute, 'Translation': combine_xyz_1})


    subtract_6 = nw.new_node(Nodes.Math, input_kwargs={0: separate_xyz.outputs["Y"], 1: multiply},
                             attrs={'operation': 'SUBTRACT'})

    subtract_7 = nw.new_node(Nodes.Math, input_kwargs={0: separate_xyz.outputs["Z"], 1: multiply},
                             attrs={'operation': 'SUBTRACT'})

    combine_xyz_6 = nw.new_node(Nodes.CombineXYZ, input_kwargs={
        'X': group_input.outputs["Thickness"],
        'Y': subtract_6,
        'Z': subtract_7
    })


    store_named_attribute_5 = nw.new_node(Nodes.StoreNamedAttribute, input_kwargs={
        'Geometry': cube_3.outputs["Mesh"],
        'Name': 'uv_map',
        3: cube_3.outputs["UV Map"]
    }, attrs={'domain': 'CORNER', 'data_type': 'FLOAT_VECTOR'})

    subtract_8 = nw.new_node(Nodes.Math, input_kwargs={0: separate_xyz.outputs["X"], 1: multiply_1},
                             attrs={'operation': 'SUBTRACT'})

    add_7 = nw.new_node(Nodes.Math,
                        input_kwargs={0: separate_xyz_2.outputs["Y"], 1: separate_xyz_1.outputs["Y"]})

                             input_kwargs={0: separate_xyz_2.outputs["Z"], 1: separate_xyz_1.outputs["Z"]},
                             attrs={'operation': 'SUBTRACT'})


    transform_3 = nw.new_node(Nodes.Transform,
                              input_kwargs={'Geometry': store_named_attribute_5, 'Translation': combine_xyz_7})


    combine_xyz_9 = nw.new_node(Nodes.CombineXYZ, input_kwargs={
        'X': separate_xyz.outputs["X"],
        'Y': group_input.outputs["Thickness"],
        'Z': separate_xyz.outputs["Z"]
    })


    store_named_attribute_2 = nw.new_node(Nodes.StoreNamedAttribute, input_kwargs={
        'Geometry': cube_4.outputs["Mesh"],
        'Name': 'uv_map',
        3: cube_4.outputs["UV Map"]
    }, attrs={'domain': 'CORNER', 'data_type': 'FLOAT_VECTOR'})

    add_8 = nw.new_node(Nodes.Math,
                        input_kwargs={0: separate_xyz_1.outputs["X"], 1: separate_xyz_2.outputs["X"]})


    add_10 = nw.new_node(Nodes.Math,
                         input_kwargs={0: separate_xyz_1.outputs["Z"], 1: separate_xyz_2.outputs["Z"]})


    transform_4 = nw.new_node(Nodes.Transform,
                              input_kwargs={'Geometry': store_named_attribute_2, 'Translation': combine_xyz_8})


    combine_xyz_10 = nw.new_node(Nodes.CombineXYZ, input_kwargs={
        'X': separate_xyz.outputs["X"],
        'Y': group_input.outputs["Thickness"],
        'Z': separate_xyz.outputs["Z"]
    })


    store_named_attribute_3 = nw.new_node(Nodes.StoreNamedAttribute, input_kwargs={
        'Geometry': cube_5.outputs["Mesh"],
        'Name': 'uv_map',
        3: cube_5.outputs["UV Map"]
    }, attrs={'domain': 'CORNER', 'data_type': 'FLOAT_VECTOR'})

    add_11 = nw.new_node(Nodes.Math,
                         input_kwargs={0: separate_xyz_2.outputs["X"], 1: separate_xyz_1.outputs["X"]})

    subtract_10 = nw.new_node(Nodes.Math, input_kwargs={0: separate_xyz.outputs["Y"], 1: multiply_1},
                              attrs={'operation': 'SUBTRACT'})

    add_12 = nw.new_node(Nodes.Math,
                         input_kwargs={0: separate_xyz_2.outputs["Z"], 1: separate_xyz_1.outputs["Z"]})


    transform_5 = nw.new_node(Nodes.Transform,
                              input_kwargs={'Geometry': store_named_attribute_3, 'Translation': combine_xyz_11})


    join_geometry = nw.new_node(Nodes.JoinGeometry, input_kwargs={
        'Geometry': [switch_2.outputs[6], switch_1.outputs[6], switch.outputs[6], switch_3.outputs[6],
            switch_4.outputs[6], switch_5.outputs[6]]
    })

    group_output = nw.new_node(Nodes.GroupOutput, input_kwargs={'Geometry': join_geometry},
                               attrs={'is_active_output': True})


    combine_xyz = nw.new_node(Nodes.CombineXYZ, input_kwargs={
        'X': group_input.outputs["Depth"],
        'Y': group_input.outputs["Width"],
        'Z': group_input.outputs["Height"]
    })

    hollowcube = nw.new_node(nodegroup_hollow_cube().name, input_kwargs={
        'Size': combine_xyz,
        'Thickness': group_input.outputs["DoorThickness"],
        'Switch2': True,
        'Switch4': True
    })




    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ, input_kwargs={
        'X': group_input.outputs["DoorThickness"],
        'Y': group_input.outputs["Width"],
        'Z': group_input.outputs["Height"]
    })




    center = nw.new_node(nodegroup_center().name, input_kwargs={
        'Geometry': cube,
        'Vector': position,
        'MarginX': -1.0000,
        'MarginY': 0.1000,
        'MarginZ': 0.1500
    })




    multiply = nw.new_node(Nodes.Math, input_kwargs={0: group_input.outputs["Width"], 1: 0.0500},
                           attrs={'operation': 'MULTIPLY'})

    multiply_1 = nw.new_node(Nodes.Math, input_kwargs={0: group_input.outputs["Width"], 1: 0.8000},
                             attrs={'operation': 'MULTIPLY'})


                         input_kwargs={'width': multiply, 'length': multiply_1, 'thickness': multiply_2})

    add = nw.new_node(Nodes.Math,
                      input_kwargs={0: group_input.outputs["Depth"], 1: group_input.outputs["DoorThickness"]})

    multiply_3 = nw.new_node(Nodes.Math, input_kwargs={0: group_input.outputs["Width"], 1: 0.1000},
                             attrs={'operation': 'MULTIPLY'})

    multiply_4 = nw.new_node(Nodes.Math, input_kwargs={0: group_input.outputs["Height"], 1: 0.9500},
                             attrs={'operation': 'MULTIPLY'})


    transform_1 = nw.new_node(Nodes.Transform, input_kwargs={
        'Geometry': handle,
        'Translation': combine_xyz_13,
        'Rotation': (0.0000, 1.5708, 0.0000)
    })


    add_1 = nw.new_node(Nodes.Math,
                        input_kwargs={0: group_input.outputs["Depth"], 1: group_input.outputs["DoorThickness"]})

    multiply_5 = nw.new_node(Nodes.Math, input_kwargs={0: group_input.outputs["Width"]},
                             attrs={'operation': 'MULTIPLY'})


    multiply_6 = nw.new_node(Nodes.Math, input_kwargs={0: group_input.outputs["Height"], 1: 0.0500},
                             attrs={'operation': 'MULTIPLY'})

    text = nw.new_node(nodegroup_text().name, input_kwargs={
        'Translation': combine_xyz_12,
        'String': group_input.outputs["BrandName"],
        'Size': multiply_6
    })



    geometry_to_instance = nw.new_node('GeometryNodeGeometryToInstance',
                                       input_kwargs={'Geometry': join_geometry_3})



    rotate_instances = nw.new_node(Nodes.RotateInstances, input_kwargs={
        'Instances': geometry_to_instance,
        'Rotation': combine_xyz_3,
        'Pivot Point': combine_xyz_4
    })


    multiply_7 = nw.new_node(Nodes.Math, input_kwargs={0: group_input.outputs["DoorThickness"], 1: 2.1000},
                             attrs={'operation': 'MULTIPLY'})

    subtract = nw.new_node(Nodes.Math, input_kwargs={0: group_input.outputs["Depth"], 1: multiply_7},
                           attrs={'operation': 'SUBTRACT'})

    multiply_8 = nw.new_node(Nodes.Math, input_kwargs={0: group_input.outputs["DoorThickness"], 1: 2.1000},
                             attrs={'operation': 'MULTIPLY'})

    subtract_1 = nw.new_node(Nodes.Math, input_kwargs={0: group_input.outputs["Width"], 1: multiply_8},
                             attrs={'operation': 'SUBTRACT'})

    dishrack = nw.new_node(nodegroup_dish_rack().name, input_kwargs={
        'Radius': group_input.outputs["RackRadius"],
        'Amount': 4,
        'Height': 0.1000
    })


    duplicate_elements = nw.new_node(Nodes.DuplicateElements, input_kwargs={
        'Geometry': geometry_to_instance_1,
        'Amount': group_input.outputs["RackAmount"]
    }, attrs={'domain': 'INSTANCE'})

    multiply_9 = nw.new_node(Nodes.Math, input_kwargs={0: group_input.outputs["Depth"]},
                             attrs={'operation': 'MULTIPLY'})

    multiply_10 = nw.new_node(Nodes.Math, input_kwargs={0: group_input.outputs["Width"]},
                              attrs={'operation': 'MULTIPLY'})


    multiply_11 = nw.new_node(Nodes.Math, input_kwargs={0: group_input.outputs["DoorThickness"], 1: 2.0000},
                              attrs={'operation': 'MULTIPLY'})

    subtract_2 = nw.new_node(Nodes.Math, input_kwargs={0: group_input.outputs["Height"], 1: multiply_11},
                             attrs={'operation': 'SUBTRACT'})




    combine_xyz_5 = nw.new_node(Nodes.CombineXYZ,
                                input_kwargs={'X': multiply_9, 'Y': multiply_10, 'Z': multiply_12})

    set_position = nw.new_node(Nodes.SetPosition, input_kwargs={
        'Geometry': duplicate_elements.outputs["Geometry"],
        'Offset': combine_xyz_5
    })



    add_4 = nw.new_node(Nodes.Math,
                        input_kwargs={0: group_input.outputs["Depth"], 1: group_input.outputs["DoorThickness"]})




    combine_xyz_6 = nw.new_node(Nodes.CombineXYZ,
                                input_kwargs={'X': reroute_10, 'Y': reroute_11, 'Z': reroute_8})









    geometry = nw.new_node(Nodes.RealizeInstances, [join_geometry])
    group_output = nw.new_node(Nodes.GroupOutput, input_kwargs={'Geometry': geometry})
