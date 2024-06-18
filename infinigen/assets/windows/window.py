from numpy.random import uniform as U, normal as N, randint as RI, uniform

from infinigen.core.util.blender import deep_clone_obj

    def __init__(self, factory_seed, coarse=False, curtain=None, shutter=None):
            self.params = self.sample_parameters()
            self.curtain = curtain
            self.shutter = shutter
    @staticmethod
    def sample_parameters():
            "FrameMaterial": surface.shaderfunc_to_material(shader_frame_material_choice, vertical=True),
    def sample_asset_params(self, dimensions=None, open=None, curtain=None, shutter=None):
        if dimensions is None:
            width = U(1, 4)
            height = U(1, 4)
            frame_thickness = U(0.05, 0.15)
        else:
            width, height, frame_thickness = dimensions

        panel_h_amount = RI(1, 2)
        v_ = width / height * panel_h_amount
        panel_v_amount = int(uniform(v_ * 1.6, v_ * 2.5))

        if open is None:
            open = U(0, 1) < 0.5

        if shutter is None:
            shutter = U(0, 1) < 0.5

        if curtain is None:
            curtain = U(0, 1) < 0.5
        if curtain:
            open = False
        sub_frame_thickness = U(0.01, frame_thickness)
        open_type = RI(0, 3)
        open_offset = 0
        oe_offset = 0
        if open_type == 0:
            if frame_thickness < sub_frame_thickness * 2:
                open_type = RI(1, 2)
            else:
                oe_offset = U(sub_frame_thickness / 2, (frame_thickness - 2 * sub_frame_thickness) / 2)
                if open:
                    open_offset = U(0, width / panel_h_amount)
                else:
                    open_offset = 0

        curtain_interval_number = int(width / U(0.08, 0.2))
        curtain_mid_l = -U(0, width / 2)
        curtain_mid_r = U(0, width / 2)
            "Width": width,
            "Height": height,
            "FrameThickness": frame_thickness,
            "PanelHAmount": panel_h_amount,
            "PanelVAmount": panel_v_amount,
            "SubFrameThickness": sub_frame_thickness,
            "OpenHAngle": open_h_angle,
            "OpenVAngle": open_v_angle,
            "OpenOffset": open_offset,
            "OEOffset": oe_offset,
            "Curtain": curtain,
            "CurtainIntervalNumber": curtain_interval_number,
            "CurtainMidL": curtain_mid_l,
            "CurtainMidR": curtain_mid_r,
            "Shutter": shutter,
        }

    def create_asset(self, dimensions=None, open=None, realized=True, **params):
        butil.modify_mesh(
            obj, 
            'NODES', 
            node_group=nodegroup_window_geometry(),
            ng_inputs=self.sample_asset_params(dimensions, open, self.curtain,self.shutter), 
            apply=realized
        )
        
        obj.rotation_euler[0] = np.pi / 2
        butil.apply_transform(obj, True)
        obj_ =deep_clone_obj(obj)
        if max(obj.dimensions) > 8:
            butil.delete(obj)
            obj = obj_
        else:
            butil.delete(obj_)
    # Code generated using version 2.6.5 of the node_transpiler

    group_input_1 = nw.new_node(Nodes.GroupInput, expose_input=[('NodeSocketFloatDistance', 'Width', 2.0000),
        ('NodeSocketFloatDistance', 'Height', 2.0000), ('NodeSocketFloatDistance', 'FrameWidth', 0.1000),
        ('NodeSocketFloatDistance', 'FrameThickness', 0.1000), ('NodeSocketInt', 'PanelHAmount', 0),
        ('NodeSocketInt', 'PanelVAmount', 0), ('NodeSocketFloatDistance', 'SubFrameWidth', 0.0500),
        ('NodeSocketFloatDistance', 'SubFrameThickness', 0.0500), ('NodeSocketInt', 'SubPanelHAmount', 3),
        ('NodeSocketInt', 'SubPanelVAmount', 2), ('NodeSocketFloat', 'GlassThickness', 0.0100),
        ('NodeSocketFloat', 'OpenHAngle', 0.5000), ('NodeSocketFloat', 'OpenVAngle', 0.5000),
        ('NodeSocketFloat', 'OpenOffset', 0.5000), ('NodeSocketFloat', 'OEOffset', 0.0500),
        ('NodeSocketBool', 'Curtain', False), ('NodeSocketFloat', 'CurtainFrameDepth', 0.5000),
        ('NodeSocketFloat', 'CurtainDepth', 0.0300), ('NodeSocketFloat', 'CurtainIntervalNumber', 20.0000),
        ('NodeSocketFloatDistance', 'CurtainFrameRadius', 0.0100), ('NodeSocketFloat', 'CurtainMidL', -0.5000),
        ('NodeSocketFloat', 'CurtainMidR', 0.5000), ('NodeSocketBool', 'Shutter', True),
        ('NodeSocketFloatDistance', 'ShutterPanelRadius', 0.0050),
        ('NodeSocketFloatDistance', 'ShutterWidth', 0.0500),
        ('NodeSocketFloatDistance', 'ShutterThickness', 0.0050), ('NodeSocketFloat', 'ShutterRotation', 0.0000),
        ('NodeSocketFloat', 'ShutterInterval', 0.0500), ('NodeSocketMaterial', 'FrameMaterial', None),
        ('NodeSocketMaterial', 'CurtainFrameMaterial', None), ('NodeSocketMaterial', 'CurtainMaterial', None),
        ('NodeSocketMaterial', 'Material', None)])

    windowpanel = nw.new_node(nodegroup_window_panel().name, input_kwargs={
        'Width': group_input_1.outputs["Width"],
        'Height': group_input_1.outputs["Height"],
        'FrameWidth': group_input_1.outputs["FrameWidth"],
        'FrameThickness': group_input_1.outputs["FrameThickness"],
        'PanelWidth': group_input_1.outputs["FrameWidth"],
        'PanelThickness': group_input_1.outputs["FrameThickness"],
        'PanelHAmount': group_input_1.outputs["PanelHAmount"],
        'PanelVAmount': group_input_1.outputs["PanelVAmount"],
        'FrameMaterial': group_input_1.outputs["FrameMaterial"],
        'Material': group_input_1.outputs["Material"]
    })

    multiply = nw.new_node(Nodes.Math, input_kwargs={
        0: group_input_1.outputs["FrameWidth"],
        1: group_input_1.outputs["PanelVAmount"]
    }, attrs={'operation': 'MULTIPLY'})

    subtract = nw.new_node(Nodes.Math, input_kwargs={0: group_input_1.outputs["Width"], 1: multiply},
                           attrs={'operation': 'SUBTRACT'})

    divide = nw.new_node(Nodes.Math, input_kwargs={0: subtract, 1: group_input_1.outputs["PanelVAmount"]},
                         attrs={'operation': 'DIVIDE'})

    subtract_1 = nw.new_node(Nodes.Math, input_kwargs={0: divide, 1: group_input_1.outputs["SubFrameWidth"]},
                             attrs={'operation': 'SUBTRACT'})

    multiply_1 = nw.new_node(Nodes.Math, input_kwargs={
        0: group_input_1.outputs["FrameWidth"],
        1: group_input_1.outputs["PanelHAmount"]
    }, attrs={'operation': 'MULTIPLY'})

    subtract_2 = nw.new_node(Nodes.Math, input_kwargs={0: group_input_1.outputs["Height"], 1: multiply_1},
                             attrs={'operation': 'SUBTRACT'})

    divide_1 = nw.new_node(Nodes.Math, input_kwargs={0: subtract_2, 1: group_input_1.outputs["PanelHAmount"]},
                           attrs={'operation': 'DIVIDE'})

    subtract_3 = nw.new_node(Nodes.Math, input_kwargs={0: divide_1, 1: group_input_1.outputs["SubFrameWidth"]},
                             attrs={'operation': 'SUBTRACT'})

    windowpanel_1 = nw.new_node(nodegroup_window_panel().name, input_kwargs={
        'Width': subtract_1,
        'Height': subtract_3,
        'FrameWidth': group_input_1.outputs["SubFrameWidth"],
        'FrameThickness': group_input_1.outputs["SubFrameThickness"],
        'PanelWidth': group_input_1.outputs["SubFrameWidth"],
        'PanelThickness': group_input_1.outputs["SubFrameThickness"],
        'PanelHAmount': group_input_1.outputs["SubPanelHAmount"],
        'PanelVAmount': group_input_1.outputs["SubPanelVAmount"],
        'WithGlass': True,
        'GlassThickness': group_input_1.outputs["GlassThickness"],
        'FrameMaterial': group_input_1.outputs["FrameMaterial"],
        'Material': group_input_1.outputs["Material"]
    })

    windowshutter = nw.new_node(nodegroup_window_shutter().name, input_kwargs={
        'Width': subtract_1,
        'Height': subtract_3,
        'FrameWidth': group_input_1.outputs["FrameWidth"],
        'FrameThickness': group_input_1.outputs["FrameThickness"],
        'PanelWidth': group_input_1.outputs["ShutterPanelRadius"],
        'PanelThickness': group_input_1.outputs["ShutterPanelRadius"],
        'ShutterWidth': group_input_1.outputs["ShutterWidth"],
        'ShutterThickness': group_input_1.outputs["ShutterThickness"],
        'ShutterInterval': group_input_1.outputs["ShutterInterval"],
        'ShutterRotation': group_input_1.outputs["ShutterRotation"],
        'FrameMaterial': group_input_1.outputs["FrameMaterial"]
    })

    switch = nw.new_node(Nodes.Switch,
                         input_kwargs={1: group_input_1.outputs["Shutter"], 14: windowpanel_1, 15: windowshutter
                         })

    multiply_2 = nw.new_node(Nodes.Math, input_kwargs={0: group_input_1.outputs["Width"], 1: -0.5000},
                             attrs={'operation': 'MULTIPLY'})

    divide_2 = nw.new_node(Nodes.Math, input_kwargs={
        0: group_input_1.outputs["Width"],
        1: group_input_1.outputs["PanelVAmount"]
    }, attrs={'operation': 'DIVIDE'})



    multiply_4 = nw.new_node(Nodes.Math, input_kwargs={0: group_input_1.outputs["Height"], 1: -0.5000},
                             attrs={'operation': 'MULTIPLY'})

    divide_3 = nw.new_node(Nodes.Math, input_kwargs={
        0: group_input_1.outputs["Height"],
        1: group_input_1.outputs["PanelHAmount"]
    }, attrs={'operation': 'DIVIDE'})




    transform = nw.new_node(Nodes.Transform,
                            input_kwargs={'Geometry': switch.outputs[6], 'Translation': combine_xyz})


    multiply_6 = nw.new_node(Nodes.Math, input_kwargs={
        0: group_input_1.outputs["PanelHAmount"],
        1: group_input_1.outputs["PanelVAmount"]
    }, attrs={'operation': 'MULTIPLY'})

                                     input_kwargs={'Geometry': geometry_to_instance, 'Amount': multiply_6},
                                     attrs={'domain': 'INSTANCE'})


                           input_kwargs={0: duplicate_elements.outputs["Duplicate Index"], 1: reroute},
                           attrs={'operation': 'DIVIDE'})




                         input_kwargs={0: duplicate_elements.outputs["Duplicate Index"], 1: reroute},
                         attrs={'operation': 'MODULO'})




    multiply_9 = nw.new_node(Nodes.Math, input_kwargs={0: power, 1: group_input_1.outputs["OEOffset"]},
                             attrs={'operation': 'MULTIPLY'})

    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ,
                                input_kwargs={'X': multiply_7, 'Y': multiply_8, 'Z': multiply_9})

    set_position = nw.new_node(Nodes.SetPosition, input_kwargs={
        'Geometry': duplicate_elements.outputs["Geometry"],
        'Offset': combine_xyz_1
    })


    multiply_10 = nw.new_node(Nodes.Math, input_kwargs={0: group_input_1.outputs["OpenVAngle"], 1: power_1},
                              attrs={'operation': 'MULTIPLY'})



    multiply_11 = nw.new_node(Nodes.Math, input_kwargs={0: divide, 1: modulo_1},
                              attrs={'operation': 'MULTIPLY'})



    multiply_12 = nw.new_node(Nodes.Math, input_kwargs={0: divide_1, 1: modulo_2},
                              attrs={'operation': 'MULTIPLY'})



    rotate_instances = nw.new_node(Nodes.RotateInstances, input_kwargs={
        'Instances': set_position,
        'Rotation': combine_xyz_3,
        'Pivot Point': combine_xyz_2
    })

    multiply_13 = nw.new_node(Nodes.Math, input_kwargs={0: group_input_1.outputs["OpenHAngle"]},
                              attrs={'operation': 'MULTIPLY'})




    rotate_instances_1 = nw.new_node(Nodes.RotateInstances, input_kwargs={
        'Instances': rotate_instances,
        'Rotation': combine_xyz_5,
        'Pivot Point': combine_xyz_6
    })


    multiply_15 = nw.new_node(Nodes.Math, input_kwargs={0: power_2, 1: group_input_1.outputs["OpenOffset"]},
                              attrs={'operation': 'MULTIPLY'})


    set_position_1 = nw.new_node(Nodes.SetPosition,
                                 input_kwargs={'Geometry': rotate_instances_1, 'Offset': combine_xyz_4})


    multiply_16 = nw.new_node(Nodes.Math, input_kwargs={0: group_input_1.outputs["Width"]},
                              attrs={'operation': 'MULTIPLY'})

    multiply_17 = nw.new_node(Nodes.Math, input_kwargs={0: multiply_16, 1: -1.0000},
                              attrs={'operation': 'MULTIPLY'})

                              input_kwargs={0: group_input_1.outputs["CurtainFrameDepth"], 1: -1.0000},
                              attrs={'operation': 'MULTIPLY'})

    curtain = nw.new_node(nodegroup_curtain().name, input_kwargs={
        'Width': group_input_1.outputs["Width"],
        'Depth': group_input_1.outputs["CurtainDepth"],
        'Height': group_input_1.outputs["Height"],
        'IntervalNumber': group_input_1.outputs["CurtainIntervalNumber"],
        'Radius': group_input_1.outputs["CurtainFrameRadius"],
        'L1': multiply_17,
        'R1': group_input_1.outputs["CurtainMidL"],
        'L2': group_input_1.outputs["CurtainMidR"],
        'R2': multiply_16,
        'FrameDepth': multiply_18,
        'CurtainFrameMaterial': group_input_1.outputs["CurtainFrameMaterial"],
        'CurtainMaterial': group_input_1.outputs["CurtainMaterial"]
    })

    multiply_19 = nw.new_node(Nodes.Math, input_kwargs={0: group_input_1.outputs["FrameThickness"]},
                              attrs={'operation': 'MULTIPLY'})

    add_6 = nw.new_node(Nodes.Math,
                        input_kwargs={0: group_input_1.outputs["CurtainFrameDepth"], 1: multiply_19})


    transform_geometry = nw.new_node(Nodes.Transform,
                                     input_kwargs={'Geometry': curtain, 'Translation': combine_xyz_7})

    join_geometry_1 = nw.new_node(Nodes.JoinGeometry,
                                  input_kwargs={'Geometry': [transform_geometry, join_geometry]})

    switch_1 = nw.new_node(Nodes.Switch, input_kwargs={
        1: group_input_1.outputs["Curtain"],
        14: join_geometry,
        15: join_geometry_1
    })



    group_output = nw.new_node(Nodes.GroupOutput, input_kwargs={
        'Geometry': realize_instances,
        'Bounding Box': bounding_box.outputs["Bounding Box"]
    }, attrs={'is_active_output': True})
    group_input = nw.new_node(Nodes.GroupInput, expose_input=[('NodeSocketFloat', 'Width', -1.0000),
        ('NodeSocketFloat', 'Height', 0.5000), ('NodeSocketFloat', 'Amount', 0.5000)])

    multiply = nw.new_node(Nodes.Math, input_kwargs={0: group_input.outputs["Width"]},
                           attrs={'operation': 'MULTIPLY'})

    multiply_1 = nw.new_node(Nodes.Math, input_kwargs={0: group_input.outputs["Height"], 1: -0.5000},
                             attrs={'operation': 'MULTIPLY'})


    multiply_2 = nw.new_node(Nodes.Math, input_kwargs={0: group_input.outputs["Width"], 1: -0.5000},
                             attrs={'operation': 'MULTIPLY'})




    duplicate_elements = nw.new_node(Nodes.DuplicateElements, input_kwargs={
        'Geometry': geometry_to_instance,
        'Amount': group_input.outputs["Amount"]
    }, attrs={'domain': 'INSTANCE'})



    divide = nw.new_node(Nodes.Math, input_kwargs={0: group_input.outputs["Height"], 1: add_1},
                         attrs={'operation': 'DIVIDE'})



    set_position = nw.new_node(Nodes.SetPosition, input_kwargs={
        'Geometry': duplicate_elements.outputs["Geometry"],
        'Offset': combine_xyz_2
    })

    group_output = nw.new_node(Nodes.GroupOutput, input_kwargs={'Curve': set_position},
                               attrs={'is_active_output': True})

    group_input = nw.new_node(Nodes.GroupInput, expose_input=[('NodeSocketFloat', 'Width', 0.5000),
        ('NodeSocketFloat', 'Depth', 0.1000), ('NodeSocketFloatDistance', 'Height', 0.1000),
        ('NodeSocketFloat', 'IntervalNumber', 0.5000), ('NodeSocketFloatDistance', 'Radius', 1.0000),
        ('NodeSocketFloat', 'L1', 0.5000), ('NodeSocketFloat', 'R1', 0.0000), ('NodeSocketFloat', 'L2', 0.0000),
        ('NodeSocketFloat', 'R2', 0.5000), ('NodeSocketFloat', 'FrameDepth', 0.0000),
        ('NodeSocketMaterial', 'CurtainFrameMaterial', None), ('NodeSocketMaterial', 'CurtainMaterial', None)])




    multiply_1 = nw.new_node(Nodes.Math, input_kwargs={0: group_input.outputs["Width"]},
                             attrs={'operation': 'MULTIPLY'})

    multiply_2 = nw.new_node(Nodes.Math, input_kwargs={0: multiply_1, 1: -1.0000},
                             attrs={'operation': 'MULTIPLY'})





    set_position_2 = nw.new_node(Nodes.SetPosition, input_kwargs={
        'Geometry': ico_sphere.outputs["Mesh"],
        'Offset': sample_curve_1.outputs["Position"]
    })

    combine_xyz_9 = nw.new_node(Nodes.CombineXYZ,
                                input_kwargs={'X': multiply_1, 'Z': group_input.outputs["FrameDepth"]})


    combine_xyz_8 = nw.new_node(Nodes.CombineXYZ,
                                input_kwargs={'X': multiply_2, 'Z': group_input.outputs["FrameDepth"]})


    join_geometry_3 = nw.new_node(Nodes.JoinGeometry,
                                  input_kwargs={'Geometry': [curve_line, curve_line_4, curve_line_3]})


    curve_to_mesh_1 = nw.new_node(Nodes.CurveToMesh, input_kwargs={
        'Curve': join_geometry_3,
        'Profile Curve': curve_circle.outputs["Curve"],
        'Fill Caps': True
    })



    set_position_3 = nw.new_node(Nodes.SetPosition, input_kwargs={
        'Geometry': ico_sphere_1.outputs["Mesh"],
        'Offset': sample_curve.outputs["Position"]
    })

    join_geometry_2 = nw.new_node(Nodes.JoinGeometry,
                                  input_kwargs={'Geometry': [set_position_2, curve_to_mesh_1, set_position_3]})

    multiply_3 = nw.new_node(Nodes.Math, input_kwargs={0: group_input.outputs["Height"], 1: -0.4700},
                             attrs={'operation': 'MULTIPLY'})


    set_position_1 = nw.new_node(Nodes.SetPosition,
                                 input_kwargs={'Geometry': join_geometry_2, 'Offset': combine_xyz_3})

    set_material_1 = nw.new_node(Nodes.SetMaterial, input_kwargs={
        'Geometry': set_position_1,
        'Material': group_input.outputs["CurtainFrameMaterial"]
    })









    join_geometry_1 = nw.new_node(Nodes.JoinGeometry,
                                  input_kwargs={'Geometry': [resample_curve, resample_curve_1]})


    capture_attribute = nw.new_node(Nodes.CaptureAttribute, input_kwargs={
        'Geometry': join_geometry_1,
        2: spline_parameter_1.outputs["Factor"]
    })


    multiply_4 = nw.new_node(Nodes.Math, input_kwargs={0: group_input.outputs["IntervalNumber"], 1: 6.2800},
                             attrs={'operation': 'MULTIPLY'})

    divide = nw.new_node(Nodes.Math, input_kwargs={0: multiply_4, 1: group_input.outputs["Width"]},
                         attrs={'operation': 'DIVIDE'})

    multiply_5 = nw.new_node(Nodes.Math, input_kwargs={0: spline_parameter.outputs["Length"], 1: divide},
                             attrs={'operation': 'MULTIPLY'})



    multiply_6 = nw.new_node(Nodes.Math, input_kwargs={0: sine, 1: group_input.outputs["Depth"]},
                             attrs={'operation': 'MULTIPLY'})


    set_position = nw.new_node(Nodes.SetPosition, input_kwargs={
        'Geometry': capture_attribute.outputs["Geometry"],
        'Offset': combine_xyz_2
    })


    quadrilateral = nw.new_node('GeometryNodeCurvePrimitiveQuadrilateral',
                                input_kwargs={'Width': reroute, 'Height': 0.0020})



    divide_1 = nw.new_node(Nodes.Math, input_kwargs={0: separate_xyz.outputs["X"], 1: reroute},
                           attrs={'operation': 'DIVIDE'})

    capture_attribute_1 = nw.new_node(Nodes.CaptureAttribute,
                                      input_kwargs={'Geometry': quadrilateral, 2: divide_1})

    curve_to_mesh = nw.new_node(Nodes.CurveToMesh, input_kwargs={
        'Curve': set_position,
        'Profile Curve': capture_attribute_1.outputs["Geometry"]
    })

    combine_xyz_12 = nw.new_node(Nodes.CombineXYZ, input_kwargs={
        'X': capture_attribute_1.outputs[2],
        'Y': capture_attribute.outputs[2]
    })

    store_named_attribute = nw.new_node(Nodes.StoreNamedAttribute, input_kwargs={
        'Geometry': curve_to_mesh,
        'Name': 'UVMap',
        3: combine_xyz_12
    }, attrs={'domain': 'CORNER', 'data_type': 'FLOAT2'})

    set_material = nw.new_node(Nodes.SetMaterial, input_kwargs={
        'Geometry': store_named_attribute,
        'Material': group_input.outputs["CurtainMaterial"]
    })

    multiply_7 = nw.new_node(Nodes.Math, input_kwargs={0: reroute_1, 1: 1.3000},
                             attrs={'operation': 'MULTIPLY'})


    curve_to_mesh_2 = nw.new_node(Nodes.CurveToMesh, input_kwargs={
        'Curve': curve_line,
        'Profile Curve': curve_circle_1.outputs["Curve"]
    })



    set_position_4 = nw.new_node(Nodes.SetPosition,
                                 input_kwargs={'Geometry': curve_to_mesh_2, 'Offset': combine_xyz_10})

    join_geometry = nw.new_node(Nodes.JoinGeometry,
                                input_kwargs={'Geometry': [set_material_1, difference.outputs["Mesh"]]})

    set_shade_smooth = nw.new_node(Nodes.SetShadeSmooth,
                                   input_kwargs={'Geometry': join_geometry, 'Shade Smooth': False})

    group_output = nw.new_node(Nodes.GroupOutput, input_kwargs={'Geometry': set_shade_smooth},
                               attrs={'is_active_output': True})
    group_input = nw.new_node(Nodes.GroupInput, expose_input=[('NodeSocketFloatDistance', 'Width', 2.0000),
        ('NodeSocketFloatDistance', 'Height', 2.0000), ('NodeSocketFloatDistance', 'FrameWidth', 0.1000),
        ('NodeSocketFloatDistance', 'FrameThickness', 0.1000),
        ('NodeSocketFloatDistance', 'PanelWidth', 0.1000),
        ('NodeSocketFloatDistance', 'PanelThickness', 0.1000),
        ('NodeSocketFloatDistance', 'ShutterWidth', 0.1000),
        ('NodeSocketFloatDistance', 'ShutterThickness', 0.1000), ('NodeSocketFloat', 'ShutterInterval', 0.5000),
        ('NodeSocketFloat', 'ShutterRotation', 0.0000), ('NodeSocketMaterial', 'FrameMaterial', None)])

    quadrilateral = nw.new_node('GeometryNodeCurvePrimitiveQuadrilateral', input_kwargs={
        'Width': group_input.outputs["Width"],
        'Height': group_input.outputs["Height"]
    })


    multiply = nw.new_node(Nodes.Math, input_kwargs={0: group_input.outputs["FrameWidth"], 1: sqrt},
                           attrs={'operation': 'MULTIPLY'})

    quadrilateral_1 = nw.new_node('GeometryNodeCurvePrimitiveQuadrilateral', input_kwargs={
        'Width': multiply,
        'Height': group_input.outputs["FrameThickness"]
    })

    curve_to_mesh = nw.new_node(Nodes.CurveToMesh,
                                input_kwargs={'Curve': quadrilateral, 'Profile Curve': quadrilateral_1})

                           input_kwargs={0: group_input.outputs["Width"], 1: group_input.outputs["FrameWidth"]},
                           attrs={'operation': 'SUBTRACT'})

    combine_xyz = nw.new_node(Nodes.CombineXYZ, input_kwargs={
        'X': subtract,
        'Y': group_input.outputs["ShutterWidth"],
        'Z': group_input.outputs["ShutterThickness"]
    })


    geometry_to_instance = nw.new_node('GeometryNodeGeometryToInstance',
                                       input_kwargs={'Geometry': cube.outputs["Mesh"]})

    subtract_1 = nw.new_node(Nodes.Math, input_kwargs={
        0: group_input.outputs["Height"],
        1: group_input.outputs["FrameWidth"]
    }, attrs={'operation': 'SUBTRACT'})

    divide = nw.new_node(Nodes.Math, input_kwargs={0: subtract_1, 1: group_input.outputs["ShutterInterval"]},
                         attrs={'operation': 'DIVIDE'})


    shutter_number = nw.new_node(Nodes.Math, input_kwargs={0: floor, 1: 1.0000}, label='ShutterNumber',
                                 attrs={'operation': 'SUBTRACT'})

                                     input_kwargs={'Geometry': geometry_to_instance, 'Amount': shutter_number},
                                     attrs={'domain': 'INSTANCE'})

    shutter_true_interval = nw.new_node(Nodes.Math, input_kwargs={0: subtract_1, 1: floor},
                                        label='ShutterTrueInterval', attrs={'operation': 'DIVIDE'})

    multiply_1 = nw.new_node(Nodes.Math, input_kwargs={
        0: duplicate_elements.outputs["Duplicate Index"],
        1: shutter_true_interval
    }, attrs={'operation': 'MULTIPLY'})

    multiply_2 = nw.new_node(Nodes.Math, input_kwargs={0: subtract_1, 1: -0.5000},
                             attrs={'operation': 'MULTIPLY'})




    set_position = nw.new_node(Nodes.SetPosition, input_kwargs={
        'Geometry': duplicate_elements.outputs["Geometry"],
        'Offset': combine_xyz_1
    })



    rotate_instances = nw.new_node(Nodes.RotateInstances,
                                   input_kwargs={'Instances': set_position, 'Rotation': combine_xyz_5})

    multiply_3 = nw.new_node(Nodes.Math, input_kwargs={0: shutter_true_interval, 1: 2.0000},
                             attrs={'operation': 'MULTIPLY'})

    subtract_2 = nw.new_node(Nodes.Math, input_kwargs={0: subtract_1, 1: multiply_3},
                             attrs={'operation': 'SUBTRACT'})

    combine_xyz_2 = nw.new_node(Nodes.CombineXYZ, input_kwargs={
        'X': group_input.outputs["PanelWidth"],
        'Y': subtract_2,
        'Z': group_input.outputs["PanelThickness"]
    })


    multiply_4 = nw.new_node(Nodes.Math, input_kwargs={0: group_input.outputs["ShutterWidth"]},
                             attrs={'operation': 'MULTIPLY'})



    geometry_to_instance_1 = nw.new_node('GeometryNodeGeometryToInstance',
                                         input_kwargs={'Geometry': curve_line})


    rotate_instances_1 = nw.new_node(Nodes.RotateInstances, input_kwargs={
        'Instances': geometry_to_instance_1,
        'Rotation': combine_xyz_4
    })



    set_position_1 = nw.new_node(Nodes.SetPosition, input_kwargs={
        'Geometry': cube_1.outputs["Mesh"],
        'Offset': sample_curve.outputs["Position"]
    })

    join_geometry_2 = nw.new_node(Nodes.JoinGeometry,
                                  input_kwargs={'Geometry': [curve_to_mesh, rotate_instances, set_position_1]})

    set_material = nw.new_node(Nodes.SetMaterial, input_kwargs={
        'Geometry': join_geometry_2,
        'Material': group_input.outputs["FrameMaterial"]
    })

    set_shade_smooth = nw.new_node(Nodes.SetShadeSmooth,
                                   input_kwargs={'Geometry': set_material, 'Shade Smooth': False})


    group_output = nw.new_node(Nodes.GroupOutput, input_kwargs={'Geometry': realize_instances_1},
                               attrs={'is_active_output': True})

    group_input = nw.new_node(Nodes.GroupInput, expose_input=[('NodeSocketFloatDistance', 'Width', 2.0000),
        ('NodeSocketFloatDistance', 'Height', 2.0000), ('NodeSocketFloatDistance', 'FrameWidth', 0.1000),
        ('NodeSocketFloatDistance', 'FrameThickness', 0.1000),
        ('NodeSocketFloatDistance', 'PanelWidth', 0.1000),
        ('NodeSocketFloatDistance', 'PanelThickness', 0.1000), ('NodeSocketInt', 'PanelHAmount', 0),
        ('NodeSocketInt', 'PanelVAmount', 0), ('NodeSocketBool', 'WithGlass', False),
        ('NodeSocketFloat', 'GlassThickness', 0.0000), ('NodeSocketMaterial', 'FrameMaterial', None),
        ('NodeSocketMaterial', 'Material', None)])

    quadrilateral = nw.new_node('GeometryNodeCurvePrimitiveQuadrilateral', input_kwargs={
        'Width': group_input.outputs["Width"],
        'Height': group_input.outputs["Height"]
    })


    multiply = nw.new_node(Nodes.Math, input_kwargs={0: group_input.outputs["FrameWidth"], 1: sqrt},
                           attrs={'operation': 'MULTIPLY'})

    quadrilateral_1 = nw.new_node('GeometryNodeCurvePrimitiveQuadrilateral', input_kwargs={
        'Width': multiply,
        'Height': group_input.outputs["FrameThickness"]
    })

    curve_to_mesh = nw.new_node(Nodes.CurveToMesh,
                                input_kwargs={'Curve': quadrilateral, 'Profile Curve': quadrilateral_1})


    lineseq = nw.new_node(nodegroup_line_seq().name, input_kwargs={
        'Width': group_input.outputs["Width"],
        'Height': group_input.outputs["Height"],
        'Amount': add
    })


    subtract = nw.new_node(Nodes.Math, input_kwargs={0: group_input.outputs["PanelThickness"], 1: 0.0010},
                           attrs={'operation': 'SUBTRACT'})

    quadrilateral_2 = nw.new_node('GeometryNodeCurvePrimitiveQuadrilateral',
                                  input_kwargs={'Width': reroute, 'Height': subtract})

    curve_to_mesh_1 = nw.new_node(Nodes.CurveToMesh,
                                  input_kwargs={'Curve': lineseq, 'Profile Curve': quadrilateral_2})


    lineseq_1 = nw.new_node(nodegroup_line_seq().name, input_kwargs={
        'Width': group_input.outputs["Height"],
        'Height': group_input.outputs["Width"],
        'Amount': add_1
    })

    transform = nw.new_node(Nodes.Transform,
                            input_kwargs={'Geometry': lineseq_1, 'Rotation': (0.0000, 0.0000, 1.5708)})


    quadrilateral_3 = nw.new_node('GeometryNodeCurvePrimitiveQuadrilateral',
                                  input_kwargs={'Width': reroute, 'Height': subtract_1})

    curve_to_mesh_2 = nw.new_node(Nodes.CurveToMesh,
                                  input_kwargs={'Curve': transform, 'Profile Curve': quadrilateral_3})

    join_geometry_3 = nw.new_node(Nodes.JoinGeometry,
                                  input_kwargs={'Geometry': [curve_to_mesh_1, curve_to_mesh_2]})

    join_geometry_2 = nw.new_node(Nodes.JoinGeometry,
                                  input_kwargs={'Geometry': [curve_to_mesh, join_geometry_3]})

    set_material_1 = nw.new_node(Nodes.SetMaterial, input_kwargs={
        'Geometry': join_geometry_2,
        'Material': group_input.outputs["FrameMaterial"]
    })

    combine_xyz = nw.new_node(Nodes.CombineXYZ, input_kwargs={
        'X': group_input.outputs["Width"],
        'Y': group_input.outputs["Height"],
        'Z': group_input.outputs["GlassThickness"]
    })


    store_named_attribute = nw.new_node(Nodes.StoreNamedAttribute, input_kwargs={
        'Geometry': cube.outputs["Mesh"],
        'Name': 'uv_map',
        3: cube.outputs["UV Map"]
    }, attrs={'domain': 'CORNER', 'data_type': 'FLOAT_VECTOR'})

    set_material = nw.new_node(Nodes.SetMaterial, input_kwargs={
        'Geometry': store_named_attribute,
        'Material': group_input.outputs["Material"]
    })


    switch = nw.new_node(Nodes.Switch, input_kwargs={
        1: group_input.outputs["WithGlass"],
        14: set_material_1,
        15: join_geometry
    })

    set_shade_smooth = nw.new_node(Nodes.SetShadeSmooth,
                                   input_kwargs={'Geometry': switch.outputs[6], 'Shade Smooth': False})

    group_output = nw.new_node(Nodes.GroupOutput, input_kwargs={'Geometry': set_shade_smooth},
                               attrs={'is_active_output': True})

    principled_bsdf = nw.new_node(Nodes.PrincipledBSDF, input_kwargs={
        'Base Color': color_category('textile'),
        'Transmission': np.random.uniform(0, 1),
        'Transmission Roughness': 1.0
    })

    material_output = nw.new_node(Nodes.MaterialOutput, input_kwargs={'Surface': principled_bsdf},
                                  attrs={'is_active_output': True})

    principled_bsdf = nw.new_node(Nodes.PrincipledBSDF,
                                  input_kwargs={'Base Color': (0.1840, 0.0000, 0.8000, 1.0000)})

    material_output = nw.new_node(Nodes.MaterialOutput, input_kwargs={'Surface': principled_bsdf},
                                  attrs={'is_active_output': True})

    principled_bsdf = nw.new_node(Nodes.PrincipledBSDF,
                                  input_kwargs={'Base Color': (0.8000, 0.5033, 0.0057, 1.0000)})

    material_output = nw.new_node(Nodes.MaterialOutput, input_kwargs={'Surface': principled_bsdf},
                                  attrs={'is_active_output': True})

    principled_bsdf = nw.new_node(Nodes.PrincipledBSDF, input_kwargs={
        'Base Color': (0.0094, 0.0055, 0.8000, 1.0000),
        'Roughness': 0.0000
    })

    material_output = nw.new_node(Nodes.MaterialOutput, input_kwargs={'Surface': principled_bsdf},
                                  attrs={'is_active_output': True})
