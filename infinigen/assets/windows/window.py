# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: 
# - Hongyu Wen: primary author
# - Alexander Raistrick: update window glass

import bpy
import random
import mathutils
import numpy as np
from numpy.random import uniform as U, normal as N, randint as RI, uniform

from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.nodes import node_utils
from infinigen.core.util.blender import deep_clone_obj
from infinigen.core.util.color import color_category
from infinigen.core import surface
from infinigen.core.util import blender as butil

from infinigen.core.util.math import FixedSeed, clip_gaussian
from infinigen.core.placement.factory import AssetFactory
from infinigen.assets.materials import metal_shader_list, wood_shader_list

from infinigen.assets.utils.autobevel import BevelSharp

def shader_window_glass(nw: NodeWrangler):

    """ Non-refractive glass shader, since windows consist of a one-sided mesh currently and would not properly 
        refract-then un-refract the light
    """

    roughness = clip_gaussian(0, 0.015, 0, 0.03, 0.03)
    transmission = uniform(0.05, 0.12)

    # non-refractive glass
    transparent_bsdf = nw.new_node(Nodes.TransparentBSDF)
    shader = nw.new_node(Nodes.GlossyBSDF, input_kwargs={'Roughness': roughness})
    shader = nw.new_node(Nodes.MixShader, input_kwargs={'Fac': transmission, 1: transparent_bsdf, 2: shader})

    # complete pass-through for non-camera rays, for render efficiency
    light_path = nw.new_node(Nodes.LightPath)
    shader = nw.new_node(Nodes.MixShader, input_kwargs={'Fac': light_path.outputs["Is Camera Ray"], 1: transparent_bsdf, 2: shader})

    material_output = nw.new_node(Nodes.MaterialOutput, input_kwargs={'Surface': shader}, attrs={'is_active_output': True})

class WindowFactory(AssetFactory):
    def __init__(self, factory_seed, coarse=False, curtain=None, shutter=None):
        super(WindowFactory, self).__init__(factory_seed, coarse=coarse)

        with FixedSeed(factory_seed):
            self.params = self.sample_parameters()
            self.beveler = BevelSharp()
            self.curtain = curtain
            self.shutter = shutter

    @staticmethod
    def sample_parameters():
        frame_width = U(0.05, 0.1)
        sub_frame_width = U(0.01, frame_width)
        sub_frame_h_amount = RI(1, 2)
        sub_frame_v_amount = RI(1, 2)
        glass_thickness = U(0.01, 0.03)

        shutter_panel_radius = U(0.001, 0.003)
        shutter_width = U(0.03, 0.05)
        shutter_thickness = U(0.003, 0.007)
        shutter_rotation = U(0, 1)
        shutter_inverval = shutter_width + U(0.001, 0.003)

        curtain_frame_depth = U(0.05, 0.1)
        curtain_depth = U(0.03, curtain_frame_depth)
        curtain_frame_radius = U(0.01, 0.02)

        shader_frame_material_choice = random.choice(wood_shader_list)
        shader_curtain_frame_material_choice = random.choice(metal_shader_list)
        shader_curtain_material_choice = shader_curtain_material

        params = {
            "FrameWidth": frame_width,
            "SubFrameWidth": sub_frame_width,
            "SubPanelHAmount": sub_frame_h_amount,
            "SubPanelVAmount": sub_frame_v_amount,
            "GlassThickness": glass_thickness,
            "CurtainFrameDepth": curtain_frame_depth,
            "CurtainDepth": curtain_depth,
            "CurtainFrameRadius": curtain_frame_radius,
            "ShutterPanelRadius": shutter_panel_radius,
            "ShutterWidth": shutter_width,
            "ShutterThickness": shutter_thickness,
            "ShutterRotation": shutter_rotation,
            "ShutterInterval": shutter_inverval,
            "FrameMaterial": surface.shaderfunc_to_material(shader_frame_material_choice, vertical=True),
            "CurtainFrameMaterial": surface.shaderfunc_to_material(shader_curtain_frame_material_choice),
            "CurtainMaterial": surface.shaderfunc_to_material(shader_curtain_material_choice),
            "Material": surface.shaderfunc_to_material(shader_window_glass)
        }
        return params

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
        open_h_angle = U(0, 0.3) if open_type == 1 and open else 0
        open_v_angle = -U(0, 0.3) if open_type == 2 and open else 0

        curtain_interval_number = int(width / U(0.08, 0.2))
        curtain_mid_l = -U(0, width / 2)
        curtain_mid_r = U(0, width / 2)
        return {
            **self.params,
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
        obj = butil.spawn_cube()
        
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
        self.beveler(obj)
        if max(obj.dimensions) > 8:
            butil.delete(obj)
            obj = obj_
        else:
            butil.delete(obj_)

        bpy.ops.object.light_add(
            type='AREA', 
            radius=1, 
            align='WORLD', 
            location=(0,0,0),
            scale=(1,1,1)
        )
        portal = bpy.context.active_object

        w, _, h = obj.dimensions
        portal.scale = (w, h, 1)
        portal.data.cycles.is_portal = True
        portal.rotation_euler = (-np.pi/2, 0, 0)
        butil.parent_to(portal, obj, no_inverse=True)
        portal.hide_viewport = True

        return obj


@node_utils.to_nodegroup('nodegroup_window_geometry', singleton=True, type='GeometryNodeTree')
def nodegroup_window_geometry(nw: NodeWrangler):
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

    multiply_3 = nw.new_node(Nodes.Math, input_kwargs={0: divide_2}, attrs={'operation': 'MULTIPLY'})

    add = nw.new_node(Nodes.Math, input_kwargs={0: multiply_2, 1: multiply_3})

    multiply_4 = nw.new_node(Nodes.Math, input_kwargs={0: group_input_1.outputs["Height"], 1: -0.5000},
                             attrs={'operation': 'MULTIPLY'})

    divide_3 = nw.new_node(Nodes.Math, input_kwargs={
        0: group_input_1.outputs["Height"],
        1: group_input_1.outputs["PanelHAmount"]
    }, attrs={'operation': 'DIVIDE'})

    multiply_5 = nw.new_node(Nodes.Math, input_kwargs={0: divide_3}, attrs={'operation': 'MULTIPLY'})

    add_1 = nw.new_node(Nodes.Math, input_kwargs={0: multiply_4, 1: multiply_5})

    combine_xyz = nw.new_node(Nodes.CombineXYZ, input_kwargs={'X': add, 'Y': add_1})

    transform = nw.new_node(Nodes.Transform,
                            input_kwargs={'Geometry': switch.outputs[6], 'Translation': combine_xyz})

    geometry_to_instance = nw.new_node('GeometryNodeGeometryToInstance', input_kwargs={'Geometry': transform})

    multiply_6 = nw.new_node(Nodes.Math, input_kwargs={
        0: group_input_1.outputs["PanelHAmount"],
        1: group_input_1.outputs["PanelVAmount"]
    }, attrs={'operation': 'MULTIPLY'})

    duplicate_elements = nw.new_node(Nodes.DuplicateElements,
                                     input_kwargs={'Geometry': geometry_to_instance, 'Amount': multiply_6},
                                     attrs={'domain': 'INSTANCE'})

    reroute = nw.new_node(Nodes.Reroute, input_kwargs={'Input': group_input_1.outputs["PanelHAmount"]})

    divide_4 = nw.new_node(Nodes.Math,
                           input_kwargs={0: duplicate_elements.outputs["Duplicate Index"], 1: reroute},
                           attrs={'operation': 'DIVIDE'})

    floor = nw.new_node(Nodes.Math, input_kwargs={0: divide_4}, attrs={'operation': 'FLOOR'})

    add_2 = nw.new_node(Nodes.Math, input_kwargs={0: divide, 1: group_input_1.outputs["FrameWidth"]})

    multiply_7 = nw.new_node(Nodes.Math, input_kwargs={0: floor, 1: add_2}, attrs={'operation': 'MULTIPLY'})

    modulo = nw.new_node(Nodes.Math,
                         input_kwargs={0: duplicate_elements.outputs["Duplicate Index"], 1: reroute},
                         attrs={'operation': 'MODULO'})

    add_3 = nw.new_node(Nodes.Math, input_kwargs={0: divide_1, 1: group_input_1.outputs["FrameWidth"]})

    multiply_8 = nw.new_node(Nodes.Math, input_kwargs={0: modulo, 1: add_3}, attrs={'operation': 'MULTIPLY'})

    power = nw.new_node(Nodes.Math, input_kwargs={0: -1.0000, 1: floor}, attrs={'operation': 'POWER'})

    multiply_9 = nw.new_node(Nodes.Math, input_kwargs={0: power, 1: group_input_1.outputs["OEOffset"]},
                             attrs={'operation': 'MULTIPLY'})

    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ,
                                input_kwargs={'X': multiply_7, 'Y': multiply_8, 'Z': multiply_9})

    set_position = nw.new_node(Nodes.SetPosition, input_kwargs={
        'Geometry': duplicate_elements.outputs["Geometry"],
        'Offset': combine_xyz_1
    })

    power_1 = nw.new_node(Nodes.Math, input_kwargs={0: -1.0000, 1: floor}, attrs={'operation': 'POWER'})

    multiply_10 = nw.new_node(Nodes.Math, input_kwargs={0: group_input_1.outputs["OpenVAngle"], 1: power_1},
                              attrs={'operation': 'MULTIPLY'})

    combine_xyz_3 = nw.new_node(Nodes.CombineXYZ, input_kwargs={'Y': multiply_10})

    modulo_1 = nw.new_node(Nodes.Math, input_kwargs={0: floor, 1: 2.0000}, attrs={'operation': 'MODULO'})

    multiply_11 = nw.new_node(Nodes.Math, input_kwargs={0: divide, 1: modulo_1},
                              attrs={'operation': 'MULTIPLY'})

    add_4 = nw.new_node(Nodes.Math, input_kwargs={0: multiply_2, 1: multiply_11})

    modulo_2 = nw.new_node(Nodes.Math, input_kwargs={0: multiply_8, 1: 2.0000}, attrs={'operation': 'MODULO'})

    multiply_12 = nw.new_node(Nodes.Math, input_kwargs={0: divide_1, 1: modulo_2},
                              attrs={'operation': 'MULTIPLY'})

    add_5 = nw.new_node(Nodes.Math, input_kwargs={0: multiply_4, 1: multiply_12})

    combine_xyz_2 = nw.new_node(Nodes.CombineXYZ, input_kwargs={'X': add_4, 'Y': add_5})

    rotate_instances = nw.new_node(Nodes.RotateInstances, input_kwargs={
        'Instances': set_position,
        'Rotation': combine_xyz_3,
        'Pivot Point': combine_xyz_2
    })

    multiply_13 = nw.new_node(Nodes.Math, input_kwargs={0: group_input_1.outputs["OpenHAngle"]},
                              attrs={'operation': 'MULTIPLY'})

    combine_xyz_5 = nw.new_node(Nodes.CombineXYZ, input_kwargs={'X': multiply_13})

    multiply_14 = nw.new_node(Nodes.Math, input_kwargs={0: add_3, 1: -0.5000}, attrs={'operation': 'MULTIPLY'})

    combine_xyz_6 = nw.new_node(Nodes.CombineXYZ, input_kwargs={'Y': multiply_14})

    rotate_instances_1 = nw.new_node(Nodes.RotateInstances, input_kwargs={
        'Instances': rotate_instances,
        'Rotation': combine_xyz_5,
        'Pivot Point': combine_xyz_6
    })

    power_2 = nw.new_node(Nodes.Math, input_kwargs={0: -1.0000, 1: floor}, attrs={'operation': 'POWER'})

    multiply_15 = nw.new_node(Nodes.Math, input_kwargs={0: power_2, 1: group_input_1.outputs["OpenOffset"]},
                              attrs={'operation': 'MULTIPLY'})

    combine_xyz_4 = nw.new_node(Nodes.CombineXYZ, input_kwargs={'X': multiply_15})

    set_position_1 = nw.new_node(Nodes.SetPosition,
                                 input_kwargs={'Geometry': rotate_instances_1, 'Offset': combine_xyz_4})

    join_geometry = nw.new_node(Nodes.JoinGeometry, input_kwargs={'Geometry': [windowpanel, set_position_1]})

    multiply_16 = nw.new_node(Nodes.Math, input_kwargs={0: group_input_1.outputs["Width"]},
                              attrs={'operation': 'MULTIPLY'})

    multiply_17 = nw.new_node(Nodes.Math, input_kwargs={0: multiply_16, 1: -1.0000},
                              attrs={'operation': 'MULTIPLY'})

    multiply_18 = nw.new_node(Nodes.Math,
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

    combine_xyz_7 = nw.new_node(Nodes.CombineXYZ, input_kwargs={'Z': add_6})

    transform_geometry = nw.new_node(Nodes.Transform,
                                     input_kwargs={'Geometry': curtain, 'Translation': combine_xyz_7})

    join_geometry_1 = nw.new_node(Nodes.JoinGeometry,
                                  input_kwargs={'Geometry': [transform_geometry, join_geometry]})

    switch_1 = nw.new_node(Nodes.Switch, input_kwargs={
        1: group_input_1.outputs["Curtain"],
        14: join_geometry,
        15: join_geometry_1
    })

    realize_instances = nw.new_node(Nodes.RealizeInstances, input_kwargs={'Geometry': switch_1.outputs[6]})

    bounding_box = nw.new_node(Nodes.BoundingBox, input_kwargs={'Geometry': realize_instances})

    group_output = nw.new_node(Nodes.GroupOutput, input_kwargs={
        'Geometry': realize_instances,
        'Bounding Box': bounding_box.outputs["Bounding Box"]
    }, attrs={'is_active_output': True})


@node_utils.to_nodegroup('nodegroup_line_seq', singleton=False, type='GeometryNodeTree')
def nodegroup_line_seq(nw: NodeWrangler):
    # Code generated using version 2.6.5 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput, expose_input=[('NodeSocketFloat', 'Width', -1.0000),
        ('NodeSocketFloat', 'Height', 0.5000), ('NodeSocketFloat', 'Amount', 0.5000)])

    multiply = nw.new_node(Nodes.Math, input_kwargs={0: group_input.outputs["Width"]},
                           attrs={'operation': 'MULTIPLY'})

    multiply_1 = nw.new_node(Nodes.Math, input_kwargs={0: group_input.outputs["Height"], 1: -0.5000},
                             attrs={'operation': 'MULTIPLY'})

    combine_xyz = nw.new_node(Nodes.CombineXYZ, input_kwargs={'X': multiply, 'Y': multiply_1})

    multiply_2 = nw.new_node(Nodes.Math, input_kwargs={0: group_input.outputs["Width"], 1: -0.5000},
                             attrs={'operation': 'MULTIPLY'})

    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ, input_kwargs={'X': multiply_2, 'Y': multiply_1})

    curve_line = nw.new_node(Nodes.CurveLine, input_kwargs={'Start': combine_xyz, 'End': combine_xyz_1})

    geometry_to_instance = nw.new_node('GeometryNodeGeometryToInstance', input_kwargs={'Geometry': curve_line})

    duplicate_elements = nw.new_node(Nodes.DuplicateElements, input_kwargs={
        'Geometry': geometry_to_instance,
        'Amount': group_input.outputs["Amount"]
    }, attrs={'domain': 'INSTANCE'})

    add = nw.new_node(Nodes.Math, input_kwargs={0: duplicate_elements.outputs["Duplicate Index"], 1: 1.0000})

    add_1 = nw.new_node(Nodes.Math, input_kwargs={0: group_input.outputs["Amount"], 1: 1.0000})

    divide = nw.new_node(Nodes.Math, input_kwargs={0: group_input.outputs["Height"], 1: add_1},
                         attrs={'operation': 'DIVIDE'})

    multiply_3 = nw.new_node(Nodes.Math, input_kwargs={0: add, 1: divide}, attrs={'operation': 'MULTIPLY'})

    combine_xyz_2 = nw.new_node(Nodes.CombineXYZ, input_kwargs={'Y': multiply_3})

    set_position = nw.new_node(Nodes.SetPosition, input_kwargs={
        'Geometry': duplicate_elements.outputs["Geometry"],
        'Offset': combine_xyz_2
    })

    group_output = nw.new_node(Nodes.GroupOutput, input_kwargs={'Curve': set_position},
                               attrs={'is_active_output': True})


@node_utils.to_nodegroup('nodegroup_curtain', singleton=False, type='GeometryNodeTree')
def nodegroup_curtain(nw: NodeWrangler):
    # Code generated using version 2.6.5 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput, expose_input=[('NodeSocketFloat', 'Width', 0.5000),
        ('NodeSocketFloat', 'Depth', 0.1000), ('NodeSocketFloatDistance', 'Height', 0.1000),
        ('NodeSocketFloat', 'IntervalNumber', 0.5000), ('NodeSocketFloatDistance', 'Radius', 1.0000),
        ('NodeSocketFloat', 'L1', 0.5000), ('NodeSocketFloat', 'R1', 0.0000), ('NodeSocketFloat', 'L2', 0.0000),
        ('NodeSocketFloat', 'R2', 0.5000), ('NodeSocketFloat', 'FrameDepth', 0.0000),
        ('NodeSocketMaterial', 'CurtainFrameMaterial', None), ('NodeSocketMaterial', 'CurtainMaterial', None)])

    reroute_1 = nw.new_node(Nodes.Reroute, input_kwargs={'Input': group_input.outputs["Radius"]})

    multiply = nw.new_node(Nodes.Math, input_kwargs={0: reroute_1, 1: 2.0000}, attrs={'operation': 'MULTIPLY'})

    ico_sphere = nw.new_node(Nodes.MeshIcoSphere, input_kwargs={'Radius': multiply, 'Subdivisions': 4})

    multiply_1 = nw.new_node(Nodes.Math, input_kwargs={0: group_input.outputs["Width"]},
                             attrs={'operation': 'MULTIPLY'})

    multiply_2 = nw.new_node(Nodes.Math, input_kwargs={0: multiply_1, 1: -1.0000},
                             attrs={'operation': 'MULTIPLY'})

    combine_xyz = nw.new_node(Nodes.CombineXYZ, input_kwargs={'X': multiply_2})

    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ, input_kwargs={'X': multiply_1})

    curve_line = nw.new_node(Nodes.CurveLine, input_kwargs={'Start': combine_xyz, 'End': combine_xyz_1})

    sample_curve_1 = nw.new_node(Nodes.SampleCurve, input_kwargs={'Curves': curve_line, 'Factor': 1.0000})

    set_position_2 = nw.new_node(Nodes.SetPosition, input_kwargs={
        'Geometry': ico_sphere.outputs["Mesh"],
        'Offset': sample_curve_1.outputs["Position"]
    })

    combine_xyz_9 = nw.new_node(Nodes.CombineXYZ,
                                input_kwargs={'X': multiply_1, 'Z': group_input.outputs["FrameDepth"]})

    curve_line_4 = nw.new_node(Nodes.CurveLine, input_kwargs={'Start': combine_xyz_1, 'End': combine_xyz_9})

    combine_xyz_8 = nw.new_node(Nodes.CombineXYZ,
                                input_kwargs={'X': multiply_2, 'Z': group_input.outputs["FrameDepth"]})

    curve_line_3 = nw.new_node(Nodes.CurveLine, input_kwargs={'Start': combine_xyz, 'End': combine_xyz_8})

    join_geometry_3 = nw.new_node(Nodes.JoinGeometry,
                                  input_kwargs={'Geometry': [curve_line, curve_line_4, curve_line_3]})

    curve_circle = nw.new_node(Nodes.CurveCircle, input_kwargs={'Radius': group_input.outputs["Radius"]})

    curve_to_mesh_1 = nw.new_node(Nodes.CurveToMesh, input_kwargs={
        'Curve': join_geometry_3,
        'Profile Curve': curve_circle.outputs["Curve"],
        'Fill Caps': True
    })

    ico_sphere_1 = nw.new_node(Nodes.MeshIcoSphere, input_kwargs={'Radius': multiply, 'Subdivisions': 4})

    sample_curve = nw.new_node(Nodes.SampleCurve, input_kwargs={'Curves': curve_line})

    set_position_3 = nw.new_node(Nodes.SetPosition, input_kwargs={
        'Geometry': ico_sphere_1.outputs["Mesh"],
        'Offset': sample_curve.outputs["Position"]
    })

    join_geometry_2 = nw.new_node(Nodes.JoinGeometry,
                                  input_kwargs={'Geometry': [set_position_2, curve_to_mesh_1, set_position_3]})

    multiply_3 = nw.new_node(Nodes.Math, input_kwargs={0: group_input.outputs["Height"], 1: -0.4700},
                             attrs={'operation': 'MULTIPLY'})

    combine_xyz_3 = nw.new_node(Nodes.CombineXYZ, input_kwargs={'Y': multiply_3})

    set_position_1 = nw.new_node(Nodes.SetPosition,
                                 input_kwargs={'Geometry': join_geometry_2, 'Offset': combine_xyz_3})

    set_material_1 = nw.new_node(Nodes.SetMaterial, input_kwargs={
        'Geometry': set_position_1,
        'Material': group_input.outputs["CurtainFrameMaterial"]
    })

    combine_xyz_4 = nw.new_node(Nodes.CombineXYZ, input_kwargs={'X': group_input.outputs["L1"]})

    combine_xyz_5 = nw.new_node(Nodes.CombineXYZ, input_kwargs={'X': group_input.outputs["R1"]})

    curve_line_1 = nw.new_node(Nodes.CurveLine, input_kwargs={'Start': combine_xyz_4, 'End': combine_xyz_5})

    resample_curve = nw.new_node(Nodes.ResampleCurve, input_kwargs={'Curve': curve_line_1, 'Count': 200})

    combine_xyz_6 = nw.new_node(Nodes.CombineXYZ, input_kwargs={'X': group_input.outputs["L2"]})

    combine_xyz_7 = nw.new_node(Nodes.CombineXYZ, input_kwargs={'X': group_input.outputs["R2"]})

    curve_line_2 = nw.new_node(Nodes.CurveLine, input_kwargs={'Start': combine_xyz_6, 'End': combine_xyz_7})

    resample_curve_1 = nw.new_node(Nodes.ResampleCurve, input_kwargs={'Curve': curve_line_2, 'Count': 200})

    join_geometry_1 = nw.new_node(Nodes.JoinGeometry,
                                  input_kwargs={'Geometry': [resample_curve, resample_curve_1]})

    spline_parameter_1 = nw.new_node(Nodes.SplineParameter)

    capture_attribute = nw.new_node(Nodes.CaptureAttribute, input_kwargs={
        'Geometry': join_geometry_1,
        2: spline_parameter_1.outputs["Factor"]
    })

    spline_parameter = nw.new_node(Nodes.SplineParameter)

    multiply_4 = nw.new_node(Nodes.Math, input_kwargs={0: group_input.outputs["IntervalNumber"], 1: 6.2800},
                             attrs={'operation': 'MULTIPLY'})

    divide = nw.new_node(Nodes.Math, input_kwargs={0: multiply_4, 1: group_input.outputs["Width"]},
                         attrs={'operation': 'DIVIDE'})

    multiply_5 = nw.new_node(Nodes.Math, input_kwargs={0: spline_parameter.outputs["Length"], 1: divide},
                             attrs={'operation': 'MULTIPLY'})

    add = nw.new_node(Nodes.Math, input_kwargs={0: multiply_5, 1: 1.6800})

    sine = nw.new_node(Nodes.Math, input_kwargs={0: add}, attrs={'operation': 'SINE'})

    multiply_6 = nw.new_node(Nodes.Math, input_kwargs={0: sine, 1: group_input.outputs["Depth"]},
                             attrs={'operation': 'MULTIPLY'})

    combine_xyz_2 = nw.new_node(Nodes.CombineXYZ, input_kwargs={'Z': multiply_6})

    set_position = nw.new_node(Nodes.SetPosition, input_kwargs={
        'Geometry': capture_attribute.outputs["Geometry"],
        'Offset': combine_xyz_2
    })

    reroute = nw.new_node(Nodes.Reroute, input_kwargs={'Input': group_input.outputs["Height"]})

    quadrilateral = nw.new_node('GeometryNodeCurvePrimitiveQuadrilateral',
                                input_kwargs={'Width': reroute, 'Height': 0.0020})

    position = nw.new_node(Nodes.InputPosition)

    separate_xyz = nw.new_node(Nodes.SeparateXYZ, input_kwargs={'Vector': position})

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

    curve_circle_1 = nw.new_node(Nodes.CurveCircle, input_kwargs={'Radius': multiply_7})

    curve_to_mesh_2 = nw.new_node(Nodes.CurveToMesh, input_kwargs={
        'Curve': curve_line,
        'Profile Curve': curve_circle_1.outputs["Curve"]
    })

    add_1 = nw.new_node(Nodes.Math, input_kwargs={0: multiply_3, 1: group_input.outputs["Radius"]})

    combine_xyz_10 = nw.new_node(Nodes.CombineXYZ, input_kwargs={'Y': add_1})

    set_position_4 = nw.new_node(Nodes.SetPosition,
                                 input_kwargs={'Geometry': curve_to_mesh_2, 'Offset': combine_xyz_10})

    difference = nw.new_node(Nodes.MeshBoolean, input_kwargs={'Mesh 1': set_material, 'Mesh 2': set_position_4})

    join_geometry = nw.new_node(Nodes.JoinGeometry,
                                input_kwargs={'Geometry': [set_material_1, difference.outputs["Mesh"]]})

    set_shade_smooth = nw.new_node(Nodes.SetShadeSmooth,
                                   input_kwargs={'Geometry': join_geometry, 'Shade Smooth': False})

    group_output = nw.new_node(Nodes.GroupOutput, input_kwargs={'Geometry': set_shade_smooth},
                               attrs={'is_active_output': True})


@node_utils.to_nodegroup('nodegroup_window_shutter', singleton=False, type='GeometryNodeTree')
def nodegroup_window_shutter(nw: NodeWrangler):
    # Code generated using version 2.6.5 of the node_transpiler

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

    sqrt = nw.new_node(Nodes.Math, input_kwargs={0: 2.0000}, attrs={'operation': 'SQRT'})

    multiply = nw.new_node(Nodes.Math, input_kwargs={0: group_input.outputs["FrameWidth"], 1: sqrt},
                           attrs={'operation': 'MULTIPLY'})

    quadrilateral_1 = nw.new_node('GeometryNodeCurvePrimitiveQuadrilateral', input_kwargs={
        'Width': multiply,
        'Height': group_input.outputs["FrameThickness"]
    })

    curve_to_mesh = nw.new_node(Nodes.CurveToMesh,
                                input_kwargs={'Curve': quadrilateral, 'Profile Curve': quadrilateral_1})

    subtract = nw.new_node(Nodes.Math,
                           input_kwargs={0: group_input.outputs["Width"], 1: group_input.outputs["FrameWidth"]},
                           attrs={'operation': 'SUBTRACT'})

    combine_xyz = nw.new_node(Nodes.CombineXYZ, input_kwargs={
        'X': subtract,
        'Y': group_input.outputs["ShutterWidth"],
        'Z': group_input.outputs["ShutterThickness"]
    })

    cube = nw.new_node(Nodes.MeshCube, input_kwargs={'Size': combine_xyz})

    geometry_to_instance = nw.new_node('GeometryNodeGeometryToInstance',
                                       input_kwargs={'Geometry': cube.outputs["Mesh"]})

    subtract_1 = nw.new_node(Nodes.Math, input_kwargs={
        0: group_input.outputs["Height"],
        1: group_input.outputs["FrameWidth"]
    }, attrs={'operation': 'SUBTRACT'})

    divide = nw.new_node(Nodes.Math, input_kwargs={0: subtract_1, 1: group_input.outputs["ShutterInterval"]},
                         attrs={'operation': 'DIVIDE'})

    floor = nw.new_node(Nodes.Math, input_kwargs={0: divide}, attrs={'operation': 'FLOOR'})

    shutter_number = nw.new_node(Nodes.Math, input_kwargs={0: floor, 1: 1.0000}, label='ShutterNumber',
                                 attrs={'operation': 'SUBTRACT'})

    duplicate_elements = nw.new_node(Nodes.DuplicateElements,
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

    add = nw.new_node(Nodes.Math, input_kwargs={0: multiply_2, 1: shutter_true_interval})

    add_1 = nw.new_node(Nodes.Math, input_kwargs={0: multiply_1, 1: add})

    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ, input_kwargs={'Y': add_1})

    set_position = nw.new_node(Nodes.SetPosition, input_kwargs={
        'Geometry': duplicate_elements.outputs["Geometry"],
        'Offset': combine_xyz_1
    })

    reroute = nw.new_node(Nodes.Reroute, input_kwargs={'Input': group_input.outputs["ShutterRotation"]})

    combine_xyz_5 = nw.new_node(Nodes.CombineXYZ, input_kwargs={'X': reroute})

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

    cube_1 = nw.new_node(Nodes.MeshCube, input_kwargs={'Size': combine_xyz_2})

    multiply_4 = nw.new_node(Nodes.Math, input_kwargs={0: group_input.outputs["ShutterWidth"]},
                             attrs={'operation': 'MULTIPLY'})

    combine_xyz_3 = nw.new_node(Nodes.CombineXYZ, input_kwargs={'Y': multiply_4})

    curve_line = nw.new_node(Nodes.CurveLine, input_kwargs={'End': combine_xyz_3})

    geometry_to_instance_1 = nw.new_node('GeometryNodeGeometryToInstance',
                                         input_kwargs={'Geometry': curve_line})

    combine_xyz_4 = nw.new_node(Nodes.CombineXYZ, input_kwargs={'X': reroute})

    rotate_instances_1 = nw.new_node(Nodes.RotateInstances, input_kwargs={
        'Instances': geometry_to_instance_1,
        'Rotation': combine_xyz_4
    })

    realize_instances = nw.new_node(Nodes.RealizeInstances, input_kwargs={'Geometry': rotate_instances_1})

    sample_curve = nw.new_node(Nodes.SampleCurve, input_kwargs={'Curves': realize_instances, 'Factor': 1.0000})

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

    realize_instances_1 = nw.new_node(Nodes.RealizeInstances, input_kwargs={'Geometry': set_shade_smooth})

    group_output = nw.new_node(Nodes.GroupOutput, input_kwargs={'Geometry': realize_instances_1},
                               attrs={'is_active_output': True})


@node_utils.to_nodegroup('nodegroup_window_panel', singleton=False, type='GeometryNodeTree')
def nodegroup_window_panel(nw: NodeWrangler):
    # Code generated using version 2.6.5 of the node_transpiler

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

    sqrt = nw.new_node(Nodes.Math, input_kwargs={0: 2.0000}, attrs={'operation': 'SQRT'})

    multiply = nw.new_node(Nodes.Math, input_kwargs={0: group_input.outputs["FrameWidth"], 1: sqrt},
                           attrs={'operation': 'MULTIPLY'})

    quadrilateral_1 = nw.new_node('GeometryNodeCurvePrimitiveQuadrilateral', input_kwargs={
        'Width': multiply,
        'Height': group_input.outputs["FrameThickness"]
    })

    curve_to_mesh = nw.new_node(Nodes.CurveToMesh,
                                input_kwargs={'Curve': quadrilateral, 'Profile Curve': quadrilateral_1})

    add = nw.new_node(Nodes.Math, input_kwargs={0: group_input.outputs["PanelHAmount"], 1: -1.0000})

    lineseq = nw.new_node(nodegroup_line_seq().name, input_kwargs={
        'Width': group_input.outputs["Width"],
        'Height': group_input.outputs["Height"],
        'Amount': add
    })

    reroute = nw.new_node(Nodes.Reroute, input_kwargs={'Input': group_input.outputs["PanelWidth"]})

    subtract = nw.new_node(Nodes.Math, input_kwargs={0: group_input.outputs["PanelThickness"], 1: 0.0010},
                           attrs={'operation': 'SUBTRACT'})

    quadrilateral_2 = nw.new_node('GeometryNodeCurvePrimitiveQuadrilateral',
                                  input_kwargs={'Width': reroute, 'Height': subtract})

    curve_to_mesh_1 = nw.new_node(Nodes.CurveToMesh,
                                  input_kwargs={'Curve': lineseq, 'Profile Curve': quadrilateral_2})

    add_1 = nw.new_node(Nodes.Math, input_kwargs={0: group_input.outputs["PanelVAmount"], 1: -1.0000})

    lineseq_1 = nw.new_node(nodegroup_line_seq().name, input_kwargs={
        'Width': group_input.outputs["Height"],
        'Height': group_input.outputs["Width"],
        'Amount': add_1
    })

    transform = nw.new_node(Nodes.Transform,
                            input_kwargs={'Geometry': lineseq_1, 'Rotation': (0.0000, 0.0000, 1.5708)})

    subtract_1 = nw.new_node(Nodes.Math, input_kwargs={0: subtract, 1: 0.0010}, attrs={'operation': 'SUBTRACT'})

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

    cube = nw.new_node(Nodes.MeshCube, input_kwargs={'Size': combine_xyz})

    store_named_attribute = nw.new_node(Nodes.StoreNamedAttribute, input_kwargs={
        'Geometry': cube.outputs["Mesh"],
        'Name': 'uv_map',
        3: cube.outputs["UV Map"]
    }, attrs={'domain': 'CORNER', 'data_type': 'FLOAT_VECTOR'})

    set_material = nw.new_node(Nodes.SetMaterial, input_kwargs={
        'Geometry': store_named_attribute,
        'Material': group_input.outputs["Material"]
    })

    join_geometry = nw.new_node(Nodes.JoinGeometry, input_kwargs={'Geometry': [set_material, set_material_1]})

    switch = nw.new_node(Nodes.Switch, input_kwargs={
        1: group_input.outputs["WithGlass"],
        14: set_material_1,
        15: join_geometry
    })

    set_shade_smooth = nw.new_node(Nodes.SetShadeSmooth,
                                   input_kwargs={'Geometry': switch.outputs[6], 'Shade Smooth': False})

    group_output = nw.new_node(Nodes.GroupOutput, input_kwargs={'Geometry': set_shade_smooth},
                               attrs={'is_active_output': True})


def shader_curtain_material(nw: NodeWrangler):
    # Code generated using version 2.6.5 of the node_transpiler

    principled_bsdf = nw.new_node(Nodes.PrincipledBSDF, input_kwargs={
        'Base Color': color_category('textile'),
        'Transmission': np.random.uniform(0, 1),
        'Transmission Roughness': 1.0
    })

    material_output = nw.new_node(Nodes.MaterialOutput, input_kwargs={'Surface': principled_bsdf},
                                  attrs={'is_active_output': True})


def shader_curtain_frame_material(nw: NodeWrangler):
    # Code generated using version 2.6.5 of the node_transpiler

    principled_bsdf = nw.new_node(Nodes.PrincipledBSDF,
                                  input_kwargs={'Base Color': (0.1840, 0.0000, 0.8000, 1.0000)})

    material_output = nw.new_node(Nodes.MaterialOutput, input_kwargs={'Surface': principled_bsdf},
                                  attrs={'is_active_output': True})


def shader_frame_material(nw: NodeWrangler):
    # Code generated using version 2.6.5 of the node_transpiler

    principled_bsdf = nw.new_node(Nodes.PrincipledBSDF,
                                  input_kwargs={'Base Color': (0.8000, 0.5033, 0.0057, 1.0000)})

    material_output = nw.new_node(Nodes.MaterialOutput, input_kwargs={'Surface': principled_bsdf},
                                  attrs={'is_active_output': True})


def shader_glass_material(nw: NodeWrangler):
    # Code generated using version 2.6.5 of the node_transpiler

    principled_bsdf = nw.new_node(Nodes.PrincipledBSDF, input_kwargs={
        'Base Color': (0.0094, 0.0055, 0.8000, 1.0000),
        'Roughness': 0.0000
    })

    material_output = nw.new_node(Nodes.MaterialOutput, input_kwargs={'Surface': principled_bsdf},
                                  attrs={'is_active_output': True})
