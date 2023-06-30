import bpy
import mathutils
import numpy as np
from nodes.node_wrangler import Nodes, NodeWrangler
from nodes import node_utils
from nodes.color import color_category, hsv2rgba
from surfaces import surface

from assets.insects.utils.geom_utils import nodegroup_symmetric_clone
from assets.insects.parts.head.dragonfly_head import nodegroup_dragon_fly_head 
from assets.insects.parts.body.dragonfly_body import nodegroup_dragonfly_body 
from assets.insects.parts.tail.dragonfly_tail import nodegroup_dragonfly_tail 
from assets.insects.parts.leg.dragonfly_leg import nodegroup_leg_control, nodegroup_dragonfly_leg
from assets.insects.parts.wing.dragonfly_wing import nodegroup_dragonfly_wing

from util.math import FixedSeed
from placement.factory import AssetFactory
from util import blender as butil

def geometry_dragonfly(nw: NodeWrangler, **kwargs):
    # Code generated using version 2.4.3 of the node_transpiler
    value_head_scale = nw.new_node(Nodes.Value)
    value_head_scale.outputs[0].default_value = kwargs["Head Scale"]

    dragonflyhead = nw.new_node(nodegroup_dragon_fly_head(base_color=kwargs["Base Color"], eye_color=kwargs["Eye Color"], v=kwargs["V"]).name)
        
    combine_xyz_8 = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'X': kwargs["Head Roll"], 'Y': kwargs["Head Pitch"], 'Z': 1.5708})
    
    transform_8 = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': dragonflyhead, 'Translation': (0.0, -0.3, 0.0), 'Rotation': combine_xyz_8, 'Scale': value_head_scale})
    
    transform_13 = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': transform_8, 'Scale': (1.1, 1.0, 1.0)})
    
    dragonflybody = nw.new_node(nodegroup_dragonfly_body(base_color=kwargs["Body Color"], v=kwargs["V"]).name,
        input_kwargs={'Body Length': kwargs["Body Length"], 'Random Seed': kwargs["Body Seed"]})
    
    store_named_attribute = nw.new_node(Nodes.StoreNamedAttribute,
        input_kwargs={'Geometry': dragonflybody.outputs["Geometry"], 'Name': 'spline parameter', 3: dragonflybody.outputs["spline parameter"]})
    
    store_named_attribute_1 = nw.new_node(Nodes.StoreNamedAttribute,
        input_kwargs={'Geometry': store_named_attribute, 'Name': 'body seed', 3: kwargs["Body Seed"]})
    
    transform = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': store_named_attribute_1, 'Rotation': (1.5708, 0.0, 0.0)})
        
    multiply = nw.new_node(Nodes.Math,
        input_kwargs={0: kwargs["Tail Length"]},
        attrs={'operation': 'MULTIPLY'})
    
    multiply_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: kwargs["Tail Tip Z"], 1: -0.5},
        attrs={'operation': 'MULTIPLY'})
    
    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'X': multiply, 'Z': multiply_1})
    
    combine_xyz = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'X': kwargs["Tail Length"], 'Z': kwargs["Tail Tip Z"]})
    
    dragonflytail = nw.new_node(nodegroup_dragonfly_tail(base_color=kwargs["Base Color"], v=kwargs["V"], ring_length=kwargs['Ring Length']).name,
        input_kwargs={'Middle': combine_xyz_1, 'End': combine_xyz, 'Segment Length': 0.38, 'Random Seed': kwargs["Tail Seed"], 'Radius': kwargs["Tail Radius"]})
    
    value = nw.new_node(Nodes.Value)
    value.outputs[0].default_value = 10.0
    
    transform_1 = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': dragonflytail, 'Translation': (0.0, -10.2, 0.0), 'Rotation': (0.0, 0.0, -1.5708), 'Scale': value})
    
    join_geometry = nw.new_node(Nodes.JoinGeometry,
        input_kwargs={'Geometry': [transform, transform_1]})
        
    nodegroup = nw.new_node(nodegroup_leg_control().name,
        input_kwargs={'Openness': kwargs["Leg Openness 3"]})
    
    dragonflyleg = nw.new_node(nodegroup_dragonfly_leg().name,
        input_kwargs={'Rot claw': 0.18, 'Rot Tarsus': nodegroup.outputs["Tarsus"], 'Rot Femur': nodegroup.outputs["Femur"]})
    
    value_leg_scale = nw.new_node(Nodes.Value)
    value_leg_scale.outputs[0].default_value = kwargs["Leg Scale"]

    transform_15 = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': dragonflyleg, 'Rotation': (0.0, 0.0, -0.5236), 'Scale': value_leg_scale})
    
    combine_xyz_6 = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'Y': nodegroup.outputs["Shoulder"], 'Z': -0.5861})
    
    transform_2 = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': transform_15, 'Translation': (0.38, 0.0, 0.0), 'Rotation': combine_xyz_6})
    
    symmetric_clone = nw.new_node(nodegroup_symmetric_clone().name,
        input_kwargs={'Geometry': transform_2, 'Scale': (-1.0, 1.0, 1.0)})
    
    value_1 = nw.new_node(Nodes.Value)
    value_1.outputs[0].default_value = 1.2
    
    transform_3 = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': symmetric_clone.outputs["Both"], 'Translation': (0.0, -4.6, -2.26), 'Scale': value_1})
    
    nodegroup_1 = nw.new_node(nodegroup_leg_control().name,
        input_kwargs={'Openness': kwargs["Leg Openness 2"]})
    
    dragonflyleg_1 = nw.new_node(nodegroup_dragonfly_leg().name,
        input_kwargs={'Rot claw': 0.18, 'Rot Tarsus': nodegroup_1.outputs["Tarsus"], 'Rot Femur': nodegroup_1.outputs["Femur"]})
    
    transform_16 = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': dragonflyleg_1, 'Rotation': (0.0, 0.0, -0.1745), 'Scale': value_leg_scale})
    
    combine_xyz_5 = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'Y': nodegroup_1.outputs["Shoulder"], 'Z': 0.174})
    
    transform_5 = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': transform_16, 'Translation': (0.38, 0.0, 0.0), 'Rotation': combine_xyz_5})
    
    symmetric_clone_1 = nw.new_node(nodegroup_symmetric_clone().name,
        input_kwargs={'Geometry': transform_5, 'Scale': (-1.0, 1.0, 1.0)})
    
    value_2 = nw.new_node(Nodes.Value)
    value_2.outputs[0].default_value = 1.18
    
    transform_4 = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': symmetric_clone_1.outputs["Both"], 'Translation': (0.0, -3.62, -2.26), 'Scale': value_2})
    
    nodegroup_2 = nw.new_node(nodegroup_leg_control().name,
        input_kwargs={'Openness': kwargs["Leg Openness 1"]})
    
    dragonflyleg_2 = nw.new_node(nodegroup_dragonfly_leg().name,
        input_kwargs={'Rot claw': 1.0, 'Rot Tarsus': nodegroup_2.outputs["Tarsus"], 'Rot Femur': nodegroup_2.outputs["Femur"]})
    
    transform_14 = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': dragonflyleg_2, 'Rotation': (0.0, 0.0, 0.3491), 'Scale': value_leg_scale})
    
    combine_xyz_4 = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'Y': nodegroup_2.outputs["Shoulder"], 'Z': 0.663})
    
    transform_6 = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': transform_14, 'Translation': (0.38, 0.0, 0.0), 'Rotation': combine_xyz_4})
    
    symmetric_clone_2 = nw.new_node(nodegroup_symmetric_clone().name,
        input_kwargs={'Geometry': transform_6, 'Scale': (-1.0, 1.0, 1.0)})
    
    value_3 = nw.new_node(Nodes.Value)
    value_3.outputs[0].default_value = 1.04
    
    transform_7 = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': symmetric_clone_2.outputs["Both"], 'Translation': (0.0, -2.66, -2.26), 'Scale': value_3})
    
    join_geometry_1 = nw.new_node(Nodes.JoinGeometry,
        input_kwargs={'Geometry': [join_geometry, transform_3, transform_4, transform_7]})
    
    join_geometry_2 = nw.new_node(Nodes.JoinGeometry,
        input_kwargs={'Geometry': [transform_13, join_geometry_1]})
    
    dragonflywing = nw.new_node(nodegroup_dragonfly_wing().name)

    value_wing_yaw = nw.new_node(Nodes.Value)
    value_wing_yaw.outputs[0].default_value = kwargs["Wing Yaw"]
        
    combine_xyz_2 = nw.new_node(Nodes.CombineXYZ,

    value_wing_scale = nw.new_node(Nodes.Value)
    value_wing_scale.outputs[0].default_value = kwargs["Wing Scale"]
    
    transform_9 = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': dragonflywing, 'Translation': (0.22, 0.0, 0.0), 'Rotation': combine_xyz_2, 'Scale': value_wing_scale})
    
    symmetric_clone_3 = nw.new_node(nodegroup_symmetric_clone().name,
        input_kwargs={'Geometry': transform_9, 'Scale': (-1.0, 1.0, 1.0)})
    
    value_5 = nw.new_node(Nodes.Value)
    value_5.outputs[0].default_value = 5.4
    
    transform_10 = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': symmetric_clone_3.outputs["Both"], 'Translation': (0.0, -2.4, 1.8), 'Scale': value_5})
    
    dragonflywing_1 = nw.new_node(nodegroup_dragonfly_wing().name)
    
    add = nw.new_node(Nodes.Math,
    
    subtract = nw.new_node(Nodes.Math,
        input_kwargs={1: value_wing_yaw},
        attrs={'operation': 'SUBTRACT'})
    
    combine_xyz_3 = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'Y': add, 'Z': subtract})
    
    transform_12 = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': dragonflywing_1, 'Translation': (0.22, 0.0, 0.0), 'Rotation': combine_xyz_3, 'Scale': value_wing_scale})
    
    symmetric_clone_4 = nw.new_node(nodegroup_symmetric_clone().name,
        input_kwargs={'Geometry': transform_12, 'Scale': (-1.0, 1.0, 1.0)})
    
    value_6 = nw.new_node(Nodes.Value)
    value_6.outputs[0].default_value = 6.0
    
    transform_11 = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': symmetric_clone_4.outputs["Both"], 'Translation': (0.0, -4.18, 1.8), 'Scale': value_6})
    
    join_geometry_3 = nw.new_node(Nodes.JoinGeometry,
        input_kwargs={'Geometry': [join_geometry_2, transform_10, transform_11]})

    realize_instances = nw.new_node(Nodes.RealizeInstances, input_kwargs={'Geometry': join_geometry_3})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': realize_instances})



        with FixedSeed(factory_seed):
            self.genome = self.sample_geo_genome()

    @staticmethod
    def sample_geo_genome():
        base_color_rgba = hsv2rgba(base_color)

        eye_color = np.copy(base_color)
        eye_color_rgba = hsv2rgba(eye_color)

        body_color = np.copy(base_color)
        body_color_rgba = hsv2rgba(body_color)

        return {
                'Base Color': base_color_rgba,
                'Body Color': body_color_rgba,
                'Eye Color': eye_color_rgba,
            }


        bpy.ops.mesh.primitive_plane_add(
            size=2, enter_editmode=False, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
        obj = bpy.context.active_object

        phenome = self.genome.copy()

        surface.add_geomod(obj, geometry_dragonfly, apply=False, input_kwargs=phenome)

        obj = bpy.context.object

        return obj