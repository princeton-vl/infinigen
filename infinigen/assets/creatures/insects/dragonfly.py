# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Yiming Zuo


import bpy
import mathutils
import gin
import numpy as np
from numpy.random import uniform as U, normal as N, randint

from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.nodes import node_utils
from infinigen.core.util.color import color_category, hsv2rgba
from infinigen.core import surface

from .utils.geom_utils import nodegroup_symmetric_clone
from .parts.head.dragonfly_head import nodegroup_dragon_fly_head 
from .parts.body.dragonfly_body import nodegroup_dragonfly_body 
from .parts.tail.dragonfly_tail import nodegroup_dragonfly_tail 
from .parts.leg.dragonfly_leg import nodegroup_leg_control, nodegroup_dragonfly_leg
from .parts.wing.dragonfly_wing import nodegroup_dragonfly_wing

from infinigen.core.placement import animation_policy

from infinigen.core.util.math import FixedSeed
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.util import blender as butil

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
        input_kwargs={'Geometry': dragonflybody.outputs["Geometry"], 'Name': 'spline parameter', 'Value': dragonflybody.outputs["spline parameter"]})
    
    store_named_attribute_1 = nw.new_node(Nodes.StoreNamedAttribute,
        input_kwargs={'Geometry': store_named_attribute, 'Name': 'body seed', 'Value': kwargs["Body Seed"]})
    
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

    scene_time = nw.new_node('GeometryNodeInputSceneTime')
    multiply_2 = nw.new_node(Nodes.Math,
        input_kwargs={0: scene_time.outputs["Seconds"], 1: 2 * np.pi * kwargs["Flap Freq"]},
        attrs={'operation': 'MULTIPLY'})
    sine = nw.new_node(Nodes.Math, input_kwargs={0: multiply_2}, attrs={'operation': 'SINE'})
    wing_roll = nw.new_node(Nodes.Math, input_kwargs={0: sine, 1: kwargs["Flap Mag"]}, attrs={'operation': 'MULTIPLY'})
    
    value_wing_yaw = nw.new_node(Nodes.Value)
    value_wing_yaw.outputs[0].default_value = kwargs["Wing Yaw"]
        
    combine_xyz_2 = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'Y': wing_roll, 'Z': value_wing_yaw})

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
        input_kwargs={0: wing_roll, 1: 0.0524})
    
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


@gin.configurable
class DragonflyFactory(AssetFactory):


    def __init__(self, factory_seed, coarse=False, bvh=None, **_):
        super(DragonflyFactory, self).__init__(factory_seed, coarse=coarse)
        self.bvh = bvh
        with FixedSeed(factory_seed):
            self.genome = self.sample_geo_genome()
            y = U(20, 60)
            self.scale = 0.015 * N(1, 0.1)
            self.policy = animation_policy.AnimPolicyRandomForwardWalk(
                forward_vec=(1, 0, 0), speed=U(7, 10), 
                step_range=(0.2, 7), yaw_dist=("uniform", -y, y), rot_vars=[0,0,0])

    @staticmethod
    def sample_geo_genome():
        base_color = np.array((U(0.1, 0.6), 0.9, 0.8))
        base_color[1] += N(0.0, 0.05)
        base_color[2] += N(0.0, 0.05)
        base_color_rgba = hsv2rgba(base_color)

        eye_color = np.copy(base_color)
        eye_color[0] += N(0.0, 0.1)
        eye_color[1] += N(0.0, 0.05)
        eye_color[2] += N(0.0, 0.05)
        eye_color_rgba = hsv2rgba(eye_color)

        body_color = np.copy(base_color)
        body_color[0] += N(0.0, 0.1)
        body_color[1] += N(0.0, 0.05)
        body_color[2] += N(0.0, 0.05)
        body_color_rgba = hsv2rgba(body_color)

        return {
                'Tail Length': U(2.5, 3.5),
                'Tail Tip Z': U(-0.4, 0.3),
                'Tail Seed': U(-100, 100),
                'Tail Radius': U(0.7, 0.9),
                'Body Length': U(8.0, 10.0),
                'Body Seed': U(-100, 100),
                'Flap Freq': U(20, 50),
                'Flap Mag': U(0.15, 0.25),
                'Wing Yaw': U(0.43, 0.7),
                'Wing Scale': U(0.9, 1.1),
                'Leg Scale': U(0.9, 1.1),
                'Leg Openness 1': U(0.0, 1.0),
                'Leg Openness 2': U(0.0, 1.0),
                'Leg Openness 3': U(0.0, 1.0),
                'Head Scale': U(1.6, 1.8),
                'Head Roll': U(-0.2, 0.2),
                'Head Pitch': U(-0.6, 0.6),
                'Base Color': base_color_rgba,
                'Body Color': body_color_rgba,
                'Eye Color': eye_color_rgba,
                'V': U(0.0, 0.5),
                'Ring Length': U(0.0, 0.3),
            }

    def create_placeholder(self, i, loc, rot):
        
        p = butil.spawn_cube(size=1)
        p.location = loc
        p.rotation_euler = rot

        if self.bvh is not None:
            p.location.z += U(0.5, 2)
            animation_policy.animate_trajectory(p, self.bvh, self.policy)

        return p

    def create_asset(self, placeholder, **params):

        bpy.ops.mesh.primitive_plane_add(
            size=2, enter_editmode=False, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
        obj = bpy.context.active_object

        phenome = self.genome.copy()

        surface.add_geomod(obj, geometry_dragonfly, apply=False, input_kwargs=phenome)

        obj = bpy.context.object
        obj.scale *= N(1, 0.1) * self.scale

        obj.parent = placeholder
        obj.location.x += 0.6
        obj.rotation_euler.z = -np.pi / 2 # TODO: dragonfly should have been defined facing +X

        return obj