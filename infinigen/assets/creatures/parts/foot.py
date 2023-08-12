# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick


import bpy

import numpy as np
from numpy.random import uniform, normal as N

from infinigen.assets.creatures.util.genome import Joint, IKParams

from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.nodes import node_utils
from infinigen.assets.creatures.util.nodegroups.curve import nodegroup_simple_tube, nodegroup_simple_tube_v2
from infinigen.assets.creatures.util.nodegroups.attach import nodegroup_surface_muscle, nodegroup_attach_part
from infinigen.assets.creatures.util.nodegroups.math import nodegroup_deg2_rad

from infinigen.assets.creatures.util.creature import Part, PartFactory
from infinigen.assets.creatures.util.part_util import nodegroup_to_part
from infinigen.assets.utils.tag import tag_object, tag_nodegroup

@node_utils.to_nodegroup('nodegroup_tiger_toe', singleton=False, type='GeometryNodeTree')
def nodegroup_tiger_toe(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketVector', 'length_rad1_rad2', (0.18, 0.045, 0.024)),
            ('NodeSocketFloatDistance', 'Toebean Radius', 0.03),
            ('NodeSocketFloat', 'Claw Curl Deg', 30.0),
            ('NodeSocketVector', 'Claw Pct Length Rad1 Rad2', (0.0, 0.0, 0.0)),
            ('NodeSocketFloat', 'Toe Curl Scalar', 1.0)])
    
    scale = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: (-50.0, 25.0, 35.0), 'Scale': group_input.outputs["Toe Curl Scalar"]},
        attrs={'operation': 'SCALE'})
    
    separate_xyz = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': group_input.outputs["length_rad1_rad2"]})
    
    scale_1 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: separate_xyz.outputs["X"], 'Scale': 0.18},
        attrs={'operation': 'SCALE'})
    
    toe = nw.new_node(nodegroup_simple_tube().name,
        input_kwargs={'Origin': (-0.05, 0.0, 0.0), 'Angles Deg': scale.outputs["Vector"], 'Seg Lengths': scale_1.outputs["Vector"], 'Start Radius': separate_xyz.outputs["Y"], 'End Radius': separate_xyz.outputs["Z"]},
        label='Toe')
    
    uv_sphere = nw.new_node(Nodes.MeshUVSphere,
        input_kwargs={'Segments': 16, 'Rings': 8, 'Radius': group_input.outputs["Toebean Radius"]})
    
    transform_1 = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': uv_sphere, 'Scale': (1.5, 1.0, 0.6)})
    
    attach_part = nw.new_node(nodegroup_attach_part().name,
        input_kwargs={'Skin Mesh': toe.outputs["Geometry"], 'Skeleton Curve': toe.outputs["Skeleton Curve"], 'Geometry': transform_1, 'Length Fac': 0.5037, 'Ray Rot': (0.0, 1.5708, 0.0), 'Rad': 0.9})
    
    transform = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': uv_sphere, 'Scale': (1.0, 0.7, 0.6)})
    
    attach_part_1 = nw.new_node(nodegroup_attach_part().name,
        input_kwargs={'Skin Mesh': toe.outputs["Geometry"], 'Skeleton Curve': toe.outputs["Skeleton Curve"], 'Geometry': transform, 'Length Fac': 0.8, 'Ray Rot': (0.0, 1.5708, 0.0), 'Rad': 0.7})
    
    combine_xyz = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'X': separate_xyz.outputs["Z"], 'Y': separate_xyz.outputs["Z"], 'Z': 3.0})
    
    toe_top = nw.new_node(nodegroup_surface_muscle().name,
        input_kwargs={'Skin Mesh': toe.outputs["Geometry"], 'Skeleton Curve': toe.outputs["Skeleton Curve"], 'Coord 0': (0.56, -1.5708, 0.3), 'Coord 1': (0.7, -1.5708, 1.0), 'Coord 2': (0.95, -1.5708, 0.0), 'StartRad, EndRad, Fullness': combine_xyz, 'ProfileHeight, StartTilt, EndTilt': (0.9, 0.0, 0.0)},
        label='Toe Top')
    
    join_geometry_1 = nw.new_node(Nodes.JoinGeometry,
        input_kwargs={'Geometry': [toe.outputs["Geometry"], attach_part.outputs["Geometry"], attach_part_1.outputs["Geometry"], toe_top]})
    
    scale_2 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: (1.0, -2.0, -1.0), 'Scale': group_input.outputs["Claw Curl Deg"]},
        attrs={'operation': 'SCALE'})
    
    multiply = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: group_input.outputs["length_rad1_rad2"], 1: group_input.outputs["Claw Pct Length Rad1 Rad2"]},
        attrs={'operation': 'MULTIPLY'})
    
    separate_xyz_1 = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': multiply.outputs["Vector"]})
    
    scale_3 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: (0.33, 0.33, 0.33), 'Scale': separate_xyz_1.outputs["X"]},
        attrs={'operation': 'SCALE'})
    
    claw = nw.new_node(nodegroup_simple_tube().name,
        input_kwargs={'Origin': (-0.007, 0.0, 0.0), 'Angles Deg': scale_2.outputs["Vector"], 'Seg Lengths': scale_3.outputs["Vector"], 'Start Radius': separate_xyz_1.outputs["Y"], 'End Radius': separate_xyz_1.outputs["Z"]},
        label='Claw')
    
    attach_part_2 = nw.new_node(nodegroup_attach_part().name,
        input_kwargs={'Skin Mesh': toe.outputs["Geometry"], 'Skeleton Curve': toe.outputs["Skeleton Curve"], 'Geometry': claw.outputs["Geometry"], 'Length Fac': 0.85})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': join_geometry_1, 'Skeleton Curve': toe.outputs["Skeleton Curve"], 'Claw': attach_part_2.outputs["Geometry"]})


@node_utils.to_nodegroup('nodegroup_foot', singleton=False, type='GeometryNodeTree')
def nodegroup_foot(nw: NodeWrangler):
    # Code generated using version 2.5.1 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
                              expose_input=[('NodeSocketInt', 'Num Toes', 3),
                                            ('NodeSocketVector', 'length_rad1_rad2', (0.2700, 0.0400, 0.0900)),
                                            ('NodeSocketVector', 'Toe Rotate', (0.0000, -1.57, 0.0000)),
                                            ('NodeSocketVector', 'Toe Length Rad1 Rad2', (0.3000, 0.0450, 0.0250)),
                                            ('NodeSocketFloat', 'Toe Splay', 0.0000),
                                            ('NodeSocketFloatDistance', 'Toebean Radius', 0.0300),
                                            ('NodeSocketFloat', 'Claw Curl Deg', 30.0000),
                                            ('NodeSocketVector', 'Claw Pct Length Rad1 Rad2', (0.3000, 0.5000, 0.0000)),
                                            ('NodeSocketVector', 'Thumb Pct', (1.0000, 1.0000, 1.0000)),
                                            ('NodeSocketFloat', 'Toe Curl Scalar', 1.0000)])

    simple_tube_v2 = nw.new_node(nodegroup_simple_tube_v2().name,
                                 input_kwargs={'length_rad1_rad2': group_input.outputs["length_rad1_rad2"],
                                               'angles_deg': (10.0000, 8.0000, -25.0000)})

    separate_xyz = nw.new_node(Nodes.SeparateXYZ,
                               input_kwargs={'Vector': group_input.outputs["length_rad1_rad2"]})

    multiply_add = nw.new_node(Nodes.VectorMath,
                               input_kwargs={0: separate_xyz.outputs["Z"], 1: (0.0000, -0.4500, 0.1000),
                                             2: (-0.0700, 0.0000, 0.0000)},
                               attrs={'operation': 'MULTIPLY_ADD'})

    add = nw.new_node(Nodes.VectorMath,
                      input_kwargs={0: simple_tube_v2.outputs["Endpoint"], 1: multiply_add.outputs["Vector"]})

    multiply_add_1 = nw.new_node(Nodes.VectorMath,
                                 input_kwargs={0: separate_xyz.outputs["Z"], 1: (0.0000, 0.4500, 0.1000),
                                               2: (-0.0700, 0.0000, 0.0000)},
                                 attrs={'operation': 'MULTIPLY_ADD'})

    add_1 = nw.new_node(Nodes.VectorMath,
                        input_kwargs={0: simple_tube_v2.outputs["Endpoint"], 1: multiply_add_1.outputs["Vector"]})

    mesh_line = nw.new_node(Nodes.MeshLine,
                            input_kwargs={'Count': group_input.outputs["Num Toes"],
                                          'Start Location': add.outputs["Vector"], 'Offset': add_1.outputs["Vector"]},
                            attrs={'mode': 'END_POINTS'})

    tigertoe = nw.new_node(nodegroup_tiger_toe().name,
                           input_kwargs={'length_rad1_rad2': group_input.outputs["Toe Length Rad1 Rad2"],
                                         'Toebean Radius': group_input.outputs["Toebean Radius"],
                                         'Claw Curl Deg': group_input.outputs["Claw Curl Deg"],
                                         'Claw Pct Length Rad1 Rad2': group_input.outputs["Claw Pct Length Rad1 Rad2"],
                                         'Toe Curl Scalar': group_input.outputs["Toe Curl Scalar"]})

    instance_on_points = nw.new_node(Nodes.InstanceOnPoints,
                                     input_kwargs={'Points': mesh_line, 'Instance': tigertoe.outputs["Geometry"]})

    rotate_instances_1 = nw.new_node(Nodes.RotateInstances,
                                     input_kwargs={'Instances': instance_on_points,
                                                   'Rotation': group_input.outputs["Toe Rotate"]})

    index = nw.new_node(Nodes.Index)

    add_2 = nw.new_node(Nodes.Math,
                        input_kwargs={0: group_input.outputs["Num Toes"], 1: -1.0000})

    divide = nw.new_node(Nodes.Math,
                         input_kwargs={0: index, 1: add_2},
                         attrs={'operation': 'DIVIDE'})

    scale = nw.new_node(Nodes.VectorMath,
                        input_kwargs={0: (0.0000, 0.0000, -1.0000), 'Scale': group_input.outputs["Toe Splay"]},
                        attrs={'operation': 'SCALE'})

    scale_1 = nw.new_node(Nodes.VectorMath,
                          input_kwargs={0: (0.0000, 0.0000, 1.0000), 'Scale': group_input.outputs["Toe Splay"]},
                          attrs={'operation': 'SCALE'})

    map_range = nw.new_node(Nodes.MapRange,
                            input_kwargs={'Vector': divide, 9: scale.outputs["Vector"], 10: scale_1.outputs["Vector"]},
                            attrs={'data_type': 'FLOAT_VECTOR'})

    deg2rad = nw.new_node(nodegroup_deg2_rad().name,
                          input_kwargs={'Deg': map_range.outputs["Vector"]})

    rotate_instances = nw.new_node(Nodes.RotateInstances,
                                   input_kwargs={'Instances': rotate_instances_1, 'Rotation': deg2rad})

    realize_instances = nw.new_node(Nodes.RealizeInstances,
                                    input_kwargs={'Geometry': rotate_instances})

    uv_sphere = nw.new_node(Nodes.MeshUVSphere,
                            input_kwargs={'Segments': 16, 'Rings': 8, 'Radius': 0.01500})

    add_3 = nw.new_node(Nodes.VectorMath,
                        input_kwargs={0: simple_tube_v2.outputs["Endpoint"], 1: (-0.0200, 0.0000, 0.0000)})

    transform = nw.new_node(Nodes.Transform,
                            input_kwargs={'Geometry': uv_sphere, 'Translation': add_3.outputs["Vector"],
                                          'Scale': (0.7000, 1.0000, 1.0000)})

    reroute = nw.new_node(Nodes.Reroute,
                          input_kwargs={'Input': simple_tube_v2.outputs["Geometry"]})

    reroute_1 = nw.new_node(Nodes.Reroute,
                            input_kwargs={'Input': simple_tube_v2.outputs["Skeleton Curve"]})

    multiply = nw.new_node(Nodes.VectorMath,
                           input_kwargs={0: group_input.outputs["Toe Length Rad1 Rad2"],
                                         1: group_input.outputs["Thumb Pct"]},
                           attrs={'operation': 'MULTIPLY'})

    tigertoe_1 = nw.new_node(nodegroup_tiger_toe().name,
                             input_kwargs={'length_rad1_rad2': multiply.outputs["Vector"],
                                           'Toebean Radius': group_input.outputs["Toebean Radius"],
                                           'Claw Curl Deg': group_input.outputs["Claw Curl Deg"],
                                           'Claw Pct Length Rad1 Rad2': group_input.outputs[
                                               "Claw Pct Length Rad1 Rad2"],
                                           'Toe Curl Scalar': group_input.outputs["Toe Curl Scalar"]})

    value_2 = nw.new_node(Nodes.Value)
    value_2.outputs[0].default_value = 0.3000

    vector_1 = nw.new_node(Nodes.Vector)
    vector_1.vector = (90.0000, 90.0000, 90.0000)

    value_1 = nw.new_node(Nodes.Value)
    value_1.outputs[0].default_value = 0.8000

    vector = nw.new_node(Nodes.Vector)
    vector.vector = (90.0000, 1.4300, -55.6800)

    attach_part = nw.new_node(nodegroup_attach_part().name,
                              input_kwargs={'Skin Mesh': reroute, 'Skeleton Curve': reroute_1,
                                            'Geometry': tigertoe_1.outputs["Geometry"], 'Length Fac': value_2,
                                            'Ray Rot': vector_1, 'Rad': value_1, 'Part Rot': vector,
                                            'Do Tangent Rot': True})

    join_geometry = nw.new_node(Nodes.JoinGeometry,
                                input_kwargs={
                                    'Geometry': [realize_instances, transform, attach_part.outputs["Geometry"],
                                                 simple_tube_v2.outputs["Geometry"]]})

    instance_on_points_1 = nw.new_node(Nodes.InstanceOnPoints,
                                       input_kwargs={'Points': mesh_line, 'Instance': tigertoe.outputs["Claw"]})

    rotate_instances_2 = nw.new_node(Nodes.RotateInstances,
                                     input_kwargs={'Instances': instance_on_points_1,
                                                   'Rotation': group_input.outputs["Toe Rotate"]})

    rotate_instances_3 = nw.new_node(Nodes.RotateInstances,
                                     input_kwargs={'Instances': rotate_instances_2, 'Rotation': deg2rad})

    realize_instances_1 = nw.new_node(Nodes.RealizeInstances,
                                      input_kwargs={'Geometry': rotate_instances_3})

    attach_part_1 = nw.new_node(nodegroup_attach_part().name,
                                input_kwargs={'Skin Mesh': reroute, 'Skeleton Curve': reroute_1,
                                              'Geometry': tigertoe_1.outputs["Claw"], 'Length Fac': value_2,
                                              'Ray Rot': vector_1, 'Rad': value_1, 'Part Rot': vector,
                                              'Do Tangent Rot': True})

    join_geometry_1 = nw.new_node(Nodes.JoinGeometry,
                                  input_kwargs={'Geometry': [realize_instances_1, attach_part_1.outputs["Geometry"]]})

    group_output = nw.new_node(Nodes.GroupOutput,
                               input_kwargs={'Geometry': join_geometry,
                                             'Skeleton Curve': simple_tube_v2.outputs["Skeleton Curve"],
                                             'Base Mesh': simple_tube_v2.outputs["Geometry"], 'Claws': join_geometry_1})

class Foot(PartFactory):

    def __init__(self, params=None, bald=False):
        super().__init__(params)
        self.tags = ['foot']
        if bald:
            self.tags.append('bald')

    def sample_params(self):
        return {
            'length_rad1_rad2': np.array((0.27, 0.04, 0.09)) * N(1, (0.2, 0.05, 0.05), 3),
            'Num Toes': max(int(N(4, 1)), 2),
            'Toe Length Rad1 Rad2': np.array((0.3, 0.045, 0.025)) * N(1, 0.1, 3),
            'Toe Rotate': (0., -N(0.7, 0.15), 0.),
            'Toe Splay': 20.0 * N(1, 0.2),
            'Toebean Radius': 0.03 * N(1, 0.2),
            'Claw Curl Deg': 30 * N(1, 0.4),
            'Claw Pct Length Rad1 Rad2': np.array((0.3, 0.5, 0.0)) * N(1, 0.1, 3)
        }

    def make_part(self, params):

        part = nodegroup_to_part(nodegroup_foot, params, split_extras=True)
        part.iks = {1.0: IKParams('foot', rotation_weight=0.1, chain_parts=2, chain_length=-1)}
        part.settings['rig_extras'] = True
        tag_object(part.obj, 'foot')
        return part