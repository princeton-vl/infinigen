import bpy

import numpy as np
from numpy.random import uniform, normal as N

from assets.creatures.genome import Joint, IKParams

from nodes.node_wrangler import Nodes, NodeWrangler
from nodes import node_utils
from assets.creatures.nodegroups.curve import nodegroup_simple_tube, nodegroup_simple_tube_v2
from assets.creatures.nodegroups.attach import nodegroup_surface_muscle, nodegroup_attach_part
from assets.creatures.nodegroups.math import nodegroup_deg2_rad

from assets.creatures.creature import Part, PartFactory
from assets.creatures.util.part_util import nodegroup_to_part

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

def nodegroup_foot(nw: NodeWrangler):

    group_input = nw.new_node(Nodes.GroupInput,
    simple_tube_v2 = nw.new_node(nodegroup_simple_tube_v2().name,
    separate_xyz = nw.new_node(Nodes.SeparateXYZ,
    multiply_add = nw.new_node(Nodes.VectorMath,
    add = nw.new_node(Nodes.VectorMath,
    multiply_add_1 = nw.new_node(Nodes.VectorMath,
    add_1 = nw.new_node(Nodes.VectorMath,
    mesh_line = nw.new_node(Nodes.MeshLine,
    tigertoe = nw.new_node(nodegroup_tiger_toe().name,
    instance_on_points = nw.new_node(Nodes.InstanceOnPoints,
    rotate_instances_1 = nw.new_node(Nodes.RotateInstances,
    index = nw.new_node(Nodes.Index)
    add_2 = nw.new_node(Nodes.Math,
    divide = nw.new_node(Nodes.Math,
    scale = nw.new_node(Nodes.VectorMath,
    scale_1 = nw.new_node(Nodes.VectorMath,
    map_range = nw.new_node(Nodes.MapRange,
    deg2rad = nw.new_node(nodegroup_deg2_rad().name,
    rotate_instances = nw.new_node(Nodes.RotateInstances,
    realize_instances = nw.new_node(Nodes.RealizeInstances,
    uv_sphere = nw.new_node(Nodes.MeshUVSphere,
    add_3 = nw.new_node(Nodes.VectorMath,
    transform = nw.new_node(Nodes.Transform,
    reroute = nw.new_node(Nodes.Reroute,
    reroute_1 = nw.new_node(Nodes.Reroute,
    multiply = nw.new_node(Nodes.VectorMath,
    tigertoe_1 = nw.new_node(nodegroup_tiger_toe().name,
    value_2 = nw.new_node(Nodes.Value)
    vector_1 = nw.new_node(Nodes.Vector)
    value_1 = nw.new_node(Nodes.Value)
    vector = nw.new_node(Nodes.Vector)
    attach_part = nw.new_node(nodegroup_attach_part().name,
    join_geometry = nw.new_node(Nodes.JoinGeometry,
    instance_on_points_1 = nw.new_node(Nodes.InstanceOnPoints,
    rotate_instances_2 = nw.new_node(Nodes.RotateInstances,
    rotate_instances_3 = nw.new_node(Nodes.RotateInstances,
    realize_instances_1 = nw.new_node(Nodes.RealizeInstances,
    attach_part_1 = nw.new_node(nodegroup_attach_part().name,
    join_geometry_1 = nw.new_node(Nodes.JoinGeometry,
    group_output = nw.new_node(Nodes.GroupOutput,

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
            'Toebean Radius': 0.03 * N(1, 0.2),
            'Claw Curl Deg': 30 * N(1, 0.4),
            'Claw Pct Length Rad1 Rad2': np.array((0.3, 0.5, 0.0)) * N(1, 0.1, 3)
        }

    def make_part(self, params):

        part = nodegroup_to_part(nodegroup_foot, params, split_extras=True)
        part.iks = {1.0: IKParams('foot', rotation_weight=0.1, chain_parts=2, chain_length=-1)}
        part.settings['rig_extras'] = True
        return part