# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Beining Han


import bpy
import mathutils
from numpy.random import uniform, normal, randint
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.nodes import node_utils
from infinigen.core import surface
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.util import blender as butil
from infinigen.assets.small_plants.leaf_general import LeafFactory
from infinigen.assets.small_plants.leaf_heart import LeafHeartFactory
from infinigen.assets.materials import simple_greenery
import numpy as np
from infinigen.assets.utils.tag import tag_object, tag_nodegroup

@node_utils.to_nodegroup('nodegroup_leafon_stem', singleton=False, type='GeometryNodeTree')
def nodegroup_leaf_on_stem(nw: NodeWrangler, z_rotation=(0, 0, 0,), leaf_scale=1.0, leaf=None):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
                              expose_input=[('NodeSocketGeometry', 'Points', None)])

    endpoint_selection = nw.new_node('GeometryNodeCurveEndpointSelection',
                                     input_kwargs={'Start Size': 0})

    object_info = nw.new_node(Nodes.ObjectInfo,
                              input_kwargs={'Object': leaf})

    curve_tangent = nw.new_node(Nodes.CurveTangent)

    align_euler_to_vector = nw.new_node(Nodes.AlignEulerToVector,
                                        input_kwargs={'Vector': curve_tangent},
                                        attrs={'axis': 'Z'})

    value = nw.new_node(Nodes.Value)
    value.outputs[0].default_value = leaf_scale

    instance_on_points = nw.new_node(Nodes.InstanceOnPoints,
                                     input_kwargs={'Points': group_input.outputs["Points"],
                                                   'Selection': endpoint_selection,
                                                   'Instance': object_info.outputs["Geometry"],
                                                   'Rotation': align_euler_to_vector, 'Scale': value})

    vector_1 = nw.new_node(Nodes.Vector)
    vector_1.vector = z_rotation

    rotate_instances = nw.new_node(Nodes.RotateInstances,
                                   input_kwargs={'Instances': instance_on_points, 'Rotation': vector_1})

    group_output = nw.new_node(Nodes.GroupOutput,
                               input_kwargs={'Instances': rotate_instances})


@node_utils.to_nodegroup('nodegroup_stem_geometry', singleton=False, type='GeometryNodeTree')
def nodegroup_stem_geometry(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
                              expose_input=[('NodeSocketGeometry', 'Curve', None)])

    spline_parameter = nw.new_node(Nodes.SplineParameter)

    map_range = nw.new_node(Nodes.MapRange,
                            input_kwargs={'Value': spline_parameter.outputs["Factor"], 3: 1.0, 4: 0.4})

    set_curve_radius = nw.new_node(Nodes.SetCurveRadius,
                                   input_kwargs={'Curve': group_input.outputs["Curve"],
                                                 'Radius': map_range.outputs["Result"]})

    curve_circle = nw.new_node(Nodes.CurveCircle,
                               input_kwargs={'Resolution': 12, 'Radius': 0.03})

    curve_to_mesh = nw.new_node(Nodes.CurveToMesh,
                                input_kwargs={'Curve': set_curve_radius, 'Profile Curve': curve_circle.outputs["Curve"],
                                              'Fill Caps': True})

    group_output = nw.new_node(Nodes.GroupOutput,
                               input_kwargs={'Mesh': tag_nodegroup(nw, curve_to_mesh, 'stem')})


def geo_face_colors(nw: NodeWrangler, **kwargs):
    # Code generated using version 2.4.3 of the node_transpiler

    rotation_scale = kwargs["stem_rotation"]
    leaf_num = kwargs["leaf_num"]
    leaf = kwargs["leaf"]
    mid_z = uniform(0.35, 0.65, size=(1,))[0]
    mid_x = normal(0., rotation_scale, size=(1,))[0]
    mid_y = normal(0., rotation_scale, size=(1,))[0]
    vector_2 = nw.new_node(Nodes.Vector)
    vector_2.vector = (mid_x, mid_y, mid_z)

    top_x = normal(0., rotation_scale, size=(1,))[0]
    top_y = normal(0., rotation_scale, size=(1,))[0]
    vector = nw.new_node(Nodes.Vector)
    vector.vector = (top_x, top_y, 1.0)

    quadratic_bezier = nw.new_node(Nodes.QuadraticBezier,
                                   input_kwargs={'Resolution': 25, 'Start': (0.0, 0.0, 0.0), 'Middle': vector_2,
                                                 'End': vector})

    noise_texture = nw.new_node(Nodes.NoiseTexture,
                                input_kwargs={'Scale': 1.0, 'Roughness': 0.2})

    add = nw.new_node(Nodes.VectorMath,
                      input_kwargs={0: noise_texture.outputs["Fac"], 1: (-0.5, -0.5, -0.5)})

    spline_parameter_1 = nw.new_node(Nodes.SplineParameter)

    multiply = nw.new_node(Nodes.VectorMath,
                           input_kwargs={0: add.outputs["Vector"], 1: spline_parameter_1.outputs["Factor"]},
                           attrs={'operation': 'MULTIPLY'})

    set_position = nw.new_node(Nodes.SetPosition,
                               input_kwargs={'Geometry': quadratic_bezier, 'Offset': multiply.outputs["Vector"]})

    stemgeometry = nw.new_node(nodegroup_stem_geometry().name,
                               input_kwargs={'Curve': set_position})

    leaf_scale = uniform(0.15, 0.35, size=(1,))[0] * kwargs["leaf_scale"]
    leaves = []
    rotation = 0
    for _ in range(leaf_num):
        leaves.append(nw.new_node(nodegroup_leaf_on_stem(z_rotation=(0, 0, rotation), leaf_scale=leaf_scale, leaf=leaf).name,
                                  input_kwargs={'Points': set_position}))
        rotation += 6.28 / leaf_num

    join_geometry = nw.new_node(Nodes.JoinGeometry,
                                input_kwargs={'Geometry': leaves + [stemgeometry]})

    realize_instances = nw.new_node(Nodes.RealizeInstances,
                                    input_kwargs={'Geometry': join_geometry})

    colored = nw.new_node(Nodes.SetMaterial,
                               input_kwargs={'Geometry': realize_instances,
                                             'Material': surface.shaderfunc_to_material(simple_greenery.shader_simple_greenery)})

    group_output = nw.new_node(Nodes.GroupOutput,
                               input_kwargs={'Geometry': colored})


class NumLeafGrassFactory(AssetFactory):
    def __init__(self, factory_seed, coarse=False):
        super(NumLeafGrassFactory, self).__init__(factory_seed, coarse=coarse)
        self.leaf_num = [2, 3, 4]
        self.leaf_model = [LeafFactory, LeafHeartFactory]

    def create_asset(self, **params):
        bpy.ops.mesh.primitive_plane_add(
            size=1, enter_editmode=False, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
        obj = bpy.context.active_object

        lf_seed = randint(0, 1000, size=(1,))[0]
        leaf_num = np.random.choice(self.leaf_num, size=(1,), p=[0.2, 0.4, 0.4])[0]
        z_offset = normal(0, 0.05, size=(1,))[0]
        if leaf_num == 2:
            leaf_model = LeafFactory(genome={"leaf_width": 0.95, "width_rand": 0.1,
                                             "z_scaling": z_offset}, factory_seed=lf_seed)
            leaf = leaf_model.create_asset()
            params["leaf_scale"] = 2.0
        elif leaf_num == 3:
            leaf_model = LeafHeartFactory(genome={"leaf_width": 1.1, "width_rand": 0.05,
                                                  "z_scaling": z_offset}, factory_seed=lf_seed)
            leaf = leaf_model.create_asset()
            params["leaf_scale"] = 1.0
        else:
            leaf_model = LeafHeartFactory(genome={"leaf_width": 0.85, "width_rand": 0.05,
                                                  "z_scaling": z_offset}, factory_seed=lf_seed)
            leaf = leaf_model.create_asset()
            params["leaf_scale"] = 1.0

        params["leaf"] = leaf
        params["leaf_num"] = leaf_num
        params["stem_rotation"] = 0.15

        surface.add_geomod(obj, geo_face_colors, apply=True, attributes=[], input_kwargs=params)
        butil.delete([leaf])
        with butil.SelectObjects(obj):
            bpy.ops.object.material_slot_remove()
            bpy.ops.object.shade_flat()

        tag_object(obj, 'num_leaf_grass')
        return obj


# if __name__ == '__main__':
#     grass = NumLeafGrassFactory(0)
#     obj = grass.create_asset()