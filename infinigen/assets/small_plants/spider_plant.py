# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Beining Han
# Acknowledgements: This file draws inspiration from https://blenderartists.org/t/extrude-face-along-curve-with-geometry-nodes/1432653/3

import bpy
import mathutils
from numpy.random import uniform, normal, randint
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.nodes import node_utils
from infinigen.core.util.color import color_category
from infinigen.core import surface
from infinigen.core.placement.factory import AssetFactory
from infinigen.assets.materials import spider_plant
import numpy as np

from infinigen.core.util import blender as butil
from infinigen.assets.utils.tag import tag_object, tag_nodegroup

@node_utils.to_nodegroup('nodegroup_set_leaf_countour', singleton=False, type='GeometryNodeTree')
def nodegroup_set_leaf_countour(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
                              expose_input=[('NodeSocketFloat', 'Value', 1.0)])

    float_curve_2 = nw.new_node(Nodes.FloatCurve,
                                input_kwargs={'Value': group_input.outputs["Value"]})
    k = uniform(0, 0.05)
    node_utils.assign_curve(float_curve_2.mapping.curves[0],
                            [(0.0, 0.1), (0.2, 0.1 + k / 1.5), (0.4, 0.1 + k / 1.5),
                             (0.6, 0.1), (0.8, 0.1 - k), (1.0, 0.0)],
                            handles=['AUTO', 'AUTO', 'AUTO', 'AUTO', 'AUTO', 'VECTOR'])

    multiply = nw.new_node(Nodes.Math,
                           input_kwargs={0: float_curve_2, 1: uniform(0.8, 1.3)},
                           attrs={'operation': 'MULTIPLY'})

    combine_xyz_2 = nw.new_node(Nodes.CombineXYZ,
                                input_kwargs={'X': multiply})

    group_output = nw.new_node(Nodes.GroupOutput,
                               input_kwargs={'Vector': combine_xyz_2, 'Value': multiply})


@node_utils.to_nodegroup('nodegroup_leaf_z_rotation', singleton=False, type='GeometryNodeTree')
def nodegroup_leaf_z_rotation(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    position_8 = nw.new_node(Nodes.InputPosition)

    spline_parameter_1 = nw.new_node(Nodes.SplineParameter)

    map_range_1 = nw.new_node(Nodes.MapRange,
                              input_kwargs={'Value': spline_parameter_1.outputs["Factor"], 4: np.abs(normal(0, 0.6))})

    vector_rotate_6 = nw.new_node(Nodes.VectorRotate,
                                  input_kwargs={'Vector': position_8, 'Center': (0.0, 0.0, 0.5),
                                                'Angle': map_range_1.outputs["Result"]},
                                  attrs={'rotation_type': 'Z_AXIS'})

    group_output = nw.new_node(Nodes.GroupOutput,
                               input_kwargs={'Vector': vector_rotate_6})


@node_utils.to_nodegroup('nodegroup_leaf_x_rotation', singleton=False, type='GeometryNodeTree')
def nodegroup_leaf_x_rotation(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    position_5 = nw.new_node(Nodes.InputPosition)

    spline_parameter = nw.new_node(Nodes.SplineParameter)

    map_range = nw.new_node(Nodes.MapRange,
                            input_kwargs={'Value': spline_parameter.outputs["Factor"], 4: np.abs(normal(0, 1.2))})

    vector_rotate_4 = nw.new_node(Nodes.VectorRotate,
                                  input_kwargs={'Vector': position_5, 'Center': (0.0, 0.0, 0.5),
                                                'Angle': map_range.outputs["Result"]},
                                  attrs={'rotation_type': 'X_AXIS'})

    group_output = nw.new_node(Nodes.GroupOutput,
                               input_kwargs={'Vector': vector_rotate_4})


@node_utils.to_nodegroup('nodegroup_leaf_rotate_on_base', singleton=False, type='GeometryNodeTree')
def nodegroup_leaf_rotate_on_base(nw: NodeWrangler, x_R=0.):
    # Code generated using version 2.4.3 of the node_transpiler

    random_value_2 = nw.new_node(Nodes.RandomValue,
                                 input_kwargs={2: -0.3, 3: 0.3})

    add = nw.new_node(Nodes.Math,
                      input_kwargs={0: x_R, 1: random_value_2.outputs[1]})

    random_value_3 = nw.new_node(Nodes.RandomValue,
                                 input_kwargs={2: -0.6, 3: 0.6})

    noise_texture_1 = nw.new_node(Nodes.NoiseTexture)

    map_range_1 = nw.new_node(Nodes.MapRange,
                              input_kwargs={'Value': noise_texture_1.outputs["Fac"], 3: -0.5, 4: 0.5})

    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ,
                                input_kwargs={'X': add, 'Y': random_value_3.outputs[1],
                                              'Z': map_range_1.outputs["Result"]})

    group_output = nw.new_node(Nodes.GroupOutput,
                               input_kwargs={'Vector': combine_xyz_1})


@node_utils.to_nodegroup('nodegroup_leaf_scale_align', singleton=False, type='GeometryNodeTree')
def nodegroup_leaf_scale_align(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    normal = nw.new_node(Nodes.InputNormal)

    align_euler_to_vector = nw.new_node(Nodes.AlignEulerToVector,
                                        input_kwargs={'Vector': normal},
                                        attrs={'axis': 'Y'})

    noise_texture = nw.new_node(Nodes.NoiseTexture)

    map_range = nw.new_node(Nodes.MapRange,
                            input_kwargs={'Value': noise_texture.outputs["Fac"], 3: 0.6, 4: 1.1})

    group_output = nw.new_node(Nodes.GroupOutput,
                               input_kwargs={'Rotation': align_euler_to_vector, 'Result': map_range.outputs["Result"]})


@node_utils.to_nodegroup('nodegroup_leaf_geometry', singleton=False, type='GeometryNodeTree')
def nodegroup_leaf_geometry(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    quadratic_bezier = nw.new_node(Nodes.QuadraticBezier,
                                   input_kwargs={'Resolution': 100, 'Start': (0.0, 0.0, 0.0), 'Middle': (0.0, 0.0, 0.5),
                                                 'End': (0.0, 0.0, 1.0)})

    leaf_x_rotation = nw.new_node(nodegroup_leaf_x_rotation().name)

    set_position_7 = nw.new_node(Nodes.SetPosition,
                                 input_kwargs={'Geometry': quadratic_bezier, 'Offset': leaf_x_rotation})

    leaf_z_rotation = nw.new_node(nodegroup_leaf_z_rotation().name)

    set_position_2 = nw.new_node(Nodes.SetPosition,
                                 input_kwargs={'Geometry': set_position_7, 'Offset': leaf_z_rotation})

    spline_parameter_3 = nw.new_node(Nodes.SplineParameter)

    capture_attribute_3 = nw.new_node(Nodes.CaptureAttribute,
                                      input_kwargs={'Geometry': set_position_2,
                                                    2: spline_parameter_3.outputs["Factor"]})

    normal_1 = nw.new_node(Nodes.InputNormal)

    capture_attribute_2 = nw.new_node(Nodes.CaptureAttribute,
                                      input_kwargs={'Geometry': capture_attribute_3.outputs["Geometry"], 1: normal_1},
                                      attrs={'data_type': 'FLOAT_VECTOR'})

    set_leaf_countour = nw.new_node(nodegroup_set_leaf_countour().name,
                                    input_kwargs={'Value': capture_attribute_3.outputs[2]})

    set_position_8 = nw.new_node(Nodes.SetPosition,
                                 input_kwargs={'Geometry': capture_attribute_2.outputs["Geometry"],
                                               'Offset': set_leaf_countour.outputs["Vector"]})

    curve_to_mesh_2 = nw.new_node(Nodes.CurveToMesh,
                                  input_kwargs={'Curve': set_position_8, 'Fill Caps': True})

    extrude_mesh_3 = nw.new_node(Nodes.ExtrudeMesh,
                                 input_kwargs={'Mesh': curve_to_mesh_2,
                                               'Offset': capture_attribute_2.outputs["Attribute"],
                                               'Offset Scale': set_leaf_countour.outputs["Value"]},
                                 attrs={'mode': 'EDGES'})

    group_output = nw.new_node(Nodes.GroupOutput,
                               input_kwargs={'Mesh': extrude_mesh_3})


def geometry_spider_plant_nodes(nw: NodeWrangler, **kwargs):
    # Code generated using version 2.4.3 of the node_transpiler
    num_leaf_versions = kwargs["num_leaf_versions"]
    num_plant_bases = kwargs["num_plant_bases"]
    base_radius = kwargs["base_radius"]
    leaf_x_R = kwargs["leaf_x_R"]
    leaf_x_S = kwargs["leaf_x_S"]

    leaves, bases = [], []
    for _ in range(num_leaf_versions):
        leaf = nw.new_node(nodegroup_leaf_geometry().name)
        leaves.append(leaf)

    geometry_to_instance = nw.new_node('GeometryNodeGeometryToInstance',
                                       input_kwargs={'Geometry': leaves})

    for i in range(num_plant_bases):
        curve_circle = nw.new_node(Nodes.CurveCircle,
                                   input_kwargs={'Radius': base_radius[i]})

        resample_curve = nw.new_node(Nodes.ResampleCurve,
                                     input_kwargs={'Curve': curve_circle.outputs["Curve"], 'Count': randint(20, 40)})

        random_value = nw.new_node(Nodes.RandomValue,
                                   input_kwargs={2: -0.3 * base_radius[i], 3: 0.3 * base_radius[i]})

        random_value_1 = nw.new_node(Nodes.RandomValue,
                                     input_kwargs={2: -0.3 * base_radius[i], 3: 0.3 * base_radius[i]})

        combine_xyz = nw.new_node(Nodes.CombineXYZ,
                                  input_kwargs={'X': random_value.outputs[1], 'Y': random_value_1.outputs[1]})

        set_position_3 = nw.new_node(Nodes.SetPosition,
                                     input_kwargs={'Geometry': resample_curve, 'Offset': combine_xyz})

        subdivision_surface = nw.new_node(Nodes.SubdivisionSurface,
                                          input_kwargs={'Mesh': geometry_to_instance})

        leaf_scale_align = nw.new_node(nodegroup_leaf_scale_align().name)

        instance_on_points = nw.new_node(Nodes.InstanceOnPoints,
                                         input_kwargs={'Points': set_position_3, 'Instance': subdivision_surface,
                                                       'Pick Instance': True,
                                                       'Rotation': leaf_scale_align.outputs["Rotation"],
                                                       'Scale': leaf_scale_align.outputs["Result"]})

        value = nw.new_node(Nodes.Value)
        value.outputs[0].default_value = leaf_x_S[i]

        scale_instances = nw.new_node(Nodes.ScaleInstances,
                                      input_kwargs={'Instances': instance_on_points, 'Scale': value})

        leaf_rotate_on_base = nw.new_node(nodegroup_leaf_rotate_on_base(x_R=leaf_x_R[i]).name)

        rotate_instances = nw.new_node(Nodes.RotateInstances,
                                       input_kwargs={'Instances': scale_instances, 'Rotation': leaf_rotate_on_base})

        realize_instances = nw.new_node(Nodes.RealizeInstances,
                                        input_kwargs={'Geometry': rotate_instances})
        bases.append(realize_instances)

    join_geometry = nw.new_node(Nodes.JoinGeometry, input_kwargs={'Geometry': bases})

    set_shade_smooth = nw.new_node(Nodes.SetShadeSmooth,
                                   input_kwargs={'Geometry': join_geometry})

    set_material = nw.new_node(Nodes.SetMaterial,
                               input_kwargs={'Geometry': set_shade_smooth,
                                             'Material': surface.shaderfunc_to_material(spider_plant.shader_spider_plant)})

    group_output = nw.new_node(Nodes.GroupOutput,
                               input_kwargs={'Geometry': set_material})


class SpiderPlantFactory(AssetFactory):
    def __init__(self, factory_seed, coarse=False):
        super(SpiderPlantFactory, self).__init__(factory_seed, coarse=coarse)

    def get_params(self):
        params = {}
        params["num_leaf_versions"] = randint(4, 8)
        num_bases = randint(5, 12)
        params["num_plant_bases"] = num_bases
        base_radius, leaf_x_R, leaf_x_S = [], [], []
        init_base_radius = uniform(0.10, 0.20)
        diff_base_radius = init_base_radius - 0.04
        init_x_R, diff_x_R = uniform(1.2, 1.5), uniform(0.7, 1.1)
        init_x_S, diff_x_S = uniform(1.4, 2.0), uniform(0.2, 0.6)
        for i in range(params["num_plant_bases"]):
            base_radius.append(init_base_radius - (i * diff_base_radius) / num_bases)
            leaf_x_R.append(init_x_R - (i * diff_x_R) / num_bases)
            leaf_x_S.append(init_x_S - (i * diff_x_S) / num_bases)
        params["base_radius"] = base_radius
        params["leaf_x_R"] = leaf_x_R
        params["leaf_x_S"] = leaf_x_S

        return params

    def create_asset(self, **params):
        bpy.ops.mesh.primitive_plane_add(
            size=1, enter_editmode=False, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
        obj = bpy.context.active_object

        params = self.get_params()

        surface.add_geomod(obj, geometry_spider_plant_nodes, apply=True, input_kwargs=params)
        surface.add_material(obj, spider_plant.shader_spider_plant, selection=None)

        # convert to appropriate units - TODO replace this
        butil.apply_modifiers(obj)
        obj.scale = (0.1, 0.1, 0.1)
        butil.apply_transform(obj, scale=True)

        tag_object(obj, 'spider_plant')
        return obj


if __name__ == '__main__':
    fac = SpiderPlantFactory(0)
    fac.create_asset()
