# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Beining Han


import bpy
import mathutils
from numpy.random import uniform, normal, randint
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.nodes import node_utils
from infinigen.core.util.color import color_category
from infinigen.core import surface
from infinigen.assets.materials import succulent
from infinigen.core.placement.factory import AssetFactory
import numpy as np

from infinigen.core.util import blender as butil
from infinigen.assets.utils.tag import tag_object, tag_nodegroup

@node_utils.to_nodegroup('nodegroup_pedal_cross_contour_top', singleton=False, type='GeometryNodeTree')
def nodegroup_pedal_cross_contour_top(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    normal_2 = nw.new_node(Nodes.InputNormal)

    group_input = nw.new_node(Nodes.GroupInput,
                              expose_input=[('NodeSocketFloat', 'Y', 0.0),
                                            ('NodeSocketFloat', 'X', 0.0)])

    combine_xyz_3 = nw.new_node(Nodes.CombineXYZ,
                                input_kwargs={'X': group_input.outputs["X"], 'Y': group_input.outputs["Y"]})

    multiply = nw.new_node(Nodes.VectorMath,
                           input_kwargs={0: normal_2, 1: combine_xyz_3},
                           attrs={'operation': 'MULTIPLY'})

    index_1 = nw.new_node(Nodes.Index)

    greater_than = nw.new_node(Nodes.Math,
                               input_kwargs={0: index_1, 1: 63.0},
                               attrs={'operation': 'GREATER_THAN'})

    group_output = nw.new_node(Nodes.GroupOutput,
                               input_kwargs={'Vector': multiply.outputs["Vector"], 'Value': greater_than})


@node_utils.to_nodegroup('nodegroup_pedal_cross_contour_bottom', singleton=False, type='GeometryNodeTree')
def nodegroup_pedal_cross_contour_bottom(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    normal = nw.new_node(Nodes.InputNormal)

    group_input = nw.new_node(Nodes.GroupInput,
                              expose_input=[('NodeSocketFloat', 'Y', 0.0),
                                            ('NodeSocketFloat', 'X', 0.0)])

    combine_xyz = nw.new_node(Nodes.CombineXYZ,
                              input_kwargs={'X': group_input.outputs["X"], 'Y': group_input.outputs["Y"]})

    multiply = nw.new_node(Nodes.VectorMath,
                           input_kwargs={0: normal, 1: combine_xyz},
                           attrs={'operation': 'MULTIPLY'})

    index = nw.new_node(Nodes.Index)

    less_than = nw.new_node(Nodes.Math,
                            input_kwargs={0: index, 1: 64.0},
                            attrs={'operation': 'LESS_THAN'})

    group_output = nw.new_node(Nodes.GroupOutput,
                               input_kwargs={'Vector': multiply.outputs["Vector"], 'Value': less_than})


@node_utils.to_nodegroup('nodegroup_pedal_cross_contour', singleton=False, type='GeometryNodeTree')
def nodegroup_pedal_cross_contour(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    curve_circle = nw.new_node(Nodes.CurveCircle,
                               input_kwargs={'Resolution': 128, 'Radius': 0.05})

    group_input = nw.new_node(Nodes.GroupInput,
                              expose_input=[('NodeSocketFloat', 'Y_bottom', 0.0),
                                            ('NodeSocketFloat', 'X', 0.0),
                                            ('NodeSocketFloat', 'Y_top', 0.0)])

    pedal_cross_contour_bottom = nw.new_node(nodegroup_pedal_cross_contour_bottom().name,
                                             input_kwargs={'Y': group_input.outputs["Y_bottom"],
                                                           'X': group_input.outputs["X"]})

    set_position_1 = nw.new_node(Nodes.SetPosition,
                                 input_kwargs={'Geometry': curve_circle.outputs["Curve"],
                                               'Selection': pedal_cross_contour_bottom.outputs["Value"],
                                               'Offset': pedal_cross_contour_bottom.outputs["Vector"]})

    pedal_cross_contour_top = nw.new_node(nodegroup_pedal_cross_contour_top().name,
                                          input_kwargs={'Y': group_input.outputs["Y_top"],
                                                        'X': group_input.outputs["X"]})

    set_position_2 = nw.new_node(Nodes.SetPosition,
                                 input_kwargs={'Geometry': set_position_1,
                                               'Selection': pedal_cross_contour_top.outputs["Value"],
                                               'Offset': pedal_cross_contour_top.outputs["Vector"]})

    noise_texture_2 = nw.new_node(Nodes.NoiseTexture,
                                  input_kwargs={'W': 7.0, 'Detail': 15.0},
                                  attrs={'noise_dimensions': '4D'})

    scale = nw.new_node(Nodes.VectorMath,
                        input_kwargs={0: noise_texture_2.outputs["Fac"], 'Scale': uniform(0.00, 0.02)},
                        attrs={'operation': 'SCALE'})

    set_position_5 = nw.new_node(Nodes.SetPosition,
                                 input_kwargs={'Geometry': set_position_2, 'Offset': scale.outputs["Vector"]})

    group_output = nw.new_node(Nodes.GroupOutput,
                               input_kwargs={'Geometry': set_position_5})


@node_utils.to_nodegroup('nodegroup_pedal_z_contour', singleton=False, type='GeometryNodeTree')
def nodegroup_pedal_z_contour(nw: NodeWrangler, curve_param=[]):
    # Code generated using version 2.4.3 of the node_transpiler

    spline_parameter = nw.new_node(Nodes.SplineParameter)

    float_curve = nw.new_node(Nodes.FloatCurve,
                              input_kwargs={'Value': spline_parameter.outputs["Factor"]})
    node_utils.assign_curve(float_curve.mapping.curves[0],
                            [(0.0, curve_param[0]), (0.2, curve_param[1] * (1. + normal(0, 0.04))),
                             (0.4, curve_param[2] * (1. + normal(0, 0.1))), (0.6, curve_param[3] * (1. + normal(0, 0.03))),
                             (0.8, curve_param[4] * (1. + normal(0, 0.06))), (0.9, curve_param[5] * (1. + normal(0, 0.04))),
                             (1.0, 0.0)])

    group_input = nw.new_node(Nodes.GroupInput,
                              expose_input=[('NodeSocketFloat', 'Value', 0.5)])

    multiply = nw.new_node(Nodes.Math,
                           input_kwargs={0: float_curve, 1: group_input.outputs["Value"]},
                           attrs={'operation': 'MULTIPLY'})

    group_output = nw.new_node(Nodes.GroupOutput,
                               input_kwargs={'Value': multiply})


@node_utils.to_nodegroup('nodegroup_pedal_stem_curvature', singleton=False, type='GeometryNodeTree')
def nodegroup_pedal_stem_curvature(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    position_3 = nw.new_node(Nodes.InputPosition)

    spline_parameter_1 = nw.new_node(Nodes.SplineParameter)

    float_curve_1 = nw.new_node(Nodes.FloatCurve,
                                input_kwargs={'Value': spline_parameter_1.outputs["Factor"]})
    k = uniform(0.0, 0.3)
    node_utils.assign_curve(float_curve_1.mapping.curves[0],
                            [(0.0, 0.0), (0.2, 0.2 - k / 2.5), (0.4, 0.4 - k / 1.1), (0.6, 0.6 - k),
                             (0.8, 0.8 - k / 1.5), (1.0, 1.0 - k / 3.)])

    group_input = nw.new_node(Nodes.GroupInput,
                              expose_input=[('NodeSocketFloat', 'Value', 0.2)])

    multiply = nw.new_node(Nodes.Math,
                           input_kwargs={0: float_curve_1, 1: group_input.outputs["Value"]},
                           attrs={'operation': 'MULTIPLY'})

    vector_rotate = nw.new_node(Nodes.VectorRotate,
                                input_kwargs={'Vector': position_3, 'Center': (0.0, 0.0, 0.2), 'Angle': multiply},
                                attrs={'rotation_type': 'X_AXIS'})

    group_output = nw.new_node(Nodes.GroupOutput,
                               input_kwargs={'Vector': vector_rotate})


@node_utils.to_nodegroup('nodegroup_pedal_rotation_on_base_circle', singleton=False, type='GeometryNodeTree')
def nodegroup_pedal_rotation_on_base_circle(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    random_value_1 = nw.new_node(Nodes.RandomValue,
                                 input_kwargs={2: -0.1, 3: 0.1})

    group_input = nw.new_node(Nodes.GroupInput,
                              expose_input=[('NodeSocketFloat', 'Value1', -1.3),
                                            ('NodeSocketFloat', 'Value2', -1.57)])

    add = nw.new_node(Nodes.Math,
                      input_kwargs={0: random_value_1.outputs[1], 1: group_input.outputs["Value1"]})

    random_value_2 = nw.new_node(Nodes.RandomValue,
                                 input_kwargs={2: -0.3, 3: 0.3})

    add_1 = nw.new_node(Nodes.Math,
                        input_kwargs={0: random_value_2.outputs[1], 1: group_input.outputs["Value2"]})

    combine_xyz_2 = nw.new_node(Nodes.CombineXYZ,
                                input_kwargs={'X': add, 'Z': add_1})

    group_output = nw.new_node(Nodes.GroupOutput,
                               input_kwargs={'Vector': combine_xyz_2})


@node_utils.to_nodegroup('nodegroup_base_perturbation', singleton=False, type='GeometryNodeTree')
def nodegroup_base_perturbation(nw: NodeWrangler, R=1.0):
    # Code generated using version 2.4.3 of the node_transpiler

    random_value_4 = nw.new_node(Nodes.RandomValue,
                                 input_kwargs={2: -0.8 * R, 3: 0.8 * R})

    random_value = nw.new_node(Nodes.RandomValue,
                               input_kwargs={2: -0.8 * R, 3: 0.8 * R})

    random_value_1 = nw.new_node(Nodes.RandomValue,
                                 input_kwargs={2: -0.2 * R, 3: 0.2 * R})

    group_input = nw.new_node(Nodes.GroupInput,
                              expose_input=[('NodeSocketFloat', 'Value', 0.5)])

    add = nw.new_node(Nodes.Math,
                      input_kwargs={0: random_value_1.outputs[1], 1: group_input.outputs["Value"]})

    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ,
                                input_kwargs={'X': random_value_4.outputs[1], 'Y': random_value.outputs[1], 'Z': add})

    group_output = nw.new_node(Nodes.GroupOutput,
                               input_kwargs={'Vector': combine_xyz_1})


@node_utils.to_nodegroup('nodegroup_pedal_geometry', singleton=False, type='GeometryNodeTree')
def nodegroup_pedal_geometry(nw: NodeWrangler, curve_param=[]):
    # Code generated using version 2.4.3 of the node_transpiler

    curve_line = nw.new_node(Nodes.CurveLine,
                             input_kwargs={'End': (0.0, 0.0, 0.2)})

    integer = nw.new_node(Nodes.Integer,
                          attrs={'integer': 64})
    integer.integer = 64

    resample_curve = nw.new_node(Nodes.ResampleCurve,
                                 input_kwargs={'Curve': curve_line, 'Count': integer})

    group_input = nw.new_node(Nodes.GroupInput,
                              expose_input=[('NodeSocketFloat', 'Y_bottom', 0.0),
                                            ('NodeSocketFloat', 'X', 0.0),
                                            ('NodeSocketFloat', 'Y_top', 0.0),
                                            ('NodeSocketFloat', 'pedal_stem', 0.2),
                                            ('NodeSocketFloat', 'pedal_z', 0.5)])

    pedal_stem_curvature = nw.new_node(nodegroup_pedal_stem_curvature().name,
                                       input_kwargs={'Value': group_input.outputs["pedal_stem"]})

    set_position_4 = nw.new_node(Nodes.SetPosition,
                                 input_kwargs={'Geometry': resample_curve, 'Offset': pedal_stem_curvature})

    pedal_z_contour = nw.new_node(nodegroup_pedal_z_contour(curve_param=curve_param).name,
                                  input_kwargs={'Value': group_input.outputs["pedal_z"]})

    set_curve_radius = nw.new_node(Nodes.SetCurveRadius,
                                   input_kwargs={'Curve': set_position_4, 'Radius': pedal_z_contour})

    pedal_cross_contour = nw.new_node(nodegroup_pedal_cross_contour().name,
                                      input_kwargs={'Y_bottom': group_input.outputs["Y_bottom"],
                                                    'X': group_input.outputs["X"],
                                                    'Y_top': group_input.outputs["Y_top"]})

    curve_to_mesh = nw.new_node(Nodes.CurveToMesh,
                                input_kwargs={'Curve': set_curve_radius, 'Profile Curve': pedal_cross_contour,
                                              'Fill Caps': True})

    group_output = nw.new_node(Nodes.GroupOutput,
                               input_kwargs={'Mesh': curve_to_mesh})


@node_utils.to_nodegroup('nodegroup_pedal_on_base', singleton=False, type='GeometryNodeTree')
def nodegroup_pedal_on_base(nw: NodeWrangler, R=1.0):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
                              expose_input=[('NodeSocketFloatDistance', 'Radius', 0.1),
                                            ('NodeSocketFloat', 'x_R', -1.3),
                                            ('NodeSocketFloat', 'z_R', -1.57),
                                            ('NodeSocketInt', 'Resolution', 10),
                                            ('NodeSocketGeometry', 'Instance', None),
                                            ('NodeSocketVectorXYZ', 'Scale', (1.0, 1.0, 1.0)),
                                            ('NodeSocketFloat', 'base_z', 0.5)])

    curve_circle_1 = nw.new_node(Nodes.CurveCircle,
                                 input_kwargs={'Resolution': group_input.outputs["Resolution"],
                                               'Radius': group_input.outputs["Radius"]})

    base_perturbation = nw.new_node(nodegroup_base_perturbation(R=R).name,
                                    input_kwargs={'Value': group_input.outputs["base_z"]})

    set_position = nw.new_node(Nodes.SetPosition,
                               input_kwargs={'Geometry': curve_circle_1.outputs["Curve"], 'Offset': base_perturbation})

    normal_1 = nw.new_node(Nodes.InputNormal)

    align_euler_to_vector_1 = nw.new_node(Nodes.AlignEulerToVector,
                                          input_kwargs={'Vector': normal_1},
                                          attrs={'pivot_axis': 'Z'})

    random_value_3 = nw.new_node(Nodes.RandomValue,
                                 input_kwargs={2: 0.7, 3: 1.2})

    instance_on_points_1 = nw.new_node(Nodes.InstanceOnPoints,
                                       input_kwargs={'Points': set_position,
                                                     'Instance': group_input.outputs["Instance"],
                                                     'Rotation': align_euler_to_vector_1,
                                                     'Scale': random_value_3.outputs[1]})

    realize_instances_1 = nw.new_node(Nodes.RealizeInstances,
        input_kwargs={'Geometry': instance_on_points_1})

    pedal_rotation_on_base_circle = nw.new_node(nodegroup_pedal_rotation_on_base_circle().name,
                                                input_kwargs={0: group_input.outputs["x_R"],
                                                              1: group_input.outputs["z_R"]})

    rotate_instances_1 = nw.new_node(Nodes.RotateInstances,
                                     input_kwargs={'Instances': realize_instances_1,
                                                   'Rotation': pedal_rotation_on_base_circle})

    scale_instances = nw.new_node(Nodes.ScaleInstances,
                                  input_kwargs={'Instances': rotate_instances_1, 'Scale': group_input.outputs["Scale"]})

    group_output = nw.new_node(Nodes.GroupOutput,
                               input_kwargs={'Instances': scale_instances})


def geometry_succulent_nodes(nw: NodeWrangler, **kwargs):
    # Code generated using version 2.4.3 of the node_transpiler
    pedal_bases = []

    pedal_cross_coutour_y_bottom = nw.new_node(Nodes.Value,
                                               label='pedal_cross_coutour_y_bottom')
    pedal_cross_coutour_y_bottom.outputs[0].default_value = kwargs["cross_y_bottom"]

    pedal_cross_coutour_x = nw.new_node(Nodes.Value,
                                        label='pedal_cross_coutour_x')
    pedal_cross_coutour_x.outputs[0].default_value = kwargs["cross_x"]

    pedal_cross_coutour_y_top = nw.new_node(Nodes.Value,
                                            label='pedal_cross_coutour_y_top')
    pedal_cross_coutour_y_top.outputs[0].default_value = kwargs["cross_y_top"]
    pedal_stem_curvature_scale = nw.new_node(Nodes.Value,
                                             label='pedal_stem_curvature_scale')
    pedal_stem_curvature_scale.outputs[0].default_value = np.abs(normal(0, 1.0))

    pedal_z_coutour_scale = nw.new_node(Nodes.Value,
                                        label='pedal_z_coutour_scale')
    pedal_z_coutour_scale.outputs[0].default_value = uniform(0.4, 0.9)
    material = kwargs["material"]

    for i in range(kwargs["num_bases"]):
        pedal_geometry = nw.new_node(nodegroup_pedal_geometry(curve_param=kwargs["pedal_curve_param"]).name,
                                     input_kwargs={'Y_bottom': pedal_cross_coutour_y_bottom,
                                                   'X': pedal_cross_coutour_x,
                                                   'Y_top': pedal_cross_coutour_y_top,
                                                   'pedal_stem': pedal_stem_curvature_scale,
                                                   'pedal_z': pedal_z_coutour_scale})

        base_circle_radius = nw.new_node(Nodes.Value,
                                         label='base_circle_radius')
        base_circle_radius.outputs[0].default_value = kwargs["base_radius"][i]

        pedal_x_rotation = nw.new_node(Nodes.Value,
                                       label='pedal_x_rotation')
        pedal_x_rotation.outputs[0].default_value = kwargs["pedal_x_R"][i]

        base_z_rotation = nw.new_node(Nodes.Value,
                                      label='base_z_rotation')
        base_z_rotation.outputs[0].default_value = -1.57 + normal(0, 0.3)

        base_pedal_num = nw.new_node(Nodes.Integer,
                                     label='base_pedal_num',
                                     attrs={'integer': 10})
        base_pedal_num.integer = kwargs["base_pedal_num"][i]

        pedal_scale = nw.new_node(Nodes.Value,
                                  label='pedal_scale')
        pedal_scale.outputs[0].default_value = kwargs["base_pedal_scale"][i]

        base_z = nw.new_node(Nodes.Value,
                             label='base_z')
        base_z.outputs[0].default_value = kwargs["base_z"][i]

        pedal_on_base = nw.new_node(nodegroup_pedal_on_base(R=kwargs["base_radius"][i]).name,
                                    input_kwargs={'Radius': base_circle_radius, 'x_R': pedal_x_rotation,
                                                  'z_R': base_z_rotation,
                                                  'Resolution': base_pedal_num, 'Instance': pedal_geometry,
                                                  'Scale': pedal_scale, 'base_z': base_z})
        pedal_bases.append(pedal_on_base)

    join_geometry = nw.new_node(Nodes.JoinGeometry,
                                input_kwargs={'Geometry': pedal_bases})

    set_shade_smooth_1 = nw.new_node(Nodes.SetShadeSmooth,
                                     input_kwargs={'Geometry': join_geometry})

    set_material = nw.new_node(Nodes.SetMaterial,
                               input_kwargs={'Geometry': set_shade_smooth_1,
                                             'Material': surface.shaderfunc_to_material(material)})

    realized = nw.new_node(Nodes.RealizeInstances, [set_material])

    group_output = nw.new_node(Nodes.GroupOutput,
                               input_kwargs={'Geometry': realized})


class SucculentFactory(AssetFactory):
    def __init__(self, factory_seed, coarse=False):
        super(SucculentFactory, self).__init__(factory_seed, coarse=coarse)
        self.mode = np.random.choice(["thin_pedal", "thick_pedal"], p=[0.65, 0.35])

    def get_params(self, mode):
        if mode == 'thin_pedal':
            params = {}
            params["cross_y_bottom"] = uniform(0.08, 0.25)
            params["cross_y_top"] = uniform(-0.04, 0.02)
            params["cross_x"] = uniform(0.3, 0.6)
            # get geometry params on each base
            num_bases = randint(5, 8)
            params["num_bases"] = num_bases
            base_radius, pedal_x_R, base_pedal_num, base_pedal_scale, base_z = [], [], [], [], []
            init_base_radius, diff_base_radius = uniform(0.09, 0.11), 0.1
            init_x_R, diff_x_R = uniform(-1.2, -1.35), uniform(-0.7, -1.1)
            init_pedal_num = randint(num_bases, 15)
            diff_pedal_scale = uniform(0.5, 0.9)
            for i in range(num_bases):
                base_radius.append(init_base_radius - (i * diff_base_radius) / num_bases)
                pedal_x_R.append(init_x_R - (i * diff_x_R) / num_bases)
                base_pedal_num.append(init_pedal_num - i + randint(0, 2))
                base_pedal_scale.append(1. - (i * diff_pedal_scale) / num_bases)
                base_z.append(0. + i * uniform(0.005, 0.008))
            params["base_radius"] = base_radius
            params["pedal_x_R"] = pedal_x_R
            params["base_pedal_num"] = base_pedal_num
            params["base_pedal_scale"] = base_pedal_scale
            params["base_z"] = base_z

            contour_bit = randint(0, 3)
            material_bit = randint(0, 3)

            if contour_bit == 0:
                params["pedal_curve_param"] = [0.08, 0.4, 0.46, 0.36, 0.17, 0.05]
            elif contour_bit == 1:
                params["pedal_curve_param"] = [0.22, 0.37, 0.50, 0.49, 0.30, 0.08]
            elif contour_bit == 2:
                params["pedal_curve_param"] = [0.21, 0.26, 0.31, 0.36, 0.29, 0.16]
            else:
                raise NotImplemented

            if material_bit == 0:
                params["material"] = succulent.shader_green_transition_succulent
            elif material_bit == 1:
                params["material"] = succulent.shader_pink_transition_succulent
            elif material_bit == 2:
                params["material"] = succulent.shader_green_succulent
            else:
                raise NotImplemented

            return params

        elif mode == 'thick_pedal':
            params = {}
            params["cross_y_bottom"] = uniform(0.22, 0.30)
            params["cross_y_top"] = uniform(0.08, 0.15)
            params["cross_x"] = uniform(0.14, 0.16)
            # get geometry params on each base
            num_bases = randint(3, 6)
            params["num_bases"] = num_bases
            base_radius, pedal_x_R, base_pedal_num, base_pedal_scale, base_z = [], [], [], [], []
            init_base_radius, diff_base_radius = uniform(0.12, 0.14), 0.11
            init_x_R, diff_x_R = uniform(-1.3, -1.4), uniform(-0.1, -1.2)
            init_pedal_num = randint(num_bases, 12)
            diff_pedal_scale = uniform(0.6, 0.9)
            for i in range(num_bases):
                base_radius.append(init_base_radius - (i * diff_base_radius) / num_bases)
                pedal_x_R.append(init_x_R - (i * diff_x_R) / num_bases)
                base_pedal_num.append(init_pedal_num - i + randint(0, 2))
                base_pedal_scale.append(1. - (i * diff_pedal_scale) / num_bases)
                base_z.append(0. + i * uniform(0.005, 0.006))
            params["base_radius"] = base_radius
            params["pedal_x_R"] = pedal_x_R
            params["base_pedal_num"] = base_pedal_num
            params["base_pedal_scale"] = base_pedal_scale
            params["base_z"] = base_z

            contour_bit = randint(0, 2)
            material_bit = randint(0, 2)

            if contour_bit == 0:
                params["pedal_curve_param"] = [0.10, 0.36, 0.44, 0.45, 0.30, 0.24]
            elif contour_bit == 1:
                params["pedal_curve_param"] = [0.16, 0.35, 0.48, 0.42, 0.30, 0.18]
            else:
                raise NotImplemented

            if material_bit == 0:
                params["material"] = succulent.shader_yellow_succulent
            elif material_bit == 1:
                params["material"] = succulent.shader_whitish_green_succulent
            else:
                raise NotImplemented

            return params
        else:
            raise NotImplemented

    def create_asset(self, **params):
        bpy.ops.mesh.primitive_plane_add(
            size=1, enter_editmode=False, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
        obj = bpy.context.active_object

        params = self.get_params(self.mode)

        surface.add_geomod(obj, geometry_succulent_nodes, apply=True, attributes=[], input_kwargs=params)

        obj.scale = (0.2, 0.2, 0.2)
        obj.location.z += 0.01
        butil.apply_transform(obj, loc=True, scale=True)     

        tag_object(obj, 'succulent')

        return obj


if __name__ == '__main__':
    fac = SucculentFactory(0)
    fac.create_asset()