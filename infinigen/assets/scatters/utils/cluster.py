# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei


import numpy as np
from numpy.random import uniform

from infinigen.core.nodes.node_info import Nodes
from infinigen.core.nodes.node_wrangler import NodeWrangler
from infinigen.core.placement.instance_scatter import bucketed_instance, camera_cull_points
from infinigen.core import surface


def select_points(nw: NodeWrangler, geometry, density, selection, radius, min_distance):
    keypoint_density = density / 5
    keypoints = nw.new_node(Nodes.DistributePointsOnFaces,
                            input_kwargs={'Mesh': geometry, 'Selection': selection, 'Density': keypoint_density
                            }).outputs['Points']
    distance = nw.new_node(Nodes.Proximity, [keypoints], attrs={'target_element': 'POINTS'}).outputs['Distance']
    selection = nw.boolean_math('AND', nw.compare('LESS_THAN', distance, radius), selection)
    points, normal = nw.new_node(Nodes.DistributePointsOnFaces,
                                 input_kwargs={'Mesh': geometry, 'Selection': selection, 'Density': density
                                 }).outputs[:2]
    if min_distance > 0:
        points = nw.new_node(Nodes.MergeByDistance, input_kwargs={'Geometry': points, 'Distance': min_distance})
    return points, distance, normal


def instance_rotation(nw: NodeWrangler, normal, delta_normal=.1, z_rotation='musgrave'):
    perturbed_normal = nw.new_node(Nodes.VectorRotate, input_kwargs={
        'Vector': normal,
        'Rotation': nw.uniform([-delta_normal] * 3, [delta_normal] * 3)
    }, attrs={'rotation_type': 'EULER_XYZ'})
    if z_rotation == 'musgrave':
        z_rotation = nw.scalar_multiply(nw.new_node(Nodes.MusgraveTexture), 2 * np.pi)
    elif z_rotation == 'random':
        z_rotation = nw.uniform(0, 2 * np.pi)
    else:
        z_rotation = uniform(0, 2 * np.pi)
    rotation = nw.new_node(Nodes.RotateEuler, input_kwargs={
        'Rotation': nw.new_node(Nodes.AlignEulerToVector, input_kwargs={'Vector': perturbed_normal},
                                attrs={'axis': 'Z'}),
        'Axis': perturbed_normal,
        'Angle': z_rotation
    }, attrs={'type': 'AXIS_ANGLE'})
    return rotation


def cluster_scatter(nw: NodeWrangler, base_obj, collection, density, instance_index=None, radius=.02,
                    min_distance=0., buckets=((10000, 0.0)), scaling=(1, 1, 1), normal=None,
                    selection=True, ground_offset=0, realize_instances=False, material=None, perturb_normal=.1,
                    z_rotation='musgrave', transform_space='ORIGINAL', reset_children=True):
    geometry = nw.new_node(Nodes.ObjectInfo, [base_obj], attrs={'transform_space': transform_space}).outputs[
        'Geometry']
    selection = surface.eval_argument(nw, selection, geometry=geometry)
    points, distance, default_normal = select_points(nw, geometry, density, selection, radius, min_distance)
    if normal is None:
        normal = default_normal
    visible, vis_distance = camera_cull_points(nw)

    scale = surface.eval_argument(nw, scaling, distance=distance)
    rotation = instance_rotation(nw, normal, perturb_normal, z_rotation)
    instanced = bucketed_instance(nw, points, collection, vis_distance, buckets, visible, scale, rotation,
                                  instance_index, reset_children)

    if ground_offset != 0:
        instanced = nw.new_node(Nodes.TranslateInstances, [instanced], input_kwargs={
            "Translation": nw.combine(0, 0, surface.eval_argument(nw, ground_offset)),
            "Local Space": True
        })
    if realize_instances:
        instanced = nw.new_node(Nodes.RealizeInstances, [instanced])
    if material is not None:
        instanced = nw.new_node(Nodes.SetMaterial, input_kwargs={"Geometry": instanced, "Material": material})
    nw.new_node(Nodes.GroupOutput, input_kwargs={'Geometry': instanced})
