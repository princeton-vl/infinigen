# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei


from typing import Iterable

import bpy
import numpy as np

from infinigen.assets.utils.decorate import toggle_hide
from infinigen.core.nodes import node_utils
from infinigen.core.nodes.node_info import Nodes
from infinigen.core.nodes.node_wrangler import NodeWrangler
from infinigen.core import surface


def build_curve(nw: NodeWrangler, positions, circular=False, handle='VECTOR'):
    length = 1
    transferred_positions = []
    id_mesh = nw.new_node(Nodes.InputID)
    for p in positions:
        if isinstance(p, Iterable) and not isinstance(p, bpy.types.Nodes):
            length = len(p)
            transferred_positions.append(
                nw.build_float_curve(id_mesh, np.stack([np.arange(length), np.array(p)], -1), handle))
        else:
            transferred_positions.append(p)

    if circular:
        base_curve = nw.new_node(Nodes.CurveCircle, input_kwargs={'Resolution': length})
    else:
        base_curve = nw.new_node(Nodes.MeshToCurve, input_kwargs={
            'Mesh': nw.new_node(Nodes.MeshLine, input_kwargs={'Count': length}, attrs={'mode': 'END_POINTS'})
        })

    curve = nw.new_node(Nodes.SetPosition, input_kwargs={
        'Geometry': base_curve,
        'Position': nw.new_node(Nodes.CombineXYZ, transferred_positions)
    })
    return curve


def geo_radius(nw: NodeWrangler, radius, resolution=6, merge_distance=.004):
    skeleton = nw.new_node(Nodes.GroupInput, expose_input=[('NodeSocketGeometry', 'Geometry', None)])
    radius = surface.eval_argument(nw, radius)
    curve = align_tilt(nw, nw.new_node(Nodes.MeshToCurve, [skeleton]))
    skeleton = nw.new_node(Nodes.SetCurveRadius, input_kwargs={'Curve': curve, 'Radius': radius})
    geometry = nw.curve2mesh(skeleton, nw.new_node(Nodes.CurveCircle, input_kwargs={'Resolution': resolution}))
    if merge_distance > 0:
        geometry = nw.new_node(Nodes.MergeByDistance, [geometry, None, merge_distance])
    nw.new_node(Nodes.GroupOutput, input_kwargs={'Geometry': geometry})


def geo_selection(nw: NodeWrangler, selection):
    geometry = nw.new_node(Nodes.GroupInput, expose_input=[('NodeSocketGeometry', 'Geometry', None)])
    selection = surface.eval_argument(nw, selection)
    geometry = nw.new_node(Nodes.SeparateGeometry, [geometry, selection])
    nw.new_node(Nodes.GroupOutput, input_kwargs={'Geometry': geometry})


def geo_base_selection(nw: NodeWrangler, base_obj, selection, merge_threshold=0):
    geometry = nw.new_node(Nodes.ObjectInfo, [base_obj], attrs={'transform_space': 'RELATIVE'}).outputs[
        'Geometry']
    selection = surface.eval_argument(nw, selection)
    geometry = nw.new_node(Nodes.SeparateGeometry, [geometry, selection])
    if merge_threshold > 0:
        geometry = nw.new_node(Nodes.MergeByDistance, [geometry, None, merge_threshold])
    nw.new_node(Nodes.GroupOutput, input_kwargs={'Geometry': geometry})


def align_tilt(nw: NodeWrangler, curve, axis=(1, 0, 0), noise_strength=0, noise_scale=.5):
    axis = nw.vector_math('NORMALIZE', axis)
    if noise_strength != 0:
        z = nw.separate(nw.new_node(Nodes.InputPosition))[-1]
        rot_z = nw.scalar_multiply(noise_strength,
                                   nw.new_node(Nodes.NoiseTexture, input_kwargs={'W': z, 'Scale': noise_scale},
                                               attrs={'noise_dimensions': '1D'}))
        axis = nw.new_node(Nodes.VectorRotate, input_kwargs={'Vector': axis, 'Angle': rot_z},
                           attrs={'rotation_type': 'Z_AXIS'})

    normal = nw.new_node(Nodes.InputNormal)
    tangent = nw.vector_math('NORMALIZE', nw.new_node(Nodes.CurveTangent))
    axis = nw.vector_math('NORMALIZE', nw.sub(axis, nw.dot(axis, tangent)))
    cos = nw.dot(axis, normal)
    sin = nw.dot(nw.vector_math('CROSS_PRODUCT', normal, axis), tangent)
    tilt = nw.math('ARCTAN2', sin, cos)
    curve = nw.new_node(Nodes.SetCurveTilt, [curve, None, tilt])
    return curve
