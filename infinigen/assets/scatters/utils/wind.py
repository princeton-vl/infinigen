# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Yiming Zuo


from math import prod
from functools import partial

import numpy as np
from numpy.random import uniform as U, normal as N, randint
from mathutils import Vector

from infinigen.core.nodes.node_wrangler import Nodes

def wind_rotation(nw, speed=1.0, direction=None, scale=1.0, strength=30):

    if direction is None:
        direction = Vector([N(0, 1), N(0, 1), 0])

    normalize_1 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: direction},
        attrs={'operation': 'NORMALIZE'})
    vector_rotate_1 = nw.new_node(Nodes.VectorRotate,
        input_kwargs={'Vector': normalize_1.outputs["Vector"], 'Angle': 1.5708})
    position_2 = nw.new_node(Nodes.InputPosition)
    scene_time = nw.new_node('GeometryNodeInputSceneTime')
    t = nw.new_node(Nodes.Math,
        input_kwargs={0: scene_time.outputs["Seconds"], 1: speed},
        attrs={'operation': 'MULTIPLY'})
    t = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: normalize_1.outputs["Vector"], 'Scale': t},
        attrs={'operation': 'SCALE'})
    t = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: position_2, 1: t.outputs["Vector"]})
    noise_texture = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Vector': t.outputs["Vector"], 'Scale': scale})
    multiply_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: strength, 1: -0.2},
        attrs={'operation': 'MULTIPLY'})
    map_range_2 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': noise_texture.outputs["Fac"], 3: multiply_1, 4: strength})
    multiply_2 = nw.new_node(Nodes.Math,
        input_kwargs={0: strength, 1: -0.2},
        attrs={'operation': 'MULTIPLY'})
    multiply_3 = nw.new_node(Nodes.Math,
        input_kwargs={0: strength, 1: 0.2},
        attrs={'operation': 'MULTIPLY'})
    random_value_2 = nw.new_node(Nodes.RandomValue,
        input_kwargs={2: multiply_2, 3: multiply_3, 'Seed': 1})
    add_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: map_range_2.outputs["Result"], 1: random_value_2.outputs[1]})
    deg2rad = nw.new_node(Nodes.Math,
        input_kwargs={0: add_1, 1: 0.0175},
        attrs={'operation': 'MULTIPLY'})
    rotation = nw.new_node(Nodes.RotateEuler,
        input_kwargs={'Rotation': Vector((0,0,0)), 'Axis': vector_rotate_1, 'Angle': deg2rad},
        attrs={'type': 'AXIS_ANGLE'})
    return rotation

def wind(*args, **kwargs):
    return lambda nw: wind_rotation(nw, *args, **kwargs)