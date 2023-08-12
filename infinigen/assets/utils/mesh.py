# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei


import bpy
import numpy as np
from numpy.random import normal, uniform

from infinigen.assets.utils.object import new_cube
from infinigen.core.nodes.node_info import Nodes
from infinigen.core.nodes.node_wrangler import NodeWrangler
from infinigen.core import surface
from infinigen.core.util import blender as butil


def build_prism_mesh(n=6, r_min=1., r_max=1.5, height=.3, tilt=.3):
    angles = polygon_angles(n)
    a_upper = uniform(-np.pi / 12, np.pi / 12, n)
    a_lower = uniform(-np.pi / 12, np.pi / 12, n)
    z_upper = 1 + uniform(-height, height, n) + uniform(0, tilt) * np.cos(angles + uniform(-np.pi, np.pi))
    z_lower = 1 + uniform(-height, height, n) + uniform(0, tilt) * np.sin(angles + uniform(-np.pi, np.pi))
    r_upper = uniform(r_min, r_max, n)
    r_lower = uniform(r_min, r_max, n)

    vertices = np.block([[r_upper * np.cos(angles + a_upper), r_lower * np.cos(angles + a_lower), 0, 0],
                            [r_upper * np.sin(angles + a_upper), r_lower * np.sin(angles + a_lower), 0, 0],
                            [z_upper, -z_lower, 1, -1]]).T

    r = np.arange(n)
    s = np.roll(r, -1)
    faces = np.block(
        [[r, r, r + n, s + n], [s, r + n, s + n, r + n], [np.full(n, 2 * n), s, s, np.full(n, 2 * n + 1)]]).T
    mesh = bpy.data.meshes.new('prism')
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    return mesh


def build_convex_mesh(n=6, height=.2, tilt=.2):
    angles = polygon_angles(n)
    a_upper = uniform(-np.pi / 18, 0, n)
    a_lower = uniform(0, np.pi / 18, n)
    z_upper = 1 + normal(0, height, n) + uniform(0, tilt) * np.cos(angles + uniform(-np.pi, np.pi))
    z_lower = 1 + normal(0, height, n) + uniform(0, tilt) * np.cos(angles + uniform(-np.pi, np.pi))
    r = 1.8
    vertices = np.block([[r * np.cos(angles + a_upper), r * np.cos(angles + a_lower), 0, 0],
                            [r * np.sin(angles + a_upper), r * np.sin(angles + a_lower), 0, 0],
                            [z_upper, -z_lower, z_upper.max() + uniform(.1, .2),
                                -z_lower.max() - uniform(.1, .2)]]).T

    r = np.arange(n)
    s = np.roll(r, -1)
    faces = np.block(
        [[r, r, r + n, s + n], [s, r + n, s + n, r + n], [np.full(n, 2 * n), s, s, np.full(n, 2 * n + 1)]]).T
    mesh = bpy.data.meshes.new('prism')
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    return mesh


def polygon_angles(n, min_angle=np.pi / 6, max_angle=np.pi * 2 / 3):
    for _ in range(100):
        angles = np.sort(uniform(0, 2 * np.pi, n))
        difference = (angles - np.roll(angles, 1)) % (np.pi * 2)
        if (difference >= min_angle).all() and (difference <= max_angle).all():
            break
    else:
        angles = np.sort((np.arange(n) * (2 * np.pi / n) + uniform(0, np.pi * 2)) % (np.pi * 2))
    return angles
