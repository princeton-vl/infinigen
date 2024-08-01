# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei

import bpy
import numpy as np
from numpy.random import uniform

from infinigen.assets.utils.object import center


def add_lighting(asset):
    dimension = asset.dimensions * asset.scale
    radius = np.sqrt(dimension[0] * dimension[1]) / 2 * 1.5
    locations = (
        np.array(
            [
                (uniform(3, 4), -uniform(3, 4), uniform(5, 6)),
                (uniform(3, 4), uniform(3, 4), uniform(3, 4)),
                (-uniform(5, 6), uniform(-2, -3), uniform(3, 4)),
            ]
        )
        * radius
    )
    energies = [1000, 1000 / uniform(5, 10), 1000 * uniform(5, 10)]
    for loc, energy in zip(locations, energies):
        bpy.ops.object.light_add(type="SPOT")
        light = bpy.context.active_object
        light.location = loc + asset.location + center(asset) * asset.scale
        light.rotation_euler = (
            0,
            np.arctan2(np.sqrt(loc[0] ** 2 + loc[1] ** 2), loc[2]),
            -np.arctan2(-loc[0], -loc[1]) - np.pi / 2,
        )
        light.data.energy = energy * radius * radius
