# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Lingjie Mei: original Infinigen monocot growth (https://github.com/princeton-vl/infinigen/blob/224598d4a2ff95b4fc928e72c222598b7122c698/src/infinigen/assets/objects/monocot/growth.py)
# - David Yan, Jia Deng: optimized implementation

import bpy
import numpy as np
from numpy.random import uniform

from infinigen.assets.objects.monocot.growth import (
    MonocotGrowthFactory as _DefaultMonocotGrowthFactory,
)
from infinigen.assets.utils.decorate import (
    geo_extension,
    read_co,
    write_co,
)
from infinigen.assets.utils.object import origin2leftmost
from infinigen.core import surface
from infinigen.core.util import blender as butil
from infinigen.perf.assets.utils import deform
from infinigen.perf.assets.utils.draw import monocot_legacy
from infinigen.perf.core.util import blender as _blender_ext


def translate_to_leftmost(obj, offset=(0, 0, 0)):
    """Shift the vertices so the leftmost one sits at `offset`, like `origin2leftmost`
    followed by applying the location."""
    co = read_co(obj)
    co -= co[np.argmin(co[:, 0])]
    co += np.asarray(offset, dtype=float)
    write_co(obj, co)
    return obj


class MonocotGrowthFactory(_DefaultMonocotGrowthFactory):
    use_distance = False

    @staticmethod
    def decorate_leaf(
        obj,
        y_ratio=4,
        y_bend_angle=np.pi / 6,
        z_bend_angle=np.pi / 6,
        noise_scale=0.1,
        strength=0.02,
        leftmost=True,
    ):
        """Bend, cup and roughen the leaf on its vertex array and put its leftmost
        vertex at the origin, drawing the default's random numbers. The default runs
        instead when `direct_monocot_leaves` is off."""
        if monocot_legacy():
            return _DefaultMonocotGrowthFactory.decorate_leaf.default(
                obj,
                y_ratio,
                y_bend_angle,
                z_bend_angle,
                noise_scale,
                strength,
                leftmost,
            )
        co = read_co(obj)
        co = deform.apply_rotation(co, deform.rotation_y(-np.pi / 2))
        co = deform.simple_deform(co, "BEND", uniform(0.5, 1) * y_bend_angle, "Y")
        co = deform.apply_rotation(co, deform.rotation_y(np.pi / 2))
        co = deform.simple_deform(co, "BEND", uniform(-1, 1) * z_bend_angle, "Z")
        co[:, 2] += y_ratio * uniform(0, 1) * co[:, 1] * co[:, 1]
        write_co(obj, co)
        # the node group and the normal displacement first, since the X and Y
        # displacements scale with the widths of that result, as in the default
        surface.add_geomod(obj, geo_extension, apply=False)
        texture = bpy.data.textures.new(name="grasses", type="STUCCI")
        texture.noise_scale = noise_scale
        butil.modify_mesh(
            obj, "DISPLACE", apply=False, strength=strength, texture=texture
        )
        _blender_ext.apply_modifiers_via_evaluated_mesh(obj)

        dims = np.ptp(read_co(obj), axis=0)
        for direction, width in zip("XY", dims[:2]):
            texture = bpy.data.textures.new(name="grasses", type="STUCCI")
            texture.noise_scale = noise_scale
            butil.modify_mesh(
                obj,
                "DISPLACE",
                apply=False,
                strength=uniform(0.01, 0.02) * width,
                texture=texture,
                direction=direction,
            )
        _blender_ext.apply_modifiers_via_evaluated_mesh(obj)
        if leftmost:
            translate_to_leftmost(obj)
        return obj

    def build_instance(self, i, face_size):
        obj = self.build_leaf(face_size)
        if monocot_legacy():
            origin2leftmost(obj)
            obj.location[0] -= 0.01
            butil.apply_transform(obj, loc=True)
        else:
            translate_to_leftmost(obj, (-0.01, 0, 0))
        return obj
