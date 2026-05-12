# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors: Alexander Raistrick

import bpy
import numpy as np
from numpy.random import uniform

from infinigen.core.util import blender as butil


class BevelSharp:
    def __init__(
        self,
        mult=1,
        angle_min_deg=70,
        segments=None,
    ):
        self.amount = uniform(0.001, 0.006)
        self.mult = mult
        self.angle_min_deg = angle_min_deg

        if segments is None:
            segments = 4 if uniform() < 0 else 1
        self.segments = segments

    def __call__(self, obj):
        butil.select_none()
        butil.select(obj)
        with butil.ViewportMode(obj, "EDIT"):
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.tris_convert_to_quads()
            bpy.ops.mesh.select_mode(use_extend=False, use_expand=False, type="EDGE")
            bpy.ops.mesh.select_all(action="DESELECT")

            angle = np.deg2rad(self.angle_min_deg)

            bpy.ops.mesh.edges_select_sharp(sharpness=angle)

            bpy.ops.mesh.bevel(
                offset=self.amount * self.mult,
                segments=self.segments,
                affect="EDGES",
                offset_type="WIDTH",
            )
