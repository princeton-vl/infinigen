# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.
# Authors: Lingjie Mei
from __future__ import annotations

from typing import Any, ClassVar

import bpy
import numpy as np
from numpy.random import uniform

from infinigen.assets.objects.elements.doors.panel import PanelDoorFactory
from infinigen.assets.utils.decorate import write_attribute, write_co
from infinigen.assets.utils.object import new_cube, new_plane
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.placement.parameters import (
    AssetParameters,
    LegacyBridgeParameters,
    ParameterizedAssetFactory,
    apply_bridge_parameters,
    legacy_init_to_parameters,
)
from infinigen.core.util import blender as butil
from infinigen.core.util.random import log_uniform


def _louver_legacy_init(
    inst: Any, seed: int, coarse: bool, constants: Any = None
) -> None:
    PanelDoorFactory.__init__(inst, seed, coarse, constants)
    inst.x_subdivisions = 1
    inst.y_subdivisions = np.clip(np.random.binomial(5, 0.4), 1, None)
    inst.has_panel = uniform() < 0.7
    inst.has_upper_panel = uniform() < 0.5
    inst.louver_width = uniform(0.002, 0.004)
    inst.louver_margin = uniform(0.02, 0.03)
    inst.louver_size = log_uniform(0.05, 0.1)
    inst.louver_angle = uniform(np.pi / 4.5, np.pi / 3.5)
    inst.has_louver = True


class LouverDoorParameters(LegacyBridgeParameters):
    pass


class LouverDoorFactory(PanelDoorFactory):
    parameters_model: ClassVar[type[AssetParameters]] = LouverDoorParameters

    def __init__(self, factory_seed, coarse=False, constants=None):
        self._constants = constants
        AssetFactory.__init__(self, factory_seed, coarse)
        self.init_legacy_parameters()

    def _sample_init_parameters(self, seed: int) -> LouverDoorParameters:
        return legacy_init_to_parameters(
            LouverDoorParameters,
            LouverDoorFactory,
            seed,
            self.coarse,
            self._constants,
            init_fn=_louver_legacy_init,
        )

    def apply_parameters(
        self, params: LouverDoorParameters, *, spawn_scope: bool = True
    ) -> None:
        apply_bridge_parameters(self, params, spawn_scope=spawn_scope)

    def louver(self, obj, panel):
        x_min, x_max, y_min, y_max = panel["dimension"]
        cutter = new_cube(location=(1, 1, 1))
        butil.apply_transform(cutter, loc=True)
        write_attribute(cutter, 1, "louver", "FACE")
        cutter.location = (
            x_min - self.louver_margin,
            -self.louver_width,
            y_min - self.louver_margin,
        )
        cutter.scale = [
            (x_max - x_min) / 2 + self.louver_margin,
            self.depth / 2 + self.louver_width,
            (y_max - y_min) / 2 + self.louver_margin,
        ]
        butil.apply_transform(cutter, loc=True)
        butil.modify_mesh(obj, "BOOLEAN", object=cutter, operation="DIFFERENCE")

        hole = new_cube(location=(1, 1, 1))
        butil.apply_transform(hole, loc=True)
        write_attribute(hole, 1, "louver", "FACE")
        hole.location = x_min, -self.louver_width * 2, y_min
        hole.scale = (
            (x_max - x_min) / 2,
            self.depth / 2 + self.louver_width * 2,
            (y_max - y_min) / 2,
        )
        butil.apply_transform(hole, loc=True)
        butil.modify_mesh(cutter, "BOOLEAN", object=hole, operation="DIFFERENCE")
        butil.delete(hole)

        louver = new_plane()
        x = x_min, x_max, x_min, x_max
        y = 0, 0, self.depth, self.depth
        y_upper = y_min + self.depth * np.tan(self.louver_angle)
        z = y_min, y_min, y_upper, y_upper
        write_co(louver, np.stack([x, y, z], -1))
        butil.modify_mesh(louver, "SOLIDIFY", thickness=self.louver_width, offset=0)
        butil.modify_mesh(
            louver,
            "ARRAY",
            use_relative_offset=False,
            use_constant_offset=True,
            constant_offset_displace=(0, 0, self.louver_size),
            count=int(np.ceil((y_max - y_min) / self.louver_size) + 0.5),
        )
        louver.location[-1] -= self.depth * np.tan(self.louver_angle) / 2
        butil.apply_transform(louver, True)
        butil.select_none()
        with butil.ViewportMode(louver, "EDIT"):
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.bisect(
                plane_co=(0, 0, y_min),
                plane_no=(0, 0, 1),
                use_fill=True,
                clear_inner=True,
            )
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.bisect(
                plane_co=(0, 0, y_max),
                plane_no=(0, 0, 1),
                use_fill=True,
                clear_outer=True,
            )
        write_attribute(louver, 1, "louver", "FACE")
        return [cutter, louver]

    def make_panels(self):
        panels = super(LouverDoorFactory, self).make_panels()
        if len(panels) == 1:
            panels[0]["func"] = self.louver
        elif len(panels) == 2:
            if not self.has_panel:
                panels[0]["func"] = self.louver
            panels[1]["func"] = self.louver
        else:
            if self.has_upper_panel:
                panels = [panels[0], panels[-1]]
            else:
                panels = [panels[0]]
            for panel in panels:
                panel["func"] = self.louver
        return panels
