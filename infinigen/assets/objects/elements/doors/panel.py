# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei
from __future__ import annotations

from typing import Any, ClassVar

import bpy
import numpy as np
from numpy.random import uniform

from infinigen.assets.objects.elements.doors.base import BaseDoorFactory
from infinigen.assets.utils.decorate import read_area, select_faces, write_attribute
from infinigen.assets.utils.mesh import prepare_for_boolean
from infinigen.assets.utils.object import new_cube
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.placement.parameters import (
    AssetParameters,
    LegacyBridgeParameters,
    ParameterizedAssetFactory,
    apply_bridge_parameters,
    legacy_init_to_parameters,
)
from infinigen.core.surface import read_attr_data, write_attr_data
from infinigen.core.util import blender as butil


def _panel_door_legacy_init(
    inst: Any, seed: int, coarse: bool, constants: Any = None
) -> None:
    BaseDoorFactory.__init__(inst, seed, coarse, constants)
    inst.x_subdivisions = 1 if uniform() < 0.5 else 2
    inst.y_subdivisions = np.clip(np.random.binomial(5, 0.45), 1, None)


class PanelDoorParameters(LegacyBridgeParameters):
    pass


class PanelDoorFactory(ParameterizedAssetFactory, BaseDoorFactory):
    parameters_model: ClassVar[type[AssetParameters]] = PanelDoorParameters

    def __init__(self, factory_seed, coarse=False, constants=None):
        self._constants = constants
        AssetFactory.__init__(self, factory_seed, coarse)
        self.init_legacy_parameters()

    def _sample_init_parameters(self, seed: int) -> PanelDoorParameters:
        return legacy_init_to_parameters(
            PanelDoorParameters,
            PanelDoorFactory,
            seed,
            self.coarse,
            self._constants,
            init_fn=_panel_door_legacy_init,
        )

    def apply_parameters(
        self, params: PanelDoorParameters, *, spawn_scope: bool = True
    ) -> None:
        apply_bridge_parameters(self, params, spawn_scope=spawn_scope)

    def bevel(self, obj, panel):
        x_min, x_max, y_min, y_max = panel["dimension"]
        assert x_min <= x_max and y_min <= y_max
        cutter = new_cube()
        butil.apply_transform(cutter, loc=True)
        if panel["attribute_name"] is not None:
            write_attribute(cutter, 1, panel["attribute_name"], "FACE")
        cutter.location = (
            (x_max + x_min) / 2,
            self.bevel_width * 0.5 - 0.1,
            (y_max + y_min) / 2,
        )
        cutter.scale = (x_max - x_min) / 2 - 2e-3, 0.1, (y_max - y_min) / 2 - 2e-3
        butil.apply_transform(cutter, loc=True)
        write_attr_data(
            cutter, "cut", np.ones(len(cutter.data.polygons), dtype=int), "INT", "FACE"
        )
        butil.modify_mesh(obj, "BOOLEAN", object=cutter, operation="DIFFERENCE")
        prepare_for_boolean(obj)
        cutter.location[1] += 0.2 + self.depth - self.bevel_width
        butil.apply_transform(cutter, loc=True)
        butil.modify_mesh(obj, "BOOLEAN", object=cutter, operation="DIFFERENCE")
        prepare_for_boolean(obj)
        butil.delete(cutter)
        select_faces(obj, (read_area(obj) > 0.01) & (read_attr_data(obj, "cut")))
        with butil.ViewportMode(obj, "EDIT"):
            bpy.ops.mesh.inset(thickness=self.shrink_width)
            bpy.ops.mesh.inset(thickness=self.bevel_width, depth=self.bevel_width)
        obj.data.attributes.remove(obj.data.attributes["cut"])
        return []

    def make_panels(self):
        panels = []
        x_cuts = np.random.randint(1, 4, self.x_subdivisions)
        x_cuts = np.cumsum(x_cuts / x_cuts.sum())
        y_cuts = np.sort(np.random.randint(2, 5, self.y_subdivisions))[::-1]
        y_cuts = np.cumsum(y_cuts / y_cuts.sum())
        for j in range(len(y_cuts)):
            for i in range(len(x_cuts)):
                x_min = self.panel_margin + (self.width - self.panel_margin) * (
                    x_cuts[i - 1] if i > 0 else 0
                )
                x_max = (self.width - self.panel_margin) * x_cuts[i]
                y_min = self.panel_margin + (self.height - self.panel_margin) * (
                    y_cuts[j - 1] if j > 0 else 0
                )
                y_max = (self.height - self.panel_margin) * y_cuts[j]
                panels.append(
                    {
                        "dimension": (x_min, x_max, y_min, y_max),
                        "func": self.bevel,
                        "attribute_name": None,
                    }
                )
        return panels


def _glass_panel_door_legacy_init(
    inst: Any, seed: int, coarse: bool, constants: Any = None
) -> None:
    _panel_door_legacy_init(inst, seed, coarse, constants)
    inst.x_subdivisions = 2
    inst.y_subdivisions = np.clip(np.random.binomial(5, 0.5), 2, None)
    inst.merge_glass = inst.y_subdivisions < 4
    inst.has_glass = True


class GlassPanelDoorParameters(LegacyBridgeParameters):
    pass


class GlassPanelDoorFactory(PanelDoorFactory):
    parameters_model: ClassVar[type[AssetParameters]] = GlassPanelDoorParameters

    def __init__(self, factory_seed, coarse=False, constants=None):
        self._constants = constants
        AssetFactory.__init__(self, factory_seed, coarse)
        self.init_legacy_parameters()

    def _sample_init_parameters(self, seed: int) -> GlassPanelDoorParameters:
        return legacy_init_to_parameters(
            GlassPanelDoorParameters,
            GlassPanelDoorFactory,
            seed,
            self.coarse,
            self._constants,
            init_fn=_glass_panel_door_legacy_init,
        )

    def apply_parameters(
        self, params: GlassPanelDoorParameters, *, spawn_scope: bool = True
    ) -> None:
        apply_bridge_parameters(self, params, spawn_scope=spawn_scope)

    def make_panels(self):
        panels = super(GlassPanelDoorFactory, self).make_panels()
        if self.merge_glass:
            first_dimension = panels[-self.x_subdivisions]["dimension"]
            last_dimension = panels[-1]["dimension"]
            merged = {
                "dimension": (
                    first_dimension[0],
                    last_dimension[1],
                    first_dimension[2],
                    last_dimension[3],
                ),
                "func": self.bevel,
                "attribute_name": "glass",
            }
            return [merged, *panels[: self.x_subdivisions]]
        else:
            for panel in panels[-self.x_subdivisions :]:
                panel["attribute_name"] = "glass"
            return panels
