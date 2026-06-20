# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei
from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
from numpy.random import uniform

from infinigen.core.placement.factory import AssetFactory
from infinigen.core.placement.parameters import (
    AssetParameters,
    LegacyBridgeParameters,
    ParameterizedAssetFactory,
    apply_bridge_parameters,
    legacy_init_to_parameters,
)

from .panel import PanelDoorFactory, _panel_door_legacy_init


def _lite_door_legacy_init(
    inst: Any, seed: int, coarse: bool, constants: Any = None
) -> None:
    _panel_door_legacy_init(inst, seed, coarse, constants)
    r = uniform()
    subdivide_glass = False
    if r <= 1 / 6:
        dimension = 0, 1, uniform(0.4, 0.6), 1
        subdivide_glass = True
    elif r <= 1 / 3:
        dimension = 0, 1, 0, 1
        subdivide_glass = True
    elif r <= 1 / 2:
        dimension = 0, uniform(0.3, 0.4), uniform(0.4, 0.6), 1
    elif r <= 2 / 3:
        dimension = 0, uniform(0.3, 0.4), uniform(0.4, 0.6), 1
    elif r <= 5 / 6:
        dimension = 0, 1, 0, 1
    else:
        x = uniform(0.3, 0.35)
        dimension = x, 1 - x, uniform(0.7, 0.8), 1
    inst.x_min, inst.x_max, inst.y_min, inst.y_max = dimension
    if subdivide_glass:
        inst.x_subdivisions = np.random.choice([1, 3])
        inst.y_subdivisions = int(
            inst.height / inst.width * inst.x_subdivisions
        ) + np.random.randint(-1, 2)
    else:
        inst.x_subdivisions = 1
        inst.y_subdivisions = 1
    inst.has_glass = True


class LiteDoorParameters(LegacyBridgeParameters):
    pass


class LiteDoorFactory(PanelDoorFactory):
    parameters_model: ClassVar[type[AssetParameters]] = LiteDoorParameters

    def __init__(self, factory_seed, coarse=False, constants=None):
        self._constants = constants
        AssetFactory.__init__(self, factory_seed, coarse)
        self.init_legacy_parameters()

    def _sample_init_parameters(self, seed: int) -> LiteDoorParameters:
        return legacy_init_to_parameters(
            LiteDoorParameters,
            LiteDoorFactory,
            seed,
            self.coarse,
            self._constants,
            init_fn=_lite_door_legacy_init,
        )

    def apply_parameters(
        self, params: LiteDoorParameters, *, spawn_scope: bool = True
    ) -> None:
        apply_bridge_parameters(self, params, spawn_scope=spawn_scope)

    def make_panels(self):
        x_range = (
            np.linspace(self.x_min, self.x_max, self.x_subdivisions + 1)
            * (self.width - self.panel_margin * 2)
            + self.panel_margin
        )
        y_range = (
            np.linspace(self.y_min, self.y_max, self.y_subdivisions + 1)
            * (self.height - self.panel_margin * 2)
            + self.panel_margin
        )
        panels = []
        for x_min, x_max in zip(x_range[:-1], x_range[1:]):
            for y_min, y_max in zip(y_range[:-1], y_range[1:]):
                panels.append(
                    {
                        "dimension": (x_min, x_max, y_min, y_max),
                        "func": self.bevel,
                        "attribute_name": "glass",
                    }
                )
        return panels
