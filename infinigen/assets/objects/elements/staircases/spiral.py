# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors:
# - Lingjie Mei
# - Karhan Kayan: fix constants

from __future__ import annotations

from typing import ClassVar

import numpy as np
from numpy.random import uniform

import infinigen.core.util.blender as butil
from infinigen.assets.objects.elements.staircases.curved import (
    CurvedStaircaseFactory,
    _curved_legacy_init,
)
from infinigen.assets.utils.decorate import read_co, remove_vertices, write_attribute
from infinigen.assets.utils.nodegroup import geo_radius
from infinigen.assets.utils.object import new_line, separate_loose
from infinigen.core import surface
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.placement.parameters import (
    LegacyBridgeParameters,
    ParameterizedAssetFactory,
    apply_bridge_parameters,
    legacy_init_to_parameters,
)
from infinigen.core.util.math import FixedSeed
from infinigen.core.util.random import log_uniform


class SpiralStaircaseParameters(LegacyBridgeParameters):
    pass


def _spiral_staircase_legacy_init(
    inst: SpiralStaircaseFactory,
    seed: int,
    coarse: bool,
    constants=None,
) -> None:
    inst._init_constants = constants
    _curved_legacy_init(inst, seed, coarse, constants)
    with FixedSeed(seed):
        inst.column_radius = inst.radius - inst.step_width + uniform(0.05, 0.08)
        inst.has_column = True
        inst.handrail_alphas = [1 - inst.handrail_offset / inst.step_width]


class SpiralStaircaseFactory(CurvedStaircaseFactory):
    parameters_model: ClassVar[type[LegacyBridgeParameters]] = SpiralStaircaseParameters
    support_types = "column"

    def __init__(self, factory_seed, coarse=False, constants=None):
        self._init_constants = constants
        AssetFactory.__init__(self, factory_seed, coarse)
        self.init_legacy_parameters()

    def _sample_init_parameters(self, seed: int) -> SpiralStaircaseParameters:
        return legacy_init_to_parameters(
            SpiralStaircaseParameters,
            SpiralStaircaseFactory,
            seed,
            self.coarse,
            init_fn=_spiral_staircase_legacy_init,
            constants=self._init_constants,
        )

    def apply_parameters(
        self, params: SpiralStaircaseParameters, *, spawn_scope: bool = True
    ) -> None:
        apply_bridge_parameters(self, params, spawn_scope=spawn_scope)

    def build_size_config(self):
        while True:
            self.full_angle = np.random.randint(1, 5) * np.pi / 2
            self.n = np.random.randint(13, 21)
            self.step_height = self.constants.wall_height / self.n
            self.theta = self.full_angle / self.n
            self.step_length = self.step_height * log_uniform(1, 1.2)
            self.radius = self.step_length / self.theta
            if 0.9 < self.radius < 1.5:
                self.step_width = self.radius * uniform(0.9, 0.95)
                break

    def make_column(self):
        obj = new_line(self.n, self.step_height * self.n + self.post_height)
        obj.rotation_euler[1] = -np.pi / 2
        butil.apply_transform(obj)
        surface.add_geomod(
            obj, geo_radius, apply=True, input_args=[self.column_radius, 16]
        )
        write_attribute(obj, 1, "steps", "FACE")
        return obj

    def unmake_spiral(self, obj):
        obj = super().unmake_spiral(obj)
        x, y, z = read_co(obj).T
        margin = 0.1
        if (x >= 0).sum() >= (x <= 0).sum():
            remove_vertices(obj, lambda x, y, z: x < margin)
        else:
            remove_vertices(obj, lambda x, y, z: x > -margin)
        obj = separate_loose(obj)
        return obj
