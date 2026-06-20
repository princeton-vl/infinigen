# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors:
# - Lingjie Mei
# - Karhan Kayan: fix constants

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from infinigen.assets.objects.elements.staircases.straight import (
    StraightStaircaseFactory,
)
from infinigen.assets.utils.decorate import read_co, write_co
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.placement.parameters import (
    AssetParameters,
    LegacyBridgeParameters,
    ParameterizedAssetFactory,
    apply_bridge_parameters,
    legacy_init_to_parameters,
)
from infinigen.core.util.random import log_uniform


def _curved_legacy_init(
    inst: Any, seed: int, coarse: bool, constants: Any = None
) -> None:
    from infinigen.assets.objects.elements.staircases.straight import (
        _straight_staircase_legacy_init,
    )

    inst.full_angle, inst.radius, inst.theta = 0, 0, 0
    _straight_staircase_legacy_init(inst, seed, coarse, constants)
    inst.has_spiral = True


class CurvedStaircaseParameters(LegacyBridgeParameters):
    pass


class CurvedStaircaseFactory(StraightStaircaseFactory):
    parameters_model: ClassVar[type[AssetParameters]] = CurvedStaircaseParameters
    support_types = (
        "weighted_choice",
        (2, "single-rail"),
        (2, "double-rail"),
        (4, "side"),
        (4, "solid"),
        (4, "hole"),
    )

    handrail_types = "weighted_choice", (2, "horizontal-post"), (2, "vertical-post")

    def __init__(self, factory_seed, coarse=False, constants=None):
        self._constants_arg = constants
        AssetFactory.__init__(self, factory_seed, coarse)
        self.init_legacy_parameters()

    def _sample_init_parameters(self, seed: int) -> CurvedStaircaseParameters:
        return legacy_init_to_parameters(
            CurvedStaircaseParameters,
            CurvedStaircaseFactory,
            seed,
            self.coarse,
            self._constants_arg,
            init_fn=_curved_legacy_init,
        )

    def apply_parameters(
        self, params: CurvedStaircaseParameters, *, spawn_scope: bool = True
    ) -> None:
        apply_bridge_parameters(self, params, spawn_scope=spawn_scope)

    def build_size_config(self):
        while True:
            self.full_angle = np.random.randint(1, 5) * np.pi / 2
            self.n = np.random.randint(13, 21)
            self.step_height = self.constants.wall_height / self.n
            self.theta = self.full_angle / self.n
            self.step_length = self.step_height * log_uniform(1, 1.5)
            self.step_width = log_uniform(0.9, 1.5)
            self.radius = self.step_length / self.theta
            if self.radius / self.step_width > 1.5:
                break

    def make_spiral(self, obj):
        x, y, z = read_co(obj).T
        u = x + self.radius - self.step_width
        t = y / self.step_length * self.theta
        write_co(obj, np.stack([u * np.cos(t), u * np.sin(t), z], -1))

    def unmake_spiral(self, obj):
        co = read_co(obj)
        x, y, z = co.T
        u = np.linalg.norm(co[:, :2], axis=-1)
        t = np.arctan2(y, x)
        margins, ts = [], []
        for o in np.linspace(0, np.pi * 2, 8):
            t_ = (t - o) % (np.pi * 2) + o
            margins.append(np.max(t_) - np.min(t_))
            ts.append(t_)
        t = ts[np.argmin(margins)]
        x = u - self.radius + self.step_width
        y = t * self.step_length / self.theta
        co = np.stack([x, y, z], -1)
        write_co(obj, co)
        return obj

    @property
    def upper(self):
        return np.pi / 2 + self.full_angle
