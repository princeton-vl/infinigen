# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei
from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import shapely
import shapely.affinity

from infinigen.assets.objects.elements.staircases.straight import (
    StraightStaircaseFactory,
)
from infinigen.assets.utils.decorate import read_co
from infinigen.assets.utils.object import join_objects
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.placement.parameters import (
    AssetParameters,
    LegacyBridgeParameters,
    ParameterizedAssetFactory,
    apply_bridge_parameters,
    legacy_init_to_parameters,
)
from infinigen.core.util import blender as butil


def _cantilever_legacy_init(
    inst: Any, seed: int, coarse: bool, constants: Any = None
) -> None:
    from infinigen.assets.objects.elements.staircases.straight import (
        _straight_staircase_legacy_init,
    )

    _straight_staircase_legacy_init(inst, seed, coarse, constants)


class CantileverStaircaseParameters(LegacyBridgeParameters):
    pass


class CantileverStaircaseFactory(StraightStaircaseFactory):
    parameters_model: ClassVar[type[AssetParameters]] = CantileverStaircaseParameters
    support_types = "wall"
    handrail_types = "weighted_choice", (2, "horizontal-post"), (2, "vertical-post")

    def __init__(self, factory_seed, coarse=False, constants=None):
        self._constants_arg = constants
        AssetFactory.__init__(self, factory_seed, coarse)
        self.init_legacy_parameters()

    def _sample_init_parameters(self, seed: int) -> CantileverStaircaseParameters:
        return legacy_init_to_parameters(
            CantileverStaircaseParameters,
            CantileverStaircaseFactory,
            seed,
            self.coarse,
            self._constants_arg,
            init_fn=_cantilever_legacy_init,
        )

    def apply_parameters(
        self, params: CantileverStaircaseParameters, *, spawn_scope: bool = True
    ) -> None:
        apply_bridge_parameters(self, params, spawn_scope=spawn_scope)

    def valid_contour(self, offset, contour, doors, lower=True):
        valid = super().valid_contour(offset, contour, doors, lower)
        if not valid or not lower:
            return valid
        obj = join_objects([self.make_line_offset(0), self.make_line_offset(1)])
        co = read_co(obj)[:, :-1]
        butil.delete(obj)
        if self.mirror:
            co[:, 0] = -co[:, 0]
        points = [
            shapely.affinity.translate(
                shapely.affinity.rotate(p, self.rot_z, (0, 0)), *offset
            )
            for p in shapely.points(co)
        ]
        others = [shapely.ops.nearest_points(p, contour.boundary)[0] for p in points]
        distance = np.array(
            [np.abs(p.x - o.x) + np.abs(p.y - o.y) for p, o in zip(points, others)]
        )
        return (distance < 0.1).sum() / len(distance) > 0.5
