# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei
from __future__ import annotations

from typing import Any, ClassVar

import bmesh
import bpy
import numpy as np
from numpy.random import uniform

from infinigen.assets.utils.decorate import subsurf, write_co
from infinigen.assets.utils.object import new_grid
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

from .base import TablewareFactory


def _fork_legacy_init(inst: Any, seed: int, coarse: bool) -> None:
    TablewareFactory.__init__(inst, seed, coarse)
    inst.x_length = log_uniform(0.4, 0.8)
    inst.x_tip = uniform(0.15, 0.2)
    inst.y_length = log_uniform(0.05, 0.08)
    inst.z_depth = log_uniform(0.02, 0.04)
    inst.z_offset = uniform(0.0, 0.05)
    inst.thickness = log_uniform(0.008, 0.015)
    inst.has_guard = uniform(0, 1) < 0.4
    inst.guard_type = "round" if uniform(0, 1) < 0.6 else "double"
    inst.n_cuts = np.random.randint(1, 3) if uniform(0, 1) < 0.3 else 3
    inst.guard_depth = log_uniform(0.2, 1.0) * inst.thickness
    inst.scale = log_uniform(0.15, 0.25)
    inst.has_cut = True


class ForkParameters(LegacyBridgeParameters):
    pass


class ForkFactory(ParameterizedAssetFactory, TablewareFactory):
    parameters_model: ClassVar[type[AssetParameters]] = ForkParameters
    x_end = 0.15
    is_fragile = True

    def __init__(self, factory_seed, coarse=False):
        AssetFactory.__init__(self, factory_seed, coarse)
        self.init_legacy_parameters()

    def _sample_init_parameters(self, seed: int) -> ForkParameters:
        return legacy_init_to_parameters(
            ForkParameters,
            ForkFactory,
            seed,
            self.coarse,
            init_fn=_fork_legacy_init,
        )

    def apply_parameters(
        self, params: ForkParameters, *, spawn_scope: bool = True
    ) -> None:
        apply_bridge_parameters(self, params, spawn_scope=spawn_scope)

    def create_asset(self, **params) -> bpy.types.Object:
        x_anchors = np.array(
            [
                self.x_tip,
                uniform(-0.04, -0.02),
                -0.08,
                -0.12,
                -self.x_end,
                -self.x_end - self.x_length,
                -self.x_end - self.x_length * log_uniform(1.2, 1.4),
            ]
        )
        y_anchors = np.array(
            [
                self.y_length * log_uniform(0.8, 1.0),
                self.y_length * log_uniform(1.0, 1.2),
                self.y_length * log_uniform(0.6, 1.0),
                self.y_length * log_uniform(0.2, 0.4),
                log_uniform(0.01, 0.02),
                log_uniform(0.02, 0.05),
                log_uniform(0.01, 0.02),
            ]
        )
        z_anchors = np.array(
            [
                0,
                -self.z_depth,
                -self.z_depth,
                0,
                self.z_offset,
                self.z_offset + uniform(-0.02, 0.04),
                self.z_offset + uniform(-0.02, 0),
            ]
        )
        n = 2 * (self.n_cuts + 1)
        obj = new_grid(x_subdivisions=len(x_anchors) - 1, y_subdivisions=n - 1)
        x = np.concatenate([x_anchors] * n)
        y = np.ravel(y_anchors[np.newaxis, :] * np.linspace(1, -1, n)[:, np.newaxis])
        z = np.concatenate([z_anchors] * n)
        write_co(obj, np.stack([x, y, z], -1))
        if self.has_cut:
            self.make_cuts(obj)
        butil.modify_mesh(obj, "SOLIDIFY", thickness=self.thickness)
        subsurf(obj, 1)

        def selection(nw, x):
            return nw.compare("LESS_THAN", x, -self.x_end)

        if self.guard_type == "double":
            selection = self.make_double_sided(selection)
        self.add_guard(obj, selection)
        subsurf(obj, 1)
        obj.scale = [self.scale] * 3
        butil.apply_transform(obj)
        return obj

    def make_cuts(self, obj):
        with butil.ViewportMode(obj, "EDIT"):
            bm = bmesh.from_edit_mesh(obj.data)
            front_verts = []
            for v in bm.verts:
                if abs(v.co[0] - self.x_tip) < 1e-3:
                    front_verts.append(v)
            front_verts = sorted(front_verts, key=lambda v: v.co[1])
            geom = []
            for f in bm.faces:
                vs = list(v for v in f.verts if v in front_verts)
                if len(vs) == 2:
                    if min(front_verts.index(vs[0]), front_verts.index(vs[1])) % 2 == 1:
                        geom.append(f)
            bmesh.ops.delete(bm, geom=geom, context="FACES")
            bmesh.update_edit_mesh(obj.data)


class SpatulaFactory(ForkFactory):
    def __init__(self, factory_seed, coarse=False):
        super(SpatulaFactory, self).__init__(factory_seed, coarse)
        self.has_cut = False
        self.z_depth = uniform(0, 0.05)
        self.y_length = log_uniform(0.08, 0.12)
