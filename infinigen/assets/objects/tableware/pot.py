# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei
from __future__ import annotations

from typing import Any, ClassVar

import bpy
import numpy as np
from numpy.random import uniform

from infinigen.assets.utils.decorate import read_co, subsurf, write_attribute
from infinigen.assets.utils.object import join_objects, new_bbox
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.placement.parameters import (
    AssetParameters,
    LegacyBridgeParameters,
    ParameterizedAssetFactory,
    apply_bridge_parameters,
    legacy_init_to_parameters,
)
from infinigen.core.util import blender as butil
from infinigen.core.util.math import FixedSeed
from infinigen.core.util.random import log_uniform

from .pan import PanFactory


def _pot_legacy_init(inst: Any, seed: int, coarse: bool) -> None:
    pan_inst = PanFactory.__new__(PanFactory)
    AssetFactory.__init__(pan_inst, seed, coarse)
    pan_params = PanFactory._sample_init_parameters(pan_inst, seed)
    PanFactory.apply_parameters(inst, pan_params, spawn_scope=False)
    inst.has_handle = True
    inst.pre_level = 2
    inst.guard_type = "round"
    with FixedSeed(seed):
        inst.depth = log_uniform(0.6, 2.0)
        inst.r_expand = 1
        inst.r_mid = 1
        inst.has_bar = uniform(0, 1) < 0.5
        inst.has_handle = not inst.has_handle
        inst.has_guard = not inst.has_bar
        inst.bar_height = inst.depth * uniform(0.75, 0.85)
        inst.bar_radius = log_uniform(0.2, 0.3)
        inst.bar_x = 1 + uniform(-inst.bar_radius, inst.bar_radius) * 0.05
        inst.bar_inner_radius = log_uniform(0.2, 0.4) * inst.bar_radius
        scale = log_uniform(0.6, 1.5)
        inst.bar_scale = (
            log_uniform(0.6, 1.0) * scale,
            1 * scale,
            log_uniform(0.6, 1.2) * scale,
        )
        inst.bar_taper = log_uniform(0.3, 0.8)
        inst.bar_y_rotation = uniform(-np.pi / 6, 0)
        inst.bar_x_offset = inst.bar_radius * uniform(-0.1, 0.1)
        inst.guard_type = "round"
        inst.guard_depth = log_uniform(0.5, 1.0) * inst.thickness
        inst.scale = log_uniform(0.1, 0.15)


class PotParameters(LegacyBridgeParameters):
    pass


class PotFactory(PanFactory):
    parameters_model: ClassVar[type[AssetParameters]] = PotParameters

    def __init__(self, factory_seed, coarse=False):
        AssetFactory.__init__(self, factory_seed, coarse)
        self.has_handle = True
        self.pre_level = 2
        self.guard_type = "round"
        self.init_legacy_parameters()

    def _sample_init_parameters(self, seed: int) -> PotParameters:
        return legacy_init_to_parameters(
            PotParameters,
            PotFactory,
            seed,
            self.coarse,
            init_fn=_pot_legacy_init,
        )

    def apply_parameters(
        self, params: PotParameters, *, spawn_scope: bool = True
    ) -> None:
        apply_bridge_parameters(self, params, spawn_scope=spawn_scope)

    def post_init(self) -> None:
        if not hasattr(self, "has_bar"):
            return
        self.has_handle = not self.has_bar
        self.has_guard = not self.has_bar

        self.bar_x = 1 + uniform(-self.bar_radius, self.bar_radius) * 0.05
        self.bar_inner_radius = log_uniform(0.2, 0.4) * self.bar_radius
        self.bar_x_offset = self.bar_radius * uniform(-0.1, 0.1)

    def create_asset(self, **params) -> bpy.types.Object:
        obj = self.make_base()
        if self.has_bar:
            self.add_bar(obj)
        obj.scale = [self.scale] * 3
        butil.apply_transform(obj)
        return obj

    def create_placeholder(self, **kwargs) -> bpy.types.Object:
        if self.has_bar:
            radius_ = (
                1
                + self.bar_x_offset
                + self.bar_radius
                + self.bar_inner_radius
                + self.thickness
            )
            obj = new_bbox(
                -radius_,
                radius_,
                -1 - self.thickness,
                1 + self.thickness,
                0,
                self.depth,
            )
        elif self.has_handle:
            obj = new_bbox(
                -1 - self.thickness,
                1 + self.thickness + self.x_handle,
                -1 - self.thickness,
                1 + self.thickness,
                0,
                self.depth,
            )
        else:
            obj = new_bbox(
                -1 - self.thickness,
                1 + self.thickness,
                -1 - self.thickness,
                1 + self.thickness,
                0,
                self.depth,
            )
        obj.scale = (self.scale,) * 3
        butil.apply_transform(obj)
        return obj

    def add_bar(self, obj):
        bars = []
        for side in [-1, 1]:
            bpy.ops.mesh.primitive_torus_add(
                location=(side * (1 + self.bar_x_offset), 0, self.bar_height),
                major_radius=self.bar_radius,
                minor_radius=self.bar_inner_radius,
            )
            bar = bpy.context.active_object
            bar.scale = self.bar_scale
            butil.modify_mesh(
                bar,
                "SIMPLE_DEFORM",
                deform_method="TAPER",
                angle=self.bar_taper,
                deform_axis="X",
            )
            bar.rotation_euler = 0, self.bar_y_rotation, 0 if side == 1 else np.pi
            butil.apply_transform(bar)

            butil.modify_mesh(bar, "BOOLEAN", object=obj, operation="DIFFERENCE")
            butil.select_none()
            objs = butil.split_object(bar)
            i = np.argmax([np.max(read_co(o)[:, 0] * side) for o in objs])
            bar = objs[i]
            objs.remove(bar)
            butil.delete(objs)
            subsurf(bar, 1)
            write_attribute(bar, lambda nw: 1, "guard", "FACE")
            bars.append(bar)
        return join_objects([obj, *bars])
