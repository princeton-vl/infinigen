# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei
from __future__ import annotations

from typing import Annotated, Any, ClassVar, Literal

import bpy
import numpy as np
from numpy.random import uniform
from pydantic import Field

from infinigen.assets.composition import material_assignments
from infinigen.assets.utils.decorate import subsurf
from infinigen.assets.utils.object import join_objects, new_base_cylinder, new_cube
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.placement.parameters import AssetParameters, ParameterizedAssetFactory
from infinigen.core.util import blender as butil
from infinigen.core.util.random import log_uniform, weighted_sample


class HardwareParameters(AssetParameters):
    attachment_radius: Annotated[
        float, Field(ge=0.02, le=0.03, json_schema_extra={"editable": True})
    ]
    attachment_depth: Annotated[
        float, Field(ge=0.01, le=0.015, json_schema_extra={"editable": True})
    ]
    radius: Annotated[float, Field(ge=0.01, le=0.015, json_schema_extra={"editable": True})]
    depth: Annotated[float, Field(ge=0.06, le=0.1, json_schema_extra={"editable": True})]
    is_circular_draw: Annotated[
        float, Field(ge=0.0, le=1.0, json_schema_extra={"editable": True})
    ]
    hardware_type: Literal["hook", "holder", "bar", "ring"] = Field(
        json_schema_extra={"editable": False}
    )
    hook_length: Annotated[float, Field(ge=2.0, le=4.0, json_schema_extra={"editable": True})]
    holder_length: Annotated[
        float, Field(ge=0.15, le=0.25, json_schema_extra={"editable": True})
    ]
    bar_length: Annotated[float, Field(ge=0.4, le=0.8, json_schema_extra={"editable": True})]
    extension_length: Annotated[
        float, Field(ge=2.0, le=3.0, json_schema_extra={"editable": True})
    ]
    ring_radius: Annotated[float, Field(ge=2.0, le=6.0, json_schema_extra={"editable": True})]
    scratch_draw: Annotated[
        float, Field(ge=0.0, le=1.0, json_schema_extra={"editable": True})
    ]
    edge_wear_draw: Annotated[
        float, Field(ge=0.0, le=1.0, json_schema_extra={"editable": True})
    ]
    ring_minor_scale: Annotated[
        float, Field(ge=0.4, le=0.7, json_schema_extra={"editable": True})
    ] = 0.55
    surface_material_gen: Any = Field(json_schema_extra={"editable": False})
    scratch: Any | None = Field(default=None, json_schema_extra={"editable": False})
    edge_wear: Any | None = Field(default=None, json_schema_extra={"editable": False})


class HardwareFactory(ParameterizedAssetFactory, AssetFactory):
    parameters_model: ClassVar[type[AssetParameters]] = HardwareParameters

    def __init__(self, factory_seed, coarse=False):
        super(HardwareFactory, self).__init__(factory_seed, coarse)
        self.init_legacy_parameters()

    def _sample_init_parameters(self, seed: int) -> HardwareParameters:
        attachment_radius = uniform(0.02, 0.03)
        scratch_prob, edge_wear_prob = material_assignments.wear_tear_prob
        scratch_fn, edge_wear_fn = material_assignments.wear_tear
        scratch_draw = uniform()
        edge_wear_draw = uniform()
        return HardwareParameters(
            seed=seed,
            attachment_radius=attachment_radius,
            attachment_depth=uniform(0.01, 0.015),
            radius=uniform(0.01, 0.015),
            depth=uniform(0.06, 0.1),
            is_circular_draw=uniform(),
            hardware_type=np.random.choice(["hook", "holder", "bar", "ring"]),
            hook_length=uniform(2, 4),
            holder_length=uniform(0.15, 0.25),
            bar_length=uniform(0.4, 0.8),
            extension_length=uniform(2, 3),
            ring_radius=log_uniform(2, 6),
            scratch_draw=scratch_draw,
            edge_wear_draw=edge_wear_draw,
            surface_material_gen=weighted_sample(material_assignments.metal_neutral)(),
            scratch=None if scratch_draw > scratch_prob else scratch_fn(),
            edge_wear=None if edge_wear_draw > edge_wear_prob else edge_wear_fn(),
        )

    def _sample_spawn_parameters(
        self, params: HardwareParameters, seed: int, i: int
    ) -> HardwareParameters:
        return params.model_copy(update={"ring_minor_scale": uniform(0.4, 0.7)})

    def apply_parameters(
        self, params: HardwareParameters, *, spawn_scope: bool = True
    ) -> None:
        self.attachment_radius = params.attachment_radius
        self.attachment_depth = params.attachment_depth
        self.radius = params.radius
        self.depth = params.depth
        self.is_circular = params.is_circular_draw < 0.5
        self.hardware_type = params.hardware_type
        self.hook_length = params.attachment_radius * params.hook_length
        self.holder_length = params.holder_length
        self.bar_length = params.bar_length
        self.extension_length = params.attachment_radius * params.extension_length
        self.ring_radius = params.ring_radius * params.attachment_radius
        self.ring_minor_scale = params.ring_minor_scale
        self.surface_material_gen = params.surface_material_gen
        self.scratch = params.scratch
        self.edge_wear = params.edge_wear
        self._use_fixed_spawn_draws = spawn_scope

    def make_attachment(self):
        base = new_base_cylinder() if self.is_circular else new_cube()
        base.scale = (
            self.attachment_radius,
            self.attachment_radius,
            self.attachment_depth / 2,
        )
        base.rotation_euler[0] = np.pi / 2
        base.location[1] = -self.attachment_depth / 2
        butil.apply_transform(base, True)

        rod = new_base_cylinder() if self.is_circular else new_cube()
        rod.scale = self.radius, self.radius, self.depth / 2
        rod.rotation_euler[0] = np.pi / 2
        rod.location[1] = -self.depth / 2
        butil.apply_transform(rod, True)
        obj = join_objects([base, rod])
        return obj

    def make_hook(self):
        obj = new_base_cylinder() if self.is_circular else new_cube()
        obj.scale = self.radius, self.radius, self.hook_length / 2
        butil.apply_transform(obj)
        return obj

    def make_holder(self):
        obj = new_base_cylinder() if self.is_circular else new_cube()
        obj.scale = (
            self.radius,
            self.radius,
            (self.holder_length + self.extension_length) / 2,
        )
        obj.rotation_euler[1] = np.pi / 2
        obj.location[0] = (self.holder_length - self.extension_length) / 2
        butil.apply_transform(obj, True)
        return obj

    def make_bar(self):
        obj = new_base_cylinder() if self.is_circular else new_cube()
        obj.scale = (
            self.radius,
            self.radius,
            self.bar_length / 2 + self.extension_length,
        )
        obj.rotation_euler[1] = np.pi / 2
        obj.location[0] = self.bar_length / 2
        butil.apply_transform(obj, True)
        return obj

    def make_ring(self):
        minor_scale = (
            self.ring_minor_scale
            if self._use_fixed_spawn_draws
            else uniform(0.4, 0.7)
        )
        bpy.ops.mesh.primitive_torus_add(
            major_segments=128,
            major_radius=self.ring_radius,
            minor_radius=self.radius * minor_scale,
        )
        obj = bpy.context.active_object
        obj.rotation_euler[0] = np.pi / 2
        obj.location = 0, self.attachment_depth, -self.ring_radius
        butil.apply_transform(obj, True)
        subsurf(obj, 2)
        return obj

    def create_asset(self, **params) -> bpy.types.Object:
        self.surface = self.surface_material_gen()

        match self.hardware_type:
            case "hook":
                extra = self.make_hook()
            case "holder":
                extra = self.make_holder()
            case "bar":
                extra = self.make_bar()
            case "ring":
                extra = self.make_ring()
            case _:
                return self.make_attachment()
        extra.scale = [1 + 1e-3] * 3
        extra.location[1] = -self.depth
        butil.apply_transform(extra, True)
        parts = [self.make_attachment(), extra]
        if self.hardware_type == "bar":
            attachment_ = self.make_attachment()
            attachment_.location[0] = self.bar_length
            butil.apply_transform(attachment_, True)
            parts.append(attachment_)
        obj = join_objects(parts)
        obj.rotation_euler[-1] = np.pi / 2
        butil.apply_transform(obj)
        return obj

    def finalize_assets(self, assets):
        self.surface.apply(assets, metal_color="plain")
        if self.scratch:
            self.scratch.apply(assets)
        if self.edge_wear:
            self.edge_wear.apply(assets)
