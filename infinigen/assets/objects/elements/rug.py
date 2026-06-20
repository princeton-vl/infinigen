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
from infinigen.assets.materials.art import ArtRug
from infinigen.assets.utils.object import new_base_circle, new_bbox, new_plane
from infinigen.assets.utils.uv import wrap_sides
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.placement.parameters import AssetParameters, ParameterizedAssetFactory
from infinigen.core.util import blender as butil
from infinigen.core.util.random import clip_gaussian, weighted_sample


class RugParameters(AssetParameters):
    width: Annotated[float, Field(ge=2.0, le=6.0, json_schema_extra={"editable": True})]
    length: Annotated[float, Field(ge=1.0, le=1.5, json_schema_extra={"editable": True})]
    rug_shape: Literal["rectangle", "circle", "rounded", "ellipse"] = Field(
        json_schema_extra={"editable": False}
    )
    rounded_buffer: Annotated[
        float, Field(ge=0.1, le=0.5, json_schema_extra={"editable": True})
    ]
    thickness: Annotated[
        float, Field(ge=0.01, le=0.02, json_schema_extra={"editable": True})
    ]
    surface: Any = Field(json_schema_extra={"editable": False})


class RugFactory(ParameterizedAssetFactory, AssetFactory):
    parameters_model: ClassVar[type[AssetParameters]] = RugParameters

    def __init__(self, factory_seed, coarse=False):
        super(RugFactory, self).__init__(factory_seed, coarse)
        self.init_legacy_parameters()

    def _sample_init_parameters(self, seed: int) -> RugParameters:
        width = clip_gaussian(3, 1, 2, 6)
        rug_shape = np.random.choice(["rectangle", "circle", "rounded", "ellipse"])
        length = 1.0 if rug_shape == "circle" else uniform(1, 1.5)
        surface_gen_class = weighted_sample(material_assignments.rug_fabric)
        surface = surface_gen_class()()
        if surface == ArtRug:
            surface = surface(seed)
        return RugParameters(
            seed=seed,
            width=width,
            length=length,
            rug_shape=rug_shape,
            rounded_buffer=uniform(0.1, 0.5),
            thickness=uniform(0.01, 0.02),
            surface=surface,
        )

    def apply_parameters(
        self, params: RugParameters, *, spawn_scope: bool = True
    ) -> None:
        self.width = params.width
        self.length = (
            params.width
            if params.rug_shape == "circle"
            else params.width * params.length
        )
        self.rug_shape = params.rug_shape
        self.rounded_buffer = params.rounded_buffer
        self.thickness = params.thickness
        self.surface = params.surface
        self._use_fixed_spawn_draws = spawn_scope

    def build_shape(self):
        match self.rug_shape:
            case "rectangle":
                obj = new_plane()
                obj.scale = self.length / 2, self.width / 2, 1
                butil.apply_transform(obj, True)
            case "rounded":
                obj = new_plane()
                obj.scale = self.length / 2, self.width / 2, 1
                butil.apply_transform(obj, True)
                butil.modify_mesh(
                    obj, "BEVEL", width=self.rounded_buffer * self.width, segments=16
                )
            case _:
                obj = new_base_circle(vertices=128)
                with butil.ViewportMode(obj, "EDIT"):
                    bpy.ops.mesh.select_all(action="SELECT")
                    bpy.ops.mesh.edge_face_add()
                obj.scale = self.length / 2, self.width / 2, 1
                butil.apply_transform(obj, True)
        return obj

    def create_placeholder(self, **kwargs) -> bpy.types.Object:
        return new_bbox(
            -self.length / 2,
            self.length / 2,
            -self.width / 2,
            self.width / 2,
            0,
            self.thickness,
        )

    def create_asset(self, **params) -> bpy.types.Object:
        obj = self.build_shape()
        wrap_sides(obj, self.surface, "z", "x", "y")
        butil.modify_mesh(obj, "SOLIDIFY", thickness=self.thickness, offset=1)
        return obj
