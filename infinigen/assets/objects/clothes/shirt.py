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
from infinigen.assets.materials.art import ArtFabric
from infinigen.assets.utils.decorate import (
    read_center,
    read_normal,
    remove_faces,
    subsurf,
    write_co,
)
from infinigen.assets.utils.draw import remesh_fill
from infinigen.assets.utils.object import new_circle
from infinigen.assets.utils.uv import wrap_front_back
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.placement.parameters import AssetParameters, ParameterizedAssetFactory
from infinigen.core.util import blender as butil
from infinigen.core.util.random import log_uniform, weighted_sample


class ShirtParameters(AssetParameters):
    width: Annotated[float, Field(ge=0.45, le=0.55, json_schema_extra={"editable": True})]
    size: Annotated[float, Field(ge=0.25, le=0.3, json_schema_extra={"editable": True})]
    size_neck: Annotated[float, Field(ge=0.1, le=0.15, json_schema_extra={"editable": True})]
    sleeve_angle: Annotated[
        float, Field(ge=0.523599, le=0.785398, json_schema_extra={"editable": True})
    ]
    sleeve_width: Annotated[float, Field(ge=0.14, le=0.18, json_schema_extra={"editable": True})]
    thickness: Annotated[float, Field(ge=0.02, le=0.03, json_schema_extra={"editable": True})]
    shirt_type: Literal["short", "long"] = Field(json_schema_extra={"editable": False})
    sleeve_length: float = Field(json_schema_extra={"editable": False})
    surface: Any = Field(json_schema_extra={"editable": False})
    y_anchors: Annotated[float, Field(ge=0.3, le=0.7, json_schema_extra={"editable": True})] = (
        0.5
    )
    bevel_width_factor: Annotated[
        float, Field(ge=0.1, le=0.15, json_schema_extra={"editable": True})
    ] = 0.125


class ShirtFactory(ParameterizedAssetFactory, AssetFactory):
    parameters_model: ClassVar[type[AssetParameters]] = ShirtParameters

    def __init__(self, factory_seed, coarse=False):
        super().__init__(factory_seed, coarse)
        self.init_legacy_parameters()

    def _sample_sleeve_length(self, shirt_type: str, size: float) -> float:
        match shirt_type:
            case "short":
                return size / 2 + uniform(-0.35, -0.3)
            case _:
                return size / 2 + uniform(-0.05, 0.0)

    def _sample_init_parameters(self, seed: int) -> ShirtParameters:
        width = log_uniform(0.45, 0.55)
        size_offset = uniform(0.25, 0.3)
        size = width + size_offset
        size_neck_frac = uniform(0.1, 0.15)
        shirt_type = np.random.choice(["short", "long"])
        surface_gen_class = weighted_sample(material_assignments.pants)
        surface_material_gen = surface_gen_class()
        surface = surface_material_gen()
        if surface == ArtFabric:
            surface = surface(seed)
        return ShirtParameters(
            seed=seed,
            width=width,
            size=size_offset,
            size_neck=size_neck_frac,
            sleeve_angle=uniform(np.pi / 6, np.pi / 4),
            sleeve_width=uniform(0.14, 0.18),
            thickness=log_uniform(0.02, 0.03),
            shirt_type=shirt_type,
            sleeve_length=self._sample_sleeve_length(shirt_type, size),
            surface=surface,
        )

    def _sample_spawn_parameters(
        self, params: ShirtParameters, seed: int, i: int
    ) -> ShirtParameters:
        return params.model_copy(
            update={
                "y_anchors": uniform(0.3, 0.7),
                "bevel_width_factor": uniform(0.1, 0.15),
            }
        )

    def apply_parameters(
        self, params: ShirtParameters, *, spawn_scope: bool = True
    ) -> None:
        self.width = params.width
        self.size = params.width + params.size
        self.size_neck = params.size_neck * self.size
        self.type = params.shirt_type
        self.sleeve_length = params.sleeve_length
        self.sleeve_width = params.sleeve_width
        self.sleeve_angle = params.sleeve_angle
        self.thickness = params.thickness
        self.surface = params.surface
        self._use_fixed_spawn_draws = spawn_scope
        if spawn_scope:
            self.y_anchors = params.y_anchors
            self.bevel_width_factor = params.bevel_width_factor

    def create_asset(self, **params) -> bpy.types.Object:
        y_anchors_neck = (
            self.y_anchors
            if self._use_fixed_spawn_draws
            else uniform(0.3, 0.7)
        )
        bevel_factor = (
            self.bevel_width_factor
            if self._use_fixed_spawn_draws
            else uniform(0.1, 0.15)
        )

        x_anchors = (
            0,
            self.width / 2,
            self.width / 2,
            self.width / 2 + self.sleeve_length * np.sin(self.sleeve_angle),
            self.width / 2
            + self.sleeve_length * np.sin(self.sleeve_angle)
            + self.sleeve_width * np.cos(self.sleeve_angle),
            self.width / 2,
            self.width / 4,
            0,
        )

        y_anchors = (
            0,
            0,
            self.size - self.sleeve_width / np.sin(self.sleeve_angle),
            self.size
            - self.sleeve_width / np.sin(self.sleeve_angle)
            - self.sleeve_length * np.cos(self.sleeve_angle),
            self.size
            - self.sleeve_width / np.sin(self.sleeve_angle)
            - self.sleeve_length * np.cos(self.sleeve_angle)
            + self.sleeve_width * np.sin(self.sleeve_angle),
            self.size,
            self.size + self.size_neck,
            self.size + self.size_neck * y_anchors_neck,
        )

        obj = new_circle(vertices=len(x_anchors))
        with butil.ViewportMode(obj, "EDIT"):
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.edge_face_add()
            bpy.ops.mesh.flip_normals()
        write_co(obj, np.stack([x_anchors, y_anchors, np.zeros_like(x_anchors)], -1))
        butil.modify_mesh(obj, "MIRROR", use_axis=(True, False, False))
        remesh_fill(obj, 0.02)
        butil.modify_mesh(obj, "SOLIDIFY", thickness=self.thickness)
        x, y, z = read_center(obj).T
        x_, y_, z_ = read_normal(obj).T
        remove_faces(obj, (y_ < -0.5) | ((y_ > 0.5) & (x_ * x < 0)))
        with butil.ViewportMode(obj, "EDIT"), butil.Suppress():
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.remove_doubles(threshold=1e-3)
        butil.modify_mesh(obj, "BEVEL", width=self.sleeve_width * bevel_factor)
        subsurf(obj, 1)
        wrap_front_back(obj, self.surface)
        return obj
