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
    distance2boundary,
    read_normal,
    remove_faces,
    subsurf,
    write_co,
)
from infinigen.assets.utils.draw import remesh_fill
from infinigen.assets.utils.object import new_circle
from infinigen.assets.utils.uv import wrap_top_bottom
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.placement.parameters import AssetParameters, ParameterizedAssetFactory
from infinigen.core.util import blender as butil
from infinigen.core.util.random import log_uniform, weighted_sample


class PantsParameters(AssetParameters):
    width: Annotated[float, Field(ge=0.45, le=0.55, json_schema_extra={"editable": True})]
    size_extra: Annotated[float, Field(ge=0.0, le=0.05, json_schema_extra={"editable": True})]
    size: float = Field(json_schema_extra={"editable": False})
    thickness: Annotated[
        float, Field(ge=0.02, le=0.03, json_schema_extra={"editable": True})
    ]
    neck_shrink: Annotated[float, Field(ge=0.1, le=0.15, json_schema_extra={"editable": True})]
    pants_type: Literal["underwear", "shorts", "pants"] = Field(
        json_schema_extra={"editable": False}
    )
    length: float = Field(json_schema_extra={"editable": False})
    surface: Any = Field(json_schema_extra={"editable": False})


class PantsFactory(ParameterizedAssetFactory, AssetFactory):
    parameters_model: ClassVar[type[AssetParameters]] = PantsParameters

    def __init__(self, factory_seed, coarse=False):
        super(PantsFactory, self).__init__(factory_seed, coarse)
        self.init_legacy_parameters()

    def _sample_init_parameters(self, seed: int) -> PantsParameters:
        width = log_uniform(0.45, 0.55)
        size_extra = uniform(0, 0.05)
        size = width / 2 + size_extra
        pants_type = np.random.choice(["underwear", "shorts", "pants"])
        match pants_type:
            case "underwear":
                length = size + uniform(-0.02, 0.02)
            case "shorts":
                length = size + uniform(0.05, 0.1)
            case _:
                length = size + uniform(0.5, 0.7)
        surface_gen_class = weighted_sample(material_assignments.pants)
        surface_material_gen = surface_gen_class()
        surface = surface_material_gen()
        if surface == ArtFabric:
            surface = surface(seed)
        return PantsParameters(
            seed=seed,
            width=width,
            size_extra=size_extra,
            size=size,
            thickness=log_uniform(0.02, 0.03),
            neck_shrink=uniform(0.1, 0.15),
            pants_type=pants_type,
            length=length,
            surface=surface,
        )

    def apply_parameters(
        self, params: PantsParameters, *, spawn_scope: bool = True
    ) -> None:
        self.width = params.width
        self.size = params.size
        self.type = params.pants_type
        self.length = params.length
        self.neck_shrink = params.neck_shrink
        self.thickness = params.thickness
        self.surface = params.surface
        self._use_fixed_spawn_draws = spawn_scope

    def create_asset(self, **params) -> bpy.types.Object:
        x_anchors = (
            0,
            self.width / 2,
            self.width / 2 * (1 + self.neck_shrink),
            self.width / 2 * self.neck_shrink * 2,
            0,
        )
        y_anchors = 0, 0, -self.length, -self.length, -self.size

        obj = new_circle(vertices=len(x_anchors))
        with butil.ViewportMode(obj, "EDIT"):
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.edge_face_add()
        write_co(obj, np.stack([x_anchors, y_anchors, np.zeros_like(x_anchors)], -1))
        butil.modify_mesh(obj, "MIRROR", use_axis=(True, False, False))
        remesh_fill(obj, 0.02)
        distance2boundary(obj)
        butil.modify_mesh(obj, "SOLIDIFY", thickness=self.thickness, offset=0)
        x_, y_, z_ = read_normal(obj).T
        remove_faces(obj, (y_ < -0.99) | (y_ > 0.99))
        with butil.ViewportMode(obj, "EDIT"), butil.Suppress():
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.remove_doubles(threshold=1e-3)
            bpy.ops.mesh.normals_make_consistent(inside=False)
            bpy.ops.mesh.select_mode(type="EDGE")
            bpy.ops.mesh.select_loose()
            bpy.ops.mesh.delete(type="EDGE")
        wrap_top_bottom(obj, self.surface)
        subsurf(obj, 1)
        return obj
