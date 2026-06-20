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
from infinigen.assets.materials.art import Art
from infinigen.assets.utils.object import join_objects, new_bbox, new_plane
from infinigen.assets.utils.uv import wrap_sides
from infinigen.core import surface
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.placement.parameters import AssetParameters, ParameterizedAssetFactory
from infinigen.core.util import blender as butil
from infinigen.core.util.blender import deep_clone_obj
from infinigen.core.util.random import log_uniform, weighted_sample


class WallArtParameters(AssetParameters):
    width: Annotated[float, Field(ge=0.4, le=2.0, json_schema_extra={"editable": True})]
    height: Annotated[float, Field(ge=0.4, le=2.0, json_schema_extra={"editable": True})]
    thickness: Annotated[
        float, Field(ge=0.02, le=0.05, json_schema_extra={"editable": True})
    ]
    depth: Annotated[float, Field(ge=0.01, le=0.02, json_schema_extra={"editable": True})]
    frame_bevel_segments: Literal[0, 1, 4] = Field(json_schema_extra={"editable": False})
    frame_bevel_width: Annotated[
        float, Field(ge=0.0025, le=0.01, json_schema_extra={"editable": True})
    ]
    surface_material_gen: Any = Field(json_schema_extra={"editable": False})
    surface: Any = Field(json_schema_extra={"editable": False})
    frame_surface_gen: Any = Field(json_schema_extra={"editable": False})
    scratch: Any | None = Field(default=None, json_schema_extra={"editable": False})
    edge_wear: Any | None = Field(default=None, json_schema_extra={"editable": False})


class WallArtFactory(ParameterizedAssetFactory, AssetFactory):
    parameters_model: ClassVar[type[AssetParameters]] = WallArtParameters

    def __init__(self, factory_seed, coarse=False):
        super(WallArtFactory, self).__init__(factory_seed, coarse)
        self.init_legacy_parameters()

    def _sample_materials(self, seed: int) -> dict[str, Any]:
        surface_gen_class = weighted_sample(material_assignments.abstract_art)
        surface_material_gen = surface_gen_class()
        surface = surface_material_gen()
        if surface == Art:
            surface = surface(seed)
        frame_surface_gen = weighted_sample(material_assignments.frame)()
        scratch_prob, edge_wear_prob = material_assignments.wear_tear_prob
        scratch_fn, edge_wear_fn = material_assignments.wear_tear
        scratch_draw = uniform()
        edge_wear_draw = uniform()
        return {
            "surface_material_gen": surface_material_gen,
            "surface": surface,
            "frame_surface_gen": frame_surface_gen,
            "scratch": None if scratch_draw > scratch_prob else scratch_fn(),
            "edge_wear": None if edge_wear_draw > edge_wear_prob else edge_wear_fn(),
        }

    def _sample_init_parameters(self, seed: int) -> WallArtParameters:
        depth = uniform(0.01, 0.02)
        materials = self._sample_materials(seed)
        return WallArtParameters(
            seed=seed,
            width=log_uniform(0.4, 2),
            height=log_uniform(0.4, 2),
            thickness=uniform(0.02, 0.05),
            depth=depth,
            frame_bevel_segments=int(np.random.choice([0, 1, 4])),
            frame_bevel_width=uniform(depth / 4, depth / 2),
            **materials,
        )

    def apply_parameters(
        self, params: WallArtParameters, *, spawn_scope: bool = True
    ) -> None:
        self.width = params.width
        self.height = params.height
        self.thickness = params.thickness
        self.depth = params.depth
        self.frame_bevel_segments = params.frame_bevel_segments
        self.frame_bevel_width = params.frame_bevel_width
        self.surface_material_gen = params.surface_material_gen
        self.surface = params.surface
        self.frame_surface_gen = params.frame_surface_gen
        self.scratch = params.scratch
        self.edge_wear = params.edge_wear
        self._use_fixed_spawn_draws = spawn_scope

    def assign_materials(self):
        materials = self._sample_materials(self.factory_seed)
        self.surface_material_gen = materials["surface_material_gen"]
        self.surface = materials["surface"]
        self.frame_surface_gen = materials["frame_surface_gen"]
        self.scratch = materials["scratch"]
        self.edge_wear = materials["edge_wear"]

    def create_placeholder(self, **params):
        return new_bbox(
            -0.01,
            0.15,
            -self.width / 2 - self.thickness,
            self.width / 2 + self.thickness,
            -self.height / 2 - self.thickness,
            self.height / 2 + self.thickness,
        )

    def create_asset(self, placeholder, **params) -> bpy.types.Object:
        self.frame_surface = self.frame_surface_gen()

        obj = new_plane()
        obj.scale = self.width / 2, self.height / 2, 1
        obj.rotation_euler = np.pi / 2, 0, np.pi / 2
        butil.apply_transform(obj, True)

        frame = deep_clone_obj(obj)
        wrap_sides(obj, self.surface, "x", "y", "z")
        butil.select_none()
        with butil.ViewportMode(frame, "EDIT"):
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.delete(type="ONLY_FACE")
        butil.modify_mesh(frame, "SOLIDIFY", thickness=self.thickness, offset=1)
        with butil.ViewportMode(frame, "EDIT"):
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.bridge_edge_loops()
        butil.modify_mesh(frame, "SOLIDIFY", thickness=self.depth, offset=1)
        if self.frame_bevel_segments > 0:
            butil.modify_mesh(
                frame,
                "BEVEL",
                width=self.frame_bevel_width,
                segments=self.frame_bevel_segments,
            )

        surface.assign_material(frame, self.frame_surface)
        obj = join_objects([obj, frame])
        return obj

    def finalize_assets(self, assets):
        if self.scratch:
            self.scratch.apply(assets)
        if self.edge_wear:
            self.edge_wear.apply(assets)


class MirrorFactory(WallArtFactory):
    def _sample_materials(self, seed: int) -> dict[str, Any]:
        surface_gen_class = weighted_sample(material_assignments.mirrors)
        surface_material_gen = surface_gen_class()
        surface_mat = surface_material_gen()
        if surface_mat == Art:
            surface_mat = surface_mat(seed)
        frame_surface_gen = weighted_sample(material_assignments.frame)()
        scratch_prob, edge_wear_prob = material_assignments.wear_tear_prob
        scratch_fn, edge_wear_fn = material_assignments.wear_tear
        scratch_draw = uniform()
        edge_wear_draw = uniform()
        return {
            "surface_material_gen": surface_material_gen,
            "surface": surface_mat,
            "frame_surface_gen": frame_surface_gen,
            "scratch": None if scratch_draw > scratch_prob else scratch_fn(),
            "edge_wear": None if edge_wear_draw > edge_wear_prob else edge_wear_fn(),
        }
