# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei

from __future__ import annotations

from typing import Annotated, Any, ClassVar, Literal

import bmesh
import bpy
import numpy as np
from numpy.random import uniform
from pydantic import Field

from infinigen.assets.composition import material_assignments
from infinigen.assets.materials.plastic import Plastic, PlasticTranslucent
from infinigen.assets.utils.decorate import subsurf, write_attribute
from infinigen.assets.utils.object import join_objects, new_circle, new_cylinder
from infinigen.core import surface
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.placement.parameters import AssetParameters, ParameterizedAssetFactory
from infinigen.core.util import blender as butil
from infinigen.core.util.math import FixedSeed
from infinigen.core.util.random import weighted_sample


class JarParameters(AssetParameters):
    z_length: Annotated[float, Field(ge=0.15, le=0.2, json_schema_extra={"editable": True})]
    x_length: Annotated[float, Field(ge=0.03, le=0.06, json_schema_extra={"editable": True})]
    thickness: Annotated[
        float, Field(ge=0.002, le=0.004, json_schema_extra={"editable": True})
    ]
    n_base: Literal[4, 6, 64] = Field(json_schema_extra={"editable": False})
    x_cap_ratio: Annotated[float, Field(ge=0.6, le=0.9, json_schema_extra={"editable": True})]
    x_cap: float = Field(json_schema_extra={"editable": False})
    z_cap: Annotated[float, Field(ge=0.05, le=0.08, json_schema_extra={"editable": True})]
    z_neck: Annotated[float, Field(ge=0.15, le=0.2, json_schema_extra={"editable": True})]
    cap_subsurf_draw: Annotated[
        float, Field(ge=0.0, le=1.0, json_schema_extra={"editable": True})
    ]
    scratch_draw: Annotated[
        float, Field(ge=0.0, le=1.0, json_schema_extra={"editable": True})
    ]
    edge_wear_draw: Annotated[
        float, Field(ge=0.0, le=1.0, json_schema_extra={"editable": True})
    ]
    clear_draw: Annotated[
        float, Field(ge=0.0, le=1.0, json_schema_extra={"editable": True})
    ]
    profile_shape_factor: Annotated[
        float, Field(ge=0.0, le=0.1, json_schema_extra={"editable": True})
    ] = 0.0
    cap_z_ratio: Annotated[
        float, Field(ge=0.5, le=0.8, json_schema_extra={"editable": True})
    ] = 0.65
    surface: Any = Field(json_schema_extra={"editable": False})
    cap_surface: Any = Field(json_schema_extra={"editable": False})
    scratch: Any | None = Field(default=None, json_schema_extra={"editable": False})
    edge_wear: Any | None = Field(default=None, json_schema_extra={"editable": False})


class JarFactory(ParameterizedAssetFactory, AssetFactory):
    parameters_model: ClassVar[type[AssetParameters]] = JarParameters

    def __init__(self, factory_seed=None, coarse=False):
        super().__init__(factory_seed, coarse)
        self.init_legacy_parameters()

    def _sample_init_parameters(self, seed: int) -> JarParameters:
        scratch_prob, edge_wear_prob = material_assignments.wear_tear_prob
        scratch_fn, edge_wear_fn = material_assignments.wear_tear
        scratch_draw = uniform()
        edge_wear_draw = uniform()
        n_base = int(np.random.choice([4, 6, 64]))
        x_cap_ratio = uniform(0.6, 0.9)
        return JarParameters(
            seed=seed,
            z_length=uniform(0.15, 0.2),
            x_length=uniform(0.03, 0.06),
            thickness=uniform(0.002, 0.004),
            n_base=n_base,
            x_cap_ratio=x_cap_ratio,
            x_cap=x_cap_ratio * np.cos(np.pi / n_base),
            z_cap=uniform(0.05, 0.08),
            z_neck=uniform(0.15, 0.2),
            cap_subsurf_draw=uniform(),
            scratch_draw=scratch_draw,
            edge_wear_draw=edge_wear_draw,
            clear_draw=uniform(),
            surface=weighted_sample(material_assignments.jar)(),
            cap_surface=weighted_sample(material_assignments.appliance_handle)(),
            scratch=None if scratch_draw > scratch_prob else scratch_fn(),
            edge_wear=None if edge_wear_draw > edge_wear_prob else edge_wear_fn(),
        )

    def _sample_spawn_parameters(
        self, params: JarParameters, seed: int, i: int
    ) -> JarParameters:
        return params.model_copy(
            update={
                "profile_shape_factor": uniform(0, 0.1),
                "cap_z_ratio": uniform(0.5, 0.8),
            }
        )

    def apply_parameters(
        self, params: JarParameters, *, spawn_scope: bool = True
    ) -> None:
        self.z_length = params.z_length
        self.x_length = params.x_length
        self.thickness = params.thickness
        self.n_base = params.n_base
        self.x_cap_ratio = params.x_cap_ratio
        self.x_cap = params.x_cap_ratio * np.cos(np.pi / params.n_base)
        self.z_cap = params.z_cap
        self.z_neck = params.z_neck
        self.cap_subsurf = params.cap_subsurf_draw < 0.5
        self.surface = params.surface
        self.cap_surface = params.cap_surface
        self.scratch = params.scratch
        self.edge_wear = params.edge_wear
        self.clear = params.clear_draw < 0.5
        self._use_fixed_spawn_draws = spawn_scope
        if spawn_scope:
            self.profile_shape_factor = params.profile_shape_factor
            self.cap_z_ratio = params.cap_z_ratio

    def create_asset(self, **params) -> bpy.types.Object:
        profile_shape_factor = (
            self.profile_shape_factor
            if self._use_fixed_spawn_draws
            else uniform(0, 0.1)
        )
        cap_z_ratio = self.cap_z_ratio if self._use_fixed_spawn_draws else uniform(0.5, 0.8)

        obj = new_cylinder(vertices=self.n_base)
        obj.scale = self.x_length, self.x_length, self.z_length
        butil.apply_transform(obj, True)
        with butil.ViewportMode(obj, "EDIT"):
            bm = bmesh.from_edit_mesh(obj.data)
            geom = [f for f in bm.faces if f.normal[-1] > 0.5]
            bmesh.ops.delete(bm, geom=geom, context="FACES_KEEP_BOUNDARY")
            bmesh.update_edit_mesh(obj.data)
            bpy.ops.mesh.select_mode(type="EDGE")
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.region_to_loop()
        subsurf(obj, 2, True)
        top = new_circle(location=(0, 0, 0))
        top.scale = [self.x_cap * self.x_length] * 3
        top.location[-1] = (1 + self.z_neck) * self.z_length
        butil.apply_transform(top)
        butil.select_none()
        obj = join_objects([obj, top])
        with butil.ViewportMode(obj, "EDIT"):
            bpy.ops.mesh.select_mode(type="EDGE")
            bpy.ops.mesh.bridge_edge_loops(
                number_cuts=5, profile_shape_factor=profile_shape_factor
            )
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.region_to_loop()
            bpy.ops.mesh.extrude_edges_move(
                TRANSFORM_OT_translate={"value": (0, 0, self.z_cap * self.z_length)}
            )
        subsurf(obj, 2)
        butil.modify_mesh(obj, "SOLIDIFY", thickness=self.thickness)

        cap = new_cylinder(vertices=64)
        cap.scale = (
            *([self.x_cap * self.x_length + 1e-3] * 2),
            self.z_cap * self.z_length,
        )
        cap.location[-1] = (1 + self.z_neck + self.z_cap * cap_z_ratio) * self.z_length
        butil.apply_transform(cap, True)
        subsurf(obj, 1, self.cap_subsurf)
        write_attribute(cap, 1, "cap", "FACE")
        obj = join_objects([obj, cap])
        return obj

    def finalize_assets(self, assets):
        kwargs = (
            dict(clear=self.clear)
            if isinstance(self.surface, Plastic)
            or isinstance(self.surface, PlasticTranslucent)
            else {}
        )
        surface.assign_material(assets, self.surface(**kwargs))
        surface.assign_material(assets, self.cap_surface(), selection="cap")
        if self.scratch:
            self.scratch.apply(assets)
        if self.edge_wear:
            self.edge_wear.apply(assets)
