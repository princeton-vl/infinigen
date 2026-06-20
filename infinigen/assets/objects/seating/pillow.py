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
from infinigen.assets.materials import art
from infinigen.assets.scatters import clothes
from infinigen.assets.utils.decorate import (
    read_normal,
    read_selected,
    select_faces,
    set_shade_smooth,
    subsurf,
)
from infinigen.assets.utils.object import (
    center,
    join_objects,
    new_base_circle,
    new_grid,
)
from infinigen.assets.utils.uv import unwrap_faces
from infinigen.core import surface
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.placement.parameters import AssetParameters, ParameterizedAssetFactory
from infinigen.core.util import blender as butil
from infinigen.core.util.random import log_uniform, weighted_sample
from infinigen.core.util.random import random_general as rg


class PillowParameters(AssetParameters):
    shape: Literal["square", "rectangle", "circle", "torus"] = Field(
        json_schema_extra={"editable": False}
    )
    width: Annotated[float, Field(ge=0.4, le=0.7, json_schema_extra={"editable": True})]
    size: float = Field(json_schema_extra={"editable": False})
    bevel_width: Annotated[
        float, Field(ge=0.02, le=0.05, json_schema_extra={"editable": True})
    ]
    thickness: Annotated[
        float, Field(ge=0.006, le=0.008, json_schema_extra={"editable": True})
    ]
    extrude_thickness: float = Field(default=0.0, json_schema_extra={"editable": False})
    has_seam_draw: Annotated[
        float, Field(ge=0.0, le=1.0, json_schema_extra={"editable": True})
    ]
    seam_radius: Annotated[
        float, Field(ge=0.01, le=0.02, json_schema_extra={"editable": True})
    ]
    surface: Any = Field(json_schema_extra={"editable": False})
    torus_inner_radius: Annotated[
        float, Field(ge=0.2, le=0.4, json_schema_extra={"editable": True})
    ] = 0.3
    pressure: Annotated[float, Field(ge=1.0, le=12.0, json_schema_extra={"editable": True})] = (
        1.5
    )
    tension_stiffness: Annotated[
        float, Field(ge=0.0, le=5.0, json_schema_extra={"editable": True})
    ] = 0.0


class PillowFactory(ParameterizedAssetFactory, AssetFactory):
    parameters_model: ClassVar[type[AssetParameters]] = PillowParameters

    shapes = (
        "weighted_choice",
        (4, "square"),
        (4, "rectangle"),
        (1, "circle"),
        (1, "torus"),
    )

    def __init__(self, factory_seed, coarse=False):
        super(PillowFactory, self).__init__(factory_seed, coarse)
        self.init_legacy_parameters()

    def _sample_init_parameters(self, seed: int) -> PillowParameters:
        shape = rg(self.shapes)
        width = uniform(0.4, 0.7)
        size = width if shape == "square" else width * log_uniform(0.6, 0.8)
        surface_gen_class = weighted_sample(material_assignments.fabrics)
        surface_mat = surface_gen_class()()
        if surface_mat == art.ArtFabric:
            surface_mat = surface_mat(self.factory_seed)
        return PillowParameters(
            seed=seed,
            shape=shape,
            width=width,
            size=size,
            bevel_width=uniform(0.02, 0.05),
            thickness=log_uniform(0.006, 0.008),
            extrude_thickness=0.0,
            has_seam_draw=uniform(),
            seam_radius=uniform(0.01, 0.02),
            surface=surface_mat,
        )

    def _sample_spawn_parameters(
        self, params: PillowParameters, seed: int, i: int
    ) -> PillowParameters:
        extrude_draw = uniform()
        extrude_thickness = (
            params.thickness * log_uniform(1, 8) if extrude_draw < 0.5 else 0.0
        )
        pressure = uniform(8, 12) if params.shape == "torus" else uniform(1, 2)
        return params.model_copy(
            update={
                "extrude_thickness": extrude_thickness,
                "torus_inner_radius": uniform(0.2, 0.4),
                "pressure": pressure,
                "tension_stiffness": uniform(0, 5),
            }
        )

    def apply_parameters(
        self, params: PillowParameters, *, spawn_scope: bool = True
    ) -> None:
        self.shape = params.shape
        self.width = params.width
        self.size = params.size
        self.bevel_width = params.bevel_width
        self.thickness = params.thickness
        self.extrude_thickness = params.extrude_thickness
        self.has_seam = params.has_seam_draw < 0.3 and params.shape != "torus"
        self.seam_radius = params.seam_radius
        self.surface = params.surface
        self.torus_inner_radius = params.torus_inner_radius
        self.pressure = params.pressure
        self.tension_stiffness = params.tension_stiffness
        self._use_fixed_spawn_draws = spawn_scope

    def create_asset(self, **params) -> bpy.types.Object:
        torus_inner_radius = (
            self.torus_inner_radius
            if self._use_fixed_spawn_draws
            else uniform(0.2, 0.4)
        )
        pressure = (
            self.pressure
            if self._use_fixed_spawn_draws
            else (uniform(8, 12) if self.shape == "torus" else uniform(1, 2))
        )
        tension_stiffness = (
            self.tension_stiffness if self._use_fixed_spawn_draws else uniform(0, 5)
        )
        extrude_thickness = self.extrude_thickness
        if not self._use_fixed_spawn_draws:
            extrude_thickness = (
                self.thickness * log_uniform(1, 8) if uniform() < 0.5 else 0
            )

        match self.shape:
            case "circle":
                obj = new_base_circle(vertices=128)
                with butil.ViewportMode(obj, "EDIT"):
                    bpy.ops.mesh.fill_grid()
            case "torus":
                obj = new_base_circle(vertices=128)
                inner = new_base_circle(vertices=128, radius=torus_inner_radius)
                obj = join_objects([obj, inner])
                with butil.ViewportMode(obj, "EDIT"):
                    bpy.ops.mesh.select_all(action="SELECT")
                    bpy.ops.mesh.bridge_edge_loops(
                        number_cuts=12, interpolation="LINEAR"
                    )
                obj = bpy.context.active_object
            case _:
                obj = new_grid(x_subdivisions=32, y_subdivisions=32)
        obj.scale = self.width / 2, self.size / 2, 1
        butil.apply_transform(obj, True)
        unwrap_faces(obj)
        butil.modify_mesh(obj, "SOLIDIFY", thickness=self.thickness, offset=0)
        normal = read_normal(obj)

        group = obj.vertex_groups.new(name="pin")
        if self.has_seam:
            with butil.ViewportMode(obj, "EDIT"):
                bpy.ops.mesh.select_mode(type="FACE")
                select_faces(
                    obj, lambda x, y, z: (x**2 + y**2 < self.seam_radius**2) & (z > 0)
                )
                bpy.ops.mesh.region_to_loop()
                bpy.ops.mesh.select_mode(type="VERT")
            selection = read_selected(obj)
            group.add(np.nonzero(selection)[0].tolist(), 1, "REPLACE")
        select_faces(obj, np.abs(normal[:, -1]) < 0.1)

        clothes.cloth_sim(
            obj,
            tension_stiffness=tension_stiffness,
            gravity=0,
            use_pressure=True,
            uniform_pressure_force=pressure,
            vertex_group_mass="pin" if self.has_seam else "",
        )
        if extrude_thickness > 0:
            with butil.ViewportMode(obj, "EDIT"):
                bpy.ops.mesh.extrude_region_shrink_fatten(
                    TRANSFORM_OT_shrink_fatten={"value": extrude_thickness}
                )
        obj.location = -center(obj)
        butil.apply_transform(obj, True)
        subsurf(obj, 2)
        set_shade_smooth(obj)
        return obj

    def make_circle(self):
        obj = new_base_circle(vertices=128)
        with butil.ViewportMode(obj, "EDIT"):
            bpy.ops.mesh.fill_grid()
            select_faces(obj, lambda x, y, z: x**2 + y**2 < self.seam_radius**2)
            bpy.ops.mesh.region_to_loop()
        return obj

    def make_gird(self):
        obj = new_grid(x_subdivisions=64, y_subdivisions=64)
        with butil.ViewportMode(obj, "EDIT"):
            select_faces(
                obj,
                lambda x, y, z: (np.abs(x) < self.seam_radius)
                & (np.abs(y) < self.seam_radius),
            )
            bpy.ops.mesh.region_to_loop()
        return obj

    def finalize_assets(self, assets):
        if isinstance(assets, bpy.types.Object):
            assets = [assets]
        for obj in assets:
            surface.assign_material(obj, self.surface)
