# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei

from __future__ import annotations

import bmesh
from typing import Annotated, Any, ClassVar, Literal

import bpy
import numpy as np
import shapely
from numpy.random import uniform
from pydantic import Field
from shapely import Point, affinity

from infinigen.assets.composition import material_assignments
from infinigen.assets.materials import text
from infinigen.assets.utils.decorate import write_co
from infinigen.assets.utils.object import join_objects, new_circle, new_cylinder
from infinigen.assets.utils.uv import wrap_four_sides
from infinigen.core import surface
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.placement.parameters import AssetParameters, ParameterizedAssetFactory
from infinigen.core.util import blender as butil
from infinigen.core.util.random import log_uniform, weighted_sample


class CanParameters(AssetParameters):
    x_length: Annotated[float, Field(ge=0.05, le=0.1, json_schema_extra={"editable": True})]
    z_length: Annotated[float, Field(ge=0.025, le=0.25, json_schema_extra={"editable": True})]
    shape: Literal["circle", "rectangle"] = Field(json_schema_extra={"editable": False})
    skewness: Annotated[float, Field(ge=1.0, le=2.5, json_schema_extra={"editable": True})]
    texture_shared_draw: Annotated[
        float, Field(ge=0.0, le=1.0, json_schema_extra={"editable": True})
    ]
    scratch_draw: Annotated[float, Field(ge=0.0, le=1.0, json_schema_extra={"editable": True})]
    edge_wear_draw: Annotated[float, Field(ge=0.0, le=1.0, json_schema_extra={"editable": True})]
    surface: Any = Field(json_schema_extra={"editable": False})
    wrap_surface: Any = Field(json_schema_extra={"editable": False})
    scratch: Any | None = Field(default=None, json_schema_extra={"editable": False})
    edge_wear: Any | None = Field(default=None, json_schema_extra={"editable": False})
    cap_scale: Annotated[
        float, Field(ge=0.96, le=0.98, json_schema_extra={"editable": True})
    ] = 0.97
    cap_extrude: Annotated[
        float, Field(ge=0.005, le=0.01, json_schema_extra={"editable": True})
    ] = 0.0075
    rect_side_frac: Annotated[
        float, Field(ge=0.2, le=0.8, json_schema_extra={"editable": True})
    ] = 0.5
    wrap_low_frac: Annotated[
        float, Field(ge=0.0, le=0.1, json_schema_extra={"editable": True})
    ] = 0.05
    wrap_high_frac: Annotated[
        float, Field(ge=0.9, le=1.0, json_schema_extra={"editable": True})
    ] = 0.95


class CanFactory(ParameterizedAssetFactory, AssetFactory):
    parameters_model: ClassVar[type[AssetParameters]] = CanParameters

    def __init__(self, factory_seed, coarse=False):
        super().__init__(factory_seed, coarse)
        self.init_legacy_parameters()

    def _sample_init_parameters(self, seed: int) -> CanParameters:
        x_length = log_uniform(0.05, 0.1)
        scratch_prob, edge_wear_prob = material_assignments.wear_tear_prob
        scratch_fn, edge_wear_fn = material_assignments.wear_tear
        scratch_draw = uniform()
        edge_wear_draw = uniform()
        wrap_surface = weighted_sample(material_assignments.graphicdesign)()()
        if wrap_surface == text.Text:
            wrap_surface = text.Text(seed, False)
        return CanParameters(
            seed=seed,
            x_length=x_length,
            z_length=x_length * log_uniform(0.5, 2.5),
            shape=np.random.choice(["circle", "rectangle"]),
            skewness=uniform(1, 2.5) if uniform() < 0.5 else 1,
            texture_shared_draw=uniform(),
            scratch_draw=scratch_draw,
            edge_wear_draw=edge_wear_draw,
            surface=weighted_sample(material_assignments.metals)()(),
            wrap_surface=wrap_surface,
            scratch=None if scratch_draw > scratch_prob else scratch_fn(),
            edge_wear=None if edge_wear_draw > edge_wear_prob else edge_wear_fn(),
        )

    def _sample_spawn_parameters(
        self, params: CanParameters, seed: int, i: int
    ) -> CanParameters:
        return params.model_copy(
            update={
                "cap_scale": uniform(0.96, 0.98),
                "cap_extrude": uniform(0.005, 0.01),
                "rect_side_frac": uniform(0.2, 0.8),
                "wrap_low_frac": uniform(0, 0.1),
                "wrap_high_frac": uniform(0.9, 1.0),
            }
        )

    def apply_parameters(
        self, params: CanParameters, *, spawn_scope: bool = True
    ) -> None:
        self.x_length = params.x_length
        self.z_length = params.z_length
        self.shape = params.shape
        self.skewness = params.skewness
        self.texture_shared = params.texture_shared_draw < 0.2
        self.surface = params.surface
        self.wrap_surface = params.wrap_surface
        self.scratch = params.scratch
        self.edge_wear = params.edge_wear
        self._use_fixed_spawn_draws = spawn_scope
        if spawn_scope:
            self.cap_scale = params.cap_scale
            self.cap_extrude = params.cap_extrude
            self.rect_side_frac = params.rect_side_frac
            self.wrap_low_frac = params.wrap_low_frac
            self.wrap_high_frac = params.wrap_high_frac

    def create_asset(self, **params) -> bpy.types.Object:
        coords = self.make_coords()
        obj = new_circle(vertices=len(coords))
        write_co(obj, np.array([[x, y, 0] for x, y in coords]))
        with butil.ViewportMode(obj, "EDIT"):
            bpy.ops.mesh.select_mode(type="EDGE")
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.edge_face_add()
        butil.modify_mesh(obj, "SOLIDIFY", thickness=self.z_length)
        surface.add_geomod(
            obj,
            self.geo_cap,
            apply=True,
            input_kwargs={
                "cap_scale": self.cap_scale if self._use_fixed_spawn_draws else uniform(0.96, 0.98),
                "cap_extrude": self.cap_extrude if self._use_fixed_spawn_draws else uniform(0.005, 0.01),
            },
        )
        surface.assign_material(obj, self.surface)
        wrap = self.make_wrap(coords)
        obj = join_objects([obj, wrap])
        return obj

    @staticmethod
    def geo_cap(nw: NodeWrangler, cap_scale: float, cap_extrude: float):
        geometry = nw.new_node(
            Nodes.GroupInput, expose_input=[("NodeSocketGeometry", "Geometry", None)]
        )
        selection = nw.compare(
            "GREATER_THAN",
            nw.math("ABSOLUTE", nw.separate(nw.new_node(Nodes.InputNormal))[-1]),
            1 - 1e-3,
        )
        geometry, top = nw.new_node(
            Nodes.ExtrudeMesh, [geometry, selection, None, 0]
        ).outputs[:2]
        geometry = nw.new_node(
            Nodes.ScaleElements,
            input_kwargs={
                "Geometry": geometry,
                "Selection": top,
                "Scale": cap_scale,
            },
        )
        geometry = nw.new_node(
            Nodes.ExtrudeMesh, [geometry, top, None, -cap_extrude]
        ).outputs[0]
        nw.new_node(Nodes.GroupOutput, input_kwargs={"Geometry": geometry})

    def make_coords(self):
        match self.shape:
            case "circle":
                p = Point(0, 0).buffer(self.x_length, quad_segs=64)
            case _:
                side = self.x_length * (
                    self.rect_side_frac
                    if self._use_fixed_spawn_draws
                    else uniform(0.2, 0.8)
                )
                p = shapely.box(-side, -side, side, side).buffer(
                    self.x_length - side, quad_segs=16
                )
        p = affinity.scale(p, yfact=1 / self.skewness)
        coords = p.boundary.segmentize(0.01).coords[:][:-1]
        return coords

    def make_wrap(self, coords):
        obj = new_cylinder(vertices=len(coords))
        with butil.ViewportMode(obj, "EDIT"):
            bm = bmesh.from_edit_mesh(obj.data)
            geom = [f for f in bm.faces if len(f.verts) > 4]
            bmesh.ops.delete(bm, geom=geom, context="FACES_ONLY")
            bmesh.update_edit_mesh(obj.data)
        if self._use_fixed_spawn_draws:
            lowest = self.z_length * self.wrap_low_frac
            highest = self.z_length * self.wrap_high_frac
        else:
            lowest = self.z_length * uniform(0, 0.1)
            highest = self.z_length * uniform(0.9, 1.0)
        write_co(
            obj,
            np.concatenate(
                [np.array([[x, y, lowest], [x, y, highest]]) for x, y in coords]
            ),
        )
        obj.scale = 1 + 1e-3, 1 + 1e-3, 1
        butil.apply_transform(obj)
        wrap_four_sides(obj, self.wrap_surface, self.texture_shared)
        return obj
