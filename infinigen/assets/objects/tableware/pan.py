# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors:
# - Lingjie Mei
# - Karhan Kayan: fix cutter bug

from __future__ import annotations

from typing import Annotated, Any, ClassVar

import bmesh
import bpy
import numpy as np
from numpy.random import uniform
from pydantic import Field

from infinigen.assets.utils.decorate import subsurf
from infinigen.assets.utils.object import (
    join_objects,
    new_base_circle,
    new_base_cylinder,
    origin2lowest,
)
from infinigen.core.placement.parameters import AssetParameters, ParameterizedAssetFactory
from infinigen.core.util import blender as butil
from infinigen.core.util.random import log_uniform

from .base import TablewareFactory, apply_tableware_base, sample_tableware_base


class PanParameters(AssetParameters):
    r_expand: Annotated[float, Field(ge=1.0, le=1.2, json_schema_extra={"editable": True})]
    depth: Annotated[float, Field(ge=0.3, le=0.8, json_schema_extra={"editable": True})]
    r_mid: Annotated[float, Field(ge=1.0, le=1.3, json_schema_extra={"editable": True})]
    has_handle_hole_draw: Annotated[
        float, Field(ge=0.0, le=1.0, json_schema_extra={"editable": True})
    ]
    x_handle: Annotated[float, Field(ge=1.2, le=2.0, json_schema_extra={"editable": True})]
    z_handle_frac: Annotated[
        float, Field(ge=0.0, le=0.2, json_schema_extra={"editable": True})
    ]
    z_handle_mid_frac: Annotated[
        float, Field(ge=0.6, le=0.8, json_schema_extra={"editable": True})
    ]
    s_handle: Annotated[float, Field(ge=0.8, le=1.2, json_schema_extra={"editable": True})]
    thickness: Annotated[float, Field(ge=0.04, le=0.06, json_schema_extra={"editable": True})]
    has_guard_draw: Annotated[
        float, Field(ge=0.0, le=1.0, json_schema_extra={"editable": True})
    ]
    x_guard_extra: Annotated[
        float, Field(ge=0.0, le=0.2, json_schema_extra={"editable": True})
    ]
    guard_depth_mult: Annotated[
        float, Field(ge=1.0, le=2.0, json_schema_extra={"editable": True})
    ]
    scale: Annotated[float, Field(ge=0.1, le=0.15, json_schema_extra={"editable": True})]
    lower_thresh: Annotated[float, Field(ge=0.5, le=0.8, json_schema_extra={"editable": True})]
    scratch_draw: Annotated[
        float, Field(ge=0.0, le=1.0, json_schema_extra={"editable": True})
    ]
    edge_wear_draw: Annotated[
        float, Field(ge=0.0, le=1.0, json_schema_extra={"editable": True})
    ]
    n_vertices: Annotated[float, Field(ge=4.0, le=8.0, json_schema_extra={"editable": True})] = (
        4.0
    )
    grid_offset: Annotated[int, Field(ge=0, le=6, json_schema_extra={"editable": True})] = 0
    hole_scale: Annotated[float, Field(ge=0.06, le=0.1, json_schema_extra={"editable": True})] = (
        0.06
    )
    hole_location_frac: Annotated[
        float, Field(ge=0.8, le=0.9, json_schema_extra={"editable": True})
    ] = 0.8
    surface: Any = Field(json_schema_extra={"editable": False})
    inside_surface: Any = Field(json_schema_extra={"editable": False})
    guard_surface: Any = Field(json_schema_extra={"editable": False})
    scratch: Any | None = Field(default=None, json_schema_extra={"editable": False})
    edge_wear: Any | None = Field(default=None, json_schema_extra={"editable": False})
    has_guard: bool = Field(default=False, json_schema_extra={"editable": False})
    metal_color: str | None = Field(default=None, json_schema_extra={"editable": False})


class PanFactory(ParameterizedAssetFactory, TablewareFactory):
    parameters_model: ClassVar[type[AssetParameters]] = PanParameters

    def __init__(self, factory_seed, coarse=False):
        super().__init__(factory_seed, coarse)
        self.has_handle = True
        self.pre_level = 2
        self.guard_type = "round"
        self.init_legacy_parameters()

    def _sample_init_parameters(self, seed: int) -> PanParameters:
        base = sample_tableware_base(seed)
        r_expand = 1 if uniform(0, 1) < 0.2 else log_uniform(1.0, 1.2)
        depth = log_uniform(0.3, 0.8)
        if r_expand == 1:
            r_mid = log_uniform(1.0, 1.3)
        else:
            r_mid = 1 + (r_expand - 1) * (
                uniform(0.5, 0.85) if uniform(0, 1) < 0.5 else 0.5
            )
        x_handle = log_uniform(1.2, 2.0)
        z_handle_frac = uniform(0, 0.2)
        thickness = log_uniform(0.04, 0.06)
        return PanParameters(
            seed=seed,
            r_expand=r_expand,
            depth=depth,
            r_mid=r_mid,
            has_handle_hole_draw=uniform(),
            x_handle=x_handle,
            z_handle_frac=z_handle_frac,
            z_handle_mid_frac=uniform(0.6, 0.8),
            s_handle=log_uniform(0.8, 1.2),
            thickness=thickness,
            has_guard_draw=uniform(0, 1),
            x_guard_extra=uniform(0, 0.2),
            guard_depth_mult=log_uniform(1.0, 2.0),
            scale=log_uniform(0.1, 0.15),
            lower_thresh=base["lower_thresh"],
            scratch_draw=base["scratch_draw"],
            edge_wear_draw=base["edge_wear_draw"],
            surface=base["surface"],
            inside_surface=base["inside_surface"],
            guard_surface=base["guard_surface"],
            scratch=None,
            edge_wear=None,
            has_guard=False,
            metal_color=None,
        )

    def _sample_spawn_parameters(
        self, params: PanParameters, seed: int, i: int
    ) -> PanParameters:
        n_factor = log_uniform(4, 8)
        n = 4 * int(n_factor)
        return params.model_copy(
            update={
                "n_vertices": n_factor,
                "grid_offset": int(np.random.randint(n // 4)),
                "hole_scale": uniform(0.06, 0.1),
                "hole_location_frac": uniform(0.8, 0.9),
            }
        )

    def apply_parameters(
        self, params: PanParameters, *, spawn_scope: bool = True
    ) -> None:
        apply_tableware_base(self, params)
        self.r_expand = params.r_expand
        self.depth = params.depth
        self.r_mid = params.r_mid
        self.has_handle_hole = params.has_handle_hole_draw < 0.6
        self.x_handle = params.x_handle
        self.z_handle = params.x_handle * params.z_handle_frac
        self.z_handle_mid = params.z_handle_mid_frac * self.z_handle
        self.s_handle = params.s_handle
        self.has_guard = params.has_guard_draw < 0.8
        self.x_guard = params.r_expand + params.x_guard_extra * params.x_handle
        self.guard_depth = params.guard_depth_mult * params.thickness
        self.metal_color = params.metal_color
        self._use_fixed_spawn_draws = spawn_scope
        if spawn_scope:
            self._n = 4 * int(params.n_vertices)
            self._grid_offset = params.grid_offset
            self._hole_scale = params.hole_scale
            self._hole_location_frac = params.hole_location_frac

    def create_asset(self, **params) -> bpy.types.Object:
        obj = self.make_base()
        origin2lowest(obj, vertical=True)
        obj.scale = [self.scale] * 3
        butil.apply_transform(obj)
        return obj

    def make_base(self):
        if self._use_fixed_spawn_draws:
            n = self._n
            grid_offset = self._grid_offset
        else:
            n = 4 * int(log_uniform(4, 8))
            grid_offset = np.random.randint(n // 4)
        base = new_base_circle(vertices=n)
        middle = new_base_circle(vertices=n)
        middle.location[-1] = self.depth / 2
        middle.scale = [self.r_mid] * 3
        upper = new_base_circle(vertices=n)
        upper.location[-1] = self.depth
        upper.scale = [self.r_expand] * 3
        butil.apply_transform(upper, loc=True)
        obj = join_objects([base, middle, upper])
        with butil.ViewportMode(obj, "EDIT"):
            bpy.ops.mesh.bridge_edge_loops()
            bm = bmesh.from_edit_mesh(obj.data)
            for v in bm.verts:
                v.select_set(np.abs(v.co[-1]) < 1e-3)
            bm.select_flush(False)
            bmesh.update_edit_mesh(obj.data)
        with butil.ViewportMode(obj, "EDIT"):
            bpy.ops.mesh.fill_grid(use_interp_simple=True, offset=grid_offset)
            bpy.ops.mesh.quads_convert_to_tris(
                quad_method="BEAUTY", ngon_method="BEAUTY"
            )
        obj.rotation_euler[-1] = np.pi / n
        butil.apply_transform(obj)
        if self.has_handle:
            self.add_handle(obj)
        self.solidify_with_inside(obj, self.thickness)

        def selection(nw, x):
            return nw.compare("GREATER_THAN", x, self.x_guard)

        self.add_guard(obj, selection)
        subsurf(obj, 1, True)
        subsurf(obj, 3)
        if self.has_handle_hole:
            self.add_handle_hole(obj)
        return obj

    def add_handle(self, obj):
        with butil.ViewportMode(obj, "EDIT"):
            bpy.ops.mesh.select_mode(type="EDGE")
            bm = bmesh.from_edit_mesh(obj.data)
            bm.edges.ensure_lookup_table()
            m = []
            for e in bm.edges:
                u, v = e.verts
                m.append(u.co[0] + v.co[0] + u.co[2] + v.co[2])
            ri = np.argmax(m)
            for e in bm.edges:
                e.select_set(e.index == ri)
            bm.select_flush(False)
            bmesh.update_edit_mesh(obj.data)

            bpy.ops.mesh.extrude_edges_move(
                TRANSFORM_OT_translate={
                    "value": (self.x_handle * 0.5, 0, self.z_handle_mid)
                }
            )
            bpy.ops.mesh.extrude_edges_move(
                TRANSFORM_OT_translate={
                    "value": (
                        self.x_handle * 0.5,
                        0,
                        (self.z_handle - self.z_handle_mid),
                    )
                }
            )
            bpy.ops.transform.resize(value=[self.s_handle] * 3)
            bpy.ops.mesh.extrude_edges_move(
                TRANSFORM_OT_translate={"value": (1e-3, 0, 0)}
            )

    def add_handle_hole(self, obj):
        if self._use_fixed_spawn_draws:
            hole_scale = self._hole_scale
            hole_location_frac = self._hole_location_frac
        else:
            hole_scale = uniform(0.06, 0.1)
            hole_location_frac = uniform(0.8, 0.9)
        cutter = new_base_cylinder()
        cutter.scale = *([hole_scale] * 2), 1
        cutter.location[0] = self.r_expand + hole_location_frac * self.x_handle
        butil.modify_mesh(obj, "BOOLEAN", object=cutter, operation="DIFFERENCE")
        butil.delete(cutter)
