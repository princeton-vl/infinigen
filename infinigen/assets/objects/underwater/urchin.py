# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei

from __future__ import annotations

from typing import Annotated, Any, ClassVar

import bpy
from numpy.random import uniform
from pydantic import Field

import infinigen.core.util.blender as butil
from infinigen.assets.objects.creatures.util.animation.driver_repeated import (
    repeated_driver,
)
from infinigen.assets.utils.decorate import geo_extension
from infinigen.assets.utils.misc import assign_material
from infinigen.assets.utils.object import new_icosphere, separate_loose
from infinigen.core import surface
from infinigen.core.nodes.node_info import Nodes
from infinigen.core.nodes.node_wrangler import NodeWrangler
from infinigen.core.placement.detail import adapt_mesh_resolution
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.placement.parameters import AssetParameters, ParameterizedAssetFactory
from infinigen.core.tagging import tag_object
from infinigen.core.util.color import hsv2rgba
from infinigen.core.util.random import log_uniform


class UrchinParameters(AssetParameters):
    base_hue: Annotated[float, Field(ge=0.0, le=1.0, json_schema_extra={"editable": True})]
    freq: Annotated[float, Field(ge=100.0, le=200.0, json_schema_extra={"editable": True})]
    z_scale_ratio: Annotated[
        float, Field(ge=0.8, le=1.0, json_schema_extra={"editable": True})
    ] = 0.9
    z_stretch_ratio: Annotated[
        float, Field(ge=0.6, le=1.2, json_schema_extra={"editable": True})
    ] = 0.9
    extrude_height: Annotated[
        float, Field(ge=1.0, le=5.0, json_schema_extra={"editable": True})
    ] = 2.0
    girdle_size: Annotated[
        float, Field(ge=0.6, le=1.0, json_schema_extra={"editable": True})
    ] = 0.8
    off: Annotated[float, Field(ge=0.0, le=1.0, json_schema_extra={"editable": True})] = 0.5
    phase: Annotated[float, Field(ge=0.2, le=0.8, json_schema_extra={"editable": True})] = 0.5
    anim_seed: Annotated[int, Field(ge=0, le=99999, json_schema_extra={"editable": True})] = (
        0
    )
    u: Annotated[float, Field(ge=0.8, le=1.0, json_schema_extra={"editable": True})] = 0.9
    v: Annotated[float, Field(ge=0.8, le=1.0, json_schema_extra={"editable": True})] = 0.9
    materials: Any = Field(json_schema_extra={"editable": False})


class UrchinFactory(ParameterizedAssetFactory, AssetFactory):
    parameters_model: ClassVar[type[AssetParameters]] = UrchinParameters

    def __init__(self, factory_seed, coarse=False):
        super().__init__(factory_seed, coarse)
        self.init_legacy_parameters()

    def _sample_init_parameters(self, seed: int) -> UrchinParameters:
        base_hue = uniform(-0.25, 0.15) % 1
        return UrchinParameters(
            seed=seed,
            base_hue=base_hue,
            freq=log_uniform(100, 200),
            materials=[
                surface.shaderfunc_to_material(shader, base_hue)
                for shader in [self.shader_spikes, self.shader_girdle, self.shader_base]
            ],
        )

    def _sample_spawn_parameters(
        self, params: UrchinParameters, seed: int, i: int
    ) -> UrchinParameters:
        anim_seed = int(uniform(0, 100000))
        phase = uniform(0.2, 0.8)
        return params.model_copy(
            update={
                "z_scale_ratio": uniform(0.8, 1.0),
                "z_stretch_ratio": log_uniform(0.6, 1.2),
                "extrude_height": log_uniform(1.0, 5.0),
                "girdle_size": uniform(0.6, 1.0),
                "off": uniform(0, 1),
                "phase": phase,
                "anim_seed": anim_seed,
                "u": phase * uniform(0.8, 1.0),
                "v": (1 - phase) * uniform(0.8, 1.0),
            }
        )

    def apply_parameters(
        self, params: UrchinParameters, *, spawn_scope: bool = True
    ) -> None:
        self.base_hue = params.base_hue
        self.freq = 1 / params.freq
        self.materials = params.materials
        self._use_fixed_spawn_draws = spawn_scope
        if spawn_scope:
            self._z_scale_ratio = params.z_scale_ratio
            self._z_stretch_ratio = params.z_stretch_ratio
            self._extrude_height = params.extrude_height
            self._girdle_size = params.girdle_size
            self._off = params.off
            self._phase = params.phase
            self._anim_seed = params.anim_seed
            self._u = params.u
            self._v = params.v

    def create_asset(self, placeholder, face_size=0.01, **params):
        obj = new_icosphere(subdivisions=4)
        surface.add_geomod(obj, geo_extension, apply=True)
        z_scale_ratio = (
            self._z_scale_ratio if self._use_fixed_spawn_draws else uniform(0.8, 1.0)
        )
        obj.scale[-1] = z_scale_ratio
        butil.apply_transform(obj)
        butil.modify_mesh(
            obj, "BEVEL", offset_type="PERCENT", width_pct=25, angle_limit=0
        )
        surface.add_geomod(
            obj,
            self._geo_extrude,
            apply=True,
            attributes=["spike", "girdle"],
            domains=["FACE"] * 2,
        )
        levels = 1
        butil.modify_mesh(
            obj, "SUBSURF", apply=True, levels=levels, render_levels=levels
        )
        obj.scale = [2 / max(obj.dimensions)] * 3
        z_stretch_ratio = (
            self._z_stretch_ratio
            if self._use_fixed_spawn_draws
            else log_uniform(0.6, 1.2)
        )
        obj.scale[-1] *= z_stretch_ratio
        butil.apply_transform(obj)
        adapt_mesh_resolution(obj, face_size, method="subdiv_by_area")
        obj = separate_loose(obj)
        butil.modify_mesh(
            obj,
            "DISPLACE",
            texture=bpy.data.textures.new(name="urchin", type="STUCCI"),
            strength=0.005,
            mid_level=0,
        )
        surface.add_geomod(
            obj,
            self.geo_material_index,
            apply=True,
            input_attributes=[None, "spike", "girdle"],
        )
        assign_material(obj, self.materials)
        self.animate_stretch(obj)
        tag_object(obj, "urchin")
        return obj

    def animate_stretch(self, obj):
        obj, mod = butil.modify_mesh(
            obj,
            "SIMPLE_DEFORM",
            False,
            return_mod=True,
            deform_method="STRETCH",
            deform_axis="Z",
        )
        driver = mod.driver_add("factor").driver
        if self._use_fixed_spawn_draws:
            t = f"{self.freq: .4f} * frame+{self._off:.4f}"
            t = f"{t}-floor({t})"
            driver.expression = (
                f"-0.1+0.2*(smoothstep(0,{self._u},{t})-"
                f"smoothstep({self._phase},{self._phase + self._v},{t}))"
            )
        else:
            driver.expression = repeated_driver(-0.1, 0.1, self.freq)

    def _geo_extrude(self, nw: NodeWrangler):
        face_prob = 0.98
        girdle_height = 0.1
        extrude_height = (
            self._extrude_height
            if self._use_fixed_spawn_draws
            else log_uniform(1.0, 5.0)
        )
        perturb = 0.1
        girdle_size = (
            self._girdle_size if self._use_fixed_spawn_draws else uniform(0.6, 1)
        )
        geometry = nw.new_node(
            Nodes.GroupInput, expose_input=[("NodeSocketGeometry", "Geometry", None)]
        )
        face_vertices = nw.new_node(Nodes.FaceNeighbors)
        selection = nw.boolean_math(
            "AND",
            nw.compare("GREATER_EQUAL", face_vertices, 5),
            nw.bernoulli(face_prob),
        )
        geometry, top, _ = nw.new_node(
            Nodes.ExtrudeMesh, [geometry, selection, None, girdle_height]
        ).outputs
        geometry, top, girdle = nw.new_node(
            Nodes.ExtrudeMesh, [geometry, top, None, 1e-3]
        ).outputs
        geometry = nw.new_node(Nodes.ScaleElements, [geometry, top, girdle_size])
        geometry, top, _ = nw.new_node(
            Nodes.ExtrudeMesh, [geometry, top, None, -girdle_height]
        ).outputs
        direction = nw.scale(
            nw.add(
                nw.new_node(Nodes.InputNormal),
                nw.uniform([-perturb] * 3, [perturb] * 3),
            ),
            nw.uniform(0.5 * extrude_height, extrude_height),
        )
        geometry, top, side = nw.new_node(
            Nodes.ExtrudeMesh, [geometry, top, direction]
        ).outputs
        geometry = nw.new_node(Nodes.ScaleElements, [geometry, top, 0.2])
        spike = nw.boolean_math("OR", top, side)
        nw.new_node(
            Nodes.GroupOutput,
            input_kwargs={"Geometry": geometry, "Spike": spike, "Girdle": girdle},
        )

    @staticmethod
    def geo_extrude(nw: NodeWrangler):
        face_prob = 0.98
        girdle_height = 0.1
        extrude_height = log_uniform(1.0, 5.0)
        perturb = 0.1
        girdle_size = uniform(0.6, 1)
        geometry = nw.new_node(
            Nodes.GroupInput, expose_input=[("NodeSocketGeometry", "Geometry", None)]
        )
        face_vertices = nw.new_node(Nodes.FaceNeighbors)
        selection = nw.boolean_math(
            "AND",
            nw.compare("GREATER_EQUAL", face_vertices, 5),
            nw.bernoulli(face_prob),
        )
        geometry, top, _ = nw.new_node(
            Nodes.ExtrudeMesh, [geometry, selection, None, girdle_height]
        ).outputs
        geometry, top, girdle = nw.new_node(
            Nodes.ExtrudeMesh, [geometry, top, None, 1e-3]
        ).outputs
        geometry = nw.new_node(Nodes.ScaleElements, [geometry, top, girdle_size])
        geometry, top, _ = nw.new_node(
            Nodes.ExtrudeMesh, [geometry, top, None, -girdle_height]
        ).outputs
        direction = nw.scale(
            nw.add(
                nw.new_node(Nodes.InputNormal),
                nw.uniform([-perturb] * 3, [perturb] * 3),
            ),
            nw.uniform(0.5 * extrude_height, extrude_height),
        )
        geometry, top, side = nw.new_node(
            Nodes.ExtrudeMesh, [geometry, top, direction]
        ).outputs
        geometry = nw.new_node(Nodes.ScaleElements, [geometry, top, 0.2])
        spike = nw.boolean_math("OR", top, side)
        nw.new_node(
            Nodes.GroupOutput,
            input_kwargs={"Geometry": geometry, "Spike": spike, "Girdle": girdle},
        )

    @staticmethod
    def shader_spikes(nw: NodeWrangler, base_hue):
        transmission = uniform(0.95, 0.99)
        subsurface = uniform(0.1, 0.2)
        roughness = uniform(0.5, 0.8)
        color = hsv2rgba(base_hue, uniform(0.5, 1.0), log_uniform(0.05, 1.0))
        principled_bsdf = nw.new_node(
            Nodes.PrincipledBSDF,
            input_kwargs={
                "Base Color": color,
                "Roughness": roughness,
                "Subsurface Weight": subsurface,
                "Subsurface Color": color,
                "Transmission Weight": transmission,
            },
        )
        return principled_bsdf

    @staticmethod
    def shader_girdle(nw: NodeWrangler, base_hue):
        roughness = uniform(0.5, 0.8)
        color = hsv2rgba(base_hue, uniform(0.4, 0.5), log_uniform(0.02, 0.1))
        principled_bsdf = nw.new_node(
            Nodes.PrincipledBSDF,
            input_kwargs={"Base Color": color, "Roughness": roughness},
        )
        return principled_bsdf

    @staticmethod
    def shader_base(nw: NodeWrangler, base_hue):
        roughness = uniform(0.5, 0.8)
        color = hsv2rgba(base_hue, uniform(0.8, 1.0), log_uniform(0.01, 0.02))
        principled_bsdf = nw.new_node(
            Nodes.PrincipledBSDF,
            input_kwargs={"Base Color": color, "Roughness": roughness},
        )
        return principled_bsdf

    @staticmethod
    def geo_material_index(nw: NodeWrangler):
        geometry, spike, girdle = nw.new_node(
            Nodes.GroupInput,
            expose_input=[
                ("NodeSocketGeometry", "Geometry", None),
                ("NodeSocketFloat", "Spike", None),
                ("NodeSocketFloat", "Girdle", None),
            ],
        ).outputs[:-1]
        geometry = nw.new_node(Nodes.SetMaterialIndex, [geometry, None, 2])
        geometry = nw.new_node(Nodes.SetMaterialIndex, [geometry, spike, 0])
        geometry = nw.new_node(Nodes.SetMaterialIndex, [geometry, girdle, 1])
        nw.new_node(Nodes.GroupOutput, input_kwargs={"Geometry": geometry})
