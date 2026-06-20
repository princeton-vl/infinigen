# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei

from __future__ import annotations

from typing import Annotated, Any, ClassVar, Type

import bpy
import numpy as np
from numpy.random import uniform
from pydantic import Field

import infinigen.core.util.blender as butil
from infinigen.assets.objects.cactus import spike
from infinigen.assets.utils.misc import assign_material
from infinigen.assets.utils.decorate import geo_extension
from infinigen.assets.utils.object import join_objects, new_cube
from infinigen.core import surface, tagging
from infinigen.core.nodes.node_utils import build_color_ramp
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.placement.detail import remesh_with_attrs
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.placement.parameters import AssetParameters, ParameterizedAssetFactory
from infinigen.core.util.color import hsv2rgba
from infinigen.core.util.random import log_uniform

from .base import BaseCactusFactory
from .columnar import ColumnarBaseCactusFactory
from .globular import GlobularBaseCactusFactory
from .kalidium import KalidiumBaseCactusFactory
from .pricky_pear import PrickyPearBaseCactusFactory


class CactusParameters(AssetParameters):
    base_hue: Annotated[float, Field(ge=0.2, le=0.4, json_schema_extra={"editable": True})]
    texture_type_draw: Annotated[
        float, Field(ge=0.0, le=1.0, json_schema_extra={"editable": True})
    ]
    factory: Any = Field(json_schema_extra={"editable": False})
    material: Any = Field(json_schema_extra={"editable": False})


class CactusFactory(ParameterizedAssetFactory, AssetFactory):
    parameters_model: ClassVar[type[AssetParameters]] = CactusParameters
    _factory_method_class: ClassVar[Type | None] = None

    def __init__(self, factory_seed, coarse=False, factory_method=None):
        super(CactusFactory, self).__init__(factory_seed, coarse)
        self._init_factory_method = factory_method
        self.init_legacy_parameters()

    def _resolve_factory_method(self, factory_method=None):
        if factory_method is not None:
            return factory_method
        if self._factory_method_class is not None:
            return self._factory_method_class
        factory_methods = [
            GlobularBaseCactusFactory,
            ColumnarBaseCactusFactory,
            PrickyPearBaseCactusFactory,
        ]
        weights = np.array([1] * len(factory_methods))
        return np.random.choice(factory_methods, p=weights / weights.sum())

    def _sample_init_parameters(self, seed: int) -> CactusParameters:
        factory_method = self._resolve_factory_method(self._init_factory_method)
        factory: BaseCactusFactory = factory_method(seed, self.coarse)
        base_hue = uniform(0.2, 0.4)
        material = surface.shaderfunc_to_material(self.shader_cactus, base_hue)
        return CactusParameters(
            seed=seed,
            base_hue=base_hue,
            texture_type_draw=uniform(),
            factory=factory,
            material=material,
        )

    def apply_parameters(
        self, params: CactusParameters, *, spawn_scope: bool = True
    ) -> None:
        self.factory = params.factory
        self.material = params.material
        self._texture_type_draw = params.texture_type_draw
        self._use_fixed_spawn_draws = spawn_scope

    def create_asset(self, face_size=0.01, realize=True, **params):
        obj = self.factory.create_asset(**params)
        remesh_with_attrs(obj, face_size)
        if self.factory.noise_strength > 0:
            texture_types = ["STUCCI", "MARBLE"]
            t = (
                texture_types[int(self._texture_type_draw * len(texture_types))]
                if self._use_fixed_spawn_draws
                else np.random.choice(texture_types)
            )
            texture = bpy.data.textures.new(name="coral", type=t)
            texture.noise_scale = log_uniform(0.1, 0.15)
            butil.modify_mesh(
                obj,
                "DISPLACE",
                True,
                strength=self.factory.noise_strength,
                mid_level=0,
                texture=texture,
            )
        assign_material(obj, self.material)
        if face_size <= 0.05 and self.factory.density > 0:
            t = spike.apply(
                obj, self.factory.points_fn, self.factory.base_radius, realize
            )
            tagging.tag_object(obj, "cactus_spike")
            obj = join_objects([obj, t])
        tagging.tag_object(obj, "cactus")
        return obj

    @staticmethod
    def shader_cactus(nw: NodeWrangler, base_hue):
        shift = uniform(-0.15, 0.15)
        bright_color = hsv2rgba((base_hue + shift) % 1, 1.0, 0.02)
        dark_color = hsv2rgba(base_hue, 0.8, 0.01)
        fresnel_color = hsv2rgba(
            (base_hue - uniform(0.05, 0.1)) % 1, 0.9, uniform(0.3, 0.5)
        )
        specular = 0.25
        fresnel = nw.scalar_multiply(nw.new_node(Nodes.Fresnel), log_uniform(0.6, 1.0))
        color = build_color_ramp(
            nw,
            nw.musgrave(log_uniform(10, 50)),
            [0.0, 0.3, 0.7, 1.0],
            [dark_color, dark_color, bright_color, bright_color],
        )
        color = nw.new_node(Nodes.MixRGB, [fresnel, color, fresnel_color])
        noise_texture = nw.new_node(Nodes.NoiseTexture, input_kwargs={"Scale": 50})
        roughness = nw.build_float_curve(noise_texture, [(0, 0.5), (1, 0.8)])
        bsdf = nw.new_node(
            Nodes.PrincipledBSDF,
            input_kwargs={
                "Base Color": color,
                "Roughness": roughness,
                "Specular IOR Level": specular,
            },
        )
        return bsdf


class GlobularCactusParameters(CactusParameters):
    generate_59: Annotated[
        float, Field(ge=0.1, le=0.15, json_schema_extra={"editable": True})
    ] = 0.125
    globular_83: Annotated[
        float, Field(ge=0.0, le=6.283185, json_schema_extra={"editable": True})
    ] = 0.0
    anchors: Annotated[
        float, Field(ge=0.2, le=0.4, json_schema_extra={"editable": True})
    ] = 0.3
    circle: Annotated[
        float, Field(ge=1.1, le=1.2, json_schema_extra={"editable": True})
    ] = 1.15
    frequency: Annotated[
        float, Field(ge=-0.2, le=0.2, json_schema_extra={"editable": True})
    ] = 0.0
    radius: Annotated[
        float, Field(ge=0.5, le=1.0, json_schema_extra={"editable": True})
    ] = 0.75
    star_resolution: Annotated[
        int, Field(ge=6, le=11, json_schema_extra={"editable": True})
    ] = 8
    scale_x: Annotated[
        float, Field(ge=0.8, le=1.5, json_schema_extra={"editable": True})
    ] = 1.0
    scale_y: Annotated[
        float, Field(ge=0.8, le=1.5, json_schema_extra={"editable": True})
    ] = 1.0
    scale_z: Annotated[
        float, Field(ge=0.8, le=1.5, json_schema_extra={"editable": True})
    ] = 1.0


class GlobularCactusFactory(CactusFactory):
    parameters_model: ClassVar[type[AssetParameters]] = GlobularCactusParameters
    _factory_method_class = GlobularBaseCactusFactory

    def _sample_init_parameters(self, seed: int) -> GlobularCactusParameters:
        params = super()._sample_init_parameters(seed)
        return GlobularCactusParameters(**params.model_dump())

    def _sample_spawn_parameters(
        self, params: GlobularCactusParameters, seed: int, i: int
    ) -> GlobularCactusParameters:
        return params.model_copy(
            update={
                "generate_59": log_uniform(0.1, 0.15),
                "globular_83": uniform(0.0, np.pi * 2),
                "anchors": uniform(0.2, 0.4),
                "circle": uniform(1.1, 1.2),
                "frequency": uniform(-0.2, 0.2),
                "radius": log_uniform(0.5, 1.0),
                "star_resolution": int(np.random.randint(6, 12)),
                "scale_x": uniform(0.8, 1.5),
                "scale_y": uniform(0.8, 1.5),
                "scale_z": uniform(0.8, 1.5),
            }
        )

    def apply_parameters(
        self, params: GlobularCactusParameters, *, spawn_scope: bool = True
    ) -> None:
        super().apply_parameters(params, spawn_scope=spawn_scope)
        if spawn_scope:
            self._globular_params = params

    def create_asset(self, face_size=0.01, realize=True, **params):
        obj = self._create_globular_mesh()
        remesh_with_attrs(obj, face_size)
        if self.factory.noise_strength > 0:
            texture_types = ["STUCCI", "MARBLE"]
            t = (
                texture_types[int(self._texture_type_draw * len(texture_types))]
                if self._use_fixed_spawn_draws
                else np.random.choice(texture_types)
            )
            texture = bpy.data.textures.new(name="coral", type=t)
            noise_scale = (
                self._globular_params.generate_59
                if self._use_fixed_spawn_draws
                else log_uniform(0.1, 0.15)
            )
            texture.noise_scale = noise_scale
            butil.modify_mesh(
                obj,
                "DISPLACE",
                True,
                strength=self.factory.noise_strength,
                mid_level=0,
                texture=texture,
            )
        assign_material(obj, self.material)
        if face_size <= 0.05 and self.factory.density > 0:
            t = spike.apply(
                obj, self.factory.points_fn, self.factory.base_radius, realize
            )
            tagging.tag_object(obj, "cactus_spike")
            obj = join_objects([obj, t])
        tagging.tag_object(obj, "cactus")
        return obj

    def _create_globular_mesh(self) -> bpy.types.Object:
        p = self._globular_params if self._use_fixed_spawn_draws else None
        star_resolution = (
            p.star_resolution if p is not None else int(np.random.randint(6, 12))
        )
        frequency = p.frequency if p is not None else uniform(-0.2, 0.2)
        circle_scale = p.circle if p is not None else uniform(1.1, 1.2)
        anchors_y = p.anchors if p is not None else uniform(0.2, 0.4)
        radius_scale = p.radius if p is not None else log_uniform(0.5, 1.0)
        rotation_z = p.globular_83 if p is not None else uniform(0, np.pi * 2)
        scale = (
            (p.scale_x, p.scale_y, p.scale_z)
            if p is not None
            else tuple(uniform(0.8, 1.5, 3))
        )

        def geo_globular(nw: NodeWrangler):
            resolution = 64
            circle = nw.new_node(Nodes.MeshCircle, [star_resolution * 3])
            selection = nw.compare(
                "EQUAL", nw.math("MODULO", nw.new_node(Nodes.Index), 2), 0
            )
            circle, selection = nw.new_node(
                Nodes.CaptureAttribute, [circle, selection]
            ).outputs[:2]
            circle = nw.new_node(
                Nodes.SetPosition,
                [
                    circle,
                    selection,
                    nw.scale(nw.new_node(Nodes.InputPosition), circle_scale),
                ],
            )
            profile_curve = nw.new_node(Nodes.MeshToCurve, [circle])
            curve = nw.new_node(
                Nodes.ResampleCurve, [nw.new_node(Nodes.CurveLine), None, resolution]
            )
            anchors = [
                (0, anchors_y),
                (uniform(0.4, 0.6), log_uniform(0.5, 0.8)),
                (uniform(0.8, 0.85), uniform(0.4, 0.6)),
                (1.0, 0.05),
            ]
            radius = nw.scalar_multiply(
                nw.build_float_curve(nw.new_node(Nodes.SplineParameter), anchors, "AUTO"),
                radius_scale,
            )
            curve = nw.new_node(Nodes.SetCurveRadius, [curve, None, radius])
            curve = nw.new_node(
                Nodes.SetCurveTilt,
                [
                    curve,
                    None,
                    nw.scalar_multiply(
                        nw.new_node(Nodes.SplineParameter), 2 * np.pi * frequency
                    ),
                ],
            )
            geometry = nw.curve2mesh(curve, profile_curve)
            nw.new_node(
                Nodes.GroupOutput,
                input_kwargs={"Geometry": geometry, "Selection": selection},
            )

        obj = new_cube()
        surface.add_geomod(obj, geo_globular, apply=True, attributes=["selection"])
        surface.add_geomod(
            obj, geo_extension, apply=True, input_kwargs={"musgrave_dimensions": "2D"}
        )
        obj.scale = scale
        obj.rotation_euler[-1] = rotation_z
        butil.apply_transform(obj)
        return obj


class ColumnarCactusFactory(CactusFactory):
    _factory_method_class = ColumnarBaseCactusFactory


class PrickyPearCactusFactory(CactusFactory):
    _factory_method_class = PrickyPearBaseCactusFactory


class KalidiumCactusFactory(CactusFactory):
    _factory_method_class = KalidiumBaseCactusFactory
