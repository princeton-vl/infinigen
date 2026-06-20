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
from infinigen.assets.utils.misc import assign_material
from infinigen.assets.utils.object import join_objects
from infinigen.core import surface
from infinigen.core.nodes.node_info import Nodes
from infinigen.core.nodes.node_utils import build_color_ramp
from infinigen.core.nodes.node_wrangler import NodeWrangler
from infinigen.core.placement.detail import remesh_with_attrs
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.placement.parameters import AssetParameters, ParameterizedAssetFactory
from infinigen.core.tagging import tag_object
from infinigen.core.util.color import hsv2rgba
from infinigen.core.util.random import log_uniform

from . import tentacles
from .base import BaseCoralFactory
from .diff_growth import (
    DiffGrowthBaseCoralFactory,
    LeatherBaseCoralFactory,
    TableBaseCoralFactory,
)
from .elkhorn import ElkhornBaseCoralFactory
from .fan import FanBaseCoralFactory
from .laplacian import CauliflowerBaseCoralFactory
from .reaction_diffusion import (
    BrainBaseCoralFactory,
    HoneycombBaseCoralFactory,
    ReactionDiffusionBaseCoralFactory,
)
from .star import StarBaseCoralFactory
from .tree import BushBaseCoralFactory, TreeBaseCoralFactory, TwigBaseCoralFactory
from .tube import TubeBaseCoralFactory


class CoralParameters(AssetParameters):
    base_hue: Annotated[float, Field(ge=-0.2, le=0.3, json_schema_extra={"editable": True})]
    has_bump_draw: Annotated[
        float, Field(ge=0.0, le=1.0, json_schema_extra={"editable": True})
    ]
    tentacle_draw: Annotated[
        float, Field(ge=0.0, le=1.0, json_schema_extra={"editable": True})
    ]
    scale_factors: tuple[float, float, float] = Field(
        json_schema_extra={"editable": False}
    )
    factory: Any = Field(json_schema_extra={"editable": False})
    material: Any = Field(json_schema_extra={"editable": False})


class CoralFactory(ParameterizedAssetFactory, AssetFactory):
    parameters_model: ClassVar[type[AssetParameters]] = CoralParameters
    _factory_method_class: ClassVar[Type | None] = None

    def __init__(self, factory_seed, coarse=False, factory_method=None):
        super(CoralFactory, self).__init__(factory_seed, coarse)
        self._init_factory_method = factory_method
        self.init_legacy_parameters()

    def _resolve_factory_method(self, factory_method=None):
        if factory_method is not None:
            return factory_method
        if self._factory_method_class is not None:
            return self._factory_method_class
        factory_methods = [
            DiffGrowthBaseCoralFactory,
            ReactionDiffusionBaseCoralFactory,
            TubeBaseCoralFactory,
            TreeBaseCoralFactory,
            CauliflowerBaseCoralFactory,
            ElkhornBaseCoralFactory,
            StarBaseCoralFactory,
        ]
        weights = np.array([0.15, 0.2, 0.15, 0.2, 0.2, 0.15, 0.2])
        return np.random.choice(factory_methods, p=weights / weights.sum())

    def _sample_init_parameters(self, seed: int) -> CoralParameters:
        factory_method = self._resolve_factory_method(self._init_factory_method)
        factory: BaseCoralFactory = factory_method(seed, self.coarse)
        base_hue = self.build_base_hue()
        material = surface.shaderfunc_to_material(self.shader_coral, base_hue)
        return CoralParameters(
            seed=seed,
            base_hue=base_hue,
            has_bump_draw=uniform(),
            tentacle_draw=uniform(),
            scale_factors=(1.0, 1.0, 1.0),
            factory=factory,
            material=material,
        )

    def _sample_spawn_parameters(
        self, params: CoralParameters, seed: int, i: int
    ) -> CoralParameters:
        return params.model_copy(
            update={"scale_factors": tuple(uniform(0.8, 1.2, 3))}
        )

    def apply_parameters(
        self, params: CoralParameters, *, spawn_scope: bool = True
    ) -> None:
        self.factory = params.factory
        self.base_hue = params.base_hue
        self.material = params.material
        self._has_bump_draw = params.has_bump_draw
        self._tentacle_draw = params.tentacle_draw
        self._use_fixed_spawn_draws = spawn_scope
        if spawn_scope:
            self._scale_factors = params.scale_factors

    def create_asset(self, face_size=0.01, realize=True, **params):
        obj = self.factory.create_asset(**params)
        scale = (
            2
            * np.array(self.factory.default_scale)
            / max(obj.dimensions[:2])
            * (
                self._scale_factors
                if self._use_fixed_spawn_draws
                else uniform(0.8, 1.2, 3)
            )
        )
        butil.apply_transform(obj)
        remesh_with_attrs(obj, face_size)
        assign_material(obj, self.material)

        has_bump = (
            self._has_bump_draw < self.factory.bump_prob
            if self._use_fixed_spawn_draws
            else uniform(0, 1) < self.factory.bump_prob
        )
        if self.factory.noise_strength > 0:
            if has_bump:
                self.apply_noise_texture(obj)
            else:
                self.apply_bump(obj)

        tag_object(obj, "coral")

        tentacle = (
            self._tentacle_draw < self.factory.tentacle_prob
            if self._use_fixed_spawn_draws
            else uniform(0, 1) < self.factory.tentacle_prob
        )
        if tentacle and not has_bump:
            t = tentacles.apply(
                obj,
                self.factory.points_fn,
                self.factory.density,
                realize,
                self.base_hue,
            )
            obj = join_objects([obj, t])

        return obj

    def apply_noise_texture(self, obj):
        t = np.random.choice(["STUCCI", "MARBLE"])
        texture = bpy.data.textures.new(name="coral", type=t)
        texture.noise_scale = log_uniform(0.01, 0.02)
        butil.modify_mesh(
            obj,
            "DISPLACE",
            True,
            strength=self.factory.noise_strength * uniform(0.9, 1.2),
            mid_level=0,
            texture=texture,
        )

    def apply_bump(self, obj):
        texture = bpy.data.textures.new(name="coral", type="VORONOI")
        texture.noise_scale = log_uniform(0.02, 0.03)
        texture.noise_intensity = log_uniform(1.5, 2)
        texture.distance_metric = "MINKOVSKY"
        texture.minkovsky_exponent = uniform(1, 1.5)
        butil.modify_mesh(
            obj,
            "DISPLACE",
            True,
            strength=-self.factory.noise_strength * uniform(1, 2),
            mid_level=1,
            texture=texture,
        )

    @staticmethod
    def build_base_hue():
        if uniform(0, 1) < 0.25:
            base_hue = uniform(0, 1)
        else:
            base_hue = uniform(-0.2, 0.3) % 1
        return base_hue

    @staticmethod
    def shader_coral(nw: NodeWrangler, base_hue):
        shift = uniform(0.05, 0.1) * (-1) ** np.random.randint(2)
        subsurface_color = hsv2rgba(uniform(0, 1), uniform(0, 1), 1.0)
        bright_color = hsv2rgba((base_hue + shift) % 1, uniform(0.7, 0.9), 0.2)
        dark_color = hsv2rgba(base_hue, uniform(0.5, 0.7), 0.1)
        light_color = hsv2rgba(
            (base_hue + uniform(-0.2, 0.2)) % 1, uniform(0.2, 0.4), 0.4
        )
        specular = uniform(0.25, 0.5)

        color = build_color_ramp(
            nw,
            nw.musgrave(uniform(10, 20)),
            [0.0, 0.3, 0.7, 1.0],
            [dark_color, dark_color, bright_color, bright_color],
        )
        color = nw.new_node(
            Nodes.MixRGB,
            [
                nw.build_float_curve(
                    nw.musgrave(uniform(10, 20)),
                    [(0, 1), (uniform(0.3, 0.4), 0), (1, 0)],
                ),
                color,
                light_color,
            ],
        )

        noise_texture = nw.new_node(Nodes.NoiseTexture, input_kwargs={"Scale": 50})
        roughness = nw.build_float_curve(noise_texture, [(0, 0.5), (1, 1.0)])
        subsurface_ratio = uniform(0, 0.05) if uniform(0, 1) > 0.5 else 0
        subsurface_radius = [uniform(0.05, 0.2)] * 3
        bsdf = nw.new_node(
            Nodes.PrincipledBSDF,
            input_kwargs={
                "Base Color": color,
                "Roughness": roughness,
                "Specular IOR Level": specular,
                "Subsurface Weight": subsurface_ratio,
                "Subsurface Radius": subsurface_radius,
                "Subsurface Color": subsurface_color,
            },
        )
        return bsdf


class LeatherCoralFactory(CoralFactory):
    _factory_method_class = LeatherBaseCoralFactory


class TableCoralParameters(CoralParameters):
    diff_growth_114: Annotated[
        float, Field(ge=1.0, le=2.0, json_schema_extra={"editable": True})
    ] = 1.5


class TableCoralFactory(CoralFactory):
    parameters_model: ClassVar[type[AssetParameters]] = TableCoralParameters
    _factory_method_class = TableBaseCoralFactory

    def _sample_init_parameters(self, seed: int) -> TableCoralParameters:
        params = super()._sample_init_parameters(seed)
        return TableCoralParameters(**params.model_dump())

    def _sample_spawn_parameters(
        self, params: TableCoralParameters, seed: int, i: int
    ) -> TableCoralParameters:
        params = super()._sample_spawn_parameters(params, seed, i)
        return params.model_copy(update={"diff_growth_114": uniform(1.0, 2.0)})

    def apply_parameters(
        self, params: TableCoralParameters, *, spawn_scope: bool = True
    ) -> None:
        super().apply_parameters(params, spawn_scope=spawn_scope)
        if spawn_scope:
            self._diff_growth_114 = params.diff_growth_114

    def create_asset(self, face_size=0.01, realize=True, **params):
        z_scale = (
            self._diff_growth_114
            if self._use_fixed_spawn_draws
            else uniform(1.0, 2.0)
        )
        saved_maker = self.factory.maker

        def maker():
            obj = DiffGrowthBaseCoralFactory.diff_growth_make(
                "flat_coral",
                1,
                max_polygons=4e2,
                repulsion_radius=2,
                inhibit_shell=1,
            )
            obj.scale = 1, 1, z_scale
            butil.apply_transform(obj)
            return obj

        self.factory.maker = maker
        try:
            return super().create_asset(face_size=face_size, realize=realize, **params)
        finally:
            self.factory.maker = saved_maker


class CauliflowerCoralFactory(CoralFactory):
    _factory_method_class = CauliflowerBaseCoralFactory


class BrainCoralFactory(CoralFactory):
    _factory_method_class = BrainBaseCoralFactory


class HoneycombCoralFactory(CoralFactory):
    _factory_method_class = HoneycombBaseCoralFactory


class BushCoralFactory(CoralFactory):
    _factory_method_class = BushBaseCoralFactory


class TwigCoralFactory(CoralFactory):
    _factory_method_class = TwigBaseCoralFactory


class TubeCoralFactory(CoralFactory):
    parameters_model: ClassVar[type[AssetParameters]] = CoralParameters
    _factory_method_class = TubeBaseCoralFactory


class FanCoralFactory(CoralFactory):
    _factory_method_class = FanBaseCoralFactory


class ElkhornCoralFactory(CoralFactory):
    _factory_method_class = ElkhornBaseCoralFactory


class StarCoralFactory(CoralFactory):
    _factory_method_class = StarBaseCoralFactory
