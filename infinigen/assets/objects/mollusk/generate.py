# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei

from __future__ import annotations

from typing import Annotated, Any, ClassVar, Type

import colorsys

import bpy
import numpy as np
from numpy.random import uniform
from pydantic import Field

from infinigen.assets.utils.decorate import subsurface2face_size
from infinigen.assets.utils.draw import shape_by_angles
from infinigen.assets.utils.misc import assign_material
from infinigen.assets.utils.object import join_objects
from infinigen.core import surface
from infinigen.core.nodes.node_utils import build_color_ramp
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.placement.parameters import AssetParameters, ParameterizedAssetFactory
from infinigen.core.tagging import tag_object
from infinigen.core.util import blender as butil
from infinigen.core.util.random import log_uniform

from .base import BaseMolluskFactory
from .shell import (
    ClamBaseFactory,
    MusselBaseFactory,
    ScallopBaseFactory,
    ShellBaseFactory,
)
from .snail import (
    AugerBaseFactory,
    ConchBaseFactory,
    NautilusBaseFactory,
    SnailBaseFactory,
    VoluteBaseFactory,
)


class MolluskParameters(AssetParameters):
    generate_71: Annotated[
        float, Field(ge=0.1, le=0.2, json_schema_extra={"editable": True})
    ] = 0.15
    generate_85: Annotated[
        float, Field(ge=0.0, le=1.0, json_schema_extra={"editable": True})
    ]
    base_hue: Annotated[float, Field(ge=0.0, le=0.2, json_schema_extra={"editable": True})]
    factory: Any = Field(json_schema_extra={"editable": False})
    material: Any = Field(json_schema_extra={"editable": False})


class MusselParameters(MolluskParameters):
    generate_88: Annotated[
        float, Field(ge=0.05, le=0.12, json_schema_extra={"editable": True})
    ]
    shell_159: Annotated[
        float, Field(ge=0.523599, le=1.047198, json_schema_extra={"editable": True})
    ] = 0.785398
    base: Annotated[float, Field(ge=0.0, le=0.785398, json_schema_extra={"editable": True})] = (
        0.392699
    )
    s: Annotated[float, Field(ge=0.6, le=0.8, json_schema_extra={"editable": True})] = 0.7
    scales: Annotated[float, Field(ge=0.6, le=0.8, json_schema_extra={"editable": True})] = (
        0.7
    )
    z_scale: Annotated[float, Field(ge=2.0, le=10.0, json_schema_extra={"editable": True})] = (
        5.0
    )


class MolluskFactory(ParameterizedAssetFactory, AssetFactory):
    parameters_model: ClassVar[type[AssetParameters]] = MolluskParameters
    _factory_method_class: ClassVar[Type | None] = None

    def __init__(self, factory_seed, coarse=False, factory_method=None):
        super(MolluskFactory, self).__init__(factory_seed, coarse)
        self._init_factory_method = factory_method
        self.init_legacy_parameters()

    def _resolve_factory_method(self, factory_method=None):
        if factory_method is not None:
            return factory_method
        if self._factory_method_class is not None:
            return self._factory_method_class
        factory_methods = [SnailBaseFactory, ShellBaseFactory]
        weights = np.array([1] * len(factory_methods))
        return np.random.choice(factory_methods, p=weights / weights.sum())

    def _sample_base_hue(self, generate_88: float | None = None) -> float:
        if uniform(0, 1) < 0.4:
            return uniform(0, 0.2)
        if generate_88 is not None:
            return generate_88
        return uniform(0.05, 0.12)

    def _build_factory(self, factory_method, seed: int) -> BaseMolluskFactory:
        return factory_method(seed, self.coarse)

    def _sample_init_parameters(self, seed: int) -> MolluskParameters:
        factory_method = self._resolve_factory_method(self._init_factory_method)
        factory = self._build_factory(factory_method, seed)
        base_hue = self._sample_base_hue()
        material = surface.shaderfunc_to_material(
            self.shader_mollusk,
            base_hue,
            factory.ratio,
            factory.x_scale,
            factory.z_scale,
            factory.distortion,
        )
        return MolluskParameters(
            seed=seed,
            generate_85=uniform(),
            base_hue=base_hue,
            factory=factory,
            material=material,
        )

    def _sample_spawn_parameters(
        self, params: MolluskParameters, seed: int, i: int
    ) -> MolluskParameters:
        return params.model_copy(update={"generate_71": log_uniform(0.1, 0.2)})

    def apply_parameters(
        self, params: MolluskParameters, *, spawn_scope: bool = True
    ) -> None:
        self.factory = params.factory
        self.material = params.material
        self.base_hue = params.base_hue
        self._texture_type_draw = params.generate_85
        self._use_fixed_spawn_draws = spawn_scope
        if spawn_scope:
            self._noise_scale = params.generate_71

    def create_asset(self, face_size=0.01, **params):
        obj = self.factory.create_asset(**params)
        self.decorate_mollusk(obj, face_size)
        return obj

    def decorate_mollusk(self, obj, face_size):
        subsurface2face_size(obj, face_size)
        butil.modify_mesh(obj, "SOLIDIFY", True, thickness=0.005)
        texture_types = ["STUCCI", "MARBLE"]
        t = (
            texture_types[int(self._texture_type_draw * len(texture_types))]
            if self._use_fixed_spawn_draws
            else np.random.choice(texture_types)
        )
        texture = bpy.data.textures.new(name="mollusk", type=t)
        noise_scale = (
            self._noise_scale if self._use_fixed_spawn_draws else log_uniform(0.1, 0.2)
        )
        texture.noise_scale = noise_scale
        butil.modify_mesh(
            obj,
            "DISPLACE",
            strength=self.factory.noise_strength,
            mid_level=0,
            texture=texture,
        )
        assign_material(obj, self.material)
        tag_object(obj, "mollusk")
        return obj

    @staticmethod
    def build_base_hue():
        if uniform(0, 1) < 0.4:
            return uniform(0, 0.2)
        else:
            return uniform(0.05, 0.12)

    @staticmethod
    def shader_mollusk(
        nw: NodeWrangler, base_hue, ratio=0, x_scale=2, z_scale=1, distortion=5
    ):
        roughness = uniform(0.2, 0.8)
        specular = 0.3
        value_scale = log_uniform(1, 20)
        saturation_scale = log_uniform(0.4, 1)

        def dark_color():
            return *colorsys.hsv_to_rgb(
                base_hue + uniform(-0.06, 0.06),
                uniform(0.6, 1.0) * saturation_scale,
                0.005 * value_scale**1.5,
            ), 1

        def light_color():
            return *colorsys.hsv_to_rgb(
                base_hue + uniform(-0.06, 0.06),
                uniform(0.6, 1.0) * saturation_scale,
                0.05 * value_scale,
            ), 1

        def color_fn(dark_prob):
            return dark_color() if uniform(0, 1) < dark_prob else light_color()

        vector = nw.new_node(
            Nodes.Attribute, attrs={"attribute_name": "vector"}
        ).outputs["Vector"]
        n = np.random.randint(3, 5)
        texture_0 = nw.new_node(
            Nodes.WaveTexture,
            input_kwargs={"Vector": vector, "Distortion": distortion, "Scale": x_scale},
            attrs={"wave_profile": "SAW", "bands_direction": "X"},
        )
        cr_0 = build_color_ramp(
            nw, texture_0, np.sort(uniform(0, 1, n)), [color_fn(0.4) for _ in range(n)]
        )
        texture_1 = nw.new_node(
            Nodes.WaveTexture,
            input_kwargs={"Vector": vector, "Distortion": distortion, "Scale": z_scale},
            attrs={"wave_profile": "SAW", "bands_direction": "Z"},
        )
        cr_1 = build_color_ramp(
            nw, texture_1, np.sort(uniform(0, 1, n)), [color_fn(0.4) for _ in range(n)]
        )
        principled_bsdf = nw.new_node(
            Nodes.PrincipledBSDF,
            input_kwargs={
                "Base Color": nw.new_node(Nodes.MixRGB, [ratio, cr_0, cr_1]),
                "Specular IOR Level": specular,
                "Roughness": roughness,
            },
        )
        return principled_bsdf


def _bind_mussel_shell_factory(
    factory: MusselBaseFactory, params: MusselParameters, fixed: bool
) -> None:
    factory.z_scale = params.z_scale
    if not fixed:
        return

    def mussel_make():
        obj = factory.build_ellipse(softness=0.5)
        obj.scale = 1, 3, 1
        butil.apply_transform(obj)
        angles = [-0.5, -uniform(0.1, 0.15), uniform(0.0, 0.25), 0.5]
        scales = [0, params.s, 1, params.scales]
        shape_by_angles(obj, np.array(angles) * np.pi, scales)
        tag_object(obj, "mussel")
        return obj

    def create_asset(**kw):
        upper = mussel_make()
        dim = np.sqrt(upper.dimensions[0] * upper.dimensions[1] + 0.01)
        upper.scale = [1 / dim] * 3
        upper.location[-1] += 0.005
        butil.apply_transform(upper, loc=True)
        lower = butil.deep_clone_obj(upper)
        lower.scale[-1] = -1
        butil.apply_transform(lower)
        lower.rotation_euler[1] = -params.base
        upper.rotation_euler[1] = -params.base - params.shell_159
        return join_objects([lower, upper])

    factory.mussel_make = mussel_make
    factory.create_asset = create_asset


class MusselFactory(MolluskFactory):
    parameters_model: ClassVar[type[AssetParameters]] = MusselParameters
    _factory_method_class = MusselBaseFactory

    def _sample_init_parameters(self, seed: int) -> MusselParameters:
        factory_method = self._resolve_factory_method(self._init_factory_method)
        generate_88 = uniform(0.05, 0.12)
        factory = self._build_factory(factory_method, seed)
        factory.z_scale = log_uniform(2, 10)
        base_hue = self._sample_base_hue(generate_88)
        material = surface.shaderfunc_to_material(
            self.shader_mollusk,
            base_hue,
            factory.ratio,
            factory.x_scale,
            factory.z_scale,
            factory.distortion,
        )
        return MusselParameters(
            seed=seed,
            generate_85=uniform(),
            generate_88=generate_88,
            base_hue=base_hue,
            z_scale=factory.z_scale,
            factory=factory,
            material=material,
        )

    def _sample_spawn_parameters(
        self, params: MusselParameters, seed: int, i: int
    ) -> MusselParameters:
        return params.model_copy(
            update={
                "generate_71": log_uniform(0.1, 0.2),
                "shell_159": uniform(np.pi / 6, np.pi / 3),
                "base": uniform(0, np.pi / 4),
                "s": uniform(0.6, 0.8),
                "scales": uniform(0.6, 0.8),
            }
        )

    def apply_parameters(
        self, params: MusselParameters, *, spawn_scope: bool = True
    ) -> None:
        super().apply_parameters(params, spawn_scope=spawn_scope)
        self.factory.z_scale = params.z_scale
        if spawn_scope:
            self._shell_159 = params.shell_159
            self._shell_base = params.base
            self._mussel_s = params.s
            self._mussel_scales = params.scales

    def create_asset(self, face_size=0.01, **params):
        if self._use_fixed_spawn_draws:
            _bind_mussel_shell_factory(
                self.factory,
                MusselParameters(
                    seed=self.factory_seed,
                    generate_85=self._texture_type_draw,
                    generate_88=self.base_hue,
                    base_hue=self.base_hue,
                    generate_71=self._noise_scale,
                    shell_159=self._shell_159,
                    base=self._shell_base,
                    s=self._mussel_s,
                    scales=self._mussel_scales,
                    z_scale=self.factory.z_scale,
                    factory=self.factory,
                    material=self.material,
                ),
                True,
            )
        return super().create_asset(face_size=face_size, **params)


class ScallopFactory(MolluskFactory):
    _factory_method_class = ScallopBaseFactory


class ClamFactory(MolluskFactory):
    _factory_method_class = ClamBaseFactory


class ConchFactory(MolluskFactory):
    _factory_method_class = ConchBaseFactory


class AugerFactory(MolluskFactory):
    _factory_method_class = AugerBaseFactory


class VoluteFactory(MolluskFactory):
    _factory_method_class = VoluteBaseFactory


class NautilusFactory(MolluskFactory):
    parameters_model: ClassVar[type[AssetParameters]] = MolluskParameters
    _factory_method_class = NautilusBaseFactory
