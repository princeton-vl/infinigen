# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei


from __future__ import annotations

from typing import Annotated, ClassVar

import bpy
import numpy as np
from numpy.random import uniform
from pydantic import Field

from infinigen.assets.objects.monocot.growth import MonocotGrowthFactory
from infinigen.assets.utils.decorate import (
    remove_vertices,
    write_attribute,
    write_material_index,
)
from infinigen.assets.utils.draw import bezier_curve, leaf, spin
from infinigen.assets.utils.mesh import polygon_angles
from infinigen.assets.utils.misc import assign_material
from infinigen.assets.utils.object import join_objects
from infinigen.core import surface
from infinigen.core.nodes.node_info import Nodes
from infinigen.core.nodes.node_wrangler import NodeWrangler
from infinigen.core.placement.detail import remesh_with_attrs
from infinigen.core.placement.factory import AssetFactory, make_asset_collection
from infinigen.core.placement.parameters import (
    AssetParameters,
    LegacyBridgeParameters,
    ParameterizedAssetFactory,
    apply_bridge_parameters,
    legacy_init_to_parameters,
)
from infinigen.core.surface import shaderfunc_to_material
from infinigen.core.tagging import tag_object
from infinigen.core.util import blender as butil
from infinigen.core.util.color import hsv2rgba
from infinigen.core.util.math import FixedSeed
from infinigen.core.util.random import log_uniform


class GrassesMonocotParameters(LegacyBridgeParameters):
    cut_prob_draw: Annotated[
        float, Field(ge=0.0, le=1.0, json_schema_extra={"editable": True})
    ] = 0.0


def _grasses_monocot_legacy_init(
    inst: GrassesMonocotFactory, seed: int, coarse: bool
) -> None:
    MonocotGrowthFactory.__init__(inst, seed, coarse)
    with FixedSeed(seed):
        inst.stem_offset = uniform(1.5, 2.0)
        inst.angle = uniform(np.pi / 6, np.pi / 3)
        inst.z_drag = uniform(0.0, 0.2)
        inst.min_y_angle = uniform(np.pi * 0.35, np.pi * 0.45)
        inst.max_y_angle = uniform(np.pi * 0.45, np.pi * 0.5)
        inst.count = int(log_uniform(16, 64))
        inst.scale_curve = [(0, 1.0), (1, 0.2)]
        inst.bend_angle = np.pi / 2


class GrassesMonocotFactory(ParameterizedAssetFactory, MonocotGrowthFactory):
    parameters_model: ClassVar[type[AssetParameters]] = GrassesMonocotParameters

    def __init__(self, factory_seed, coarse=False):
        AssetFactory.__init__(self, factory_seed, coarse)
        self.init_legacy_parameters()

    def _sample_init_parameters(self, seed: int) -> GrassesMonocotParameters:
        return legacy_init_to_parameters(
            GrassesMonocotParameters,
            GrassesMonocotFactory,
            seed,
            self.coarse,
            init_fn=_grasses_monocot_legacy_init,
        )

    def _sample_spawn_parameters(
        self, params: GrassesMonocotParameters, seed: int, i: int
    ) -> GrassesMonocotParameters:
        return params.model_copy(update={"cut_prob_draw": uniform()})

    def apply_parameters(
        self, params: GrassesMonocotParameters, *, spawn_scope: bool = True
    ) -> None:
        apply_bridge_parameters(self, params, spawn_scope=spawn_scope)

    @staticmethod
    def build_base_hue():
        if uniform(0, 1) < 0.6:
            return uniform(0.08, 0.12)
        else:
            return uniform(0.2, 0.25)

    def build_leaf(self, face_size):
        x_anchors = np.array([0, uniform(0.1, 0.2), uniform(0.5, 0.7), 1.0])
        y_anchors = np.array([0, uniform(0.02, 0.03), uniform(0.02, 0.03), 0])
        obj = leaf(x_anchors, y_anchors, face_size=face_size)

        if (self.cut_prob_draw if self._use_fixed_spawn_draws else uniform()) < 0.4:
            x_cutoff = uniform(0.5, 1.0)
            angle = uniform(-np.pi / 3, np.pi / 3)
            remove_vertices(
                obj,
                lambda x, y, z: (x - x_cutoff) * np.cos(angle) + y * np.sin(angle) > 0,
            )
        self.decorate_leaf(obj)
        tag_object(obj, "grasses")
        return obj

    @property
    def is_grass(self):
        return True


class WheatEarMonocotParameters(LegacyBridgeParameters):
    pass


def _wheat_ear_monocot_legacy_init(
    inst: WheatEarMonocotFactory, seed: int, coarse: bool
) -> None:
    MonocotGrowthFactory.__init__(inst, seed, coarse)
    with FixedSeed(seed):
        inst.stem_offset = uniform(0.4, 0.5)
        inst.angle = uniform(np.pi / 6, np.pi / 4)
        inst.min_y_angle = uniform(np.pi / 4, np.pi / 3)
        inst.max_y_angle = np.pi / 2
        inst.leaf_prob = uniform(0.9, 1)
        inst.count = int(log_uniform(96, 128))
        inst.bend_angle = np.pi


class WheatEarMonocotFactory(ParameterizedAssetFactory, MonocotGrowthFactory):
    parameters_model: ClassVar[type[LegacyBridgeParameters]] = WheatEarMonocotParameters

    def __init__(self, factory_seed, coarse=False):
        AssetFactory.__init__(self, factory_seed, coarse)
        self.init_legacy_parameters()

    def _sample_init_parameters(self, seed: int) -> WheatEarMonocotParameters:
        return legacy_init_to_parameters(
            WheatEarMonocotParameters,
            WheatEarMonocotFactory,
            seed,
            self.coarse,
            init_fn=_wheat_ear_monocot_legacy_init,
        )

    def apply_parameters(
        self, params: WheatEarMonocotParameters, *, spawn_scope: bool = True
    ) -> None:
        apply_bridge_parameters(self, params, spawn_scope=spawn_scope)

    @staticmethod
    def build_base_hue():
        return uniform(0.12, 0.28)

    def build_leaf(self, face_size):
        x_anchors = np.array([0, 0.05, 0.1])
        y_anchors = np.array([0, uniform(0.01, 0.015), 0])
        curves = []
        for angle in polygon_angles(np.random.randint(4, 6)):
            anchors = [x_anchors, np.cos(angle) * y_anchors, np.sin(angle) * y_anchors]
            curves.append(bezier_curve(anchors))
        obj = butil.join_objects(curves)
        with butil.ViewportMode(obj, "EDIT"):
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.convex_hull()
        remesh_with_attrs(obj, face_size / 2)
        tag_object(obj, "wheat_ear")
        return obj


class WheatMonocotParameters(LegacyBridgeParameters):
    ear_bend_angle: Annotated[
        float, Field(ge=0.0, le=np.pi, json_schema_extra={"editable": True})
    ] = 0.0


def _wheat_monocot_legacy_init(
    inst: WheatMonocotFactory, seed: int, coarse: bool
) -> None:
    _grasses_monocot_legacy_init(inst, seed, coarse)
    with FixedSeed(seed):
        inst.ear_factory = WheatEarMonocotFactory(seed, coarse)
        inst.scale_curve = [(0, 1.0), (1, 0.6)]
        inst.leaf_range = 0.1, 0.7


class WheatMonocotFactory(GrassesMonocotFactory):
    parameters_model: ClassVar[type[AssetParameters]] = WheatMonocotParameters

    def __init__(self, factory_seed, coarse=False):
        AssetFactory.__init__(self, factory_seed, coarse)
        self.init_legacy_parameters()

    def _sample_init_parameters(self, seed: int) -> WheatMonocotParameters:
        return legacy_init_to_parameters(
            WheatMonocotParameters,
            WheatMonocotFactory,
            seed,
            self.coarse,
            init_fn=_wheat_monocot_legacy_init,
        )

    def _sample_spawn_parameters(
        self, params: WheatMonocotParameters, seed: int, i: int
    ) -> WheatMonocotParameters:
        return params.model_copy(
            update={"ear_bend_angle": uniform(0, params.ear_factory.bend_angle)}
        )

    def apply_parameters(
        self, params: WheatMonocotParameters, *, spawn_scope: bool = True
    ) -> None:
        apply_bridge_parameters(self, params, spawn_scope=spawn_scope)

    @staticmethod
    def build_base_hue():
        return uniform(0.08, 0.12)

    def create_asset(self, **params):
        obj = super().create_raw(**params)
        ear = self.ear_factory.create_asset(**params)
        bend_angle = (
            self.ear_bend_angle
            if self._use_fixed_spawn_draws
            else uniform(0, self.ear_factory.bend_angle)
        )
        butil.modify_mesh(
            ear,
            "SIMPLE_DEFORM",
            deform_method="BEND",
            angle=bend_angle,
        )
        ear.location[-1] = self.stem_offset - 0.02
        obj = join_objects([obj, ear])
        self.decorate_monocot(obj)
        tag_object(obj, "wheat")
        return obj


class MaizeMonocotParameters(LegacyBridgeParameters):
    pass


def _maize_monocot_legacy_init(
    inst: MaizeMonocotFactory, seed: int, coarse: bool
) -> None:
    _grasses_monocot_legacy_init(inst, seed, coarse)
    inst.stem_offset = uniform(2.0, 2.5)
    inst.scale_curve = [(0, 1.0), (1, 0.6)]
    inst.leaf_range = 0.1, 0.7


class MaizeMonocotFactory(GrassesMonocotFactory):
    parameters_model: ClassVar[type[AssetParameters]] = MaizeMonocotParameters

    def __init__(self, factory_seed, coarse=False):
        AssetFactory.__init__(self, factory_seed, coarse)
        self.init_legacy_parameters()

    def _sample_init_parameters(self, seed: int) -> MaizeMonocotParameters:
        return legacy_init_to_parameters(
            MaizeMonocotParameters,
            MaizeMonocotFactory,
            seed,
            self.coarse,
            init_fn=_maize_monocot_legacy_init,
        )

    def apply_parameters(
        self, params: MaizeMonocotParameters, *, spawn_scope: bool = True
    ) -> None:
        apply_bridge_parameters(self, params, spawn_scope=spawn_scope)

    def build_leaf(self, face_size):
        x_anchors = np.array([0, uniform(0.1, 0.2), uniform(0.5, 0.7), 1.0])
        y_anchors = np.array([0, uniform(0.03, 0.06), uniform(0.03, 0.06), 0])
        obj = leaf(x_anchors, y_anchors, face_size=face_size)
        self.decorate_leaf(obj)
        tag_object(obj, "maize_leaf")
        return obj

    def build_husk(self):
        x_anchors = 0, uniform(0.04, 0.05), uniform(0.03, 0.03), 0
        z_anchors = 0, 0.01, uniform(0.24, 0.3), uniform(0.35, 0.4)
        anchors = x_anchors, 0, z_anchors
        husk = spin(anchors)
        texture = bpy.data.textures.new(name="husk", type="STUCCI")
        texture.noise_scale = 0.01
        butil.modify_mesh(husk, "DISPLACE", strength=0.02, texture=texture)
        husk.location[-1] = self.stem_offset - 0.02
        husk.rotation_euler[0] = uniform(0, np.pi * 0.2)
        tag_object(husk, "maize_husk")
        return husk

    def create_asset(self, **params):
        obj = super().create_raw(**params)
        husk = self.build_husk()
        obj = join_objects([obj, husk])
        self.decorate_monocot(obj)
        tag_object(obj, "maize")
        return obj


class ReedEarMonocotFactory(MonocotGrowthFactory):
    def __init__(self, factory_seed, coarse=False):
        super(ReedEarMonocotFactory, self).__init__(factory_seed, coarse)
        with FixedSeed(factory_seed):
            self.stem_offset = uniform(0.3, 0.4)
            self.min_y_angle = uniform(np.pi / 4, np.pi / 3)
            self.max_y_angle = self.min_y_angle + np.pi / 12
            self.count = int(log_uniform(48, 96))
            self.radius = 0.002

    def build_leaf(self, face_size):
        x_anchors = np.array([0, uniform(0.02, 0.03), 0.05])
        y_anchors = np.array([0, uniform(0.005, 0.01), 0])
        obj = leaf(x_anchors, y_anchors, face_size=face_size)
        return obj

    def create_raw(self, **params):
        obj = super(ReedEarMonocotFactory, self).create_raw(**params)
        write_attribute(obj, 1, "ear", "FACE")
        tag_object(obj, "reed_ear")
        return obj


class ReedBranchMonocotFactory(MonocotGrowthFactory):
    max_branches = 6

    def __init__(self, factory_seed, coarse=False):
        super(ReedBranchMonocotFactory, self).__init__(factory_seed, coarse)
        with FixedSeed(factory_seed):
            self.stem_offset = uniform(0.6, 0.8)
            self.ear_factory = ReedEarMonocotFactory(self.factory_seed)
            self.scale_curve = (0, 1), (0.5, 0.6), (1, 0.1)
            self.min_y_angle = uniform(-np.pi / 10, -np.pi / 8)
            self.max_y_angle = uniform(-np.pi / 6, -np.pi / 8)
            self.angle = 0
            self.radius = 0.005

    def make_collection(self, face_size):
        return make_asset_collection(
            self.ear_factory.create_raw, 2, "leaves", verbose=False, face_size=face_size
        )


class ReedMonocotFactory(GrassesMonocotFactory):
    def __init__(self, factory_seed, coarse=False):
        super(ReedMonocotFactory, self).__init__(factory_seed, coarse)
        with FixedSeed(factory_seed):
            self.stem_offset = uniform(3.0, 4.0)
            self.scale_curve = [(0, 1.2), (1, 0.8)]
            self.branch_factory = ReedBranchMonocotFactory(factory_seed, coarse)
            self.branch_material = shaderfunc_to_material(self.shader_ear)

    @staticmethod
    def build_base_hue():
        return uniform(0.08, 0.12)

    def create_asset(self, **params):
        obj = super().create_raw(**params)
        branch = self.branch_factory.create_asset(**params)
        self.branch_factory.decorate_monocot(branch)
        branch.location[-1] = self.stem_offset - 0.02
        obj = join_objects([obj, branch])
        butil.modify_mesh(obj, "WELD", merge_threshold=1e-3)
        self.decorate_monocot(obj)

        assign_material(obj, [self.material, self.branch_material])
        write_material_index(
            obj, surface.read_attr_data(obj, "ear", "FACE").astype(int)[:, 0]
        )
        tag_object(obj, "reed")
        return obj

    @staticmethod
    def shader_ear(nw: NodeWrangler):
        color = hsv2rgba(uniform(0.06, 0.1), uniform(0.2, 0.5), log_uniform(0.2, 0.5))
        specular = uniform(0.0, 0.2)
        clearcoat = 0 if uniform(0, 1) < 0.8 else uniform(0.2, 0.5)
        noise_texture = nw.new_node(Nodes.NoiseTexture, input_kwargs={"Scale": 50})
        roughness = nw.build_float_curve(noise_texture, [(0, 0.5), (1, 0.8)])
        bsdf = nw.new_node(
            Nodes.PrincipledBSDF,
            input_kwargs={
                "Base Color": color,
                "Roughness": roughness,
                "Specular IOR Level": specular,
                "Coat Weight": clearcoat,
                "Subsurface Weight": 0.01,
                "Subsurface Radius": (0.01, 0.01, 0.01),
            },
        )
        return bsdf
