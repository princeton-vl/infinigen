# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei

from __future__ import annotations

from typing import Annotated, Any, ClassVar

import bpy
import numpy as np
from numpy.random import uniform
from pydantic import Field

import infinigen.core.util.blender as butil
from infinigen.assets.objects.monocot.growth import MonocotGrowthFactory
from infinigen.assets.utils.decorate import displace_vertices, distance2boundary
from infinigen.assets.utils.draw import cut_plane, leaf
from infinigen.assets.utils.object import join_objects
from infinigen.core.placement.parameters import AssetParameters, ParameterizedAssetFactory
from infinigen.core.surface import shaderfunc_to_material
from infinigen.core.tagging import tag_object
from infinigen.core.util.blender import deep_clone_obj
from infinigen.core.util.color import hsv2rgba
from infinigen.core.util.random import log_uniform


class AgaveMonocotParameters(AssetParameters):
    stem_offset: Annotated[float, Field(ge=0.0, le=0.5, json_schema_extra={"editable": True})]
    angle: Annotated[float, Field(ge=0.349066, le=0.523599, json_schema_extra={"editable": True})]
    z_drag: Annotated[float, Field(ge=0.05, le=0.1, json_schema_extra={"editable": True})]
    min_y_angle: Annotated[
        float, Field(ge=0.314159, le=0.471239, json_schema_extra={"editable": True})
    ]
    max_y_angle: Annotated[
        float, Field(ge=1.256637, le=1.633628, json_schema_extra={"editable": True})
    ]
    count: Annotated[float, Field(ge=32.0, le=64.0, json_schema_extra={"editable": True})]
    scale_curve_low: Annotated[
        float, Field(ge=0.8, le=1.0, json_schema_extra={"editable": True})
    ]
    scale_curve_high: Annotated[
        float, Field(ge=0.6, le=1.0, json_schema_extra={"editable": True})
    ]
    bud_angle: Annotated[
        float, Field(ge=0.392699, le=0.785398, json_schema_extra={"editable": True})
    ]
    cut_prob_draw: Annotated[
        float, Field(ge=0.0, le=1.0, json_schema_extra={"editable": True})
    ]
    cut_prob: Annotated[float, Field(ge=0.0, le=0.4, json_schema_extra={"editable": True})] = (
        0.0
    )
    leaf_prob: Annotated[float, Field(ge=0.8, le=0.9, json_schema_extra={"editable": True})]
    z_scale: Annotated[float, Field(ge=1.0, le=1.2, json_schema_extra={"editable": True})]
    base_hue: Annotated[float, Field(ge=0.12, le=0.32, json_schema_extra={"editable": True})]
    material: Any = Field(json_schema_extra={"editable": False})


class AgaveMonocotFactory(ParameterizedAssetFactory, MonocotGrowthFactory):
    use_distance = True
    parameters_model: ClassVar[type[AssetParameters]] = AgaveMonocotParameters

    def __init__(self, factory_seed, coarse=False):
        super(AgaveMonocotFactory, self).__init__(factory_seed, coarse)
        self.init_legacy_parameters()

    def _sample_init_parameters(self, seed: int) -> AgaveMonocotParameters:
        base_hue = uniform(0.12, 0.32)
        cut_draw = uniform(0, 1)
        cut_prob = 0 if cut_draw < 0.5 else uniform(0.2, 0.4)
        leaf_prob = uniform(0.8, 0.9)
        z_scale = uniform(1.0, 1.2)
        bright_color = hsv2rgba(base_hue, uniform(0.6, 0.8), log_uniform(0.05, 0.1))
        dark_color = hsv2rgba(
            (base_hue + uniform(-0.03, 0.03)) % 1,
            uniform(0.8, 1.0),
            log_uniform(0.05, 0.2),
        )
        material = shaderfunc_to_material(
            self.shader_monocot, dark_color, bright_color, self.use_distance
        )
        return AgaveMonocotParameters(
            seed=seed,
            stem_offset=uniform(0.0, 0.5),
            angle=uniform(np.pi / 9, np.pi / 6),
            z_drag=uniform(0.05, 0.1),
            min_y_angle=uniform(np.pi * 0.1, np.pi * 0.15),
            max_y_angle=uniform(np.pi * 0.4, np.pi * 0.52),
            count=log_uniform(32, 64),
            scale_curve_low=uniform(0.8, 1.0),
            scale_curve_high=uniform(0.6, 1.0),
            bud_angle=uniform(np.pi / 8, np.pi / 4),
            cut_prob_draw=cut_draw,
            cut_prob=cut_prob,
            leaf_prob=leaf_prob,
            z_scale=z_scale,
            base_hue=base_hue,
            material=material,
        )

    def apply_parameters(
        self, params: AgaveMonocotParameters, *, spawn_scope: bool = True
    ) -> None:
        self.stem_offset = params.stem_offset
        self.angle = params.angle
        self.z_drag = params.z_drag
        self.min_y_angle = params.min_y_angle
        self.max_y_angle = params.max_y_angle
        self.count = int(params.count)
        self.scale_curve = [
            (0, params.scale_curve_low),
            (0.5, 1),
            (1, params.scale_curve_high),
        ]
        self.bud_angle = params.bud_angle
        self.cut_prob = params.cut_prob
        self.leaf_prob = params.leaf_prob
        self.z_scale = params.z_scale
        self.material = params.material
        self.radius = 0.01
        self.leaf_range = (0, 1)
        self.perturb = 0.05
        self.bend_angle = np.pi / 4
        self.twist_angle = np.pi / 6
        self.align_factor = 0
        self.align_direction = (1, 0, 0)
        self._use_fixed_spawn_draws = spawn_scope

    @staticmethod
    def build_base_hue():
        return uniform(0.12, 0.32)

    def build_leaf(self, face_size):
        x_anchors = 0, 0.2 * np.cos(self.bud_angle), uniform(1.0, 1.4), 1.5
        y_anchors = 0, 0.2 * np.sin(self.bud_angle), uniform(0.1, 0.15), 0
        obj = leaf(x_anchors, y_anchors, face_size=face_size)
        distance = distance2boundary(obj)

        lower = deep_clone_obj(obj)
        z_offset = -log_uniform(0.08, 0.16)
        z_ratio = uniform(1.5, 2.5)
        displace_vertices(
            lower, lambda x, y, z: (0, 0, (1 - (1 - distance) ** z_ratio) * z_offset)
        )
        obj = join_objects([lower, obj])
        butil.modify_mesh(obj, "WELD", merge_threshold=2e-4)

        if uniform(0, 1) < self.cut_prob:
            angle = uniform(-np.pi / 3, np.pi / 3)
            cut_center = np.array([uniform(1.0, 1.4), 0, 0])
            cut_normal = np.array([np.cos(angle), np.sin(angle), 0])
            obj, cut = cut_plane(obj, cut_center, cut_normal)
            obj = join_objects([obj, cut])
            with butil.ViewportMode(obj, "EDIT"), butil.Suppress():
                bpy.ops.mesh.select_all(action="SELECT")
                bpy.ops.mesh.region_to_loop()
                bpy.ops.mesh.remove_doubles(threshold=1e-2)

        self.decorate_leaf(obj)
        tag_object(obj, "agave")
        return obj
