# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei
from __future__ import annotations

from typing import Any, ClassVar

import bpy
import bmesh
import numpy as np
from numpy.random import uniform

from infinigen.assets.composition import material_assignments
from infinigen.assets.objects.bathroom.bathtub import (
    BathtubFactory,
    BathtubParameters,
    _bathtub_legacy_init,
)
from infinigen.assets.objects.table_decorations import TapFactory
from infinigen.assets.utils.decorate import read_co, subdivide_edge_ring, subsurf
from infinigen.assets.utils.object import (
    join_objects,
    new_base_cylinder,
    new_bbox,
    new_cube,
)
from infinigen.core import surface
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.placement.parameters import (
    AssetParameters,
    LegacyBridgeParameters,
    ParameterizedAssetFactory,
    apply_bridge_parameters,
    legacy_init_to_parameters,
)
from infinigen.core.util import blender as butil
from infinigen.core.util.math import FixedSeed
from infinigen.core.util.random import log_uniform, weighted_sample


def _bathroom_sink_legacy_init(inst: Any, seed: int, coarse: bool) -> None:
    AssetFactory.__init__(inst, seed, coarse)
    bathtub_params = legacy_init_to_parameters(
        BathtubParameters,
        BathtubFactory,
        seed,
        coarse,
        init_fn=_bathtub_legacy_init,
    )
    apply_bridge_parameters(inst, bathtub_params, spawn_scope=False)
    with FixedSeed(seed):
        inst.width = uniform(0.6, 0.9)
        inst.size = inst.width * log_uniform(0.55, 0.8)
        inst.depth = inst.width * log_uniform(0.2, 0.4)
        inst.contour_fn = inst.make_box_contour
        inst.sink_types = np.random.choice(["undermount", "drop-in", "vessel"])
        inst.has_stand = False
        match inst.sink_types:
            case "undermount":
                inst.bathtub_type = "freestanding"
                inst.has_extrude = uniform() < 0.7
            case "drop-in":
                inst.bathtub_type = "alcove"
                inst.has_extrude = True
            case _:
                inst.bathtub_type = np.random.choice(["alcove", "freestanding"])
                inst.has_extrude = uniform() < 0.7
                inst.has_stand = True
        inst.tap_factory = TapFactory(inst.factory_seed)
        inst.disp_x = [inst.disp_x[0], inst.disp_x[0]]
        inst.alcove_levels = 0 if uniform() < 0.5 else np.random.randint(2, 4)
        inst.thickness = 0.01 if inst.has_base else uniform(0.01, 0.03)
        inst.size_extrude = uniform(0.2, 0.35)
        inst.tap_offset = uniform(0.0, 0.05)
        inst.stand_radius = inst.width / 2 * log_uniform(0.15, 0.2)
        inst.stand_bottom = (
            inst.width * log_uniform(0.2, 0.3)
            if uniform() < 0.6
            else inst.stand_radius
        )
        inst.stand_height = uniform(0.7, 0.9) - inst.depth
        inst.is_stand_circular = uniform() < 0.5
        inst.is_hole_centered = True

        surface_gen_class = weighted_sample(material_assignments.bathroom_touchsurface)
        inst.surface_material_gen = surface_gen_class()


class BathroomSinkParameters(LegacyBridgeParameters):
    pass


class BathroomSinkFactory(BathtubFactory):
    parameters_model: ClassVar[type[AssetParameters]] = BathroomSinkParameters

    def __init__(self, factory_seed, coarse=False):
        AssetFactory.__init__(self, factory_seed, coarse)
        self.init_legacy_parameters()

    def _sample_init_parameters(self, seed: int) -> BathroomSinkParameters:
        return legacy_init_to_parameters(
            BathroomSinkParameters,
            BathroomSinkFactory,
            seed,
            self.coarse,
            init_fn=_bathroom_sink_legacy_init,
        )

    def apply_parameters(
        self, params: BathroomSinkParameters, *, spawn_scope: bool = True
    ) -> None:
        apply_bridge_parameters(self, params, spawn_scope=spawn_scope)

    def create_placeholder(self, **kwargs) -> bpy.types.Object:
        return new_bbox(
            -(self.size_extrude + 1) * self.size,
            0,
            0,
            self.width,
            -self.stand_height if self.has_stand else 0,
            self.depth,
        )

    def create_asset(self, **params) -> bpy.types.Object:
        self.surface = self.surface_material_gen()
        if self.has_base:
            obj = self.make_base()
            cutter = self.make_cutter()
            butil.modify_mesh(obj, "BOOLEAN", object=cutter, operation="DIFFERENCE")
            butil.delete(cutter)
        else:
            obj = self.make_bowl()
            self.remove_top(obj)
            butil.modify_mesh(obj, "SOLIDIFY", thickness=self.thickness)
            subsurf(obj, self.side_levels)
        obj.location = np.array(obj.location) - np.min(read_co(obj), 0)
        butil.apply_transform(obj, True)
        obj.scale = np.array([self.width, self.size, self.depth]) / np.array(
            obj.dimensions
        )
        butil.apply_transform(obj, True)
        if self.has_extrude:
            self.extrude_back(obj)
        if self.has_stand:
            self.add_stand(obj)
        hole = self.add_hole(obj)
        obj = join_objects([obj, hole])
        obj.rotation_euler[-1] = np.pi / 2
        butil.apply_transform(obj, True)
        surface.assign_material(obj, self.surface)
        if self.has_extrude:
            tap = self.tap_factory(np.random.randint(1e7))
            min_x = np.min(read_co(tap)[:, 0])
            tap.location = (
                (-1 - self.size_extrude + self.tap_offset) * self.size - min_x,
                self.width / 2,
                self.depth,
            )
            butil.apply_transform(tap, True)
            obj = join_objects([obj, tap])
        return obj

    def extrude_back(self, obj):
        with butil.ViewportMode(obj, "EDIT"):
            bpy.ops.mesh.select_mode(type="FACE")
            bpy.ops.mesh.select_all(action="DESELECT")
            bm = bmesh.from_edit_mesh(obj.data)
            for f in bm.faces:
                f.select_set(
                    f.calc_center_median()[1] > self.size / 2 and f.normal[1] > 0.1
                )
            bm.select_flush(False)
            bmesh.update_edit_mesh(obj.data)
            bpy.ops.mesh.extrude_region_move(
                TRANSFORM_OT_translate={"value": (0, self.size_extrude * self.size, 0)}
            )

    def add_stand(self, obj):
        if self.is_stand_circular:
            stand = new_base_cylinder(vertices=16)
        else:
            stand = new_cube()
        stand.scale = self.stand_radius, self.stand_radius, self.stand_height / 2
        stand.location = self.width / 2, self.size / 2, -self.stand_height / 2
        butil.apply_transform(stand, True)
        subdivide_edge_ring(stand, np.random.randint(3, 6))
        with butil.ViewportMode(stand, "EDIT"):
            bpy.ops.mesh.select_mode(type="FACE")
            bm = bmesh.from_edit_mesh(stand.data)
            for f in bm.faces:
                f.select_set(f.normal[-1] < -0.1)
            bm.select_flush(False)
            bmesh.update_edit_mesh(stand.data)
            bpy.ops.transform.resize(
                value=(
                    self.stand_bottom / self.stand_radius,
                    self.stand_bottom / self.stand_radius,
                    1,
                )
            )
        subsurf(stand, 2, True)
        subsurf(stand, 1)
        obj = join_objects([obj, stand])
        return obj

    def finalize_assets(self, assets):
        if self.scratch:
            self.scratch.apply(assets)
        if self.edge_wear:
            self.edge_wear.apply(assets)


def _standing_sink_legacy_init(inst: Any, seed: int, coarse: bool) -> None:
    _bathroom_sink_legacy_init(inst, seed, coarse)
    inst.bathtub_type = "freestanding"
    inst.has_extrude = True
    inst.has_stand = True


class StandingSinkParameters(LegacyBridgeParameters):
    pass


class StandingSinkFactory(BathroomSinkFactory):
    parameters_model: ClassVar[type[AssetParameters]] = StandingSinkParameters

    def __init__(self, factory_seed, coarse=False):
        AssetFactory.__init__(self, factory_seed, coarse)
        self.init_legacy_parameters()

    def _sample_init_parameters(self, seed: int) -> StandingSinkParameters:
        return legacy_init_to_parameters(
            StandingSinkParameters,
            StandingSinkFactory,
            seed,
            self.coarse,
            init_fn=_standing_sink_legacy_init,
        )

    def apply_parameters(
        self, params: StandingSinkParameters, *, spawn_scope: bool = True
    ) -> None:
        apply_bridge_parameters(self, params, spawn_scope=spawn_scope)
