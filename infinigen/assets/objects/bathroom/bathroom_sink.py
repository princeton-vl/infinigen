# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

import bmesh

# Authors: Lingjie Mei
import bpy
import numpy as np
from numpy.random import uniform

from infinigen.assets.material_assignments import AssetList
from infinigen.assets.objects.bathroom.bathtub import BathtubFactory
from infinigen.assets.objects.table_decorations import TapFactory
from infinigen.assets.utils.decorate import read_co, subdivide_edge_ring, subsurf
from infinigen.assets.utils.object import (
    join_objects,
    new_base_cylinder,
    new_bbox,
    new_cube,
)
from infinigen.core.util import blender as butil
from infinigen.core.util.math import FixedSeed
from infinigen.core.util.random import log_uniform


class BathroomSinkFactory(BathtubFactory):
    def __init__(self, factory_seed, coarse=False):
        super(BathroomSinkFactory, self).__init__(factory_seed, coarse)
        with FixedSeed(factory_seed):
            self.width = uniform(0.6, 0.9)
            self.size = self.width * log_uniform(0.55, 0.8)
            self.depth = self.width * log_uniform(0.2, 0.4)
            self.contour_fn = self.make_box_contour
            self.sink_types = np.random.choice(["undermount", "drop-in", "vessel"])
            self.has_stand = False
            match self.sink_types:
                case "undermount":
                    self.bathtub_type = "freestanding"
                    self.has_extrude = uniform() < 0.7
                case "drop-in":
                    self.bathtub_type = "alcove"
                    self.has_extrude = True
                case _:
                    self.bathtub_type = np.random.choice(["alcove", "freestanding"])
                    self.has_extrude = uniform() < 0.7
                    self.has_stand = True
            self.tap_factory = TapFactory(self.factory_seed)
            self.disp_x = [self.disp_x[0], self.disp_x[0]]
            self.alcove_levels = 0 if uniform() < 0.5 else np.random.randint(2, 4)
            self.thickness = 0.01 if self.has_base else uniform(0.01, 0.03)
            self.size_extrude = uniform(0.2, 0.35)
            self.tap_offset = uniform(0.0, 0.05)
            self.stand_radius = self.width / 2 * log_uniform(0.15, 0.2)
            self.stand_bottom = (
                self.width * log_uniform(0.2, 0.3)
                if uniform() < 0.6
                else self.stand_radius
            )
            self.stand_height = uniform(0.7, 0.9) - self.depth
            self.is_stand_circular = uniform() < 0.5
            self.is_hole_centered = True
            material_assignments = AssetList["BathroomSinkFactory"]()
            self.surface = material_assignments["surface"].assign_material()

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
        self.surface.apply(obj, clear=True, metal_color="plain")
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


class StandingSinkFactory(BathroomSinkFactory):
    def __init__(self, factory_seed, coarse=False):
        super(StandingSinkFactory, self).__init__(factory_seed, coarse)
        self.bathtub_type = "freestanding"
        self.has_extrude = True
        self.has_stand = True
