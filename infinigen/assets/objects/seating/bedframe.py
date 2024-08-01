# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei
import bpy
import numpy as np
from numpy.random import uniform

from infinigen.assets.material_assignments import AssetList
from infinigen.assets.objects.seating.chairs.chair import ChairFactory
from infinigen.assets.objects.seating.mattress import make_coiled
from infinigen.assets.utils.decorate import (
    read_co,
    read_normal,
    remove_faces,
    select_faces,
    subdivide_edge_ring,
    write_attribute,
    write_co,
)
from infinigen.assets.utils.object import join_objects, new_grid
from infinigen.core import surface
from infinigen.core.util import blender as butil
from infinigen.core.util.blender import deep_clone_obj
from infinigen.core.util.math import FixedSeed
from infinigen.core.util.random import log_uniform
from infinigen.core.util.random import random_general as rg


class BedFrameFactory(ChairFactory):
    scale = 1.0
    leg_decor_types = (
        "weighted_choice",
        (2, "coiled"),
        (2, "pad"),
        (1, "plain"),
        (2, "legs"),
    )
    back_types = (
        "weighted_choice",
        (3, "coiled"),
        (3, "pad"),
        (2, "whole"),
        (1, "horizontal-bar"),
        (1, "vertical-bar"),
    )

    def __init__(self, factory_seed, coarse=False):
        super().__init__(factory_seed, coarse)
        with FixedSeed(self.factory_seed):
            self.width = log_uniform(1.4, 2.4)
            self.size = uniform(2, 2.4)
            self.thickness = uniform(0.05, 0.12)
            self.has_all_legs = uniform() < 0.2
            self.leg_thickness = uniform(0.08, 0.12)
            self.leg_height = uniform(0.2, 0.6)
            self.leg_decor_type = rg(self.leg_decor_types)
            self.leg_decor_wrapped = uniform() < 0.5
            self.back_height = uniform(0.5, 1.3)
            self.seat_back = 1
            self.seat_subdivisions_x = np.random.randint(1, 4)
            self.seat_subdivisions_y = int(log_uniform(4, 10))
            self.has_arm = False
            self.leg_type = "vertical"
            self.leg_x_offset = 0
            self.leg_y_offset = 0, 0
            self.back_x_offset = 0
            self.back_y_offset = 0

            materials = AssetList["BedFrameFactory"]()
            self.surface = materials["surface"].assign_material()
            self.limb_surface = materials["limb_surface"].assign_material()

            scratch_prob, edge_wear_prob = materials["wear_tear_prob"]
            self.scratch, self.edge_wear = materials["wear_tear"]
            self.scratch = None if uniform() > scratch_prob else self.scratch
            self.edge_wear = None if uniform() > edge_wear_prob else self.edge_wear

            self.clothes_scatter = surface.NoApply
            self.dot_distance = log_uniform(0.16, 0.2)
            self.dot_size = uniform(0.005, 0.02)
            self.dot_depth = uniform(0.04, 0.08)
            self.panel_distance = uniform(0.3, 0.5)
            self.panel_margin = uniform(0.01, 0.02)
            self.post_init()

    def make_seat(self):
        obj = new_grid(
            x_subdivisions=self.seat_subdivisions_x,
            y_subdivisions=self.seat_subdivisions_y,
        )
        obj.scale = (
            (self.width - self.leg_thickness) / 2,
            (self.size - self.leg_thickness) / 2,
            1,
        )
        butil.apply_transform(obj, True)
        with butil.ViewportMode(obj, "EDIT"):
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.delete(type="ONLY_FACE")
            bpy.ops.mesh.select_mode(type="EDGE")
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.extrude_edges_move(
                TRANSFORM_OT_translate={"value": (0, 0, self.thickness)}
            )
        butil.modify_mesh(
            obj,
            "SOLIDIFY",
            thickness=self.leg_thickness - 1e-3,
            offset=0,
            solidify_mode="NON_MANIFOLD",
        )
        obj.location = 0, -self.size / 2, -self.thickness / 2
        butil.apply_transform(obj, True)
        butil.modify_mesh(obj, "BEVEL", width=self.bevel_width, segments=8)
        return obj

    def make_legs(self):
        legs = super().make_legs()
        if self.has_all_legs:
            leg_starts = np.array(
                [[-1, -0.5, 0], [0, -1, 0], [0, 0, 0], [1, -0.5, 0]]
            ) * np.array([[self.width / 2, self.size, 0]])
            leg_ends = leg_starts.copy()
            leg_ends[0, 0] -= self.leg_x_offset
            leg_ends[3, 0] += self.leg_x_offset
            leg_ends[2, 1] += self.leg_y_offset[0]
            leg_ends[1, 1] -= self.leg_y_offset[1]
            leg_ends[:, -1] = -self.leg_height
            legs += self.make_limb(leg_ends, leg_starts)
        return legs

    def make_leg_decors(self, legs):
        if self.leg_decor_type == "none":
            return super().make_leg_decors(legs)
        obj = join_objects([deep_clone_obj(_) for _ in legs])
        x, y, z = read_co(obj).T
        z = np.maximum(z, -self.leg_height * uniform(0.7, 0.9))
        write_co(obj, np.stack([x, y, z], -1))
        with butil.ViewportMode(obj, "EDIT"):
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.convex_hull()
            bpy.ops.mesh.normals_make_consistent(inside=False)
        remove_faces(obj, np.abs(read_normal(obj)[:, -1]) > 0.5)
        if self.leg_decor_wrapped:
            x, y, z = read_co(obj).T
            x[x < 0] -= self.leg_thickness / 2 + 1e-3
            x[x > 0] += self.leg_thickness / 2 + 1e-3
            y[y < -self.size / 2] -= self.leg_thickness / 2 + 1e-3
            y[y > -self.size / 2] += self.leg_thickness / 2 + 1e-3
            write_co(obj, np.stack([x, y, z], -1))
        match self.leg_decor_type:
            case "coiled":
                self.divide(obj, self.dot_distance)
                make_coiled(obj, self.dot_distance, self.dot_depth, self.dot_size)
            case "pad":
                self.divide(obj, self.panel_distance)
                with butil.ViewportMode(obj, "EDIT"):
                    bpy.ops.mesh.select_all(action="SELECT")
                    bpy.ops.mesh.inset(
                        thickness=self.panel_margin,
                        depth=self.panel_margin,
                        use_individual=True,
                    )
                butil.modify_mesh(obj, "BEVEL", segments=4)
        write_attribute(obj, 1, "panel", "FACE")
        return [obj]

    def divide(self, obj, distance):
        for i, size in enumerate(obj.dimensions):
            axis = np.zeros(3)
            axis[i] = 1
            distance = distance if i != 2 else distance * uniform(0.5, 1.0)
            subdivide_edge_ring(obj, int(np.ceil(size / distance)), axis)

    def make_back_decors(self, backs, finalize=True):
        decors = super().make_back_decors(backs)
        match self.back_type:
            case "coiled":
                obj = self.make_back(backs)
                self.divide(obj, self.dot_distance)
                make_coiled(obj, self.dot_distance, self.dot_depth, self.dot_size)
                obj.scale = (1 - 1e-3,) * 3
                write_attribute(obj, 1, "panel", "FACE")
                with butil.ViewportMode(decors[0], "EDIT"):
                    bpy.ops.mesh.select_all(action="SELECT")
                    bpy.ops.mesh.bisect(
                        plane_co=(0, 0, self.back_height),
                        plane_no=(0, 0, 1),
                        clear_inner=True,
                    )
                return [obj] + decors
            case "pad":
                obj = self.make_back(backs)
                self.divide(obj, self.panel_distance)
                with butil.ViewportMode(obj, "EDIT"):
                    select_faces(obj, np.abs(read_normal(obj)[:, 1]) > 0.5)
                    bpy.ops.mesh.inset(
                        thickness=self.panel_margin,
                        depth=self.panel_margin,
                        use_individual=True,
                    )
                butil.modify_mesh(obj, "BEVEL", segments=4)
                write_attribute(obj, 1, "panel", "FACE")
                obj.scale = (1 - 1e-3,) * 3
                with butil.ViewportMode(decors[0], "EDIT"):
                    bpy.ops.mesh.select_all(action="SELECT")
                    bpy.ops.mesh.bisect(
                        plane_co=(0, 0, self.back_height),
                        plane_no=(0, 0, 1),
                        clear_inner=True,
                    )
                return [obj] + decors
            case _:
                return decors

    def make_back(self, backs):
        obj = join_objects([deep_clone_obj(b) for b in backs])
        with butil.ViewportMode(obj, "EDIT"):
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.convex_hull()
        butil.modify_mesh(
            obj,
            "SOLIDIFY",
            thickness=np.minimum(self.thickness, self.leg_thickness),
            offset=0,
        )
        with butil.ViewportMode(obj, "EDIT"):
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.normals_make_consistent(inside=False)
        return obj
