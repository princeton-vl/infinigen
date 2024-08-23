# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei
import bmesh
import bpy
import numpy as np
from numpy.random import uniform

from infinigen.assets.materials import marble_regular, marble_voronoi
from infinigen.assets.utils.decorate import (
    read_co,
    read_edge_center,
    read_selected,
    select_edges,
    subdivide_edge_ring,
    subsurf,
    write_co,
)
from infinigen.assets.utils.object import (
    join_objects,
    new_base_circle,
    new_cylinder,
)
from infinigen.core.constraints.constraint_language.constants import RoomConstants
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.util import blender as butil
from infinigen.core.util.blender import deep_clone_obj
from infinigen.core.util.math import FixedSeed
from infinigen.core.util.random import log_uniform


class PillarFactory(AssetFactory):
    def __init__(self, factory_seed, coarse=False, constants=None):
        super().__init__(factory_seed, coarse)
        with FixedSeed(factory_seed):
            if constants is None:
                constants = RoomConstants()
            self.height = constants.wall_height - constants.wall_thickness
            self.n = np.random.randint(5, 10)
            self.radius = uniform(0.08, 0.12)
            self.outer_radius = self.radius * uniform(1.3, 1.5)
            self.lower_offset = uniform(0.05, 0.15)
            self.upper_offset = uniform(0.05, 0.15)
            self.detail_type = np.random.choice(["fluting", "reeding"])
            width = np.pi / 2 / self.n
            self.inset_width = width * log_uniform(0.1, 0.2)
            self.inset_width_ = (width - self.inset_width * 2) * uniform(-0.1, 0.3)
            self.inset_depth = uniform(0.1, 0.15)
            self.inset_scale = uniform(0.05, 0.1)
            self.outer_n = np.random.choice([1, 2, self.n])
            self.m = np.random.randint(12, 20)
            z_profile = uniform(1, 3, self.m)
            self.z_profile = np.array(
                [0, *(np.cumsum(z_profile) / np.sum(z_profile))[:-1]]
            )
            alpha = uniform(0.7, 0.85)
            r_profile = uniform(0, 1, self.m + 3)
            r_profile[[0, 1]] = 1
            r_profile[[-2, -1]] = 0
            r_profile = np.convolve(
                r_profile, np.array([(1 - alpha) / 2, alpha, (1 - alpha) / 2])
            )
            self.r_profile = (
                np.array([1, *r_profile[2:-2]]) * (self.outer_radius - self.radius)
                + self.radius
            )
            self.n_profile = np.where(
                np.arange(self.m) < np.random.randint(2, self.m - 1),
                self.outer_n,
                self.n,
            )
            self.inset_profile = uniform(0, 1, self.m) < 0.3
            self.surface = np.random.choice([marble_regular, marble_voronoi])

    def create_asset(self, **params) -> bpy.types.Object:
        obj = new_cylinder(vertices=4 * self.n)
        with butil.ViewportMode(obj, "EDIT"):
            bm = bmesh.from_edit_mesh(obj.data)
            geom = [f for f in bm.faces if len(f.verts) > 4]
            bmesh.ops.delete(bm, geom=geom, context="FACES_ONLY")
            bmesh.update_edit_mesh(obj.data)

        obj.scale = (
            self.radius,
            self.radius,
            (1 - self.lower_offset - self.upper_offset) * self.height,
        )
        obj.location[-1] = self.lower_offset * self.height
        butil.apply_transform(obj, True)
        inset_scale = 1 + self.inset_scale * (
            1 if self.detail_type == "reeding" else -1
        )
        if self.detail_type in ["fluting", "reeding"]:
            with butil.ViewportMode(obj, "EDIT"):
                bpy.ops.mesh.select_mode(type="FACE")
                bpy.ops.mesh.select_all(action="SELECT")
                bpy.ops.mesh.inset(
                    thickness=self.inset_width * self.radius, use_individual=True
                )
                bpy.ops.mesh.inset(
                    thickness=self.inset_width_ * self.radius, use_individual=True
                )
                bpy.ops.transform.resize(value=(inset_scale, inset_scale, 1))
        subdivide_edge_ring(obj, 16)
        parts = [obj]
        with butil.ViewportMode(obj, "EDIT"):
            bpy.ops.mesh.select_mode(type="EDGE")
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.region_to_loop()
        z_rot = np.pi / 2 * np.random.randint(2)
        for z, r, n, i in zip(
            self.z_profile, self.r_profile, self.n_profile, self.inset_profile
        ):
            o = new_base_circle(vertices=4 * n)
            if i:
                co = read_co(o)
                stride = np.random.choice([2, 4, 8])
                co *= np.where(np.arange(len(co)) % stride == 0, 1, inset_scale)[
                    :, np.newaxis
                ]
                write_co(o, co)
            with butil.ViewportMode(o, "EDIT"):
                bpy.ops.mesh.select_mode(type="EDGE")
                bpy.ops.mesh.select_all(action="SELECT")
                bpy.ops.mesh.subdivide(number_cuts=self.n // n - 1)
            o.location[-1] = z * self.lower_offset * self.height
            r_ = r / np.cos(np.pi / 4 / n)
            o.scale = r_, r_, 1
            o.rotation_euler[-1] = z_rot
            o_ = deep_clone_obj(o)
            o_.location[-1] = (1 - z * self.upper_offset) * self.height
            butil.apply_transform(o, True)
            butil.apply_transform(o_, True)
            parts.extend([o, o_])
        obj = join_objects(parts)
        selection = read_selected(obj, "EDGE")
        z = read_edge_center(obj)[:, -1]
        number_cuts = 0
        smoothness = uniform(1, 1.4)
        select_edges(obj, selection & (z < 0.5))
        with butil.ViewportMode(obj, "EDIT"):
            bpy.ops.mesh.bridge_edge_loops(
                number_cuts=number_cuts, smoothness=smoothness
            )
        select_edges(obj, selection & (z > 0.5))
        with butil.ViewportMode(obj, "EDIT"):
            bpy.ops.mesh.bridge_edge_loops(
                number_cuts=number_cuts, smoothness=smoothness
            )
        subsurf(obj, 1, True)
        subsurf(obj, 1)
        return obj

    def finalize_assets(self, assets):
        self.surface.apply(assets)
