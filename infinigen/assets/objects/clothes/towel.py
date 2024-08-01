# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

import bmesh

# Authors: Lingjie Mei
import bpy
import numpy as np
from numpy.random import uniform
from scipy.optimize import fsolve

from infinigen.assets.material_assignments import AssetList
from infinigen.assets.objects.elements.rug import ArtRug
from infinigen.assets.utils.decorate import (
    geo_extension,
    mirror,
    read_co,
    read_edge_direction,
    subdivide_edge_ring,
    subsurf,
    write_co,
)
from infinigen.assets.utils.object import center, new_plane
from infinigen.assets.utils.uv import wrap_sides
from infinigen.core import surface
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.util import blender as butil
from infinigen.core.util.random import log_uniform


class TowelFactory(AssetFactory):
    def __init__(self, factory_seed, coarse=False):
        super(TowelFactory, self).__init__(factory_seed, coarse)
        self.width = log_uniform(0.3, 0.6)
        self.length = self.width * log_uniform(1, 1.5)
        self.thickness = log_uniform(0.003, 0.01)
        prob = np.array([2, 1])
        self.fold_type = np.random.choice(["fold", "roll"], p=prob / prob.sum())
        self.folds = np.random.randint(2, 4)
        self.extra_thickness = self.thickness * uniform(0.2, 0.3)
        self.fold_count = 15
        self.roll_count = 256
        self.roll_total = self.compute_roll_total()
        materials = AssetList["TowelFactory"]()
        self.surface = materials["surface"].assign_material()
        if self.surface == ArtRug:
            self.surface = self.surface(self.factory_seed)

    def fold(self, obj):
        x, y, z = read_co(obj).T
        if np.max(x) - np.min(x) > np.max(y) - np.min(y):
            obj.rotation_euler[-1] = np.pi * (uniform() < 0.5)
        else:
            obj.rotation_euler[-1] = np.pi * (uniform() < 0.5) + np.pi / 2
        butil.apply_transform(obj, True)
        obj.location = *(-center(obj))[:-1], 0
        obj.location[0] += uniform(-self.thickness, self.thickness)
        butil.apply_transform(obj, True)
        n = len(obj.data.vertices)
        with butil.ViewportMode(obj, "EDIT"):
            bm = bmesh.from_edit_mesh(obj.data)
            bm.edges.ensure_lookup_table()
            selected = np.abs(read_edge_direction(obj)[:, 0]) > 1 - 1e-3
            edges = [bm.edges[i] for i in np.nonzero(selected)[0]]
            bmesh.ops.subdivide_edgering(
                bm, edges=edges, cuts=self.fold_count, smooth=2
            )
            bmesh.update_edit_mesh(obj.data)
        co = read_co(obj)
        order = np.where(
            co[n :: self.fold_count, 0] < co[n + 1 :: self.fold_count, 0], 1, -1
        )
        x_ = np.linspace(
            -self.thickness * order, self.thickness * order, self.fold_count
        ).T.ravel()
        co[n:, 0] = x_
        x, y, z = co.T
        max_z = np.max(z) + self.extra_thickness
        theta = x / self.thickness * np.pi / 2
        x__ = np.where(
            x < -self.thickness,
            x,
            np.where(
                x > self.thickness, -x, -self.thickness + (max_z - z) * np.cos(theta)
            ),
        )
        z_ = np.where(
            x < -self.thickness,
            z,
            np.where(
                x > self.thickness, max_z * 2 - z, max_z + (max_z - z) * np.sin(theta)
            ),
        )
        write_co(obj, np.stack([x__, y, z_], -1))
        if uniform() < 0.5:
            mirror(obj)
        return obj

    def compute_roll_total(self):
        c = self.length / (self.thickness + self.extra_thickness) * (4 * np.pi)

        def f(t):
            return t * np.sqrt(1 + t * t) + np.log(t + np.sqrt(1 + t * t)) - c

        return fsolve(f, np.zeros(1))[0]

    def pre_roll(self, obj):
        subdivide_edge_ring(obj, self.roll_count, (1, 0, 0))
        x, y, z = read_co(obj).T
        i = np.round((x / self.length + 0.5) * self.roll_count).astype(int)
        t = np.linspace(0, self.roll_total, self.roll_count + 1)[i]
        length = (
            (t * np.sqrt(1 + t * t) + np.log(t + np.sqrt(1 + t * t)))
            * (self.thickness + self.extra_thickness)
            / (4 * np.pi)
        )
        write_co(obj, np.stack([length, y, z], -1))
        return i

    def roll(self, obj, i):
        t = np.linspace(0, self.roll_total, self.roll_count + 1)[np.concatenate([i, i])]
        x, y, z = read_co(obj).T
        r = (self.thickness + self.extra_thickness) / (2 * np.pi) * t + np.where(
            z > self.thickness / 2, -self.thickness / 2, self.thickness / 2
        )
        write_co(obj, np.stack([r * np.cos(t), y, r * np.sin((t))], -1))

    def create_asset(self, **params) -> bpy.types.Object:
        obj = new_plane()
        if self.fold_type == "roll":
            obj.scale = self.length / 2, self.width / 2, 1
        else:
            obj.scale = self.width / 2, self.length / 2, 1
        butil.apply_transform(obj, True)
        i = None
        if self.fold_type == "roll":
            i = self.pre_roll(obj)
        wrap_sides(obj, self.surface, "z", "x", "y", strength=uniform(0.2, 0.4))
        butil.modify_mesh(obj, "SOLIDIFY", thickness=self.thickness, offset=1)
        if self.fold_type == "roll":
            self.roll(obj, i)
            subdivide_edge_ring(obj, 16, (0, 1, 0))
        else:
            for _ in range(self.folds):
                self.fold(obj)
            subdivide_edge_ring(obj, 16, (1, 0, 0))
            subdivide_edge_ring(obj, 16, (0, 1, 0))
        butil.modify_mesh(
            obj, "BEVEL", width=self.thickness * uniform(0.4, 0.8), segments=2
        )
        surface.add_geomod(
            obj, geo_extension, apply=True, input_args=[uniform(0.05, 0.1)]
        )
        subsurf(obj, 1)
        return obj
