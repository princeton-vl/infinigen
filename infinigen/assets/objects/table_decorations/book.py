# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

import math

import bmesh

# Authors: Lingjie Mei
import bpy
import numpy as np
from numpy.random import uniform

from infinigen.assets.material_assignments import AssetList
from infinigen.assets.materials import text
from infinigen.assets.utils.decorate import read_co, write_attribute, write_co
from infinigen.assets.utils.mesh import longest_ray
from infinigen.assets.utils.object import center, join_objects, new_bbox, new_cube
from infinigen.assets.utils.uv import wrap_front_back_side
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.util import blender as butil
from infinigen.core.util.math import FixedSeed
from infinigen.core.util.random import log_uniform


class BookFactory(AssetFactory):
    def __init__(self, factory_seed, coarse=False):
        super(BookFactory, self).__init__(factory_seed, coarse)
        self.rel_scale = log_uniform(1, 1.5)
        self.skewness = log_uniform(1.3, 1.8)
        self.unit = 0.0127
        self.is_paperback = uniform() < 0.5
        self.margin = uniform(0.005, 0.01)
        self.offset = 0 if uniform() < 0.5 else log_uniform(0.002, 0.008)
        self.thickness = uniform(0.002, 0.003)

        materials = AssetList["BookFactory"]()
        self.surface = materials["surface"].assign_material()
        self.cover_surface = materials["cover_surface"].assign_material()
        if self.cover_surface == text.Text:
            self.cover_surface = self.cover_surface(self.factory_seed)

        scratch_prob, edge_wear_prob = materials["wear_tear_prob"]
        self.scratch, self.edge_wear = materials["wear_tear"]
        self.scratch = None if uniform() > scratch_prob else self.scratch
        self.edge_wear = None if uniform() > edge_wear_prob else self.edge_wear

        self.texture_shared = uniform() < 0.2

    def create_asset(self, **params) -> bpy.types.Object:
        width = int(log_uniform(0.08, 0.15) * self.rel_scale / self.unit) * self.unit
        height = int(width * self.skewness / self.unit) * self.unit
        depth = uniform(0.01, 0.02) * self.rel_scale
        fn = self.make_paperback if self.is_paperback else self.make_hardcover
        # noinspection PyArgumentList
        obj = fn(width, height, depth)

        return obj

    def finalize_assets(self, assets):
        if self.scratch:
            self.scratch.apply(assets)
        if self.edge_wear:
            self.edge_wear.apply(assets)

    def make_paperback(self, width, height, depth):
        paper = self.make_paper(depth, height, width)
        obj = new_cube()
        obj.location = width / 2, height / 2, depth / 2
        obj.scale = width / 2, height / 2, depth / 2
        butil.apply_transform(obj, True)

        with butil.ViewportMode(obj, "EDIT"):
            bm = bmesh.from_edit_mesh(obj.data)
            geom = []
            for e in bm.edges:
                u, v = e.verts
                if u.co[0] > 0 and v.co[0] > 0 and u.co[-1] != v.co[-1]:
                    geom.append(e)
            bmesh.ops.delete(bm, geom=geom, context="EDGES")

        self.make_cover(obj)
        write_attribute(obj, 1, "cover", "FACE")
        obj = join_objects([paper, obj])
        return obj

    def make_paper(self, depth, height, width):
        paper = new_cube()
        paper.location = width / 2, height / 2, depth / 2
        paper.scale = width / 2 - 1e-4, height / 2, depth / 2 - 1e-4
        butil.apply_transform(paper, True)
        self.surface.apply(paper)
        return paper

    def make_hardcover(self, width, height, depth):
        paper = self.make_paper(depth, height, width)
        obj = new_cube()
        count = 8
        butil.modify_mesh(
            obj,
            "ARRAY",
            count=count,
            relative_offset_displace=(0, 0, 1),
            use_merge_vertices=True,
        )
        obj.location = 1, 1, 1
        butil.apply_transform(obj, loc=True)
        with butil.ViewportMode(obj, "EDIT"):
            bm = bmesh.from_edit_mesh(obj.data)
            geom = []
            for v in bm.verts:
                if v.co[0] > 0 and 0 < v.co[-1] < count * 2:
                    geom.append(v)
            bmesh.ops.delete(bm, geom=geom, context="VERTS")
        obj.location = 0, -self.margin, 0
        obj.scale = (
            (width + self.margin) / 2,
            height / 2 + self.margin,
            depth / 2 / count,
        )
        butil.apply_transform(obj, True)
        x, y, z = read_co(obj).T
        ratio = np.minimum(z / depth, 1 - z / depth)
        x -= 4 * ratio * (1 - ratio) * self.offset
        write_co(obj, np.stack([x, y, z]).T)
        self.make_cover(obj)
        butil.modify_mesh(obj, "SOLIDIFY", thickness=self.thickness)
        write_attribute(obj, 1, "cover", "FACE")
        obj = join_objects([paper, obj])
        return obj

    def make_cover(self, obj):
        obj.rotation_euler[0] = np.pi / 2
        butil.apply_transform(obj)
        wrap_front_back_side(obj, self.cover_surface, self.texture_shared)
        obj.rotation_euler[0] = -np.pi / 2
        butil.apply_transform(obj)


class BookColumnFactory(AssetFactory):
    def __init__(self, factory_seed, coarse=False):
        super(BookColumnFactory, self).__init__(factory_seed, coarse)
        with FixedSeed(self.factory_seed):
            self.base_factories = [
                BookFactory(np.random.randint(1e5))
                for _ in range(np.random.randint(1, 4))
            ]
            self.n_books = np.random.randint(10, 20)
            self.max_angle = uniform(0, np.pi / 9) if uniform() < 0.7 else 0
            self.max_rel_scale = max(f.rel_scale for f in self.base_factories)
            self.max_skewness = max(f.skewness for f in self.base_factories)

    def create_placeholder(self, **kwargs) -> bpy.types.Object:
        height = 0.15 * self.max_rel_scale * self.max_skewness
        return new_bbox(
            0,
            (0.02 + np.sin(self.max_angle) * height)
            * self.n_books
            * self.max_rel_scale,
            -0.15 * self.max_rel_scale,
            0,
            0,
            height,
        )

    def create_asset(self, **params) -> bpy.types.Object:
        books = []
        for i in range(self.n_books):
            factory = np.random.choice(self.base_factories)
            obj = factory.create_asset(i=i)
            x, y, z = read_co(obj).T
            obj.location = [-np.max(x), -np.min(y), -np.min(z)]
            butil.apply_transform(obj, True)
            if uniform() < 0.5:
                obj.rotation_euler = (
                    np.pi / 2 - uniform(0, self.max_angle),
                    0,
                    np.pi / 2,
                )
            else:
                obj.location[-1] = -np.max(z)
                butil.apply_transform(obj, True)
                obj.rotation_euler = (
                    np.pi / 2 + uniform(0, self.max_angle),
                    0,
                    np.pi / 2,
                )
            butil.apply_transform(obj)
            if i > 0:
                obj.location[0] = 10
                butil.apply_transform(obj, True)
                dist = longest_ray(books[-1], obj, (-1, 0, 0))
                dist_ = longest_ray(obj, books[-1], (1, 0, 0))
                offset = np.minimum(np.min(dist), np.min(dist_))
                obj.location[0] = -offset
                butil.apply_transform(obj, True)
            books.append(obj)
        obj = join_objects(books)
        obj.location[0] = -np.min(read_co(obj)[:, 0])
        butil.apply_transform(obj, True)
        return obj


def rotate(theta, x, y):
    return x * math.cos(theta) - y * math.sin(theta), x * math.sin(
        theta
    ) + y * math.cos(theta)


class BookStackFactory(AssetFactory):
    def __init__(self, factory_seed, coarse=False):
        super(BookStackFactory, self).__init__(factory_seed, coarse)
        with FixedSeed(self.factory_seed):
            self.base_factories = [
                BookFactory(np.random.randint(1e5))
                for _ in range(np.random.randint(1, 4))
            ]
            self.n_books = int(log_uniform(5, 15))
            self.max_angle = uniform(np.pi / 9, np.pi / 6) if uniform() < 0.7 else 0
            self.max_rel_scale = max(f.rel_scale for f in self.base_factories)
            self.max_skewness = max(f.skewness for f in self.base_factories)

    def create_placeholder(self, **kwargs) -> bpy.types.Object:
        x_lo = -0.15 * self.max_rel_scale / 2
        x_hi = 0.15 * self.max_rel_scale / 2
        y_lo = -0.15 * self.max_rel_scale / 2 * self.max_skewness
        y_hi = 0.15 * self.max_rel_scale / 2 * self.max_skewness

        theta = self.max_angle
        x_1, y_1 = rotate(theta, x_lo, y_lo)
        x_2, y_2 = rotate(theta, x_lo, y_hi)
        x_3, y_3 = rotate(theta, x_hi, y_lo)
        x_4, y_4 = rotate(theta, x_hi, y_hi)

        return new_bbox(
            min(min([x_1, x_2, x_3, x_4]), x_lo),
            max(max([x_1, x_2, x_3, x_4]), x_hi),
            min(min([y_1, y_2, y_3, y_4]), y_lo),
            max(max([y_1, y_2, y_3, y_4]), y_hi),
            0,
            self.n_books * 0.02 * self.max_rel_scale * 0.8,
        )

    def create_asset(self, **params) -> bpy.types.Object:
        books = []
        offset = 0
        for i in range(self.n_books):
            factory = np.random.choice(self.base_factories)
            obj = factory.create_asset(i=i)
            c = center(obj)[:-1]
            obj.location = -c[0], -c[1], offset - np.min(read_co(obj)[:, -1])
            obj.rotation_euler[-1] = uniform(-self.max_angle, self.max_angle)
            butil.apply_transform(obj, True)
            offset = np.max(read_co(obj)[:, -1])
            books.append(obj)
        return join_objects(books)
