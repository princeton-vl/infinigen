# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math

import bmesh

# Authors: Lingjie Mei
from typing import Annotated, Any, ClassVar

import bpy
import numpy as np
from numpy.random import uniform
from pydantic import Field

from infinigen.assets.composition import material_assignments
from infinigen.assets.materials import text
from infinigen.assets.materials.ceramic import plaster
from infinigen.assets.utils.decorate import read_co, write_attribute, write_co
from infinigen.assets.utils.mesh import longest_ray
from infinigen.assets.utils.object import center, join_objects, new_bbox, new_cube
from infinigen.assets.utils.uv import wrap_front_back_side
from infinigen.core import surface
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.placement.parameters import AssetParameters, ParameterizedAssetFactory
from infinigen.core.util import blender as butil
from infinigen.core.util.math import FixedSeed
from infinigen.core.util.random import log_uniform


class BookParameters(AssetParameters):
    rel_scale: Annotated[float, Field(ge=1.0, le=1.5, json_schema_extra={"editable": True})]
    skewness: Annotated[float, Field(ge=1.3, le=1.8, json_schema_extra={"editable": True})]
    is_paperback_draw: Annotated[
        float, Field(ge=0.0, le=1.0, json_schema_extra={"editable": True})
    ]
    margin: Annotated[float, Field(ge=0.005, le=0.01, json_schema_extra={"editable": True})]
    thickness: Annotated[
        float, Field(ge=0.002, le=0.003, json_schema_extra={"editable": True})
    ]
    scratch_draw: Annotated[
        float, Field(ge=0.0, le=1.0, json_schema_extra={"editable": True})
    ]
    edge_wear_draw: Annotated[
        float, Field(ge=0.0, le=1.0, json_schema_extra={"editable": True})
    ]
    texture_shared_draw: Annotated[
        float, Field(ge=0.0, le=1.0, json_schema_extra={"editable": True})
    ]
    surface_material_gen: Any = Field(json_schema_extra={"editable": False})
    cover_surface: Any = Field(json_schema_extra={"editable": False})
    scratch: Any | None = Field(default=None, json_schema_extra={"editable": False})
    edge_wear: Any | None = Field(default=None, json_schema_extra={"editable": False})
    width: Annotated[float, Field(ge=0.08, le=0.15, json_schema_extra={"editable": True})] = (
        0.1
    )
    depth: Annotated[float, Field(ge=0.01, le=0.02, json_schema_extra={"editable": True})] = (
        0.015
    )
    offset: float = Field(default=0.0, json_schema_extra={"editable": False})


class BookFactory(ParameterizedAssetFactory, AssetFactory):
    parameters_model: ClassVar[type[AssetParameters]] = BookParameters

    def __init__(self, factory_seed, coarse=False):
        super(BookFactory, self).__init__(factory_seed, coarse)
        self.unit = 0.0127
        self.init_legacy_parameters()

    def _sample_init_parameters(self, seed: int) -> BookParameters:
        scratch_prob, edge_wear_prob = material_assignments.wear_tear_prob
        scratch_fn, edge_wear_fn = material_assignments.wear_tear
        scratch_draw = uniform()
        edge_wear_draw = uniform()
        surface_material_gen = plaster.Plaster()
        cover_surface_material_gen = text.Text()
        cover_surface = cover_surface_material_gen()
        if cover_surface == text.Text:
            cover_surface = cover_surface(seed)
        return BookParameters(
            seed=seed,
            rel_scale=log_uniform(1, 1.5),
            skewness=log_uniform(1.3, 1.8),
            is_paperback_draw=uniform(),
            margin=uniform(0.005, 0.01),
            thickness=uniform(0.002, 0.003),
            scratch_draw=scratch_draw,
            edge_wear_draw=edge_wear_draw,
            texture_shared_draw=uniform(),
            surface_material_gen=surface_material_gen,
            cover_surface=cover_surface,
            scratch=None if scratch_draw > scratch_prob else scratch_fn(),
            edge_wear=None if edge_wear_draw > edge_wear_prob else edge_wear_fn(),
        )

    def _sample_spawn_parameters(
        self, params: BookParameters, seed: int, i: int
    ) -> BookParameters:
        return params.model_copy(
            update={
                "width": log_uniform(0.08, 0.15),
                "depth": uniform(0.01, 0.02),
                "offset": 0 if uniform() < 0.5 else log_uniform(0.002, 0.008),
            }
        )

    def apply_parameters(
        self, params: BookParameters, *, spawn_scope: bool = True
    ) -> None:
        self.rel_scale = params.rel_scale
        self.skewness = params.skewness
        self.is_paperback = params.is_paperback_draw < 0.5
        self.margin = params.margin
        self.thickness = params.thickness
        self.scratch = params.scratch
        self.edge_wear = params.edge_wear
        self.texture_shared = params.texture_shared_draw < 0.2
        self.surface_material_gen = params.surface_material_gen
        self.cover_surface = params.cover_surface
        self._use_fixed_spawn_draws = spawn_scope
        if spawn_scope:
            self._width = params.width
            self._depth = params.depth
            self.offset = params.offset

    def create_asset(self, **params) -> bpy.types.Object:
        self.surface = self.surface_material_gen()
        width_raw = self._width if self._use_fixed_spawn_draws else log_uniform(0.08, 0.15)
        depth_raw = self._depth if self._use_fixed_spawn_draws else uniform(0.01, 0.02)
        width = int(width_raw * self.rel_scale / self.unit) * self.unit
        height = int(width * self.skewness / self.unit) * self.unit
        depth = depth_raw * self.rel_scale
        fn = self.make_paperback if self.is_paperback else self.make_hardcover
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

        surface.assign_material(paper, self.surface)
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
        offset = (
            self.offset
            if self._use_fixed_spawn_draws
            else (0 if uniform() < 0.5 else log_uniform(0.002, 0.008))
        )
        x -= 4 * ratio * (1 - ratio) * offset
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


class BookColumnParameters(AssetParameters):
    n_base_factories: Annotated[
        int, Field(ge=1, le=3, json_schema_extra={"editable": True})
    ]
    n_books: Annotated[int, Field(ge=10, le=19, json_schema_extra={"editable": True})]
    has_tilt_draw: Annotated[
        float, Field(ge=0.0, le=1.0, json_schema_extra={"editable": True})
    ]
    max_angle: Annotated[
        float, Field(ge=0.0, le=np.pi / 9, json_schema_extra={"editable": True})
    ] = 0.0
    base_factory_seeds: tuple[int, ...] = Field(
        default=(), json_schema_extra={"editable": False}
    )
    book_factory_indices: tuple[int, ...] = Field(
        default=(), json_schema_extra={"editable": False}
    )
    book_rotation_flip: tuple[bool, ...] = Field(
        default=(), json_schema_extra={"editable": False}
    )
    book_rotation_angles: tuple[float, ...] = Field(
        default=(), json_schema_extra={"editable": False}
    )
    base_factories: Any = Field(default=None, json_schema_extra={"editable": False})
    max_rel_scale: float = Field(default=1.0, json_schema_extra={"editable": False})
    max_skewness: float = Field(default=1.0, json_schema_extra={"editable": False})


class BookColumnFactory(ParameterizedAssetFactory, AssetFactory):
    parameters_model: ClassVar[type[AssetParameters]] = BookColumnParameters

    def __init__(self, factory_seed, coarse=False):
        super(BookColumnFactory, self).__init__(factory_seed, coarse)
        self.init_legacy_parameters()

    def _sample_init_parameters(self, seed: int) -> BookColumnParameters:
        n_base = int(np.random.randint(1, 4))
        has_tilt_draw = uniform()
        max_angle = uniform(0, np.pi / 9) if has_tilt_draw < 0.7 else 0.0
        base_factory_seeds = tuple(int(np.random.randint(1e5)) for _ in range(n_base))
        base_factories = [BookFactory(s) for s in base_factory_seeds]
        return BookColumnParameters(
            seed=seed,
            n_base_factories=n_base,
            n_books=int(np.random.randint(10, 20)),
            has_tilt_draw=has_tilt_draw,
            max_angle=max_angle,
            base_factory_seeds=base_factory_seeds,
            base_factories=base_factories,
            max_rel_scale=max(f.rel_scale for f in base_factories),
            max_skewness=max(f.skewness for f in base_factories),
        )

    def _sample_spawn_parameters(
        self, params: BookColumnParameters, seed: int, i: int
    ) -> BookColumnParameters:
        n = params.n_books
        n_base = params.n_base_factories
        return params.model_copy(
            update={
                "book_factory_indices": tuple(
                    int(np.random.randint(0, n_base)) for _ in range(n)
                ),
                "book_rotation_flip": tuple(uniform() < 0.5 for _ in range(n)),
                "book_rotation_angles": tuple(uniform(0, params.max_angle) for _ in range(n)),
            }
        )

    def apply_parameters(
        self, params: BookColumnParameters, *, spawn_scope: bool = True
    ) -> None:
        self.n_books = params.n_books
        self.max_angle = params.max_angle
        self.base_factories = params.base_factories
        self.max_rel_scale = params.max_rel_scale
        self.max_skewness = params.max_skewness
        self._use_fixed_spawn_draws = spawn_scope
        if spawn_scope:
            self.book_factory_indices = params.book_factory_indices
            self.book_rotation_flip = params.book_rotation_flip
            self.book_rotation_angles = params.book_rotation_angles

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
            if self._use_fixed_spawn_draws:
                factory = self.base_factories[self.book_factory_indices[i]]
                flip = self.book_rotation_flip[i]
                angle = self.book_rotation_angles[i]
            else:
                factory = np.random.choice(self.base_factories)
                flip = uniform() < 0.5
                angle = uniform(0, self.max_angle)
            obj = factory.create_asset(i=i)
            x, y, z = read_co(obj).T
            obj.location = [-np.max(x), -np.min(y), -np.min(z)]
            butil.apply_transform(obj, True)
            if flip:
                obj.rotation_euler = (np.pi / 2 - angle, 0, np.pi / 2)
            else:
                obj.location[-1] = -np.max(z)
                butil.apply_transform(obj, True)
                obj.rotation_euler = (np.pi / 2 + angle, 0, np.pi / 2)
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
