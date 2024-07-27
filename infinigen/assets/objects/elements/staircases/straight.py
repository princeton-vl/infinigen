# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors:
# - Lingjie Mei
# - Karhan Kayan: fix constants

import bmesh
import bpy
import numpy as np
import shapely
from numpy.random import uniform
from shapely import LineString, Polygon

from infinigen.assets.materials import fabrics, glass, metal, plaster, wood
from infinigen.assets.materials.stone_and_concrete import concrete
from infinigen.assets.utils.decorate import (
    mirror,
    read_co,
    remove_faces,
    remove_vertices,
    subsurf,
    write_attribute,
    write_co,
)
from infinigen.assets.utils.mesh import canonicalize_ls, convert2ls
from infinigen.assets.utils.nodegroup import geo_radius
from infinigen.assets.utils.object import (
    data2mesh,
    join_objects,
    mesh2obj,
    new_circle,
    new_cube,
    new_line,
    separate_loose,
)
from infinigen.assets.utils.shapes import cut_polygon_by_line
from infinigen.core import surface
from infinigen.core import tags as t
from infinigen.core.constraints.constraint_language.constants import RoomConstants
from infinigen.core.nodes import Nodes, NodeWrangler
from infinigen.core.placement.detail import sharp_remesh_with_attrs
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.surface import read_attr_data, write_attr_data
from infinigen.core.tagging import PREFIX
from infinigen.core.util import blender as butil
from infinigen.core.util.math import FixedSeed, normalize
from infinigen.core.util.random import log_uniform
from infinigen.core.util.random import random_general as rg


class StraightStaircaseFactory(AssetFactory):
    support_types = (
        "weighted_choice",
        (2, "single-rail"),
        (2, "double-rail"),
        (3, "side"),
        (3, "solid"),
        (3, "hole"),
    )
    handrail_types = (
        "weighted_choice",
        (2, "glass"),
        (2, "horizontal-post"),
        (2, "vertical-post"),
    )

    def __init__(self, factory_seed, coarse=False, constants=None):
        super(StraightStaircaseFactory, self).__init__(factory_seed, coarse)
        with FixedSeed(self.factory_seed):
            if constants is None:
                constants = RoomConstants()
            self.constants = constants
            self.support_type = rg(self.support_types)
            self.n, self.step_height, self.step_width, self.step_length = 0, 0, 0, 0
            self.build_size_config()

            self.has_step = self.support_type in ["solid", "hole"]
            self.hole_size = log_uniform(0.6, 1.0)
            probs = np.array([3, 2, 2, 2])
            self.step_surface = np.random.choice(
                [wood, plaster, concrete, fabrics.fabric_random], p=probs / probs.sum()
            )

            self.has_rail = self.support_type in ["single-rail", "double-rail"]
            self.rail_offset = self.step_width * uniform(0.15, 0.3)
            self.is_rail_circular = uniform() < 0.5
            self.rail_width = log_uniform(0.08, 0.2)
            self.rail_height = log_uniform(0.08, 0.12)
            probs = np.array([3, 2, 2, 1])
            self.rail_surface = np.random.choice(
                [metal, plaster, concrete, fabrics.fabric_random], p=probs / probs.sum()
            )

            self.has_tread = not self.has_step or uniform() < 0.75
            self.tread_height = (
                uniform(0.01, 0.02) if self.has_step else uniform(0.06, 0.08)
            )
            self.tread_length = self.step_length + uniform(0.01, 0.02)
            self.tread_width = (
                self.step_width + uniform(0.01, 0.02)
                if uniform() < 0.8
                else self.step_width
            )
            probs = np.array([3, 3, 1])
            self.tread_surface = np.random.choice(
                [wood, metal, glass], p=probs / probs.sum()
            )

            self.has_sides = self.support_type in ["side", "solid", "hole"]
            self.side_type = np.random.choice(["zig-zag", "straight"])
            self.side_height = self.step_height * log_uniform(0.2, 0.8)
            self.side_thickness = uniform(0.03, 0.08)
            probs = np.array([3, 3, 1, 2])
            self.side_surface = np.random.choice(
                [wood, metal, plaster, fabrics.fabric_random], p=probs / probs.sum()
            )

            self.has_column = self.support_type == "chord"

            self.handrail_type = rg(self.handrail_types)
            self.is_handrail_circular = uniform() < 0.7
            self.handrail_width = log_uniform(0.02, 0.06)
            self.handrail_height = log_uniform(0.02, 0.06)
            self.handrail_offset = self.handrail_width * log_uniform(1, 2)
            self.handrail_extension = uniform(0.1, 0.2)
            self.handrail_alphas = [
                self.handrail_offset / self.step_width,
                1 - self.handrail_offset / self.step_width,
            ]
            probs = np.array([3, 2, 3])
            self.handrail_surface = np.random.choice(
                [wood, metal, fabrics.fabric_random], p=probs / probs.sum()
            )

            self.post_height = log_uniform(0.8, 1.2)
            self.post_k = int(np.ceil(self.step_width / self.step_length))
            self.post_width = self.handrail_width * log_uniform(0.6, 0.8)
            self.post_minor_width = self.post_width * log_uniform(0.3, 0.5)
            self.is_post_circular = uniform() < 0.5
            probs = np.array([3, 3, 2])
            self.post_surface = np.random.choice(
                [wood, metal, fabrics.fabric_random], p=probs / probs.sum()
            )
            self.has_vertical_post = self.handrail_type == "vertical-post"

            self.has_bars = self.handrail_type == "horizontal-post"
            self.bar_size = log_uniform(0.1, 0.2)
            self.n_bars = int(
                np.floor(self.post_height / self.bar_size * uniform(0.35, 0.75))
            )

            self.has_glasses = self.handrail_type == "glass"
            self.glass_height = self.post_height - uniform(0, 0.05)
            self.glass_margin = self.step_height / 2 + uniform(0, 0.05)
            self.glass_surface = glass

            self.has_spiral = False
            self.mirror = uniform() < 0.5
            self.rot_z = np.random.randint(4) * np.pi / 2
            self.end_margin = self.step_length * 8

    def build_size_config(self):
        self.n = np.random.randint(13, 21)
        self.step_height = self.constants.wall_height / self.n
        self.step_width = uniform(0.8, 1.6)
        self.step_length = self.step_height * log_uniform(0.8, 1.2)

    def make_line(self, alpha):
        obj = new_line(self.n)
        x = np.full(self.n + 1, alpha * self.step_width)
        y = self.step_length * np.arange(self.n + 1)
        z = self.step_height * np.arange(self.n + 1)
        np.stack([x, y, z], -1)
        write_co(obj, np.stack([x, y, z], -1))
        return obj

    def make_line_offset(self, alpha):
        obj = self.make_line(alpha)
        x, y, z = read_co(obj).T
        y += self.step_length / 2
        z += self.step_height
        z[-1] -= self.step_height
        write_co(obj, np.stack([x, y, z], -1))
        return obj

    def make_post_locs(self, alpha):
        temp = self.make_line_offset(alpha)
        cos = read_co(temp)
        butil.delete(temp)
        chunks = self.split(self.n - 1)
        indices = list(c[0] for c in chunks) + [self.n - 1, self.n]
        return cos[indices]

    def make_vertical_post_locs(self, alpha):
        temp = self.make_line_offset(alpha)
        cos = read_co(temp)
        butil.delete(temp)
        chunks = self.split(self.n - 1)
        indices = sum(list(c[1:].tolist() for c in chunks), []) + [self.n]
        return cos[indices]

    def split(self, start, end=None):
        return np.array_split(
            np.arange(start, end),
            np.ceil((start if end is None else end - start) / self.post_k),
        )

    @staticmethod
    def triangulate(obj):
        butil.modify_mesh(obj, "TRIANGULATE", min_vertices=3)
        levels = 1
        butil.modify_mesh(
            obj,
            "SUBSURF",
            levels=levels,
            render_levels=levels,
            subdivision_type="SIMPLE",
        )
        return obj

    def vertical_cut(self, p):
        cuts = list(LineString([(i, -100), (i, 100)]) for i in range(1, self.n - 1))
        polygons = cut_polygon_by_line(p, *cuts)
        parts = []
        for p in polygons:
            coords = p.boundary.coords[:][:-1]
            part = new_circle(vertices=len(coords))
            with butil.ViewportMode(part, "EDIT"):
                bpy.ops.mesh.edge_face_add()
            write_co(
                part,
                np.array(
                    list(
                        [0, y * self.step_length, z * self.step_height]
                        for y, z in coords
                    )
                ),
            )
            parts.append(part)
        return parts

    def make_steps(self):
        coords = [(0, 0)]
        for i in range(self.n):
            coords.extend([(i, i + 1), (i + 1, i + 1)])
        coords.extend([(self.n, 0), (0, 0)])
        p = Polygon(LineString(coords))
        if self.support_type == "hole":
            hole = Polygon(
                [
                    ((1 - self.hole_size) * self.n, 0),
                    (self.n, self.hole_size * self.n),
                    (self.n, 0),
                    ((1 - self.hole_size) * self.n, 0),
                ]
            )
            p = p.difference(hole)
        objs = self.vertical_cut(p)
        for obj in objs:
            butil.modify_mesh(obj, "SOLIDIFY", thickness=self.step_width)
            self.triangulate(obj)
            write_attribute(obj, 1, "steps", "FACE")
        return objs

    def make_rails(self):
        parts = []
        if self.support_type == "single-rail":
            alphas = [0.5]
        else:
            alphas = [
                self.rail_offset / self.step_width,
                1 - self.rail_offset / self.step_width,
            ]
        for alpha in alphas:
            obj = self.make_line(alpha)
            if self.is_rail_circular:
                surface.add_geomod(
                    obj, geo_radius, apply=True, input_args=[self.rail_width, 16]
                )
                obj.location[-1] = -self.rail_width
                butil.apply_transform(obj, loc=True)
            else:
                butil.select_none()
                with butil.ViewportMode(obj, "EDIT"):
                    bpy.ops.mesh.select_mode(type="EDGE")
                    bpy.ops.mesh.select_all(action="SELECT")
                    bpy.ops.mesh.extrude_edges_move(
                        TRANSFORM_OT_translate={"value": (0, 0, -self.rail_height * 2)}
                    )
                butil.modify_mesh(obj, "SOLIDIFY", thickness=self.rail_width, offset=0)
            self.triangulate(obj)
            write_attribute(obj, 1, "rails", "FACE")
            parts.append(obj)
        return parts

    def make_treads(self):
        tread = new_cube(location=(1, 1, 1))
        butil.apply_transform(tread, loc=True)
        tread.scale = self.tread_width / 2, self.tread_length / 2, self.tread_height / 2
        tread.location = (
            -(self.tread_width - self.step_width) / 2,
            -(self.tread_length - self.step_length),
            self.step_height,
        )
        butil.apply_transform(tread, loc=True)
        self.triangulate(tread)
        write_attribute(tread, 1, "treads", "FACE")
        treads = [tread] + list(butil.deep_clone_obj(tread) for _ in range(self.n - 1))
        for i in range(1, self.n):
            treads[i].location = 0, self.step_length * i, self.step_height * i
            butil.apply_transform(treads[i], loc=True)
        return treads

    def make_inner_sides(self):
        offset = -self.side_height / self.step_height
        if self.side_type == "zig-zag":
            coords = [(0, 0)]
            for i in range(self.n):
                coords.extend([(i, i + 1), (i + 1, i + 1)])
            l = LineString(coords)
            p = l.buffer(
                offset,
                join_style="mitre",
                single_sided=True,
            )
        else:
            p = Polygon(
                LineString(
                    [
                        (0, offset),
                        (0, 1),
                        (self.n, self.n + 1),
                        (self.n, self.n + offset),
                        (0, offset),
                    ]
                )
            )
        objs = self.vertical_cut(p)

        bottom_cutter = new_cube(location=(0, 0, -1))
        butil.apply_transform(bottom_cutter, loc=True)
        bottom_cutter.scale = [100] * 3
        butil.apply_transform(bottom_cutter)
        top_cutter = new_cube(location=(0, 0, 1))
        butil.apply_transform(top_cutter, loc=True)
        top_cutter.scale = [100] * 3
        top_cutter.location[-1] = self.n * self.step_height + self.tread_height

        for obj in objs:
            butil.modify_mesh(obj, "SOLIDIFY", thickness=self.side_thickness, offset=0)
            write_attribute(obj, 1, "sides", "FACE")
            for cutter in [top_cutter, bottom_cutter]:
                butil.modify_mesh(obj, "BOOLEAN", object=cutter, operation="DIFFERENCE")
        butil.delete([top_cutter, bottom_cutter])
        return objs

    def make_outer_sides(self):
        objs = self.make_inner_sides()
        for obj in objs:
            obj.location[0] = self.step_width
            butil.apply_transform(obj, loc=True)
        return objs

    def make_column(self):
        return

    def make_handrails(self):
        parts = []
        for alpha in self.handrail_alphas:
            obj = self.make_line_offset(alpha)
            self.make_single_handrail(obj)
            parts.append(obj)
        return parts

    def make_single_handrail(self, obj):
        self.extend_line(obj, self.handrail_extension)
        if self.is_handrail_circular:
            surface.add_geomod(
                obj,
                geo_radius,
                apply=True,
                input_args=[self.handrail_width, 32],
                input_kwargs={"to_align_tilt": False},
            )
        else:
            butil.select_none()
            with butil.ViewportMode(obj, "EDIT"):
                bpy.ops.mesh.select_mode(type="EDGE")
                bpy.ops.mesh.select_all(action="SELECT")
                bpy.ops.mesh.extrude_edges_move(
                    TRANSFORM_OT_translate={"value": (0, 0, -self.handrail_height * 2)}
                )
            butil.modify_mesh(
                obj,
                "SOLIDIFY",
                thickness=self.handrail_width * 2,
                offset=0,
                solidify_mode="NON_MANIFOLD",
            )
            butil.modify_mesh(
                obj,
                "BEVEL",
                width=self.handrail_width * uniform(0.2, 0.5),
                segments=np.random.randint(4, 7),
            )
            obj.location[-1] += self.handrail_height
        write_attribute(obj, 1, "handrails", "FACE")
        obj.location[-1] += self.post_height
        butil.apply_transform(obj, loc=True)
        self.triangulate(obj)

    @staticmethod
    def extend_line(obj, extension):
        if len(obj.data.vertices) <= 1:
            return
        with butil.ViewportMode(obj, "EDIT"):
            bm = bmesh.from_edit_mesh(obj.data)
            bm.verts.ensure_lookup_table()
            v0, v1, v2, v3 = bm.verts[0], bm.verts[1], bm.verts[-1], bm.verts[-2]
            n_0 = v0.co - v1.co
            n_0[2] = 0
            v4 = bm.verts.new(v0.co + n_0 / n_0.length * extension)
            bm.edges.new((v4, v0))
            n_1 = v2.co - v3.co
            n_1[2] = 0
            v5 = bm.verts.new(v2.co + n_1 / n_1.length * extension)
            bm.edges.new((v2, v5))
            bmesh.update_edit_mesh(obj.data)

    def make_posts(self, locs, widths):
        parts = []
        existing = np.zeros((0, 3))
        for loc, width in zip(locs, widths):
            existing = np.concatenate([existing, loc[:1]], 0)
            cos = [0]
            for i, l in enumerate(loc):
                if (
                    i > 0
                    and np.min(np.linalg.norm(existing - l[np.newaxis, :], axis=1))
                    > self.handrail_width * 2
                ):
                    cos.append(i)
                    existing = np.concatenate([existing, loc[i : i + 1]], 0)
            obj = mesh2obj(data2mesh(loc[cos]))
            with butil.ViewportMode(obj, "EDIT"):
                bpy.ops.mesh.select_all(action="SELECT")
                bpy.ops.mesh.extrude_vertices_move(
                    TRANSFORM_OT_translate={"value": (0, 0, self.post_height)}
                )
            if self.is_post_circular:
                surface.add_geomod(obj, geo_radius, apply=True, input_args=[width, 32])
            else:
                with butil.ViewportMode(obj, "EDIT"):
                    bpy.ops.mesh.select_mode(type="EDGE")
                    bpy.ops.mesh.select_all(action="SELECT")
                    bpy.ops.mesh.extrude_edges_move(
                        TRANSFORM_OT_translate={"value": (width * 2, 0, 0)}
                    )
                    bpy.ops.mesh.select_mode(type="FACE")
                    bpy.ops.mesh.select_all(action="SELECT")
                    bpy.ops.mesh.extrude_region_move(
                        TRANSFORM_OT_translate={"value": (0, width * 2, 0)}
                    )
                obj.location = -width, -width, 0
                butil.apply_transform(obj, loc=True)
            write_attribute(obj, 1, "posts", "FACE")
            parts.append(obj)
        return parts

    def make_bars(self, locs):
        parts = []
        for loc in locs:
            for loc, loc_ in zip(loc[:-1], loc[1:]):
                for i in range(self.n_bars):
                    obj = new_line()
                    write_co(obj, np.stack([loc, loc_]))
                    subsurf(obj, 4)
                    surface.add_geomod(
                        obj, geo_radius, apply=True, input_args=[self.post_minor_width]
                    )
                    obj.location[-1] += self.post_height - (i + 1) * self.bar_size
                    butil.apply_transform(obj, loc=True)
                    write_attribute(obj, 1, "posts", "FACE")
                    parts.append(obj)
        return parts

    def make_glasses(self, locs):
        parts = []
        for loc in locs:
            for loc, loc_ in zip(loc[:-1], loc[1:]):
                obj = new_line()
                write_co(obj, np.stack([loc, loc_]))
                with butil.ViewportMode(obj, "EDIT"):
                    bpy.ops.mesh.select_mode(type="EDGE")
                    bpy.ops.mesh.select_all(action="SELECT")
                    bpy.ops.mesh.extrude_edges_move(
                        TRANSFORM_OT_translate={
                            "value": (0, 0, self.glass_height - self.glass_margin)
                        }
                    )
                butil.modify_mesh(obj, "SOLIDIFY", thickness=self.post_minor_width)
                obj.location[-1] += self.glass_margin
                butil.apply_transform(obj, loc=True)
                write_attribute(obj, 1, "glasses", "FACE")
                parts.append(obj)
        return parts

    def make_spiral(self, obj):
        return obj

    def unmake_spiral(self, obj):
        return obj

    def create_placeholder(self, **kwargs) -> bpy.types.Object:
        obj = self.make_line_offset(0.5)
        if self.has_spiral:
            self.make_spiral(obj)
        self.extend_line(obj, self.end_margin)
        self.decorate_line(
            obj, self.constants.wall_thickness / 2, self.constants.door_size
        )
        if self.mirror:
            mirror(obj)
        obj.rotation_euler[-1] = self.rot_z
        butil.apply_transform(obj)
        return obj

    def create_cutter(self, **kwargs) -> bpy.types.Object:
        obj = self.make_line_offset(0.5)
        if self.has_spiral:
            self.make_spiral(obj)
        self.decorate_line(obj, 0, self.constants.door_size)
        if self.mirror:
            mirror(obj)
        obj.location[-1] = -self.constants.wall_thickness / 2
        obj.rotation_euler[-1] = self.rot_z
        butil.apply_transform(obj, True)
        return obj

    def create_asset(self, **params) -> bpy.types.Object:
        parts = []
        if self.has_step:
            parts.extend(self.make_steps())
        if self.has_rail:
            parts.extend(self.make_rails())
        if self.has_tread:
            parts.extend(self.make_treads())
        if self.has_sides:
            parts.extend(self.make_inner_sides())
            parts.extend(self.make_outer_sides())
        parts.extend(self.make_handrails())
        post_locs = list(self.make_post_locs(alpha) for alpha in self.handrail_alphas)
        if self.has_vertical_post:
            vertical_post_locs = list(
                self.make_vertical_post_locs(alpha) for alpha in self.handrail_alphas
            )
            parts.extend(
                self.make_posts(
                    post_locs + vertical_post_locs,
                    [self.post_width] * len(post_locs)
                    + [self.post_minor_width] * len(vertical_post_locs),
                )
            )
        else:
            parts.extend(self.make_posts(post_locs, [self.post_width] * len(post_locs)))
        if self.has_bars:
            parts.extend(self.make_bars(post_locs))
        if self.has_glasses:
            parts.extend(self.make_glasses(post_locs))
        obj = join_objects(parts)
        if self.has_spiral:
            self.make_spiral(obj)
        if self.has_column:
            obj = join_objects([obj, self.make_column()])
        if self.mirror:
            mirror(obj)
        obj.rotation_euler[-1] = self.rot_z
        butil.apply_transform(obj)
        return obj

    def decorate_line(self, line, low, high):
        end = np.zeros(len(line.data.vertices))
        end[[0, -1]] = 1
        write_attr_data(line, "end", end)
        with butil.ViewportMode(line, "EDIT"):
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.extrude_edges_move(
                TRANSFORM_OT_translate={"value": (0, 0, high - low)}
            )
            bpy.ops.mesh.normals_make_consistent(inside=False)
        line.location[-1] -= low
        butil.modify_mesh(
            line, "SOLIDIFY", thickness=self.step_width, offset=0, use_even_offset=True
        )
        self.triangulate(line)
        line.location[-1] -= self.constants.wall_thickness / 2
        butil.apply_transform(line, True)
        write_attribute(
            line,
            lambda nw: nw.compare("LESS_THAN", surface.eval_argument(nw, "end"), 0.99),
            "staircase_wall",
            "FACE",
            "INT",
        )
        sharp_remesh_with_attrs(line, 0.05)
        zeros = np.zeros(len(line.data.polygons), dtype=int)
        ones = np.ones(len(line.data.polygons), dtype=int)
        write_attr_data(
            line, f"{PREFIX}{t.Subpart.Ceiling.value}", zeros, "INT", "FACE"
        )
        write_attr_data(
            line, f"{PREFIX}{t.Subpart.SupportSurface.value}", zeros, "INT", "FACE"
        )
        write_attr_data(line, f"{PREFIX}{t.Subpart.Wall.value}", ones, "INT", "FACE")
        write_attr_data(line, f"{PREFIX}{t.Subpart.Visible.value}", ones, "INT", "FACE")
        with butil.ViewportMode(line, "EDIT"):
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.normals_make_consistent(inside=False)

    def finalize_assets(self, assets):
        if self.has_step:
            self.step_surface.apply(assets, selection="steps", metal_color="bw+natural")
        if self.has_tread:
            self.tread_surface.apply(
                assets, selection="treads", metal_color="bw+natural"
            )
        if self.has_rail:
            self.rail_surface.apply(assets, selection="rails")
        if self.has_sides:
            self.side_surface.apply(assets, selection="sides")
        self.handrail_surface.apply(assets, selection="handrails")
        self.post_surface.apply(assets, selection="posts")
        if self.has_glasses:
            self.glass_surface.apply(assets, selection="glasses")

    def make_guardrail(self, mesh):
        def geo_extrude(nw: NodeWrangler):
            geometry = nw.new_node(
                Nodes.GroupInput,
                expose_input=[("NodeSocketGeometry", "Geometry", None)],
            )
            x, y, _ = nw.separate(nw.new_node(Nodes.InputNormal))
            offset = nw.scale(
                -self.handrail_offset, nw.vector_math("NORMALIZE", nw.combine(x, y, 0))
            )
            geometry = nw.new_node(
                Nodes.SetPosition, [geometry], input_kwargs={"Offset": offset}
            )
            nw.new_node(Nodes.GroupOutput, input_kwargs={"Geometry": geometry})

        self.unmake_spiral(mesh)
        with butil.ViewportMode(mesh, "EDIT"):
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.normals_make_consistent(inside=False)
        surface.add_geomod(mesh, geo_extrude, apply=True)
        remove_faces(mesh, read_attr_data(mesh, "staircase_wall") == 0)
        with butil.ViewportMode(mesh, "EDIT"):
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.region_to_loop()
            bpy.ops.mesh.select_all(action="INVERT")
            bpy.ops.mesh.delete(type="EDGE")
        remove_vertices(
            mesh,
            lambda x, y, z: (z < self.constants.wall_thickness / 4)
            | (z > self.constants.wall_thickness * 3 / 4),
        )
        butil.modify_mesh(
            mesh, "WELD", merge_threshold=self.constants.wall_thickness / 4
        )
        name = mesh.name
        mesh = separate_loose(mesh)
        ls = shapely.force_2d(convert2ls(mesh))
        butil.delete(mesh)
        parts, locs, minor_locs = [], [], []
        line = canonicalize_ls(ls)
        segments = line.segmentize(self.post_k * self.step_length)
        locs.append(np.array(shapely.force_3d(segments).coords))
        line = segments.segmentize(self.step_length)
        if self.has_vertical_post:
            minor_locs.append(np.array(shapely.force_3d(line).coords))
        line = shapely.force_3d(line)
        o = new_line(len(line.coords) - 1)
        write_co(o, np.array(line.coords))
        self.make_single_handrail(o)
        parts.append(o)
        parts.extend(
            self.make_posts(
                locs + minor_locs,
                [self.post_width] * len(locs)
                + [self.post_minor_width] * len(minor_locs),
            )
        )
        if self.has_bars:
            parts.extend(self.make_bars(locs))
        if self.has_glasses:
            parts.extend(self.make_glasses(locs))
        butil.select_none()
        obj = join_objects(parts)
        self.make_spiral(obj)
        self.handrail_surface.apply(obj, selection="handrails")
        self.post_surface.apply(obj, selection="posts")
        if self.has_glasses:
            self.glass_surface.apply(obj, selection="glasses")
        obj.name = name
        return obj

    @property
    def lower(self):
        return -np.pi / 2

    @property
    def upper(self):
        return np.pi / 2

    def valid_contour(self, offset, contour, doors, lower=True):
        x, y = offset
        if len(doors) == 0:
            return True
        for door in doors:
            t = self.lower if lower else self.upper
            t = (np.pi - t if self.mirror else t) + self.rot_z
            v = np.array([np.cos(t), np.sin(t)])
            if (
                normalize(np.array([door.location[0] - x, door.location[1] - y])) @ v
                >= -0.5
            ):
                return True
        return False
