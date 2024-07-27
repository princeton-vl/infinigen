# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

import bmesh

# Authors: Lingjie Mei
import bpy
import numpy as np
from numpy.random import uniform

from infinigen.assets.material_assignments import AssetList
from infinigen.assets.utils.autobevel import BevelSharp
from infinigen.assets.utils.decorate import (
    read_center,
    read_co,
    read_normal,
    subsurf,
    write_attribute,
    write_co,
)
from infinigen.assets.utils.nodegroup import geo_radius
from infinigen.assets.utils.object import (
    join_objects,
    new_bbox,
    new_cube,
    new_cylinder,
    new_line,
)
from infinigen.core import surface
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.util import blender as butil
from infinigen.core.util.blender import deep_clone_obj
from infinigen.core.util.math import FixedSeed


class BathtubFactory(AssetFactory):
    def __init__(self, factory_seed, coarse=False):
        super(BathtubFactory, self).__init__(factory_seed, coarse)
        with FixedSeed(factory_seed):
            self.width = uniform(1.5, 2)
            self.size = uniform(0.8, 1)
            self.depth = uniform(0.55, 0.7)
            prob = np.array([2, 2])
            self.bathtub_type = np.random.choice(
                ["alcove", "freestanding"], p=prob / prob.sum()
            )  # , 'corner'
            self.contour_fn = (
                self.make_corner_contour if self.has_corner else self.make_box_contour
            )
            self.has_curve = uniform() < 0.5
            self.has_legs = uniform() < 0.5

            self.thickness = (
                uniform(0.04, 0.08) if self.has_base else uniform(0.02, 0.04)
            )
            self.disp_x = uniform(0, 0.2, 2)
            self.disp_y = uniform(0, 0.1)

            self.leg_height = uniform(0.2, 0.3) * self.depth
            self.leg_side = uniform(0.05, 0.1)
            self.leg_radius = uniform(0.02, 0.03)
            self.leg_y_scale = uniform()
            self.leg_subsurf_level = np.random.randint(3)

            self.taper_factor = uniform(-0.1, 0.1)
            self.stretch_factor = uniform(-0.2, 0.2)

            self.alcove_levels = np.random.randint(1, 3) if self.has_base else 1
            self.levels = 5
            self.side_levels = 2

            self.is_hole_centered = False
            self.hole_radius = uniform(0.015, 0.02)

            # /////////////////// assign materials ///////////////////
            material_assignments = AssetList["BathtubFactory"]()
            self.surface = material_assignments["surface"].assign_material()
            self.leg_surface = material_assignments["leg"].assign_material()
            self.hole_surface = material_assignments["hole"].assign_material()
            is_scratch = uniform() < material_assignments["wear_tear_prob"][0]
            is_edge_wear = uniform() < material_assignments["wear_tear_prob"][1]
            self.scratch = material_assignments["wear_tear"][0] if is_scratch else None
            self.edge_wear = (
                material_assignments["wear_tear"][1] if is_edge_wear else None
            )
            # ////////////////////////////////////////////////////////

            self.beveler = BevelSharp(mult=5, segments=5)

    @property
    def has_base(self):
        return self.bathtub_type != "freestanding"

    @property
    def has_corner(self):
        return self.bathtub_type == "corner"

    def create_placeholder(self, **kwargs) -> bpy.types.Object:
        return new_bbox(-self.size, 0, 0, self.width, 0, self.depth)

    def create_asset(self, **params) -> bpy.types.Object:
        if self.has_base:
            obj = self.make_base()
            cutter = self.make_cutter()
            butil.modify_mesh(obj, "BOOLEAN", object=cutter, operation="DIFFERENCE")
            butil.delete(cutter)
        else:
            obj = self.make_freestanding()
            parts = [obj]
            if self.has_legs:
                parts.extend(self.make_legs(obj))
            else:
                parts.append(self.add_base(obj))
            butil.modify_mesh(obj, "SOLIDIFY", thickness=self.thickness)
            subsurf(obj, self.side_levels)
            obj = join_objects(parts)
        hole = self.add_hole(obj)
        obj = join_objects([obj, hole])
        obj.rotation_euler[-1] = np.pi / 2
        butil.apply_transform(obj, True)

        if self.bathtub_type == "freestanding":
            butil.modify_mesh(obj, "SUBSURF", levels=1, apply=True)
        else:
            self.beveler(obj)

        return obj

    def make_freestanding(self):
        obj = self.make_bowl()
        self.remove_top(obj)
        with butil.ViewportMode(obj, "EDIT"):
            bpy.ops.mesh.select_mode(type="EDGE")
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.region_to_loop()
            bpy.ops.mesh.extrude_edges_move()
            bpy.ops.transform.resize(
                value=(
                    1 + self.thickness * 2 / self.width,
                    1 + self.thickness / self.size,
                    1,
                )
            )
        obj.location[1] -= self.size / 2
        butil.apply_transform(obj, True)
        butil.modify_mesh(
            obj, "SIMPLE_DEFORM", deform_method="TAPER", angle=self.taper_factor
        )
        butil.modify_mesh(
            obj, "SIMPLE_DEFORM", deform_method="STRETCH", angle=self.taper_factor
        )
        obj.location = (
            0,
            self.size / 2,
            -np.min(read_co(obj)[:, -1]) * uniform(0.5, 0.7),
        )
        butil.apply_transform(obj, True)
        return obj

    def remove_top(self, obj):
        butil.select_none()
        with butil.ViewportMode(obj, "EDIT"):
            bm = bmesh.from_edit_mesh(obj.data)
            geom = [f for f in bm.faces if f.calc_center_median()[-1] > self.depth]
            bmesh.ops.delete(bm, geom=geom, context="FACES_KEEP_BOUNDARY")
            bmesh.update_edit_mesh(obj.data)

    def make_legs(self, obj):
        legs = []
        co, normal = read_center(obj), read_normal(obj)
        x, y, z = co.T
        leg_height = np.min(z) + self.leg_height
        for u in [1, -1]:
            for v in [1, -1]:
                metric = np.where(z < leg_height, u * x + v * y, -np.inf)
                i = np.argmax(metric)
                p = co[i]
                n = normal[i]
                q = co[i] + self.leg_side * np.array(
                    [n[0], n[1] * self.leg_y_scale, n[2]]
                )
                r = np.array([q[0], q[1], 0])
                leg = new_line(2)
                write_co(leg, np.stack([p, q, r]))
                subsurf(leg, self.leg_subsurf_level)
                surface.add_geomod(
                    leg,
                    geo_radius,
                    apply=True,
                    input_args=[self.leg_radius, 32],
                    input_kwargs={"to_align_tilt": False},
                )
                butil.modify_mesh(
                    leg, "BEVEL", width=self.leg_radius * uniform(0.3, 0.7)
                )
                leg.location[-1] = self.leg_radius
                butil.apply_transform(leg, True)
                write_attribute(leg, 1, "leg", "FACE")
                legs.append(leg)
        return legs

    def add_base(self, obj):
        obj = deep_clone_obj(obj)
        cutter = new_cube()
        x, y, z_ = read_co(obj).T
        cutter.scale = 10, 10, np.min(z_) + self.leg_height
        butil.apply_transform(cutter, True)
        butil.modify_mesh(obj, "BOOLEAN", object=cutter, operation="INTERSECT")
        butil.delete(cutter)
        with butil.ViewportMode(obj, "EDIT"):
            bm = bmesh.from_edit_mesh(obj.data)
            geom = [f for f in bm.faces if len(f.verts) > 10]
            bmesh.ops.delete(bm, geom=geom, context="FACES_KEEP_BOUNDARY")
            bmesh.update_edit_mesh(obj.data)
            bpy.ops.mesh.select_mode(type="EDGE")
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.region_to_loop()
            bpy.ops.mesh.select_all(action="INVERT")
            bpy.ops.mesh.delete(type="EDGE")
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.extrude_edges_move(
                TRANSFORM_OT_translate={"value": (0, 0, -self.depth)}
            )
        x, y, z = read_co(obj).T
        z = np.clip(z, 0, None)
        write_co(obj, np.stack([x, y, z], -1))
        with butil.ViewportMode(obj, "EDIT"):
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.normals_make_consistent(inside=False)
        subsurf(obj, 2)
        butil.modify_mesh(obj, "SOLIDIFY", thickness=self.thickness)
        return obj

    def make_box_contour(self, t, i):
        return [
            (t + self.disp_x[0] * i, t + self.disp_y * i),
            (self.width - t - self.disp_x[1] * i, t + self.disp_y * i),
            (self.width - t - self.disp_x[1] * i, self.size - t - self.disp_y * i),
            (t + self.disp_x[0] * i, self.size - t - self.disp_y * i),
        ]

    def make_corner_contour(self, t, i):
        return [
            (t + self.disp_y * i, t + self.disp_y * i),
            (self.width - t - self.disp_x[1] * i, t + self.disp_y * i),
            (
                self.width - t - self.disp_x[1] * i,
                self.size - (t + self.disp_y * i) / np.sqrt(2),
            ),
            (
                self.size - (t + self.disp_y * i) / np.sqrt(2),
                self.width - t - self.disp_x[0] * i,
            ),
            (t + self.disp_y * i, self.width - t - self.disp_x[0] * i),
        ]

    # noinspection PyArgumentList
    def make_base(self):
        contour = self.contour_fn(0, 0)
        obj = new_cylinder(vertices=len(contour))
        co = np.concatenate(
            [np.array([[x, y, 0], [x, y, self.depth]]) for x, y in contour]
        )
        write_co(obj, co)
        return obj

    # noinspection PyArgumentList
    def make_bowl(self):
        if self.has_curve:
            lower = self.contour_fn(0, 1)
            upper = self.contour_fn(0, -1)
        else:
            lower = self.contour_fn(0, 0)
            upper = self.contour_fn(0, 0)
        obj = new_cylinder(vertices=len(lower))
        co = np.concatenate(
            [
                np.array([[x, y, 0], [z, w, self.depth * 2]])
                for (x, y), (z, w) in zip(lower[::-1], upper[::-1])
            ]
        )
        write_co(obj, co)
        subsurf(obj, self.alcove_levels, True)
        levels = self.levels - self.alcove_levels - self.side_levels
        subsurf(obj, levels)
        return obj

    # noinspection PyArgumentList
    def make_cutter(self):
        if self.has_curve:
            lower = self.contour_fn(self.thickness, 1)
            upper = self.contour_fn(self.thickness, -1)
        else:
            lower = self.contour_fn(self.thickness, 0)
            upper = self.contour_fn(self.thickness, 0)
        obj = new_cylinder(vertices=len(lower))
        co = np.concatenate(
            [
                np.array(
                    [[x, y, self.thickness], [z, w, self.depth * 2 - self.thickness]]
                )
                for (x, y), (z, w) in zip(lower[::-1], upper[::-1])
            ]
        )
        write_co(obj, co)
        subsurf(obj, self.alcove_levels, True)
        levels = self.levels - self.alcove_levels
        subsurf(obj, levels)
        return obj

    def find_hole(self, obj, x=None, y=None):
        if x is None:
            x = self.width / 2
        if y is None:
            y = self.size / 2
        up_facing = read_normal(obj)[:, -1] > 0
        center = read_center(obj)
        i = np.argmin(np.abs(center[:, :2] - np.array([[x, y]])).sum(1) - up_facing)
        return center[i]

    def add_hole(self, obj):
        match self.bathtub_type:
            case "alcove":
                location = self.find_hole(obj)
            case "freestanding":
                location = self.find_hole(obj, uniform(0.35, 0.4) * self.width)
            case _:
                location = self.find_hole(obj, self.size / 2, self.size / 2)
        if self.is_hole_centered:
            location = self.find_hole(obj)
        obj = new_cylinder()
        obj.scale = self.hole_radius, self.hole_radius, 0.005
        obj.location = location
        butil.apply_transform(obj, True)
        write_attribute(obj, 1, "hole", "FACE")
        return obj

    def finalize_assets(self, assets):
        self.surface.apply(assets, clear=True)
        if self.has_legs and not self.has_base:
            self.leg_surface.apply(assets, "leg", metal_color="bw+natural")
        self.hole_surface.apply(assets, "hole", metal_color="bw+natural")

        if self.scratch:
            self.scratch.apply(assets)
        if self.edge_wear:
            self.edge_wear.apply(assets)
