# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei
import bpy
import numpy as np
from numpy.random import uniform

from infinigen.assets.material_assignments import AssetList
from infinigen.assets.utils.decorate import (
    read_center,
    read_co,
    read_edge_center,
    read_edges,
    read_normal,
    select_edges,
    select_faces,
    select_vertices,
    subsurf,
    write_attribute,
    write_co,
)
from infinigen.assets.utils.draw import align_bezier
from infinigen.assets.utils.object import join_objects, new_bbox, new_cube, new_cylinder
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.util import blender as butil
from infinigen.core.util.blender import deep_clone_obj
from infinigen.core.util.math import FixedSeed, normalize
from infinigen.core.util.random import log_uniform


class ToiletFactory(AssetFactory):
    def __init__(self, factory_seed, coarse=False):
        super().__init__(factory_seed, coarse)
        with FixedSeed(self.factory_seed):
            self.size = uniform(0.4, 0.5)
            self.width = self.size * uniform(0.7, 0.8)
            self.height = self.size * uniform(0.8, 0.9)
            self.size_mid = uniform(0.6, 0.65)
            self.curve_scale = log_uniform(0.8, 1.2, 4)
            self.depth = self.size * uniform(0.5, 0.6)
            self.tube_scale = uniform(0.25, 0.3)
            self.thickness = uniform(0.05, 0.06)
            self.extrude_height = uniform(0.015, 0.02)
            self.stand_depth = self.depth * uniform(0.85, 0.95)
            self.stand_scale = uniform(0.7, 0.85)
            self.bottom_offset = uniform(0.5, 1.5)
            self.back_thickness = self.thickness * uniform(0, 0.8)
            self.back_size = self.size * uniform(0.55, 0.65)
            self.back_scale = uniform(0.8, 1.0)
            self.seat_thickness = uniform(0.1, 0.3) * self.thickness
            self.seat_size = self.thickness * uniform(1.2, 1.6)
            self.has_seat_cut = uniform() < 0.1
            self.tank_width = self.width * uniform(1.0, 1.2)
            self.tank_height = self.height * uniform(0.6, 1.0)
            self.tank_size = self.back_size - self.seat_size - uniform(0.02, 0.03)
            self.tank_cap_height = uniform(0.03, 0.04)
            self.tank_cap_extrude = 0 if uniform() < 0.5 else uniform(0.005, 0.01)
            self.cover_rotation = -uniform(0, np.pi / 2)
            self.hardware_type = np.random.choice(["button", "handle"])
            self.hardware_cap = uniform(0.01, 0.015)
            self.hardware_radius = uniform(0.015, 0.02)
            self.hardware_length = uniform(0.04, 0.05)
            self.hardware_on_side = uniform() < 0.5
            material_assignments = AssetList["ToiletFactory"]()
            self.surface = material_assignments["surface"].assign_material()
            self.hardware_surface = material_assignments[
                "hardware_surface"
            ].assign_material()

            is_scratch = uniform() < material_assignments["wear_tear_prob"][0]
            is_edge_wear = uniform() < material_assignments["wear_tear_prob"][1]
            self.scratch = material_assignments["wear_tear"][0] if is_scratch else None
            self.edge_wear = (
                material_assignments["wear_tear"][1] if is_edge_wear else None
            )

    @property
    def mid_offset(self):
        return (1 - self.size_mid) * self.size

    def create_placeholder(self, **kwargs) -> bpy.types.Object:
        return new_bbox(
            -self.mid_offset - self.back_size - self.tank_cap_extrude,
            self.size_mid * self.size + self.thickness + self.thickness,
            -self.width / 2 - self.thickness * 1.1,
            self.width / 2 + self.thickness * 1.1,
            -self.height,
            max(
                self.tank_height,
                -np.sin(self.cover_rotation)
                * (self.seat_size + self.size + self.thickness + self.thickness),
            ),
        )

    def create_asset(self, **params) -> bpy.types.Object:
        upper = self.build_curve()
        lower = deep_clone_obj(upper)
        lower.scale = [self.tube_scale] * 3
        lower.location = 0, self.tube_scale * self.mid_offset / 2, -self.depth
        butil.apply_transform(lower, True)
        bottom = deep_clone_obj(upper)
        bottom.scale = [self.stand_scale] * 3
        bottom.location = (
            0,
            self.tube_scale * (1 - self.size_mid) * self.size / 2 * self.bottom_offset,
            -self.height,
        )
        butil.apply_transform(bottom, True)

        obj = self.make_tube(lower, upper)
        seat, cover = self.make_seat(obj)
        stand = self.make_stand(obj, bottom)
        back = self.make_back(obj)
        tank = self.make_tank()
        butil.modify_mesh(obj, "BEVEL", segments=2)
        match self.hardware_type:
            case "button":
                hardware = self.add_button()
            case _:
                hardware = self.add_handle()
        write_attribute(hardware, 1, "hardware", "FACE")
        obj = join_objects([obj, seat, cover, stand, back, tank, hardware])
        obj.rotation_euler[-1] = np.pi / 2
        butil.apply_transform(obj)
        return obj

    def build_curve(self):
        x_anchors = [0, self.width / 2, 0]
        y_anchors = [-self.size_mid * self.size, 0, self.mid_offset]
        axes = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([1, 0, 0])]
        obj = align_bezier([x_anchors, y_anchors, 0], axes, self.curve_scale)
        butil.modify_mesh(obj, "MIRROR", use_axis=(True, False, False))
        return obj

    def make_tube(self, lower, upper):
        obj = join_objects([upper, lower])
        with butil.ViewportMode(obj, "EDIT"):
            bpy.ops.mesh.select_mode(type="EDGE")
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.bridge_edge_loops(
                number_cuts=64,
                profile_shape_factor=uniform(0.1, 0.2),
                interpolation="SURFACE",
            )
        butil.modify_mesh(
            obj,
            "SOLIDIFY",
            thickness=self.thickness,
            offset=1,
            solidify_mode="NON_MANIFOLD",
            nonmanifold_boundary_mode="FLAT",
        )
        normal = read_normal(obj)
        select_faces(obj, normal[:, -1] > 0.9)
        with butil.ViewportMode(obj, "EDIT"):
            bpy.ops.mesh.extrude_region_move(
                TRANSFORM_OT_translate={
                    "value": (0, 0, self.thickness + self.extrude_height)
                }
            )
        x, y, z = read_co(obj).T
        write_co(obj, np.stack([x, y, np.clip(z, None, self.extrude_height)], -1))
        return obj

    def make_seat(self, obj):
        seat = self.make_plane(obj)
        cover = deep_clone_obj(seat)
        butil.modify_mesh(seat, "SOLIDIFY", thickness=self.extrude_height, offset=1)
        if self.has_seat_cut:
            cutter = new_cube()
            cutter.scale = [self.thickness] * 3
            cutter.location = 0, -self.thickness / 2 - self.size_mid * self.size, 0
            butil.apply_transform(cutter, True)
            butil.select_none()
            butil.modify_mesh(seat, "BOOLEAN", object=cutter, operation="DIFFERENCE")
            butil.delete(cutter)
        butil.modify_mesh(seat, "BEVEL", segments=2)

        x, y, _ = read_edge_center(cover).T
        i = np.argmin(np.abs(x) + np.abs(y))
        selection = np.full(len(x), False)
        selection[i] = True
        select_edges(cover, selection)
        with butil.ViewportMode(cover, "EDIT"):
            bpy.ops.mesh.loop_multi_select()
            bpy.ops.mesh.fill_grid()
        butil.modify_mesh(cover, "SOLIDIFY", thickness=self.extrude_height, offset=1)
        cover.location = [
            0,
            -self.mid_offset - self.seat_size + self.extrude_height / 2,
            -self.extrude_height / 2,
        ]
        butil.apply_transform(cover, True)
        cover.rotation_euler[0] = self.cover_rotation
        cover.location = [
            0,
            self.mid_offset + self.seat_size - self.extrude_height / 2,
            self.extrude_height * 1.5,
        ]
        butil.apply_transform(cover, True)
        butil.modify_mesh(cover, "BEVEL", segments=2)
        return seat, cover

    def make_plane(self, obj):
        select_faces(obj, lambda x, y, z: z > self.extrude_height * 2 / 3)
        with butil.ViewportMode(obj, "EDIT"):
            bpy.ops.mesh.duplicate_move()
            bpy.ops.mesh.separate(type="SELECTED")
        seat = next(o for o in bpy.context.selected_objects if o != obj)
        butil.select_none()
        select_vertices(seat, lambda x, y, z: y > self.mid_offset + self.seat_thickness)
        with butil.ViewportMode(seat, "EDIT"):
            bpy.ops.mesh.extrude_edges_move(
                TRANSFORM_OT_translate={
                    "value": (0, self.seat_size + self.thickness * 2, 0)
                }
            )
        x, y, z = read_co(seat).T
        write_co(
            seat,
            np.stack([x, np.clip(y, None, self.mid_offset + self.seat_size), z], -1),
        )
        return seat

    def make_stand(self, obj, bottom):
        co = read_co(obj)[read_edges(obj).reshape(-1)].reshape(-1, 2, 3)
        horizontal = np.abs(normalize(co[:, 0] - co[:, 1])[:, -1]) < 0.1
        x, y, z = read_edge_center(obj).T
        under_depth = z < -self.stand_depth
        i = np.argmin(y - horizontal - under_depth)
        selection = np.full(len(co), False)
        selection[i] = True
        select_edges(obj, selection)
        with butil.ViewportMode(obj, "EDIT"):
            bpy.ops.mesh.loop_multi_select()
            bpy.ops.mesh.duplicate_move()
            bpy.ops.mesh.separate(type="SELECTED")
        stand = next(o for o in bpy.context.selected_objects if o != obj)
        stand = join_objects([stand, bottom])
        with butil.ViewportMode(stand, "EDIT"):
            bpy.ops.mesh.select_mode(type="EDGE")
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.bridge_edge_loops(
                number_cuts=64,
                profile_shape_factor=uniform(0.0, 0.15),
            )
        return stand

    def make_back(self, obj):
        back = read_center(obj)[:, 1] > self.mid_offset - self.back_thickness
        back_facing = read_normal(obj)[:, 1] > 0.1
        butil.select_none()
        select_faces(obj, back & back_facing)
        with butil.ViewportMode(obj, "EDIT"):
            bpy.ops.mesh.region_to_loop()
            bpy.ops.mesh.duplicate_move()
            bpy.ops.mesh.separate(type="SELECTED")
        back = next(o for o in bpy.context.selected_objects if o != obj)
        butil.modify_mesh(back, "CORRECTIVE_SMOOTH")
        butil.select_none()
        with butil.ViewportMode(back, "EDIT"):
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.extrude_edges_move(
                TRANSFORM_OT_translate={
                    "value": (0, self.back_size + self.thickness * 2, 0)
                }
            )
            bpy.ops.transform.resize(value=(self.back_scale, 1, 1))
            bpy.ops.mesh.edge_face_add()
        back.location[1] -= 0.01
        butil.apply_transform(back, True)
        x, y, z = read_co(back).T
        write_co(
            back,
            np.stack([x, np.clip(y, None, self.mid_offset + self.back_size), z], -1),
        )
        return back

    def make_tank(self):
        tank = new_cube()
        tank.scale = self.tank_width / 2, self.tank_size / 2, self.tank_height / 2
        tank.location = (
            0,
            self.mid_offset + self.back_size - self.tank_size / 2,
            self.tank_height / 2,
        )
        butil.apply_transform(tank, True)
        subsurf(tank, 2, True)
        butil.modify_mesh(tank, "BEVEL", segments=2)
        cap = new_cube()
        cap.scale = (
            self.tank_width / 2 + self.tank_cap_extrude,
            self.tank_size / 2 + self.tank_cap_extrude,
            self.tank_cap_height / 2,
        )
        cap.location = (
            0,
            self.mid_offset + self.back_size - self.tank_size / 2,
            self.tank_height,
        )
        butil.apply_transform(cap, True)
        butil.modify_mesh(
            cap, "BEVEL", width=uniform(0, self.extrude_height), segments=4
        )
        tank = join_objects([tank, cap])
        return tank

    def add_button(self):
        obj = new_cylinder()
        obj.scale = (
            self.hardware_radius,
            self.hardware_radius,
            self.tank_cap_height / 2 + 1e-3,
        )
        obj.location = (
            0,
            self.mid_offset + self.back_size - self.tank_size / 2,
            self.tank_height,
        )
        butil.apply_transform(obj, True)
        return obj

    def add_handle(self):
        obj = new_cylinder()
        obj.scale = self.hardware_radius, self.hardware_radius, self.hardware_cap
        obj.rotation_euler[0] = np.pi / 2
        butil.apply_transform(obj, True)
        lever = new_cylinder()
        lever.scale = (
            self.hardware_radius / 2,
            self.hardware_radius / 2,
            self.hardware_length,
        )
        lever.rotation_euler[1] = np.pi / 2
        lever.location = [
            -self.hardware_radius * uniform(0, 0.5),
            -self.hardware_cap,
            -self.hardware_radius * uniform(0, 0.5),
        ]
        butil.apply_transform(lever, True)
        obj = join_objects([obj, lever])
        if self.hardware_on_side:
            obj.location = [
                -self.tank_width / 2 + self.hardware_radius + uniform(0.01, 0.02),
                self.mid_offset + self.back_size - self.tank_size,
                self.tank_height - self.hardware_radius - uniform(0.02, 0.03),
            ]
        else:
            obj.location = [
                -self.tank_width / 2,
                self.mid_offset
                + self.back_size
                - self.tank_size
                + self.hardware_radius
                + uniform(0.01, 0.02),
                self.tank_height - self.hardware_radius - uniform(0.02, 0.03),
            ]
            obj.rotation_euler[-1] = -np.pi / 2
        butil.apply_transform(obj, True)
        butil.modify_mesh(obj, "BEVEL", width=uniform(0.005, 0.01), segments=2)
        return obj

    def finalize_assets(self, assets):
        self.surface.apply(assets, clear=True, metal_color="plain")
        self.hardware_surface.apply(assets, "hardware", metal_color="natural")
        if self.scratch:
            self.scratch.apply(assets)
        if self.edge_wear:
            self.edge_wear.apply(assets)
