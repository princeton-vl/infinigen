# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors:
# - Lingjie Mei: primary author

import bpy
import numpy as np
from numpy.random import uniform

from infinigen.assets.materials import glass, metal, wood
from infinigen.assets.materials.common import unique_surface
from infinigen.assets.utils.autobevel import BevelSharp
from infinigen.assets.utils.decorate import mirror, read_co, write_attribute, write_co
from infinigen.assets.utils.draw import spin
from infinigen.assets.utils.nodegroup import geo_radius
from infinigen.assets.utils.object import (
    data2mesh,
    join_objects,
    mesh2obj,
    new_cube,
    new_line,
)
from infinigen.core import surface
from infinigen.core.constraints.constraint_language.constants import RoomConstants
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.util import blender as butil
from infinigen.core.util.bevelling import add_bevel, get_bevel_edges
from infinigen.core.util.blender import deep_clone_obj
from infinigen.core.util.math import FixedSeed
from infinigen.core.util.random import log_uniform


class BaseDoorFactory(AssetFactory):
    def __init__(self, factory_seed, coarse=False, constants=None):
        super(BaseDoorFactory, self).__init__(factory_seed, coarse)
        with FixedSeed(self.factory_seed):
            if constants is None:
                constants = RoomConstants()
            self.width = constants.door_width
            self.height = constants.door_size
            self.constants = constants
            self.depth = constants.wall_thickness * log_uniform(0.25, 0.5)
            self.panel_margin = log_uniform(0.08, 0.12)
            self.bevel_width = uniform(0.005, 0.01)
            self.out_bevel = uniform() < 0.7
            self.shrink_width = log_uniform(0.005, 0.06)

            surface_fn = np.random.choice([metal, wood], p=[0.2, 0.8])
            self.surface = unique_surface(surface_fn, self.factory_seed)
            self.has_glass = False
            self.glass_surface = glass
            self.has_louver = False
            self.louver_surface = np.random.choice([metal, wood], p=[0.2, 0.8])

            self.handle_type = np.random.choice(["knob", "lever", "pull"])
            self.handle_surface = np.random.choice([metal, wood], p=[0.2, 0.8])
            self.handle_offset = self.panel_margin * 0.5
            self.handle_height = self.height * uniform(0.45, 0.5)

            self.knob_radius = uniform(0.03, 0.04)
            base_radius = uniform(1.1, 1.2)
            mid_radius = uniform(0.4, 0.5)
            self.knob_radius_mid = (
                base_radius,
                base_radius,
                mid_radius,
                mid_radius,
                1,
                uniform(0.6, 0.8),
                0,
            )
            self.knob_depth = uniform(0.08, 0.1)
            self.knob_depth_mid = [
                0,
                uniform(0.1, 0.15),
                uniform(0.25, 0.3),
                uniform(0.35, 0.45),
                uniform(0.6, 0.8),
                1,
                1 + 1e-3,
            ]

            self.lever_radius = uniform(0.03, 0.04)
            self.lever_mid_radius = uniform(0.01, 0.02)
            self.lever_depth = uniform(0.05, 0.08)
            self.lever_mid_depth = uniform(0.15, 0.25)
            self.lever_length = log_uniform(0.15, 0.2)
            self.level_type = np.random.choice(["wave", "cylinder", "bent"])

            self.pull_size = log_uniform(0.1, 0.4)
            self.pull_depth = uniform(0.05, 0.08)
            self.pull_width = log_uniform(0.08, 0.15)
            self.pull_extension = uniform(0.05, 0.15)
            self.to_pull_bevel = uniform() < 0.5
            self.pull_bevel_width = uniform(0.02, 0.04)
            self.pull_radius = uniform(0.01, 0.02)
            self.pull_type = np.random.choice(["u", "tee", "zed"])
            self.is_pull_circular = uniform() < 0.5 or self.pull_type == "zed"
            self.panel_surface = unique_surface(surface_fn, np.random.randint(1e5))
            self.auto_bevel = BevelSharp()
            self.side_bevel = log_uniform(0.005, 0.015)

            self.metal_color = metal.sample_metal_color()

    def create_asset(self, **params) -> bpy.types.Object:
        for _ in range(100):
            obj = self._create_asset()
            if max(obj.dimensions) < 5:
                return obj
            else:
                butil.delete(obj)
        else:
            raise ValueError("Bad door booleaning")

    def _create_asset(self):
        obj = new_cube(location=(1, 1, 1))
        butil.apply_transform(obj, loc=True)
        obj.scale = self.width / 2, self.depth / 2, self.height / 2
        butil.apply_transform(obj)
        panels = self.make_panels()
        extras = []
        for panel in panels:
            extras.extend(panel["func"](obj, panel))
        match self.handle_type:
            case "knob":
                extras.extend(self.make_knobs())
            case "lever":
                extras.extend(self.make_levers())
            case "pull":
                extras.extend(self.make_pulls())
        obj = join_objects([obj] + extras)
        self.auto_bevel(obj)
        obj.location = -self.width, -self.depth, 0
        butil.apply_transform(obj, True)
        obj = add_bevel(obj, get_bevel_edges(obj), offset=self.side_bevel)
        return obj

    def make_panels(self):
        return []

    def finalize_assets(self, assets):
        self.surface.apply(assets, metal_color=self.metal_color, vertical=True)
        if self.has_glass:
            self.glass_surface.apply(assets, selection="glass", clear=True)
        if self.has_louver:
            self.louver_surface.apply(
                assets, selection="louver", metal_color=self.metal_color
            )
        self.handle_surface.apply(assets, selection="handle", metal_color="natural")

    def make_knobs(self):
        x_anchors = np.array(self.knob_radius_mid) * self.knob_radius
        y_anchors = np.array(self.knob_depth_mid) * self.knob_depth
        obj = spin([x_anchors, y_anchors, 0], [0, 2, 3], axis=(0, 1, 0))
        with butil.ViewportMode(obj, "EDIT"):
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.region_to_loop()
            bpy.ops.mesh.edge_face_add()
        return self.make_handles(obj)

    def make_handles(self, obj):
        write_attribute(obj, 1, "handle", "FACE")
        obj.location = self.handle_offset, 0, self.handle_height
        butil.apply_transform(obj, loc=True)
        other = deep_clone_obj(obj)
        obj.location[1] += self.depth
        butil.apply_transform(obj, loc=True)
        mirror(other, 1)
        return [obj, other]

    def make_levers(self):
        x_anchors = (
            self.lever_radius,
            self.lever_radius,
            self.lever_mid_radius,
            self.lever_mid_radius,
            0,
        )
        y_anchors = (
            np.array([0, self.lever_mid_depth, self.lever_mid_depth, 1, 1 + 1e-3])
            * self.lever_depth
        )
        obj = spin([x_anchors, y_anchors, 0], [0, 1, 2, 3], axis=(0, 1, 0))
        with butil.ViewportMode(obj, "EDIT"):
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.region_to_loop()
            bpy.ops.mesh.fill()
        lever = new_line(4)
        if self.level_type == "wave":
            co = read_co(lever)
            co[1, -1] = -uniform(0.2, 0.3)
            co[3, -1] = uniform(0.1, 0.15)
            write_co(lever, co)
        elif self.level_type == "bent":
            co = read_co(lever)
            co[4, 1] = -uniform(0.2, 0.3)
            write_co(lever, co)
        lever.scale = [self.lever_length] * 3
        butil.apply_transform(lever)
        butil.select_none()
        with butil.ViewportMode(lever, "EDIT"):
            bpy.ops.mesh.select_mode(type="EDGE")
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.extrude_edges_move(
                TRANSFORM_OT_translate={"value": (0, 0, self.lever_mid_radius * 2)}
            )
        butil.modify_mesh(
            lever, "SOLIDIFY", lever, thickness=self.lever_mid_radius, offset=0
        )
        butil.modify_mesh(lever, "SUBSURF", render_levels=1, levels=1)
        lever.location = (
            -self.lever_mid_radius,
            self.lever_depth,
            -self.lever_mid_radius,
        )
        butil.apply_transform(lever, loc=True)
        obj = join_objects([obj, lever])
        return self.make_handles(obj)

    def make_pulls(self):
        if self.pull_type == "u":
            vertices = (
                (0, 0, self.pull_size),
                (0, self.pull_depth, self.pull_size),
                (0, self.pull_depth, 0),
            )
            edges = (0, 1), (1, 2)
        elif self.pull_type == "tee":
            vertices = (
                (0, 0, self.pull_size),
                (0, self.pull_depth, self.pull_size),
                (0, self.pull_depth, 0),
                (0, self.pull_depth, self.pull_size + self.pull_extension),
            )
            edges = (0, 1), (1, 2), (1, 3)
        else:
            vertices = (
                (0, 0, self.pull_size),
                (0, self.pull_depth, self.pull_size),
                (self.pull_width, self.pull_depth, self.pull_size),
                (self.pull_width, self.pull_depth, 0),
            )
            edges = (0, 1), (1, 2), (2, 3)
        obj = mesh2obj(data2mesh(vertices, edges))
        butil.modify_mesh(obj, "MIRROR", use_axis=(False, False, True))
        if self.to_pull_bevel:
            butil.modify_mesh(
                obj, "BEVEL", width=self.pull_bevel_width, segments=4, affect="VERTICES"
            )
        if self.is_pull_circular:
            surface.add_geomod(
                obj,
                geo_radius,
                apply=True,
                input_args=[self.pull_radius, 32],
                input_kwargs={"to_align_tilt": False},
            )
        else:
            with butil.ViewportMode(obj, "EDIT"):
                bpy.ops.mesh.select_mode(type="EDGE")
                bpy.ops.mesh.select_all(action="SELECT")
                bpy.ops.mesh.extrude_edges_move(
                    TRANSFORM_OT_translate={"value": (self.pull_radius * 2, 0, 0)}
                )
                bpy.ops.mesh.select_all(action="SELECT")
                bpy.ops.mesh.normals_make_consistent(inside=False)
            obj.location = -self.pull_radius, -self.pull_radius, -self.pull_radius
            butil.apply_transform(obj, loc=True)
            butil.modify_mesh(obj, "SOLIDIFY", thickness=self.pull_radius * 2, offset=0)
        return self.make_handles(obj)

    @property
    def casing_factory(self):
        from infinigen.assets.objects.elements import DoorCasingFactory

        factory = DoorCasingFactory(self.factory_seed, self.coarse, self.constants)
        factory.surface = self.surface
        factory.metal_color = self.metal_color
        return factory
