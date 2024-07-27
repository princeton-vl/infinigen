# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei
import bpy
import numpy as np
from numpy.random import uniform

from infinigen.assets.material_assignments import AssetList
from infinigen.assets.utils.decorate import read_center, subsurf, write_co
from infinigen.assets.utils.draw import spin
from infinigen.assets.utils.object import join_objects, new_cylinder, new_line
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.util import blender as butil
from infinigen.core.util.math import FixedSeed


class LidFactory(AssetFactory):
    def __init__(self, factory_seed, coarse=False):
        super(LidFactory, self).__init__(factory_seed, coarse)
        with FixedSeed(self.factory_seed):
            self.x_length = uniform(0.08, 0.15)
            self.z_height = self.x_length * uniform(0, 0.5)
            self.thickness = uniform(0.003, 0.005)
            self.is_glass = uniform() < 0.5
            self.hardware_type = None
            self.rim_height = uniform(1, 2) * self.thickness
            self.handle_type = np.random.choice(["handle", "knob"])
            if self.handle_type == "knob":
                self.handle_height = self.x_length * uniform(0.1, 0.15)
            else:
                self.handle_height = self.x_length * uniform(0.2, 0.25)
            self.handle_radius = self.x_length * uniform(0.15, 0.25)
            self.handle_width = self.x_length * uniform(0.25, 0.3)
            self.handle_subsurf_level = np.random.randint(0, 3)

            if self.is_glass:
                material_assignments = AssetList["GlassLidFactory"]()
            else:
                material_assignments = AssetList["LidFactory"]()
            self.surface = material_assignments["surface"].assign_material()
            self.rim_surface = material_assignments["rim_surface"].assign_material()
            self.handle_surface = material_assignments[
                "handle_surface"
            ].assign_material()

            scratch_prob, edge_wear_prob = material_assignments["wear_tear_prob"]
            self.scratch, self.edge_wear = material_assignments["wear_tear"]
            self.scratch = None if uniform() > scratch_prob else self.scratch
            self.edge_wear = None if uniform() > edge_wear_prob else self.edge_wear

    def create_asset(self, **params) -> bpy.types.Object:
        x_anchors = 0, 0.01, self.x_length / 2, self.x_length
        z_anchors = self.z_height, self.z_height, self.z_height * uniform(0.7, 0.8), 0
        obj = spin((x_anchors, 0, z_anchors))
        butil.modify_mesh(obj, "SOLIDIFY", thickness=self.thickness, offset=0)
        butil.modify_mesh(obj, "BEVEL", width=self.thickness / 2, segments=4)
        self.surface.apply(
            obj, clear=True if self.is_glass else None, metal_color="bw+natural"
        )
        parts = [obj]
        if self.is_glass:
            parts.append(self.add_rim())
        match self.handle_type:
            case "handle":
                parts.append(self.add_handle(obj))
            case _:
                parts.append(self.add_knob())
        obj = join_objects(parts)
        return obj

    def add_rim(self):
        butil.select_none()
        bpy.ops.mesh.primitive_torus_add(
            major_radius=self.x_length,
            minor_radius=self.thickness / 2,
            major_segments=128,
        )
        obj = bpy.context.active_object
        obj.scale[-1] = self.rim_height / self.thickness
        butil.apply_transform(obj)
        self.rim_surface.apply(obj)
        return obj

    def add_handle(self, obj):
        center = read_center(obj)
        i = np.argmin(
            np.abs(center[:, :2] - np.array([self.handle_width, 0])[np.newaxis, :]).sum(
                -1
            )
        )
        z_offset = center[i, -1]
        obj = new_line(3)
        write_co(
            obj,
            np.array(
                [
                    [-self.handle_width, 0, 0],
                    [-self.handle_width, 0, self.handle_height],
                    [self.handle_width, 0, self.handle_height],
                    [self.handle_width, 0, 0],
                ]
            ),
        )
        subsurf(obj, self.handle_subsurf_level)
        butil.select_none()
        with butil.ViewportMode(obj, "EDIT"):
            bpy.ops.mesh.select_mode(type="EDGE")
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.extrude_edges_move(
                TRANSFORM_OT_translate={"value": (0, self.thickness * 2, 0)}
            )
        butil.modify_mesh(obj, "SOLIDIFY", thickness=self.thickness, offset=0)
        butil.modify_mesh(obj, "BEVEL", width=self.thickness / 2, segments=4)
        obj.location = 0, -self.thickness, z_offset
        butil.apply_transform(obj, True)
        self.handle_surface.apply(obj)
        return obj

    def add_knob(self):
        obj = new_cylinder()
        obj.scale = *([self.thickness * uniform(1, 2)] * 2), self.handle_height
        obj.location[-1] = self.z_height
        butil.apply_transform(obj, True)
        butil.modify_mesh(obj, "BEVEL", width=self.thickness / 2, segments=4)
        top = new_cylinder()
        top.scale = (
            self.handle_radius,
            self.handle_radius,
            self.thickness * uniform(1, 2),
        )
        top.location[-1] = self.z_height + self.handle_height
        butil.apply_transform(top, True)
        butil.modify_mesh(top, "BEVEL", width=self.thickness / 2, segments=4)
        obj = join_objects([obj, top])
        self.handle_surface.apply(obj)
        return obj

    def finalize_assets(self, assets):
        if self.scratch:
            self.scratch.apply(assets)
        if self.edge_wear:
            self.edge_wear.apply(assets)
