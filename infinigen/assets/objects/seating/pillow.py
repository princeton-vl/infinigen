# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei
import bpy
import numpy as np
from numpy.random import uniform

from infinigen.assets.material_assignments import AssetList
from infinigen.assets.materials import art, fabrics
from infinigen.assets.scatters import clothes
from infinigen.assets.utils.decorate import (
    read_normal,
    read_selected,
    select_faces,
    set_shade_smooth,
    subsurf,
)
from infinigen.assets.utils.object import (
    center,
    join_objects,
    new_base_circle,
    new_grid,
)
from infinigen.assets.utils.uv import unwrap_faces
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.util import blender as butil
from infinigen.core.util.random import log_uniform
from infinigen.core.util.random import random_general as rg


class PillowFactory(AssetFactory):
    shapes = (
        "weighted_choice",
        (4, "square"),
        (4, "rectangle"),
        (1, "circle"),
        (1, "torus"),
    )

    def __init__(self, factory_seed, coarse=False):
        super(PillowFactory, self).__init__(factory_seed, coarse)
        self.shape = rg(self.shapes)
        self.width = uniform(0.4, 0.7)
        match self.shape:
            case "square":
                self.size = self.width
            case _:
                self.size = self.width * log_uniform(0.6, 0.8)
        self.bevel_width = uniform(0.02, 0.05)
        self.thickness = log_uniform(0.006, 0.008)
        self.extrude_thickness = (
            self.thickness * log_uniform(1, 8) if uniform() < 0.5 else 0
        )
        self.surface = np.random.choice(
            [art.ArtFabric(self.factory_seed), fabrics.fabric_random]
        )
        self.has_seam = uniform() < 0.3 and not self.shape == "torus"
        self.seam_radius = uniform(0.01, 0.02)

        materials = AssetList["PillowFactory"]()
        self.surface = materials["surface"].assign_material()
        if self.surface == art.ArtFabric:
            self.surface = self.surface(self.factory_seed)

    def create_asset(self, **params) -> bpy.types.Object:
        match self.shape:
            case "circle":
                obj = new_base_circle(vertices=128)
                with butil.ViewportMode(obj, "EDIT"):
                    bpy.ops.mesh.fill_grid()
            case "torus":
                obj = new_base_circle(vertices=128)
                inner = new_base_circle(vertices=128, radius=uniform(0.2, 0.4))
                obj = join_objects([obj, inner])
                with butil.ViewportMode(obj, "EDIT"):
                    bpy.ops.mesh.select_all(action="SELECT")
                    bpy.ops.mesh.bridge_edge_loops(
                        number_cuts=12, interpolation="LINEAR"
                    )
                obj = bpy.context.active_object
            case _:
                obj = new_grid(x_subdivisions=32, y_subdivisions=32)
        obj.scale = self.width / 2, self.size / 2, 1
        butil.apply_transform(obj, True)
        unwrap_faces(obj)
        butil.modify_mesh(obj, "SOLIDIFY", thickness=self.thickness, offset=0)
        normal = read_normal(obj)

        group = obj.vertex_groups.new(name="pin")
        if self.has_seam:
            with butil.ViewportMode(obj, "EDIT"):
                bpy.ops.mesh.select_mode(type="FACE")
                select_faces(
                    obj, lambda x, y, z: (x**2 + y**2 < self.seam_radius**2) & (z > 0)
                )
                bpy.ops.mesh.region_to_loop()
                bpy.ops.mesh.select_mode(type="VERT")
            selection = read_selected(obj)
            group.add(np.nonzero(selection)[0].tolist(), 1, "REPLACE")
        select_faces(obj, np.abs(normal[:, -1]) < 0.1)

        match self.shape:
            case "torus":
                pressure = uniform(8, 12)
            case _:
                pressure = uniform(1, 2)
        clothes.cloth_sim(
            obj,
            tension_stiffness=uniform(0, 5),
            gravity=0,
            use_pressure=True,
            uniform_pressure_force=pressure,
            vertex_group_mass="pin" if self.has_seam else "",
        )
        if self.extrude_thickness > 0:
            with butil.ViewportMode(obj, "EDIT"):
                bpy.ops.mesh.extrude_region_shrink_fatten(
                    TRANSFORM_OT_shrink_fatten={"value": self.extrude_thickness}
                )
        obj.location = -center(obj)
        butil.apply_transform(obj, True)
        subsurf(obj, 2)
        set_shade_smooth(obj)
        return obj

    def make_circle(self):
        obj = new_base_circle(vertices=128)
        with butil.ViewportMode(obj, "EDIT"):
            bpy.ops.mesh.fill_grid()
            select_faces(obj, lambda x, y, z: x**2 + y**2 < self.seam_radius**2)
            bpy.ops.mesh.region_to_loop()
        return obj

    def make_gird(self):
        obj = new_grid(x_subdivisions=64, y_subdivisions=64)
        with butil.ViewportMode(obj, "EDIT"):
            select_faces(
                obj,
                lambda x, y, z: (np.abs(x) < self.seam_radius)
                & (np.abs(y) < self.seam_radius),
            )
            bpy.ops.mesh.region_to_loop()
        return obj

    def finalize_assets(self, assets):
        self.surface.apply(assets)
