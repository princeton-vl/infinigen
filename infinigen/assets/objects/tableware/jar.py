# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

import bmesh

# Authors: Lingjie Mei
import bpy
import numpy as np
from numpy.random import uniform

from infinigen.assets.material_assignments import AssetList
from infinigen.assets.utils.decorate import subsurf, write_attribute
from infinigen.assets.utils.object import join_objects, new_circle, new_cylinder
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.util import blender as butil
from infinigen.core.util.math import FixedSeed


class JarFactory(AssetFactory):
    def __init__(self, factory_seed, coarse=False):
        super().__init__(factory_seed, coarse)
        with FixedSeed(self.factory_seed):
            self.z_length = uniform(0.15, 0.2)
            self.x_length = uniform(0.03, 0.06)
            self.thickness = uniform(0.002, 0.004)
            self.n_base = np.random.choice([4, 6, 64])
            self.x_cap = uniform(0.6, 0.9) * np.cos(np.pi / self.n_base)
            self.z_cap = uniform(0.05, 0.08)
            self.z_neck = uniform(0.15, 0.2)

            material_assignments = AssetList["JarFactory"]()
            self.surface = material_assignments["surface"].assign_material()
            self.cap_surface = material_assignments["cap_surface"].assign_material()
            scratch_prob, edge_wear_prob = material_assignments["wear_tear_prob"]
            self.scratch, self.edge_wear = material_assignments["wear_tear"]
            self.scratch = None if uniform() > scratch_prob else self.scratch
            self.edge_wear = None if uniform() > edge_wear_prob else self.edge_wear

            self.cap_subsurf = uniform() < 0.5

    def create_asset(self, **params) -> bpy.types.Object:
        obj = new_cylinder(vertices=self.n_base)
        obj.scale = self.x_length, self.x_length, self.z_length
        butil.apply_transform(obj, True)
        with butil.ViewportMode(obj, "EDIT"):
            bm = bmesh.from_edit_mesh(obj.data)
            geom = [f for f in bm.faces if f.normal[-1] > 0.5]
            bmesh.ops.delete(bm, geom=geom, context="FACES_KEEP_BOUNDARY")
            bmesh.update_edit_mesh(obj.data)
            bpy.ops.mesh.select_mode(type="EDGE")
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.region_to_loop()
        subsurf(obj, 2, True)
        top = new_circle(location=(0, 0, 0))
        top.scale = [self.x_cap * self.x_length] * 3
        top.location[-1] = (1 + self.z_neck) * self.z_length
        butil.apply_transform(top)
        butil.select_none()
        obj = join_objects([obj, top])
        with butil.ViewportMode(obj, "EDIT"):
            bpy.ops.mesh.select_mode(type="EDGE")
            bpy.ops.mesh.bridge_edge_loops(
                number_cuts=5, profile_shape_factor=uniform(0, 0.1)
            )
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.region_to_loop()
            bpy.ops.mesh.extrude_edges_move(
                TRANSFORM_OT_translate={"value": (0, 0, self.z_cap * self.z_length)}
            )
        subsurf(obj, 2)
        butil.modify_mesh(obj, "SOLIDIFY", thickness=self.thickness)

        cap = new_cylinder(vertices=64)
        cap.scale = (
            *([self.x_cap * self.x_length + 1e-3] * 2),
            self.z_cap * self.z_length,
        )
        cap.location[-1] = (
            1 + self.z_neck + self.z_cap * uniform(0.5, 0.8)
        ) * self.z_length
        butil.apply_transform(cap, True)
        subsurf(obj, 1, self.cap_subsurf)
        write_attribute(cap, 1, "cap", "FACE")
        obj = join_objects([obj, cap])
        return obj

    def finalize_assets(self, assets):
        self.surface.apply(assets, clear=uniform() < 0.5)
        self.cap_surface.apply(assets, selection="cap")
        if self.scratch:
            self.scratch.apply(assets)
        if self.edge_wear:
            self.edge_wear.apply(assets)
