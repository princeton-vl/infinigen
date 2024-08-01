# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei
import bpy
import numpy as np
from numpy.random import uniform

from infinigen.assets.material_assignments import AssetList
from infinigen.assets.materials.art import Art
from infinigen.assets.utils.object import join_objects, new_bbox, new_plane
from infinigen.assets.utils.uv import wrap_sides
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.util import blender as butil
from infinigen.core.util.blender import deep_clone_obj
from infinigen.core.util.math import FixedSeed
from infinigen.core.util.random import log_uniform


class WallArtFactory(AssetFactory):
    def __init__(self, factory_seed, coarse=False):
        super(WallArtFactory, self).__init__(factory_seed, coarse)
        with FixedSeed(self.factory_seed):
            self.width = log_uniform(0.4, 2)
            self.height = log_uniform(0.4, 2)
            self.thickness = uniform(0.02, 0.05)
            self.depth = uniform(0.01, 0.02)
            self.frame_bevel_segments = np.random.choice([0, 1, 4])
            self.frame_bevel_width = uniform(self.depth / 4, self.depth / 2)
            self.material_assignments = AssetList["WallArtFactory"]()
            self.assign_materials()

    def assign_materials(self):
        # self.surface = Art(self.factory_seed)
        assignments = self.material_assignments
        self.surface = assignments["surface"].assign_material()
        if self.surface == Art:
            self.surface = self.surface(self.factory_seed)
        self.frame_surface = assignments["frame"].assign_material()
        is_scratch = uniform() < assignments["wear_tear_prob"][0]
        is_edge_wear = uniform() < assignments["wear_tear_prob"][1]
        self.scratch = assignments["wear_tear"][0] if is_scratch else None
        self.edge_wear = assignments["wear_tear"][1] if is_edge_wear else None

    def create_placeholder(self, **params):
        return new_bbox(
            -0.01,
            0.15,
            -self.width / 2 - self.thickness,
            self.width / 2 + self.thickness,
            -self.height / 2 - self.thickness,
            self.height / 2 + self.thickness,
        )

    def create_asset(self, placeholder, **params) -> bpy.types.Object:
        obj = new_plane()
        obj.scale = self.width / 2, self.height / 2, 1
        obj.rotation_euler = np.pi / 2, 0, np.pi / 2
        butil.apply_transform(obj, True)

        frame = deep_clone_obj(obj)
        wrap_sides(obj, self.surface, "x", "y", "z")
        butil.select_none()
        with butil.ViewportMode(frame, "EDIT"):
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.delete(type="ONLY_FACE")
        butil.modify_mesh(frame, "SOLIDIFY", thickness=self.thickness, offset=1)
        with butil.ViewportMode(frame, "EDIT"):
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.bridge_edge_loops()
        butil.modify_mesh(frame, "SOLIDIFY", thickness=self.depth, offset=1)
        if self.frame_bevel_segments > 0:
            butil.modify_mesh(
                frame,
                "BEVEL",
                width=self.frame_bevel_width,
                segments=self.frame_bevel_segments,
            )
        self.frame_surface.apply(frame)
        obj = join_objects([obj, frame])
        return obj

    def finalize_assets(self, assets):
        if self.scratch:
            self.scratch.apply(assets)
        if self.edge_wear:
            self.edge_wear.apply(assets)


class MirrorFactory(WallArtFactory):
    def __init__(self, factory_seed, coarse=False):
        super(MirrorFactory, self).__init__(factory_seed, coarse)
        self.material_assignments = AssetList["MirrorFactory"]()
        self.assign_materials()
