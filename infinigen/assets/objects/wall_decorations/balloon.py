# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei
import bpy
import numpy as np
from numpy.random import uniform

from infinigen.assets.material_assignments import AssetList
from infinigen.assets.scatters import clothes
from infinigen.assets.utils.decorate import subdivide_edge_ring, subsurf
from infinigen.assets.utils.draw import remesh_fill
from infinigen.assets.utils.misc import generate_text
from infinigen.assets.utils.object import new_bbox
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.util import blender as butil
from infinigen.core.util.math import FixedSeed


class BalloonFactory(AssetFactory):
    alpha = 0.8

    def __init__(self, factory_seed, coarse=False):
        super(BalloonFactory, self).__init__(factory_seed, coarse)
        with FixedSeed(self.factory_seed):
            self.thickness = uniform(0.06, 0.1)
            material_assignments = AssetList["BalloonFactory"]()
            self.surface = material_assignments["surface"].assign_material()
            self.rel_scale = uniform(0.2, 0.3) * 4
            self.displace = uniform(0.02, 0.04)

    def create_placeholder(self, **kwargs) -> bpy.types.Object:
        bpy.ops.object.text_add()
        obj = bpy.context.active_object

        with butil.ViewportMode(obj, "EDIT"):
            for _ in "Text":
                bpy.ops.font.delete(type="PREVIOUS_OR_SELECTION")
            text = generate_text().upper()
            bpy.ops.font.text_insert(text=text)
        with butil.SelectObjects(obj):
            bpy.ops.object.convert(target="MESH")
        obj = bpy.context.active_object
        parent = new_bbox(
            -self.thickness / 2,
            self.thickness / 2,
            0,
            self.rel_scale * len(text) * self.alpha,
            0,
            self.rel_scale * self.alpha,
        )
        obj.parent = parent
        return parent

    def create_asset(self, i, placeholder, **params) -> bpy.types.Object:
        obj = placeholder.children[0]
        obj.parent = None
        remesh_fill(obj, 0.02)
        butil.modify_mesh(obj, "SOLIDIFY", thickness=self.thickness, offset=0.5)
        subdivide_edge_ring(obj, 8, (0, 0, 1))

        clothes.cloth_sim(
            obj,
            tension_stiffness=uniform(0, 5),
            gravity=0,
            use_pressure=True,
            uniform_pressure_force=uniform(10, 20),
            vertex_group_mass="pin",
        )

        subsurf(obj, 1)
        obj.scale = [self.rel_scale] * 3
        obj.rotation_euler = np.pi / 2, 0, np.pi / 2
        butil.apply_transform(obj, True)
        butil.modify_mesh(obj, "DISPLACE", strength=self.displace)
        butil.modify_mesh(obj, "SMOOTH", iterations=5)
        return obj

    def finalize_assets(self, assets):
        self.surface.apply(assets)
