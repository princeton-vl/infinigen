# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Beining Han


import numpy as np
import bpy
from infinigen.assets.trees.utils import mesh
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.util import blender as butil

C = bpy.context
D = bpy.data
from infinigen.assets.utils.tag import tag_object, tag_nodegroup

class LeafHeartFactory(AssetFactory):
    scale = 0.2

    def __init__(self, factory_seed, genome: dict = None, coarse=False):
        super(LeafHeartFactory, self).__init__(factory_seed, coarse=coarse)
        self.genome = dict(
            leaf_width=1.0,
            use_wave=True,
            z_scaling=0,
            width_rand=0.1
        )
        if genome:
            for k, g in genome.items():
                assert k in self.genome
                self.genome[k] = g

    def create_asset(self, **params) -> bpy.types.Object:

        # bpy.ops.object.mode_set(mode = 'OBJECT')
        bpy.ops.mesh.primitive_circle_add(enter_editmode=False, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
        bpy.ops.object.editmode_toggle()
        bpy.ops.mesh.edge_face_add()

        obj = bpy.context.active_object
        n = len(obj.data.vertices) // 2

        # define origin point
        mesh.select_vtx_by_idx(obj, [0, -1], deselect=True)
        bpy.ops.mesh.subdivide()

        a = np.linspace(0, np.pi, n)
        x = 16. * (np.sin(a - np.pi) ** 3) * (self.genome['leaf_width'] + np.random.randn() * self.genome['width_rand'])
        y = 13. * np.cos(a - np.pi) - 5 * np.cos(2 * (a - np.pi)) - 2 * np.cos(3 * (a - np.pi))
        x, y = x * 0.3, y * 0.3
        z = x ** 2 * self.genome['z_scaling']

        full_coords = np.concatenate([np.stack([x, y, z], 1), np.stack([-x[::-1], y[::-1], z], 1),
                                      np.array([[0, y[0], 0]])]).flatten()
        bpy.ops.object.mode_set(mode='OBJECT')
        obj.data.vertices.foreach_set('co', full_coords)

        if self.genome["use_wave"]:
            bpy.ops.object.modifier_add(type='WAVE')
            bpy.context.object.modifiers["Wave"].height = 0.8 * np.random.randn() * 0.8
            bpy.context.object.modifiers["Wave"].width = 3.5 + np.random.randn() * 1.
            bpy.context.object.modifiers["Wave"].speed = 40 + np.random.uniform(-10, 20)

        mesh.finalize_obj(obj)
        C.scene.cursor.location = obj.data.vertices[-1].co

        bpy.ops.object.origin_set(type="ORIGIN_CURSOR")

        obj.location = (0, 0, 0)
        obj.scale *= self.scale
        butil.apply_transform(obj)
        tag_object(obj, 'leaf_heart')

        return obj


# if __name__ == '__main__':
#     leaf = LeafHeartFactory(factory_seed=0)
#     leaf.create_asset()


