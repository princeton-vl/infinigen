# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Beining Han

import pdb

import numpy as np

import bpy

from infinigen.assets.trees.utils import helper, mesh, materials

from infinigen.core.placement.factory import AssetFactory
from infinigen.core.util import blender as butil

C = bpy.context
D = bpy.data
from infinigen.assets.utils.tag import tag_object, tag_nodegroup

class LeafFactory(AssetFactory):
    
    scale = 0.3

    def __init__(self, factory_seed, genome: dict=None, coarse=False):
        super(LeafFactory, self).__init__(factory_seed, coarse=coarse)
        self.genome = dict(
            leaf_width=0.5,
            alpha=0.3,
            use_wave=True,
            x_offset=0,
            flip_leaf=False,
            z_scaling=0,
            width_rand=0.33
        )
        if genome:
            for k, g in genome.items():
                assert k in self.genome
                self.genome[k] = g

    def create_asset(self, **params) -> bpy.types.Object:

        # bpy.ops.object.mode_set(mode = 'OBJECT')
        bpy.ops.mesh.primitive_circle_add(enter_editmode=False, align='WORLD',
                                        location=(0, 0, 0), scale=(1, 1, 1))
        bpy.ops.object.editmode_toggle()
        bpy.ops.mesh.edge_face_add()

        obj = bpy.context.active_object
        min_radius = .02
        radii_ref = [1]
        n = len(obj.data.vertices) // 2

        # define origin point
        mesh.select_vtx_by_idx(obj, [0, -1], deselect=True)
        bpy.ops.mesh.subdivide()

        a = np.linspace(0, np.pi, n)
        if self.genome['flip_leaf']:
            a = a[::-1]
        x = np.sin(a) * (self.genome['leaf_width'] + np.random.randn() * self.genome['width_rand']) + self.genome['x_offset']
        y = -np.cos(.9 * (a - self.genome['alpha']))
        z = x ** 2 * self.genome['z_scaling']

        full_coords = np.concatenate([np.stack([x, y, z], 1),
                                    np.stack([-x[::-1], y[::-1], z], 1),
                                    np.array([[0, y[0], 0]])]).flatten()
        bpy.ops.object.mode_set(mode='OBJECT')
        obj.data.vertices.foreach_set('co', full_coords)

        if self.genome['use_wave']:
            bpy.ops.object.modifier_add(type='WAVE')
            bpy.context.object.modifiers["Wave"].height = np.random.randn() * .3
            bpy.context.object.modifiers["Wave"].width = 0.75 + \
                np.random.randn() * .1
            bpy.context.object.modifiers["Wave"].speed = np.random.rand()

        mesh.finalize_obj(obj)
        C.scene.cursor.location = obj.data.vertices[-1].co

        bpy.ops.object.origin_set(type="ORIGIN_CURSOR")

        obj.location = (0, 0, 0)
        obj.scale *= self.scale
        butil.apply_transform(obj)
        tag_object(obj, 'leaf')
        return obj


if __name__ == '__main__':
    leaf = LeafFactory(factory_seed=0)
    leaf.create_asset()