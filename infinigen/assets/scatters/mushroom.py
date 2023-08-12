# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei


from collections.abc import Iterable

import bmesh
import bpy
import numpy as np
from mathutils import Matrix
from numpy.random import uniform

from infinigen.assets.mushroom import MushroomFactory
from infinigen.core.util import blender as butil
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core import surface


def geo_skeleton(nw: NodeWrangler, base_obj, selection, threshold=.05):
    geometry = nw.new_node(Nodes.ObjectInfo, [base_obj], attrs={'transform_space': 'RELATIVE'}).outputs[
        'Geometry']
    selection = surface.eval_argument(nw, selection)
    geometry = nw.new_node(Nodes.SeparateGeometry, [geometry, selection])
    geometry = nw.new_node(Nodes.MergeByDistance, [geometry, None, threshold])
    nw.new_node(Nodes.GroupOutput, input_kwargs={'Geometry': geometry})


def apply(objs, selection=None, **kwargs):
    factories = [MushroomFactory(np.random.randint(1e5)) for _ in range(3)]
    mushroom_keypoints = [f.build_mushrooms(np.random.randint(1e5)) for f in factories]
    base_mushrooms = [m for m, k in mushroom_keypoints]
    scattered_objects = []

    if not isinstance(objs, list):
        objs = [objs]
    if len(objs) == 0:
        return
    selections = selection if isinstance(selection, Iterable) else [selection] * len(objs)

    for obj, selection in zip(objs,selections):
        temp_obj = butil.spawn_vert('temp')
        surface.add_geomod(temp_obj, geo_skeleton, apply=True, input_args=[obj, selection])
        with butil.ViewportMode(temp_obj, 'EDIT'):
            bm = bmesh.from_edit_mesh(temp_obj.data)
            bm.verts.ensure_lookup_table()
            selected = np.random.choice(bm.verts, np.random.randint(2, 5))
            rotations, start_locs, directions = [], [], []
            for v in selected:
                normal_ratio = uniform(.4, .6)
                v: bmesh.types.BMVert
                for e in v.link_edges:
                    obj = e.other_vert(v)
                    if len(e.link_faces) == 2:
                        direction = np.array(obj.co - v.co)
                        direction = direction / np.linalg.norm(direction)
                        normal = np.mean(np.array([f.normal for f in e.link_faces]),
                                         0) * normal_ratio + np.array(
                            [0, 0, 1 - normal_ratio]) + direction * uniform(.2, .5)
                        normal = normal / np.linalg.norm(normal)
                        perp_direction = direction - np.dot(direction, normal) * normal
                        perp_direction = perp_direction / np.linalg.norm(perp_direction)
                        rotation = np.array(Matrix(np.stack([perp_direction, np.cross(normal, perp_direction),
                                                                normal])).transposed().to_euler())
                        rotations.append(rotation)
                        start_locs.append(np.array(v.co))
                        directions.append(direction)
        butil.delete(temp_obj)

        factory_index = np.random.randint(0, len(factories))
        mushrooms, keypoints = mushroom_keypoints[factory_index]
        indices = np.random.randint(0, len(mushrooms), len(rotations))
        augmented = [keypoints[i] for i in indices]
        locations, rotations, scales = factories[factory_index].find_closest(augmented, rotations, start_locs,
                                                                             directions)

        scatter_obj = butil.spawn_vert('asset:mushroom')
        for i, l, r, s in zip(indices, locations, rotations, scales):
            with butil.SelectObjects(mushrooms[i]):
                bpy.ops.object.duplicate(linked=True)
                objs = bpy.context.active_object
            objs.location = l
            objs.rotation_euler = r
            objs.scale = s
            objs.parent = scatter_obj
        scattered_objects.append(scatter_obj)

    col = butil.group_in_collection(base_mushrooms, name=f'assets:base_mushroom', reuse=False)
    col.hide_viewport = True
    col.hide_render = True
    return scattered_objects, col
