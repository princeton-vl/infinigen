# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei
# Date Signed: Jun 14 2023

import math
from itertools import chain
from statistics import mean

import bmesh
import numpy as np

from mathutils import Vector, kdtree, noise
from util import blender as butil


def grow_step(bm, vg_index=0, split_radius=.5, repulsion_radius=1., dt=.1, growth_scale=(1, 1, 1),
              noise_scale=2., growth_vec=(0, 0, 1), fac_attr=1., fac_rep=1., fac_noise=1., inhibit_base=1.,
              inhibit_shell=0.):
    kd = kdtree.KDTree(len(bm.verts))
    for i, vert in enumerate(bm.verts):
        kd.insert(vert.co, i)
    kd.balance()
    seed_vector = Vector((0, 0, np.random.randint(0, 1000)))
    growth_vec = Vector(growth_vec)
    growth_scale = Vector(growth_scale)

    def calc_vert_attraction(vert):
        result = Vector()
        for edge in vert.link_edges:
            result += edge.other_vert(vert).co - vert.co
        return result

    def calc_vert_repulsion(vert, radius):
        result = Vector()
        for (co, index, distance) in kd.find_range(vert.co, radius):
            if index != vert.index:
                result += (vert.co - co).normalized() * (math.exp(-1 * (distance / radius) + 1) - 1)
        return result

    for vert in bm.verts:
        w = vert[bm.verts.layers.deform.active].get(vg_index, 0)
        if w > 0:
            f_attr = calc_vert_attraction(vert)
            f_rep = calc_vert_repulsion(vert, repulsion_radius)
            f_noise = noise.noise_vector(vert.co * noise_scale + seed_vector)
            force = fac_attr * f_attr + fac_rep * f_rep + fac_noise * f_noise + growth_vec
            vert.co += force * dt * dt * w * growth_scale

            if inhibit_base > 0 and not vert.is_boundary:
                w = w ** (1 + inhibit_base) - 0.01
            if inhibit_shell > 0:
                w = w * pow(vert.calc_shell_factor(), -1 * inhibit_shell)
            vert[bm.verts.layers.deform.active][vg_index] = w

    edges_to_subdivide = []
    for e in bm.edges:
        avg_weight = mean(vert1[bm.verts.layers.deform.active].get(vg_index, 0) for vert1 in e.verts)
        if avg_weight > 0:
            l = e.calc_length()
            if l / split_radius > 1 / avg_weight:
                edges_to_subdivide.append(e)

    if len(edges_to_subdivide) > 0:
        # noinspection PyArgumentList
        bmesh.ops.subdivide_edges(bm, edges=edges_to_subdivide, smooth=1.0, cuts=1, use_grid_fill=True,
                                  use_single_edge=True)
        adjacent_faces = set(chain.from_iterable(e.link_faces for e in edges_to_subdivide))
        # noinspection PyArgumentList
        bmesh.ops.triangulate(bm, faces=list(adjacent_faces))


def build_diff_growth(obj, index, max_polygons=1e4, **kwargs):
    with butil.ViewportMode(obj, 'EDIT'):
        bm = bmesh.from_edit_mesh(obj.data)
        plateau = 0
        while len(bm.faces) < max_polygons:
            v = len(bm.verts)
            # noinspection PyUnresolvedReferences
            grow_step(bm, index, **kwargs)
            if v == len(bm.verts):
                plateau += 1
                if plateau > 50:
                    break
        bmesh.update_edit_mesh(obj.data)

