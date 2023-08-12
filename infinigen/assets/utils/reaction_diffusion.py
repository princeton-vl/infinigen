# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei


import math

import bmesh
import numpy as np
import tqdm
from numpy.random import uniform, normal


def reaction_diffusion(obj, weight_fn, steps=1000, dt=1., scale=.5, diff_a=.18, diff_b=.09, feed_rate=.055,
                       kill_rate=.062, perturb=.05):
    diff_a = diff_a * scale
    diff_b = diff_b * scale
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.edges.ensure_lookup_table()
    bm.verts.ensure_lookup_table()
    n = len(bm.verts)
    a = np.ones(n)
    b = weight_fn(np.stack([v.co for v in bm.verts]))
    edge_from = np.array([e.verts[0].index for e in bm.edges])
    edge_to = np.array([e.verts[1].index for e in bm.edges])
    size = max(len(v.link_edges) for v in bm.verts)
    for _ in range(steps):
        a_msg = a[edge_to] - a[edge_from]
        b_msg = b[edge_to] - b[edge_from]
        lap_a = np.bincount(edge_from, a_msg, size) - np.bincount(edge_to, a_msg, size)
        lap_b = np.bincount(edge_from, b_msg, size) - np.bincount(edge_to, b_msg, size)
        ab2 = a * b ** 2
        new_a = a + (diff_a * lap_a - ab2 + feed_rate * (1 - a)) * dt
        new_b = b + (diff_b * lap_b + ab2 - (kill_rate + feed_rate) * b) * dt
        a = new_a
        b = new_b

    a_msg = a[edge_to] - a[edge_from]
    b_msg = b[edge_to] - b[edge_from]
    lap_a = np.bincount(edge_from, a_msg, size) - np.bincount(edge_to, a_msg, size)
    lap_b = np.bincount(edge_from, b_msg, size) - np.bincount(edge_to, b_msg, size)

    a *= 1 + normal(0, perturb, n)
    b *= 1 + normal(0, perturb, n)
    lap_a *= 1 + normal(0, perturb, n)
    lap_a *= 1 + normal(0, perturb, n)

    vg_a = obj.vertex_groups.new(name='A')
    vg_b = obj.vertex_groups.new(name='B')
    vg_la = obj.vertex_groups.new(name='LA')
    vg_lb = obj.vertex_groups.new(name='LB')
    for i in range(n):
        vg_la.add([i], lap_a[i], 'REPLACE')
        vg_lb.add([i], lap_b[i], 'REPLACE')
        vg_a.add([i], a[i], 'REPLACE')
        vg_b.add([i], b[i], 'REPLACE')
    obj.vertex_groups.update()
    obj.data.update()


def feed2kill(feed):
    return math.sqrt(feed) / 2 - feed


def make_periodic_weight_fn(n_instances, stride=.1):
    def periodic_weight_fn(coords):
        multiplier = uniform(20, 100, (1, n_instances))
        center = coords[np.random.randint(0, len(coords) - 1, n_instances)]
        phi = (np.expand_dims(coords, 1) * np.expand_dims(center, 0)).sum(-1) * multiplier
        measure = np.cos(phi).sum(-1) / math.sqrt(n_instances)
        return (np.abs(measure) < stride).astype(float)

    return periodic_weight_fn
