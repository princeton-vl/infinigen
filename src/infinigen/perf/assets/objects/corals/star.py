# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Lingjie Mei: original Infinigen star coral (https://github.com/princeton-vl/infinigen/blob/224598d4a2ff95b4fc928e72c222598b7122c698/src/infinigen/assets/objects/corals/star.py)
# - David Yan, Jia Deng: optimized implementation

"""The default star coral's flowers, built on arrays from the loop table.

The default's bridges fail when the scene's mesh select mode is face or edge, leaving
the coral without flowers; these are always built.
"""

import bpy
import numpy as np
from numpy.random import uniform
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

import infinigen.core.util.blender as butil
from infinigen.assets.utils.decorate import (
    displace_vertices,
    geo_extension,
    read_co,
    read_edges,
    read_loop_edges,
)
from infinigen.assets.utils.object import join_objects, mesh2obj, new_icosphere
from infinigen.core import surface
from infinigen.core.nodes.node_info import Nodes
from infinigen.core.nodes.node_wrangler import (
    NodeWrangler,
    geometry_node_group_empty_new,
)
from infinigen.core.tagging import tag_object
from infinigen.core.util.blender import deep_clone_obj
from infinigen.perf.assets.objects.monocot.leaf_direct import mesh_from_arrays

# levels of a flower, as the default's Array modifier count less one
RESOLUTION = 16
# the levels that carry the tentacles: t = level / 16 in (0.4, 0.6)
OUTERMOST_LEVELS = (np.arange(RESOLUTION + 1) / RESOLUTION > 0.4) & (
    np.arange(RESOLUTION + 1) / RESOLUTION < 0.6
)


def build_flowers(rings):
    """Flowers on every face of `rings` as one mesh object, deleting `rings`. Level i of
    a ring with centre c is c + s^i (p - c), lifted along the ring's stored normal by
    the default's flower profile at t = i / 16. Rings draw in the default's loose-part
    order, so the random draws pair with the default's."""
    mesh = rings.data
    co = read_co(rings)
    edges = read_edges(rings)
    normals = surface.read_attr_data(rings, "custom_normal")
    faces_per_edge = np.bincount(read_loop_edges(rings), minlength=len(mesh.edges))
    boundary = np.flatnonzero(faces_per_edge == 1)
    ev = edges[boundary]
    n = len(co)
    graph = coo_matrix((np.ones(len(ev)), (ev[:, 0], ev[:, 1])), shape=(n, n))
    _, labels = connected_components(graph, directed=False)
    on_boundary = np.zeros(n, dtype=bool)
    on_boundary[ev.reshape(-1)] = True
    # loose-part separation lists the parts in the order of their first vertex
    first_vertex = np.flatnonzero(on_boundary)
    part_labels = labels[first_vertex]
    _, first = np.unique(part_labels, return_index=True)
    ring_order = part_labels[np.sort(first)]
    # neighbours along the boundary: two per vertex
    nbr = np.full((n, 2), -1, dtype=np.int64)
    slot = np.zeros(n, dtype=np.int64)
    for a, b in ev:
        nbr[a, slot[a]] = b
        slot[a] += 1
        nbr[b, slot[b]] = a
        slot[b] += 1

    group = geometry_node_group_empty_new()
    nw = NodeWrangler(group)
    curve_node = nw.build_float_curve(
        nw.new_node(Nodes.Value), [(0, 0), (0.5, 0.1), (1, 0)], "AUTO"
    )
    t = np.arange(RESOLUTION + 1) / RESOLUTION
    verts, quads, norm_attr, outer_attr = [], [], [], []
    offset = 0
    for label in ring_order:
        ring = np.flatnonzero((labels == label) & on_boundary)
        size = len(ring)
        # the loop in cyclic order from its first vertex
        order = [ring[0]]
        prev, cur = -1, ring[0]
        for _ in range(size - 1):
            nxt = nbr[cur, 0] if nbr[cur, 0] != prev else nbr[cur, 1]
            order.append(nxt)
            prev, cur = cur, nxt
        order = np.array(order)
        p = co[ring]
        center = p.mean(0)
        normal = normals[ring[0]]
        # the bridge's strips face against the ring's stored normal and the
        # annulus along it: walk the loop clockwise about the normal
        loop = co[order]
        if np.cross(loop, np.roll(loop, -1, axis=0)).sum(0) @ normal > 0:
            order = order[::-1]
        # the ring's draws, in the default's order
        s = uniform(0.3, 0.5) ** (1 / RESOLUTION)
        anchor = uniform(0.4, 0.6), uniform(0.08, 0.15)
        curve_node.mapping.curves[0].points[1].location = anchor
        curve_node.mapping.update()
        curve_node.mapping.initialize()
        mapping = curve_node.mapping
        profile = np.array([mapping.evaluate(mapping.curves[0], v) for v in t])
        scale = s ** np.arange(RESOLUTION + 1)
        levels = center + scale[:, None, None] * (p - center)[None]
        levels = levels + (profile[:, None] * normal[None])[:, None, :]
        verts.append(levels.reshape(-1, 3))
        # vertex index within the ring in cyclic order, and its successor
        j = np.searchsorted(ring, order)
        jn = np.roll(j, -1)
        base = offset + np.arange(RESOLUTION + 1)[:, None] * size
        strips = np.stack(
            [base[:-1] + j, base[:-1] + jn, base[1:] + jn, base[1:] + j], -1
        ).reshape(-1, 4)
        cap = np.stack([offset + j, base[-1] + j, base[-1] + jn, offset + jn], -1)
        quads.append(strips)
        quads.append(cap)
        norm_attr.append(np.repeat(normal[None], (RESOLUTION + 1) * size, axis=0))
        outer_attr.append(np.repeat(OUTERMOST_LEVELS.astype(np.float32), size))
        offset += (RESOLUTION + 1) * size
    bpy.data.node_groups.remove(group)

    flowers = mesh2obj(
        mesh_from_arrays(np.concatenate(verts), np.concatenate(quads), name="flowers")
    )
    surface.write_attr_data(
        flowers,
        "custom_normal",
        np.concatenate(norm_attr),
        type="FLOAT_VECTOR",
        domain="POINT",
    )
    # a float, as the default's Store Named Attribute node stores it: the remesh
    # then interpolates it and the tentacles' selection reads every non-zero value
    surface.write_attr_data(
        flowers, "outermost", np.concatenate(outer_attr), type="FLOAT", domain="POINT"
    )
    butil.delete(rings)
    return flowers


class StarBaseCoralFactory:
    """Counterpart of the default factory, defining only the method that differs."""

    def create_asset(self, face_size=0.01, **params):
        obj = new_icosphere(subdivisions=3)
        obj.location[-1] = uniform(0.25, 0.5)
        butil.apply_transform(obj, loc=True)
        surface.add_geomod(obj, self.geo_dual_mesh, apply=True)
        displace_vertices(obj, lambda x, y, z: (0, 0, -0.9 * np.clip(z, None, 0)))

        rings = deep_clone_obj(obj)
        levels = 3
        butil.modify_mesh(obj, "SUBSURF", levels=levels, render_levels=levels)
        butil.modify_mesh(rings, "SHRINKWRAP", target=obj)

        surface.add_geomod(rings, self.geo_separate_faces, apply=True)
        levels = 3
        butil.modify_mesh(rings, "SUBSURF", levels=levels, render_levels=levels)

        obj = join_objects([obj, build_flowers(rings)])
        surface.add_geomod(obj, geo_extension, apply=True)
        tag_object(obj, "star_coral")
        return obj
