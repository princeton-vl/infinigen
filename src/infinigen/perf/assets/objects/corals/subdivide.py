# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - David Yan, Jia Deng: primary authors

import numpy as np

__all__ = ["catmull_clark_limit"]


def catmull_clark_limit(co, faces, edges):
    """Level-1 Catmull-Clark limit mesh (positions, quads) of a closed manifold of
    triangles or quads. Matches the Subdivision Surface modifier with limit surface on
    to float32 rounding, in its order: coarse vertices, a point per edge of `edges`, a
    point per face, and quad (v_k, e_k, F, e_{k-1}) for corner k of each face."""
    co = np.asarray(co, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    edges = np.asarray(edges, dtype=np.int64)
    nv, nf, ne, k = len(co), len(faces), len(edges), faces.shape[1]

    # the edge of every face corner (v_k, v_{k+1}) in the mesh's edge order
    lo = np.minimum(edges[:, 0], edges[:, 1])
    hi = np.maximum(edges[:, 0], edges[:, 1])
    key = lo * nv + hi
    order = np.argsort(key, kind="stable")
    ca = faces
    cb = np.roll(faces, -1, axis=1)
    ckey = np.minimum(ca, cb) * nv + np.maximum(ca, cb)
    pos = np.searchsorted(key[order], ckey.reshape(-1))
    if not np.array_equal(key[order][pos], ckey.reshape(-1)):
        raise ValueError("face corner without a mesh edge")
    corner_edge = order[pos].reshape(nf, k)
    corner_face = np.repeat(np.arange(nf), k)
    flat_edge = corner_edge.reshape(-1)
    count = np.bincount(flat_edge, minlength=ne)
    if count.min() != 2 or count.max() != 2:
        raise ValueError("catmull_clark_limit needs a closed manifold mesh")

    # face points, edge points, vertex points of one Catmull-Clark step
    F = co[faces].mean(1)
    face_sum_e = np.zeros((ne, 3))
    np.add.at(face_sum_e, flat_edge, F[corner_face])
    E = (co[edges[:, 0]] + co[edges[:, 1]] + face_sum_e) / 4
    valence = np.bincount(edges.reshape(-1), minlength=nv).astype(np.float64)
    face_sum_v = np.zeros((nv, 3))
    np.add.at(face_sum_v, faces.reshape(-1), np.repeat(F, k, axis=0))
    mid = (co[edges[:, 0]] + co[edges[:, 1]]) / 2
    mid_sum_v = np.zeros((nv, 3))
    np.add.at(mid_sum_v, edges[:, 0], mid)
    np.add.at(mid_sum_v, edges[:, 1], mid)
    n = valence[:, None]
    V = (face_sum_v / n + 2 * mid_sum_v / n + (n - 3) * co) / n

    # limit positions on the level-1 quad mesh: a vertex of valence n goes to
    # (n^2 v + 4 sum of its edge neighbours + sum of its face diagonals) / (n (n + 5))
    E_sum_v = np.zeros((nv, 3))
    np.add.at(E_sum_v, edges[:, 0], E)
    np.add.at(E_sum_v, edges[:, 1], E)
    LV = (n * n * V + 4 * E_sum_v + face_sum_v) / (n * (n + 5))
    E_sum_f = np.zeros((nf, 3))
    for i in range(k):
        E_sum_f += E[corner_edge[:, i]]
    if k == 3:
        # edge point (valence 4): neighbours are the two vertex points and the
        # two face points, diagonals the other edge points of the two faces
        diag = np.zeros((ne, 3))
        np.add.at(diag, flat_edge, E_sum_f[corner_face] - E[flat_edge])
        LE = (16 * E + 4 * (V[edges[:, 0]] + V[edges[:, 1]] + face_sum_e) + diag) / 36
        # face point (valence 3): neighbours the edge points, diagonals the vertex points
        LF = (9 * F + 4 * E_sum_f + V[faces].sum(1)) / 24
    else:
        # quads: diagonals of an edge point are the two opposite edge points
        # of its faces, of a face point the four vertex points
        opposite = np.roll(corner_edge, 2, axis=1).reshape(-1)
        diag = np.zeros((ne, 3))
        np.add.at(diag, flat_edge, E[opposite])
        LE = (16 * E + 4 * (V[edges[:, 0]] + V[edges[:, 1]] + face_sum_e) + diag) / 36
        LF = (16 * F + 4 * E_sum_f + V[faces].sum(1)) / 36

    verts = np.concatenate([LV, LE, LF])
    quads = np.stack(
        [
            faces,
            nv + corner_edge,
            np.repeat(nv + ne + np.arange(nf)[:, None], k, axis=1),
            nv + np.roll(corner_edge, 1, axis=1),
        ],
        -1,
    ).reshape(-1, 4)
    return verts, quads
