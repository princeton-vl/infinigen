# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - David Yan, Jia Deng: primary authors

"""The planar sheet of the default `draw.leaf`, built in numpy without bpy.

It sits on the remesh's lattice, but its outline vertices and face counts differ from
the operator chain's.
"""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

AUTO_HANDLE_FAC = 2.5614  # Blender's calchandleNurb_intern constant
SUBDIVIDE_MIN_LENGTH = (
    5e-3  # draw.curve2mesh subdivides segments into about 5 mm pieces
)
SUBDIVIDE_MAX_CUTS = 64
CURVE_WELD = 1e-3  # draw.curve2mesh and draw.leaf weld at 1 mm
REMESH_RESOLUTION = 0.005  # draw.remesh_fill default
SOLIDIFY_THICKNESS = 0.1
DUALCON_SCALE = 0.9  # Blender's remesh modifier passes scale 0.9 to dualcon
CORNER_ANGLE = 60.0  # outline turns sharper than this are placed exactly (leaf tips)


def blender_handles(points, vector):
    """Left and right handles of an open Bezier spline by Blender's auto rule: the ends
    mirror their neighbour and each chord is clamped to five times the other. `vector`
    handles sit a third of the way to their neighbour."""
    points = np.asarray(points, dtype=float)
    n = len(points)
    if n < 2:
        return points.copy(), points.copy()
    p1 = np.empty_like(points)
    p1[1:] = points[:-1]
    p1[0] = 2 * points[0] - points[1]
    p3 = np.empty_like(points)
    p3[:-1] = points[1:]
    p3[-1] = 2 * points[-1] - points[-2]
    dvec_a = points - p1
    dvec_b = p3 - points
    len_a = np.linalg.norm(dvec_a, axis=-1)
    len_b = np.linalg.norm(dvec_b, axis=-1)
    len_a[len_a == 0] = 1.0
    len_b[len_b == 0] = 1.0
    tvec = dvec_b / len_b[:, None] + dvec_a / len_a[:, None]
    ln = np.linalg.norm(tvec, axis=-1) * AUTO_HANDLE_FAC
    clamp = len_a > 5 * len_b
    len_a[clamp] = 5 * len_b[clamp]
    clamp = len_b > 5 * len_a
    len_b[clamp] = 5 * len_a[clamp]
    safe = ln != 0
    ln_safe = np.where(safe, ln, 1.0)
    left = np.where(safe[:, None], points - tvec * (len_a / ln_safe)[:, None], points)
    right = np.where(safe[:, None], points + tvec * (len_b / ln_safe)[:, None], points)
    vector = np.asarray(vector, dtype=bool)
    if vector.any():
        left[vector] = points[vector] - dvec_a[vector] / 3
        right[vector] = points[vector] + dvec_b[vector] / 3
    return left, right


def _bezier(p0, h0, h1, p1, t):
    t = t[:, None]
    s = 1 - t
    return s * s * s * p0 + 3 * s * s * t * h0 + 3 * s * t * t * h1 + t * t * t * p1


def weld_chain(poly, threshold):
    """Weld modifier on a polyline: runs of points closer than `threshold` merge."""
    d = np.linalg.norm(poly[1:] - poly[:-1], axis=-1)
    new = np.concatenate([[0], np.cumsum(d > threshold)])
    counts = np.bincount(new)
    sums = np.zeros((len(counts), poly.shape[1]))
    np.add.at(sums, new, poly)
    return sums / counts[:, None]


def bezier_polyline(
    anchors,
    vector_locations=(),
    min_length=SUBDIVIDE_MIN_LENGTH,
    max_cuts=SUBDIVIDE_MAX_CUTS,
    weld=CURVE_WELD,
):
    """Polyline of `draw.bezier_curve(anchors, vector_locations)`. Segments are cut last
    to first at equal Bezier parameter, and recomputing the handles after each cut
    reshapes the segments still to come. Consecutive points within `weld` then merge."""
    anchors = np.asarray(anchors, dtype=np.float32).astype(np.float64)
    n = len(anchors)
    vector = [i in vector_locations for i in range(n)]
    length = np.linalg.norm(anchors[:-1] - anchors[1:], axis=-1)
    pts = [anchors[i] for i in range(n)]
    left, right = blender_handles(np.array(pts), np.array(vector))
    for i in reversed(range(n - 1)):
        k = min(int(length[i] / min_length) - 1, max_cuts)
        if k < 0:
            continue
        if k > 0:
            t = np.arange(1, k + 1) / (k + 1)
            q = _bezier(pts[i], right[i], left[i + 1], pts[i + 1], t)
            pts[i + 1 : i + 1] = list(q)
            vector[i + 1 : i + 1] = [vector[i + 1]] * k
        left, right = blender_handles(np.array(pts), np.array(vector))
    poly = np.array(pts)
    if weld > 0 and len(poly) > 1:
        poly = weld_chain(poly, weld)
    return poly


def leaf_outline(x_anchors, y_anchors, vector_locations=()):
    """Closed 2-D outline `draw.leaf` fills: the lower (-y) polyline forward, then the
    upper (+y) one back, with the shared ends welded."""
    x = np.asarray(x_anchors, dtype=float)
    y = np.asarray(y_anchors, dtype=float)
    z = np.zeros_like(x)
    upper = bezier_polyline(np.stack([x, y, z], -1), vector_locations)
    lower = bezier_polyline(np.stack([x, -y, z], -1), vector_locations)
    poly = np.concatenate([lower[:, :2], upper[::-1][:, :2]])
    d = np.linalg.norm(poly - np.roll(poly, 1, axis=0), axis=-1)
    d[0] = np.inf
    new = np.cumsum(d > CURVE_WELD) - 1
    if len(poly) > 1 and np.linalg.norm(poly[0] - poly[-1]) <= CURVE_WELD:
        new[new == new[-1]] = 0
    counts = np.bincount(new)
    sums = np.zeros((len(counts), 2))
    np.add.at(sums, new, poly)
    return sums / counts[:, None]


def remesh_lattice(poly, resolution=REMESH_RESOLUTION, thickness=SOLIDIFY_THICKNESS):
    """Cell size and origin of the sharp remesh lattice in `draw.remesh_fill`, sized as
    the remesh modifier sizes its octree over the solidified slab's bounding box, whose
    centre is a lattice corner."""
    lo = poly.min(0)
    hi = poly.max(0)
    maxdim = max(hi[0] - lo[0], hi[1] - lo[1], thickness)
    depth = int(np.ceil(np.log2((maxdim + 0.01) / resolution)))
    cell = maxdim / (DUALCON_SCALE * 2**depth)
    return cell, (lo + hi) / 2


def _inside_corners(poly, xs, ys):
    """Mask of the lattice corners inside `poly`, by the even-odd rule."""
    x0, y0 = poly[:, 0], poly[:, 1]
    x1, y1 = np.roll(x0, -1), np.roll(y0, -1)
    inside = np.zeros((len(ys), len(xs)), dtype=bool)
    ylo = np.minimum(y0, y1)
    yhi = np.maximum(y0, y1)
    for j, yv in enumerate(ys):
        m = (ylo <= yv) & (yv < yhi)
        if not m.any():
            continue
        xc = x0[m] + (yv - y0[m]) * (x1[m] - x0[m]) / (y1[m] - y0[m])
        xc.sort()
        inside[j] = (np.searchsorted(xc, xs, side="right") % 2) == 1
    return inside


def sheet_from_outline(poly, cell, origin, weld=0.0, boundary_samples=4):
    """Planar `(verts, (quads, tris), boundary)` sheet of `poly` on the lattice. Each
    cell with a corner inside gets a vertex at its centre, or at the mean of its outline
    samples where the outline crosses it, and each inside corner a quad. The sheet is
    welded at `weld` and cut down to its largest connected piece."""
    poly = np.asarray(poly, dtype=float)
    xmin, ymin = poly.min(0) - 2 * cell
    xmax, ymax = poly.max(0) + 2 * cell
    i0 = int(np.floor((xmin - origin[0]) / cell))
    j0 = int(np.floor((ymin - origin[1]) / cell))
    i1 = int(np.ceil((xmax - origin[0]) / cell))
    j1 = int(np.ceil((ymax - origin[1]) / cell))
    xs = origin[0] + np.arange(i0, i1 + 1) * cell
    ys = origin[1] + np.arange(j0, j1 + 1) * cell
    inside = _inside_corners(poly, xs, ys)
    ny, nx = inside.shape
    corners = (
        inside[:-1, :-1].astype(np.int8)
        + inside[:-1, 1:]
        + inside[1:, :-1]
        + inside[1:, 1:]
    )
    has = corners > 0
    cell_index = np.full(corners.shape, -1, dtype=np.int64)
    cell_index[has] = np.arange(int(has.sum()))
    jj, ii = np.nonzero(has)
    verts = np.stack([xs[ii] + 0.5 * cell, ys[jj] + 0.5 * cell], -1)
    boundary = corners[jj, ii] < 4
    pinned = np.zeros(len(verts), dtype=bool)
    if boundary.any():
        seg = np.roll(poly, -1, axis=0) - poly
        nsamp = np.maximum(
            1,
            np.ceil(np.linalg.norm(seg, axis=-1) / (cell / boundary_samples)).astype(
                int
            ),
        )
        frac = np.concatenate([np.arange(k) / k for k in nsamp])
        samples = (
            np.repeat(poly, nsamp, axis=0)
            + np.repeat(seg, nsamp, axis=0) * frac[:, None]
        )
        si = np.floor((samples[:, 0] - origin[0]) / cell).astype(int) - i0
        sj = np.floor((samples[:, 1] - origin[1]) / cell).astype(int) - j0
        ok = (si >= 0) & (si < nx - 1) & (sj >= 0) & (sj < ny - 1)
        key = sj[ok] * (nx - 1) + si[ok]
        order = np.argsort(key, kind="stable")
        key, smp = key[order], samples[ok][order]
        uniq, start, count = np.unique(key, return_index=True, return_counts=True)
        means = np.add.reduceat(smp, start, axis=0) / count[:, None]
        bkey = jj[boundary] * (nx - 1) + ii[boundary]
        pos = np.minimum(np.searchsorted(uniq, bkey), len(uniq) - 1)
        hit = uniq[pos] == bkey
        bverts = verts[boundary].copy()
        bverts[hit] = means[pos[hit]]
        if (~hit).any():
            _, nn = cKDTree(samples).query(bverts[~hit])
            bverts[~hit] = samples[nn]
        # dual contouring keeps sharp outline corners (leaf tips), so the cell
        # holding one takes the corner itself
        prev_dir = poly - np.roll(poly, 1, axis=0)
        next_dir = seg
        cos_turn = np.einsum("ij,ij->i", prev_dir, next_dir) / np.maximum(
            np.linalg.norm(prev_dir, axis=-1) * np.linalg.norm(next_dir, axis=-1), 1e-30
        )
        corner = np.nonzero(cos_turn < np.cos(np.radians(CORNER_ANGLE)))[0]
        if len(corner):
            ci_ = np.floor((poly[corner, 0] - origin[0]) / cell).astype(int) - i0
            cj_ = np.floor((poly[corner, 1] - origin[1]) / cell).astype(int) - j0
            ckey = cj_ * (nx - 1) + ci_
            where = np.searchsorted(
                bkey, ckey
            )  # bkey is ascending (jj, ii from nonzero)
            where = np.minimum(where, len(bkey) - 1)
            found = bkey[where] == ckey
            bverts[where[found]] = poly[corner[found]]
            pinned_b = np.zeros(len(bverts), dtype=bool)
            pinned_b[where[found]] = True
            pinned[boundary] = pinned_b
        verts[boundary] = bverts
    cj, ci = np.nonzero(inside[1:-1, 1:-1])
    cj += 1
    ci += 1
    faces = np.stack(
        [
            cell_index[cj - 1, ci - 1],
            cell_index[cj - 1, ci],
            cell_index[cj, ci],
            cell_index[cj, ci - 1],
        ],
        -1,
    )
    verts, faces, boundary = _weld(verts, faces, boundary, weld, pinned)
    return _largest_component(verts, faces, boundary)


def weld_groups(verts, threshold, tree=None):
    """Each vertex's group under `BLI_kdtree_3d_calc_duplicates_fast`, Blender's merge
    by distance: in index order, each untouched vertex absorbs the untouched ones within
    `threshold`, so merges never chain. Each label is the absorber's index or the
    vertex's own."""
    n = len(verts)
    label = np.arange(n)
    if n == 0 or threshold <= 0:
        return label
    tree = cKDTree(verts) if tree is None else tree
    span = np.ptp(verts, axis=0)
    area = max(float(np.prod(np.maximum(span, threshold))), threshold**2)
    expected_pairs = n * n * np.pi * threshold**2 / area
    untouched = np.ones(n, dtype=bool)
    if expected_pairs < 2e6:
        pairs = tree.query_pairs(threshold, output_type="ndarray")
        if len(pairs) == 0:
            return label
        both = np.concatenate([pairs, pairs[:, ::-1]])
        order = np.lexsort((both[:, 1], both[:, 0]))
        both = both[order]
        starts = np.searchsorted(both[:, 0], np.arange(n + 1))
        for i in np.unique(both[:, 0]):
            if not untouched[i]:
                continue
            nb = both[starts[i] : starts[i + 1], 1]
            nb = nb[untouched[nb]]
            if len(nb):
                untouched[nb] = False
                label[nb] = i
    else:
        for i in range(n):
            if not untouched[i]:
                continue
            nb = np.array(tree.query_ball_point(verts[i], threshold), dtype=np.int64)
            nb = nb[(nb != i) & untouched[nb]]
            if len(nb):
                untouched[nb] = False
                label[nb] = i
    return label


def _weld(verts, quads, boundary, threshold, pinned=None):
    """Weld modifier in mode All: each `weld_groups` group becomes one vertex at its
    mean or its pinned outline corner, and quads with repeated vertices become triangles
    or are dropped."""
    empty_tris = np.zeros((0, 3), dtype=np.int64)
    if threshold <= 0 or len(verts) == 0:
        return verts, (quads, empty_tris), boundary
    group = weld_groups(verts, threshold)
    if (group == np.arange(len(verts))).all():
        return verts, (quads, empty_tris), boundary
    _, label = np.unique(group, return_inverse=True)
    counts = np.bincount(label)
    sums = np.zeros((len(counts), verts.shape[1]))
    np.add.at(sums, label, verts)
    merged = sums / counts[:, None]
    if pinned is not None and pinned.any():
        merged[label[pinned]] = verts[pinned]
    merged_boundary = np.zeros(len(counts), dtype=bool)
    merged_boundary[label[boundary]] = True
    return merged, _clean_faces(label[quads]), merged_boundary


def _clean_faces(quads):
    ok = (
        (quads[:, 0] != quads[:, 1])
        & (quads[:, 1] != quads[:, 2])
        & (quads[:, 2] != quads[:, 3])
        & (quads[:, 3] != quads[:, 0])
        & (quads[:, 0] != quads[:, 2])
        & (quads[:, 1] != quads[:, 3])
    )
    bad = quads[~ok]
    if len(bad) == 0:
        return quads, np.zeros((0, 3), dtype=np.int64)
    # a quad with exactly one repeated pair of cyclically adjacent vertices is a triangle
    tris = []
    for k in range(4):
        a, b, c, d = (bad[:, (k + m) % 4] for m in range(4))
        m = (a == b) & (b != c) & (c != d) & (d != a) & (a != c)
        tris.append(np.stack([b[m], c[m], d[m]], -1))
    tris = np.concatenate(tris) if tris else np.zeros((0, 3), dtype=np.int64)
    return quads[ok], tris.reshape(-1, 3)


def _largest_component(verts, faces, boundary):
    quads, tris = faces
    n = len(verts)
    edges = [np.stack([quads[:, k], quads[:, (k + 1) % 4]], -1) for k in range(4)]
    edges += [np.stack([tris[:, k], tris[:, (k + 1) % 3]], -1) for k in range(3)]
    edges = np.concatenate(edges)
    graph = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n))
    ncomp, label = connected_components(graph, directed=False)
    keep = np.zeros(n, dtype=bool)
    keep[quads.ravel()] = True
    keep[tris.ravel()] = True
    if ncomp > 1 and keep.any():
        keep &= label == np.bincount(label[keep]).argmax()
    new = np.full(n, -1, dtype=np.int64)
    new[keep] = np.arange(int(keep.sum()))
    quads = new[quads]
    tris = new[tris]
    quads = quads[(quads >= 0).all(1)]
    tris = tris[(tris >= 0).all(1)]
    return verts[keep], (quads, tris), boundary[keep]


def leaf_sheet(x_anchors, y_anchors, vector_locations=(), weld=0.0):
    """`(verts, (quads, tris), boundary)` of the sheet `draw.leaf` builds for the
    anchors, welded at `weld` (`face_size / 2` in `draw.leaf`, 0 for none)."""
    poly = leaf_outline(x_anchors, y_anchors, vector_locations)
    cell, origin = remesh_lattice(poly)
    return sheet_from_outline(poly, cell, origin, weld=weld)
