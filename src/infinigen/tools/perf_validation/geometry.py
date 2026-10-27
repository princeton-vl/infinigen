# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - David Yan, Jia Deng: primary authors

"""Whether two builds made the same mesh, and if not, how far apart they are.

Builds with different mesh digests are compared by nearest-sample distance relative
to the asset's size; the distance between one build's own two sample sets is the
sampling floor.
"""

from __future__ import annotations

import hashlib
import pathlib

import numpy as np

SAMPLES = 100_000


def _meshes(objs):
    seen = set()
    for root in objs:
        for o in [root, *root.children_recursive]:
            if o.type != "MESH" or o.name in seen:
                continue
            seen.add(o.name)
            yield o


def _world_triangles(o):
    mesh = o.data
    co = np.empty(len(mesh.vertices) * 3, np.float32)
    mesh.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    m = np.array(o.matrix_world, dtype=np.float64)
    world = (co @ m[:3, :3].T + m[:3, 3]).astype(np.float32)

    loop_total = np.empty(len(mesh.polygons), np.int32)
    mesh.polygons.foreach_get("loop_total", loop_total)
    loops = np.empty(len(mesh.loops), np.int32)
    mesh.loops.foreach_get("vertex_index", loops)

    mesh.calc_loop_triangles()
    tris = np.empty(len(mesh.loop_triangles) * 3, np.int32)
    mesh.loop_triangles.foreach_get("vertices", tris)
    return world, loop_total, loops, tris.reshape(-1, 3)


def record(objs, path: pathlib.Path) -> str:
    """Write surface samples of `objs` to `path` and return their mesh digest."""
    digests, meshes = [], []
    for o in _meshes(objs):
        world, loop_total, loops, tris = _world_triangles(o)
        h = hashlib.sha256()
        for a in (world, loop_total, loops):
            h.update(np.ascontiguousarray(a).tobytes())
        digests.append(h.hexdigest())
        if len(tris):
            meshes.append((world, tris, _areas(world, tris)))
    digest = hashlib.sha256("".join(sorted(digests)).encode()).hexdigest()[:16]

    if meshes:
        lo = np.min([w.min(0) for w, _, _ in meshes], axis=0)
        hi = np.max([w.max(0) for w, _, _ in meshes], axis=0)
        diagonal = float(np.linalg.norm(hi - lo))
    else:
        diagonal = 0.0
    sets = [_sample(meshes, np.random.default_rng(seed)) for seed in (0, 1)]
    np.savez_compressed(path, a=sets[0], b=sets[1], diagonal=diagonal)
    return digest


def _areas(world, tris, chunk=1 << 20):
    area = np.empty(len(tris), np.float64)
    for i in range(0, len(tris), chunk):
        t = world[tris[i : i + chunk]].astype(np.float64)
        area[i : i + chunk] = 0.5 * np.linalg.norm(
            np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1
        )
    return area


def _sample(meshes, rng):
    """`SAMPLES` points, area-weighted over every triangle of every mesh."""
    totals = np.array([area.sum() for _, _, area in meshes])
    if not len(meshes) or totals.sum() <= 0:
        return np.zeros((0, 3), np.float32)
    counts = rng.multinomial(SAMPLES, totals / totals.sum())
    out = []
    for (world, tris, area), k in zip(meshes, counts):
        if not k:
            continue
        cdf = np.cumsum(area)
        idx = np.searchsorted(cdf, rng.random(k) * cdf[-1], side="right")
        idx = np.minimum(idx, len(tris) - 1)
        u, v = rng.random(k), rng.random(k)
        flip = u + v > 1
        u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
        t = world[tris[idx]].astype(np.float64)
        pts = (
            t[:, 0]
            + u[:, None] * (t[:, 1] - t[:, 0])
            + v[:, None] * (t[:, 2] - t[:, 0])
        )
        out.append(pts.astype(np.float32))
    return np.concatenate(out)


def _distance(p, q):
    """Symmetric nearest-sample distances between two point sets."""
    from scipy.spatial import cKDTree

    d_pq, _ = cKDTree(q).query(p)
    d_qp, _ = cKDTree(p).query(q)
    return np.concatenate([d_pq, d_qp])


def compare(path_a: pathlib.Path, path_b: pathlib.Path) -> dict | None:
    """Deviation of build b's surface from build a's, as fractions of a's size: `mean`
    and `p99` of nearest-sample distances both ways, and `floor_mean` and `floor_p99`
    of those between a's own two sample sets."""
    if not (path_a.exists() and path_b.exists()):
        return None
    a, b = np.load(path_a), np.load(path_b)
    size = float(a["diagonal"]) or 1.0
    if not len(a["a"]) or not len(b["a"]):
        return None
    dev = _distance(a["a"], b["a"]) / size
    floor = _distance(a["a"], a["b"]) / size
    return {
        "mean": float(dev.mean()),
        "p99": float(np.percentile(dev, 99)),
        "floor_mean": float(floor.mean()),
        "floor_p99": float(np.percentile(floor, 99)),
    }


def describe(a_row: dict, b_row: dict | None, a_dir, b_dir) -> str:
    """One line: identical, or how far the optimized surface is from the default."""
    if not b_row:
        return "not built"
    if "asset_triangles" in a_row and "asset_triangles" in b_row:
        return describe_assets(a_row["asset_triangles"], b_row["asset_triangles"])
    da, db = a_row.get("mesh_digest"), b_row.get("mesh_digest")
    if da is None or db is None:
        return "not recorded"
    if da == db:
        return "identical mesh"
    c = None
    if a_row.get("samples") and b_row.get("samples"):
        c = compare(
            pathlib.Path(a_dir) / a_row["samples"],
            pathlib.Path(b_dir) / b_row["samples"],
        )
    if c is None:
        return "different mesh"
    return (
        f"different mesh: deviation mean {c['mean']:.3%}, p99 {c['p99']:.3%} "
        f"of size (sampling floor {c['floor_mean']:.3%}, {c['floor_p99']:.3%})"
    )


def describe_assets(a: dict, b: dict) -> str:
    """One line for a scene: which generated assets differ, in triangles."""
    names = sorted(set(a) | set(b))
    changed = [n for n in names if a.get(n) != b.get(n)]
    ta, tb = sum(a.values()), sum(b.values())
    line = f"{len(changed)} of {len(names)} assets differ; assets {ta:,} -> {tb:,} triangles"
    if changed:
        worst = sorted(changed, key=lambda n: -abs(b.get(n, 0) - a.get(n, 0)))[:3]
        line += (
            " ("
            + ", ".join(
                f"{n.removeprefix('assets:')} {a.get(n, 0):,} -> {b.get(n, 0):,}"
                for n in worst
            )
            + ")"
        )
    return line
