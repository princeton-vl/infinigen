# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Lingjie Mei: original Infinigen reaction diffusion (https://github.com/princeton-vl/infinigen/blob/224598d4a2ff95b4fc928e72c222598b7122c698/src/infinigen/assets/utils/reaction_diffusion.py)
# - David Yan, Jia Deng: optimized implementation

"""Gray-Scott steps as one numba pass each, summing edges in `np.bincount`'s order
and updating in the default's expression order, so the fields match the default bit
for bit. Without numba the default's array passes run."""

import bpy
import numpy as np
from numpy.random import normal

import infinigen.core.util.blender as butil
from infinigen.perf.kernels import optional_import


def _steps_numpy(a, b, edge_from, edge_to, steps, diff_a, diff_b, feed, kill, dt):
    """The default's step loop on arrays."""
    n = len(a)
    for _ in range(steps):
        a_msg = a[edge_to] - a[edge_from]
        b_msg = b[edge_to] - b[edge_from]
        lap_a = np.bincount(edge_from, a_msg, n) - np.bincount(edge_to, a_msg, n)
        lap_b = np.bincount(edge_from, b_msg, n) - np.bincount(edge_to, b_msg, n)
        ab2 = a * b**2
        new_a = a + (diff_a * lap_a - ab2 + feed * (1 - a)) * dt
        new_b = b + (diff_b * lap_b + ab2 - (kill + feed) * b) * dt
        a = new_a
        b = new_b
    a_msg = a[edge_to] - a[edge_from]
    b_msg = b[edge_to] - b[edge_from]
    lap_a = np.bincount(edge_from, a_msg, n) - np.bincount(edge_to, a_msg, n)
    lap_b = np.bincount(edge_from, b_msg, n) - np.bincount(edge_to, b_msg, n)
    return a, b, lap_a, lap_b


def _steps_fused(a, b, edge_from, edge_to, steps, diff_a, diff_b, feed, kill, dt):
    """`_steps_numpy` in one pass per step, summing edges in `bincount`'s order."""
    n = a.shape[0]
    m = edge_from.shape[0]
    lap_a = np.zeros(n)
    lap_b = np.zeros(n)
    acc_a_from = np.zeros(n)
    acc_a_to = np.zeros(n)
    acc_b_from = np.zeros(n)
    acc_b_to = np.zeros(n)
    for step in range(steps + 1):
        for i in range(n):
            acc_a_from[i] = 0.0
            acc_a_to[i] = 0.0
            acc_b_from[i] = 0.0
            acc_b_to[i] = 0.0
        for e in range(m):
            f = edge_from[e]
            t = edge_to[e]
            a_msg = a[t] - a[f]
            b_msg = b[t] - b[f]
            acc_a_from[f] += a_msg
            acc_a_to[t] += a_msg
            acc_b_from[f] += b_msg
            acc_b_to[t] += b_msg
        for i in range(n):
            lap_a[i] = acc_a_from[i] - acc_a_to[i]
            lap_b[i] = acc_b_from[i] - acc_b_to[i]
        if step == steps:
            break
        for i in range(n):
            ai = a[i]
            bi = b[i]
            ab2 = ai * (bi * bi)
            a[i] = ai + (diff_a * lap_a[i] - ab2 + feed * (1 - ai)) * dt
            b[i] = bi + (diff_b * lap_b[i] + ab2 - (kill + feed) * bi) * dt
    return a, b, lap_a, lap_b


_steps_impl = None


def _steps():
    """The compiled step loop, or the array loop when numba is missing."""
    global _steps_impl
    if _steps_impl is None:
        njit = optional_import(
            "numba",
            "njit",
            hint="The reaction-diffusion steps run as numpy array passes instead "
            "(same result, about twice the time); numba comes with the 'vis' extra.",
        )
        _steps_impl = _steps_numpy if njit is None else njit(cache=True)(_steps_fused)
    return _steps_impl


def _write_vertex_group(obj, name, weights):
    """Add a vertex group of `weights`, clamped to [0, 1] as `VertexGroup.add` does,
    by converting an attribute: one operator call instead of one RNA call per vertex.
    Unlike the default, a zero weight leaves the vertex unassigned."""
    mesh = obj.data
    attr = mesh.attributes.new(name, "FLOAT", "POINT")
    attr.data.foreach_set("value", np.clip(weights, 0, 1).astype(np.float32))
    mesh.attributes.active = attr
    with butil.SelectObjects(obj):
        bpy.ops.geometry.attribute_convert(mode="VERTEX_GROUP")


def reaction_diffusion(
    obj,
    weight_fn,
    steps=1000,
    dt=1.0,
    scale=0.5,
    diff_a=0.18,
    diff_b=0.09,
    feed_rate=0.055,
    kill_rate=0.062,
    perturb=0.05,
):
    diff_a = diff_a * scale
    diff_b = diff_b * scale
    mesh = obj.data
    n = len(mesh.vertices)
    co = np.empty(n * 3, dtype=np.float32)
    mesh.vertices.foreach_get("co", co)
    edges = np.empty(len(mesh.edges) * 2, dtype=np.int32)
    mesh.edges.foreach_get("vertices", edges)
    edges = edges.reshape(-1, 2).astype(np.int64)
    a = np.ones(n)
    b = weight_fn(co.reshape(n, 3).astype(np.float64))
    a, b, lap_a, lap_b = _steps()(
        a,
        np.ascontiguousarray(b, dtype=np.float64),
        np.ascontiguousarray(edges[:, 0]),
        np.ascontiguousarray(edges[:, 1]),
        steps,
        diff_a,
        diff_b,
        feed_rate,
        kill_rate,
        dt,
    )

    a *= 1 + normal(0, perturb, n)
    b *= 1 + normal(0, perturb, n)
    lap_a *= 1 + normal(0, perturb, n)
    lap_a *= 1 + normal(0, perturb, n)

    for name, weights in (("A", a), ("B", b), ("LA", lap_a), ("LB", lap_b)):
        _write_vertex_group(obj, name, weights)
    obj.vertex_groups.update()
    obj.data.update()
