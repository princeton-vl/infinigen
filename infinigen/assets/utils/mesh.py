# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors: Lingjie Mei


import bmesh
import bpy
import numpy as np
import shapely
import trimesh
from mathutils import Vector
from numpy.random import normal, uniform
from shapely import LineString

from infinigen.assets.utils.decorate import (
    read_co,
    read_edges,
)
from infinigen.assets.utils.object import obj2trimesh, separate_loose
from infinigen.assets.utils.shapes import dissolve_limited
from infinigen.core.util import blender as butil
from infinigen.core.util.math import normalize


def build_prism_mesh(n=6, r_min=1.0, r_max=1.5, height=0.3, tilt=0.3):
    angles = polygon_angles(n)
    a_upper = uniform(-np.pi / 12, np.pi / 12, n)
    a_lower = uniform(-np.pi / 12, np.pi / 12, n)
    z_upper = (
        1
        + uniform(-height, height, n)
        + uniform(0, tilt) * np.cos(angles + uniform(-np.pi, np.pi))
    )
    z_lower = (
        1
        + uniform(-height, height, n)
        + uniform(0, tilt) * np.sin(angles + uniform(-np.pi, np.pi))
    )
    r_upper = uniform(r_min, r_max, n)
    r_lower = uniform(r_min, r_max, n)

    vertices = np.block(
        [
            [
                r_upper * np.cos(angles + a_upper),
                r_lower * np.cos(angles + a_lower),
                0,
                0,
            ],
            [
                r_upper * np.sin(angles + a_upper),
                r_lower * np.sin(angles + a_lower),
                0,
                0,
            ],
            [z_upper, -z_lower, 1, -1],
        ]
    ).T

    r = np.arange(n)
    s = np.roll(r, -1)
    faces = np.block(
        [
            [r, r, r + n, s + n],
            [s, r + n, s + n, r + n],
            [np.full(n, 2 * n), s, s, np.full(n, 2 * n + 1)],
        ]
    ).T
    mesh = bpy.data.meshes.new("prism")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    return mesh


def build_convex_mesh(n=6, height=0.2, tilt=0.2):
    angles = polygon_angles(n)
    a_upper = uniform(-np.pi / 18, 0, n)
    a_lower = uniform(0, np.pi / 18, n)
    z_upper = (
        1
        + normal(0, height, n)
        + uniform(0, tilt) * np.cos(angles + uniform(-np.pi, np.pi))
    )
    z_lower = (
        1
        + normal(0, height, n)
        + uniform(0, tilt) * np.cos(angles + uniform(-np.pi, np.pi))
    )
    r = 1.8
    vertices = np.block(
        [
            [r * np.cos(angles + a_upper), r * np.cos(angles + a_lower), 0, 0],
            [r * np.sin(angles + a_upper), r * np.sin(angles + a_lower), 0, 0],
            [
                z_upper,
                -z_lower,
                z_upper.max() + uniform(0.1, 0.2),
                -z_lower.max() - uniform(0.1, 0.2),
            ],
        ]
    ).T

    r = np.arange(n)
    s = np.roll(r, -1)
    faces = np.block(
        [
            [r, r, r + n, s + n],
            [s, r + n, s + n, r + n],
            [np.full(n, 2 * n), s, s, np.full(n, 2 * n + 1)],
        ]
    ).T
    mesh = bpy.data.meshes.new("prism")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    return mesh


def polygon_angles(n, min_angle=np.pi / 6, max_angle=np.pi * 2 / 3):
    for _ in range(100):
        angles = np.sort(uniform(0, 2 * np.pi, n))
        difference = (angles - np.roll(angles, 1)) % (np.pi * 2)
        if (difference >= min_angle).all() and (difference <= max_angle).all():
            break
    else:
        angles = np.sort(
            (np.arange(n) * (2 * np.pi / n) + uniform(0, np.pi * 2)) % (np.pi * 2)
        )
    return angles


def face_area(obj):
    with butil.ViewportMode(obj, "EDIT"):
        bm = bmesh.from_edit_mesh(obj.data)
        return sum(f.calc_area() for f in bm.faces)


def centroid(obj):
    with butil.ViewportMode(obj, "EDIT"):
        bm = bmesh.from_edit_mesh(obj.data)
        s = sum(
            (f.calc_area() * f.calc_center_median() for f in bm.faces),
            Vector((0, 0, 0)),
        )
        area = sum(f.calc_area() for f in bm.faces)
        return np.array(s / area)


def longest_ray(obj, obj_, direction):
    co = read_co(obj_)
    directions = np.array([direction] * len(co))
    mesh = obj2trimesh(obj)
    signed_distance = trimesh.proximity.longest_ray(mesh, co, directions)
    return signed_distance


def treeify(obj):
    if len(obj.data.vertices) == 0:
        return obj

    obj = separate_loose(obj)
    with butil.ViewportMode(obj, "EDIT"):
        bm = bmesh.from_edit_mesh(obj.data)
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        included = np.zeros(len(bm.verts))
        i = min((v.co[-1], i) for i, v in enumerate(bm.verts))[1]
        queue = [bm.verts[i]]
        included[i] = 1
        to_keep = []
        while queue:
            v = queue.pop()
            for e in v.link_edges:
                o = e.other_vert(v)
                if not included[o.index]:
                    included[o.index] = 1
                    to_keep.append(e)
                    queue.append(o)
        bmesh.ops.delete(
            bm, geom=list(set(bm.edges).difference(to_keep)), context="EDGES"
        )
        bmesh.update_edit_mesh(obj.data)
    return obj


def convert2ls(obj):
    with butil.ViewportMode(obj, "EDIT"):
        bm = bmesh.from_edit_mesh(obj.data)
        verts = [next(v for v in bm.verts if len(v.link_edges) == 1)]
        for i in range(len(bm.verts) - 1):
            vs = [e.other_vert(verts[-1]) for e in verts[-1].link_edges]
            if len(verts) > 1 and len(vs) > 1:
                verts.append(next(_ for _ in vs if _ != verts[-2]))
            else:
                verts.append(vs[0])
        return LineString(np.array([v.co for v in verts]))


def convert2mls(obj):
    mls = []
    for o in butil.split_object(obj):
        mls.append(convert2ls(o))
    return shapely.MultiLineString(mls)


def fix_tree(obj):
    with butil.ViewportMode(obj, "EDIT"), butil.Suppress():
        bpy.ops.mesh.remove_doubles()
        bm = bmesh.from_edit_mesh(obj.data)
        vertices_remove = []
        for v in bm.verts:
            if len(v.link_edges) == 1:
                o = v.link_edges[0].other_vert(v)
                if len(o.link_edges) > 2:
                    vertices_remove.append(v)
        bmesh.ops.delete(bm, geom=vertices_remove)
        bmesh.update_edit_mesh(obj.data)
    return obj


def longest_path(obj):
    with butil.ViewportMode(obj, "EDIT"), butil.Suppress():
        bpy.ops.mesh.remove_doubles()
        bm = bmesh.from_edit_mesh(obj.data)

        def longest_path_(u, v):
            dist = 0
            rest = [v]
            for e in v.link_edges:
                w = e.other_vert(v)
                if w != u:
                    l, r = longest_path_(v, w)
                    dist = max(dist, l)
                    rest.extend(r)
            return dist + np.linalg.norm(u.co - v.co), rest

        while True:
            for v in bm.verts:
                if len(v.link_edges) > 2:
                    ws = [e.other_vert(v) for e in v.link_edges]
                    dists, rests = list(zip(*list(longest_path_(v, w) for w in ws)))
                    geom = sum(list(rests[i] for i in np.argsort(dists)[:-2]), [])
                    bmesh.ops.delete(bm, geom=geom)
                    break
            else:
                break
        bmesh.update_edit_mesh(obj.data)
    return obj


def bevel(obj, width, **kwargs):
    preset = np.random.choice(["LINE", "SUPPORTS", "CORNICE", "CROWN", "STEPS"])
    obj, mod = butil.modify_mesh(
        obj,
        "BEVEL",
        width=width,
        segments=np.random.randint(20, 30),
        profile_type="CUSTOM",
        apply=False,
        return_mod=True,
        **kwargs,
    )
    reset_preset(mod.custom_profile, preset)
    butil.apply_modifiers(obj, mod)


def reset_preset(profile, name, n=None):
    if n is None:
        n = np.random.randint(8, 15)
    match name:
        case "LINE":
            configs = [(1.0, 0.0, 0, "AUTO", "AUTO"), (0.0, 1.0, 0, "AUTO", "AUTO")]
        case "CORNICE":
            configs = [
                (1.0, 0.0, 0, "VECTOR", "VECTOR"),
                (1.0, 0.125, 0, "VECTOR", "VECTOR"),
                (0.92, 0.16, 0, "AUTO", "AUTO"),
                (0.875, 0.25, 0, "VECTOR", "VECTOR"),
                (0.8, 0.25, 0, "VECTOR", "VECTOR"),
                (0.733, 0.433, 0, "AUTO", "AUTO"),
                (0.582, 0.522, 0, "AUTO", "AUTO"),
                (0.4, 0.6, 0, "AUTO", "AUTO"),
                (0.289, 0.727, 0, "AUTO", "AUTO"),
                (0.25, 0.925, 0, "VECTOR", "VECTOR"),
                (0.175, 0.925, 0, "VECTOR", "VECTOR"),
                (0.175, 1.0, 0, "VECTOR", "VECTOR"),
                (0.0, 1.0, 0, "VECTOR", "VECTOR"),
            ]
        case "CROWN":
            configs = [
                (1.0, 0.0, 0, "VECTOR", "VECTOR"),
                (1.0, 0.25, 0, "VECTOR", "VECTOR"),
                (0.75, 0.25, 0, "VECTOR", "VECTOR"),
                (0.75, 0.325, 0, "VECTOR", "VECTOR"),
                (0.925, 0.4, 0, "AUTO", "AUTO"),
                (0.975, 0.5, 0, "AUTO", "AUTO"),
                (0.94, 0.65, 0, "AUTO", "AUTO"),
                (0.85, 0.75, 0, "AUTO", "AUTO"),
                (0.75, 0.875, 0, "AUTO", "AUTO"),
                (0.7, 1.0, 0, "VECTOR", "VECTOR"),
                (0.0, 1.0, 0, "VECTOR", "VECTOR"),
            ]
        case "SUPPORTS":
            configs = (
                [(1.0, 0.0, 0, "VECTOR", "VECTOR"), (1.0, 0.5, 0, "VECTOR", "VECTOR")]
                + list(
                    (
                        1 - 0.5 * (1 - np.cos(i / (n - 3) * np.pi / 2)),
                        0.5 + 0.5 * np.sin(i / (n - 3) * np.pi / 2),
                        0,
                        "AUTO",
                        "AUTO",
                    )
                    for i in range(1, n - 2)
                )
                + [(0.5, 1.0, 0, "VECTOR", "VECTOR"), (0.0, 1.0, 0, "VECTOR", "VECTOR")]
            )
        case _:
            n_steps_x = n if n % 2 == 0 else n - 1
            n_steps_y = n - 2 if n % 2 == 0 else n - 1
            configs = list(
                (
                    1 - (i + 1) // 2 * 2 / n_steps_x,
                    i // 2 * 2 / n_steps_y,
                    0,
                    "VECTOR",
                    "VECTOR",
                )
                for i in range(n)
            )
    k = len(configs) - len(profile.points)
    for i in range(k):
        profile.points.add((i + 1) / (k + 1), 0)
    for p, c in zip(profile.points, configs):
        p.location = c[0], c[1]
        p.select = True
        p.handle_type_1 = c[3]
        p.handle_type_2 = c[4]
        p.select = False
    profile.points.update()


def canonicalize_ls(line):
    line = shapely.simplify(line, 0.02)
    while True:
        coords = np.array(line.coords)
        diff = coords[1:] - coords[:-1]
        diff = diff / (np.linalg.norm(diff, axis=-1, keepdims=True) + 1e-6)
        product = (diff[:-1] * diff[1:]).sum(-1)
        valid_indices = (
            np.nonzero((1 - 1e-6 > product) & (product > -0.8))[0] + 1
        ).tolist()
        ls = LineString(coords[[0] + valid_indices + [-1]])
        if ls.length < line.length:
            line = ls
        else:
            break
    return ls


def canonicalize_mls(mls):
    return shapely.MultiLineString([canonicalize_ls(ls) for ls in mls.geoms])


def separate_selected(obj, face=False):
    butil.select_none()
    with butil.ViewportMode(obj, "EDIT"):
        if face:
            bpy.ops.mesh.duplicate_move()
        bpy.ops.mesh.separate(type="SELECTED")
    o = next(o for o in bpy.context.selected_objects if o != obj)
    butil.select_none()
    return o


def snap_mesh(obj, eps=1e-3):
    while True:
        dissolve_limited(obj)
        co = read_co(obj)
        u, w = read_edges(obj).T
        d = co[:, np.newaxis] - co[np.newaxis, u]
        l = co[np.newaxis, w] - co[np.newaxis, u]
        n = normalize(l, in_place=False)
        prod = (d * n).sum(-1)
        diff = np.linalg.norm(d - prod[:, :, np.newaxis] * n, axis=-1)
        diff[u, np.arange(len(u))] = 1
        diff[w, np.arange(len(w))] = 1
        diff[prod < 0] = 1
        diff[prod > np.linalg.norm(l, axis=-1)] = 1
        es, vs = np.nonzero((diff < eps).T)
        if len(vs) == 0:
            return obj
        indices = np.concatenate([[0], np.nonzero(es[1:] != es[:-1])[0] + 1])
        vs = vs[indices]
        es = es[indices]
        with butil.ViewportMode(obj, "EDIT"):
            bm = bmesh.from_edit_mesh(obj.data)
            bm.verts.ensure_lookup_table()
            bm.edges.ensure_lookup_table()
            dis = co[w[es]] - co[u[es]]
            norms = np.linalg.norm(dis, axis=-1)
            percents = ((co[vs] - co[u[es]]) * dis).sum(-1) / (norms**2)
            edges = [bm.edges[e] for e in es]
            for e, p in zip(edges, percents):
                bmesh.ops.subdivide_edges(bm, edges=[e], cuts=1, edge_percents={e: p})
            bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=eps * 1.5)
            bmesh.update_edit_mesh(obj.data)


def prepare_for_boolean(obj):
    butil.modify_mesh(obj, "WELD", merge_threshold=1e-3)
    with butil.ViewportMode(obj, "EDIT"), butil.Suppress():
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.remove_doubles()
