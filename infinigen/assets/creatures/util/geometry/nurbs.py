# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick


import math
import logging

import bpy
import bmesh

from geomdl import NURBS, knotvector
import numpy as np

from infinigen.core.util import blender as butil
from infinigen.core.util.math import randomspacing

logger = logging.getLogger(__name__)

try:
    import bnurbs
except ImportError:
    logger.warning(f'Failed to import compiled `bnurbs` package, either installation failed or we are running a minimal install')
    bnurbs = None

def compute_cylinder_topology(n: int, m: int, uvs=False, cyclic=True, h_neighbors=None):

    # n: num vertices in vertical direction
    # m: num vertices in each loop

    # compute ring edges
    loop = np.arange(m)
    if h_neighbors is None:
        h_neighbors = np.stack([loop, np.roll(loop, -1)], axis=-1)
    ring_start_offsets = np.arange(0, n * m, m)
    ring_edges = ring_start_offsets[:, None, None] + h_neighbors[None]
    if not cyclic:
        ring_edges = ring_edges[:, :-1, :]
    ring_edges = ring_edges.reshape(-1, 2)

    # compute bridge edges
    v_neighbors = np.stack([loop, loop + m], axis=-1)
    ring_start_offsets = np.arange(0, (n - 1) * m, m)
    bridge_edges = ring_start_offsets[:, None, None] + v_neighbors[None]
    bridge_edges = bridge_edges.reshape(-1, 2)

    edges = np.concatenate([ring_edges, bridge_edges])

    # compute faces
    face_neighbors = np.concatenate(
        [h_neighbors, h_neighbors[:, ::-1] + m], axis=-1)
    faces = ring_start_offsets[:, None, None] + face_neighbors[None]
    if not cyclic:
        faces = faces[:, :-1, :]
    faces = faces.reshape(-1, 4)

    if not uvs:
        return edges, faces

    us, vs = np.meshgrid(np.linspace(0, 1, m, endpoint=True), np.linspace(0, 1, n, endpoint=True))
    uvs = np.stack([us, vs], axis=-1).reshape(-1, 2)

    return edges, faces, uvs

def apply_crease_values(obj, creases: np.array):

    n, m, c = creases.shape

    # set crease values
    with butil.ViewportMode(obj, mode='EDIT'):
        bm = bmesh.from_edit_mesh(obj.data)

        creaseLayer = bm.edges.layers.crease.verify()
        for i, e in enumerate(bm.edges):
            v1 = e.verts[0].index
            v2 = e.verts[1].index
            # channel 0 for ring edges, 1 for bridge edges
            channel = int(v2 != v1 + 1)
            e[creaseLayer] = creases[v1 // m, v1 % m, channel]

        bmesh.update_edit_mesh(obj.data)


def subdiv_mesh_nurbs(verts, level, creases=None, name='loft_mesh', cyclic_v=True) -> bpy.types.Object:

    if not cyclic_v:
        raise NotImplementedError()

    n, m, _ = verts.shape
    edges, faces, uvs = compute_cylinder_topology(n, m, cyclic=True, uvs=True)
    obj = blender_mesh_from_pydata(verts, edges, faces, uvs=uvs, name=name)

    if creases is not None:
        assert creases.shape[-1] == 2
        apply_crease_values(obj, creases)

    if level:
        butil.modify_mesh(obj, type='SUBSURF', levels=level,
                          render_levels=level, apply=False)

    return obj
    
def blender_nurbs(ctrlpts, ws=None, name='loft_nurbs', resolution=(32, 32), cyclic_v=True, kv_u=None, kv_v=None):
    
    n, m, _  = ctrlpts.shape

    if ws is None:
        ws = np.ones((n, m, 1))
    else:
        assert ws.shape == (n, m, 1)

    curve = bpy.data.curves.new(name, 'SURFACE')
    curve.dimensions = '3D'
    obj = bpy.data.objects.new(name, curve)
    bpy.context.scene.collection.objects.link(obj)

    # create each profile as its own spline
    verts_4d = np.concatenate([ctrlpts, ws], axis=-1)
    for i, profile in enumerate(verts_4d):
        spline = curve.splines.new(type='NURBS')
        spline.points.add(m - len(spline.points))
        for p, co in zip(spline.points, profile):
            p.co = co

    # bridge profiles
    for s in curve.splines:
        for p in s.points:
            p.select = True
    with butil.ViewportMode(obj, mode='EDIT'):
        bpy.ops.curve.make_segment()

    spline = obj.data.splines[0]

    spline.use_endpoint_u = True
    spline.use_cyclic_v = cyclic_v
    spline.resolution_u, spline.resolution_v = resolution
    
    if kv_u is not None:
        bnurbs.set_knotsu(spline, kv_u)
    if kv_v is not None:
        bnurbs.set_knotsv(spline, kv_v)

    return obj


def generate_knotvector(degree, n, mode='uniform', clamped=True):
    if mode == 'uniform':
        if clamped:
            middle = np.linspace(0, n, n - degree + 1)[1:-1]
        else:
            middle = np.arange(0, n + degree + 1)
    elif mode == 'piecewise_bezier':  # todo: this isn't correct
        middle = np.repeat(np.arange(0, n), degree)
    elif mode == 'random_uniform':
        if clamped:
            middle = np.sort(np.random.uniform(0, n, n - degree - 1))
        else:
            middle = np.sort(np.random.uniform(0, n, n + degree + 1))
    else:
        raise ValueError(f'Unrecognized {mode=} for generate_knotvector')

    if clamped:
        assert len(middle) == n - degree - \
            1, f'{len(middle)} != {n - degree - 1}'
        knot = np.concatenate(
            [np.full(degree + 1, 0), middle, np.full(degree + 1, n)])  # pin the ends
    else:
        knot = middle

    knot = knot / knot.max()

    return list(knot)


def blender_mesh_from_pydata(points, edges, faces, uvs=None, name="pydata_mesh"):

    mesh = bpy.data.meshes.new(name=name)
    mesh.from_pydata(points, edges, faces)

    # blender likes to implicitly create new verts/edges/faces if you mess up
    # make sure we are specifying everything precisely
    assert len(mesh.vertices) == len(points)
    if edges is not None:
        assert len(mesh.edges) == len(edges)
    if faces is not None:
        assert len(mesh.polygons) == len(faces)

    if uvs is not None:
        mesh.uv_layers.active = mesh.uv_layers.new()
        for loop in mesh.loops:
            i = loop.vertex_index
            mesh.uv_layers.active.data[loop.index].uv = uvs[i]

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)

    return obj


def blender_nurbs_to_geomdl(s: bpy.types.Spline) -> NURBS.Surface:
    surf = NURBS.Surface(normalize_kv=False)

    surf.degree_u, surf.degree_v = (s.order_u - 1, s.order_v - 1)
    surf.ctrlpts_size_u = s.point_count_u + (s.order_u - 1 if s.use_cyclic_u else 0)
    surf.ctrlpts_size_v = s.point_count_v + (s.order_v - 1 if s.use_cyclic_v else 0)

    if bnurbs is None:
        logger.warning(f'Failed to import compiled `bnurbs` package, either installation failed or we are running a minimal install')
    surf.knotvector_u = bnurbs.get_knotsu(s)
    surf.knotvector_v = bnurbs.get_knotsv(s)

    ctrlpts = np.empty((len(s.points), 4))
    for i, p in enumerate(s.points):
        ctrlpts[i] = p.co

    # geomdl has no notion of cyclic, needs to duplicate points
    # IMPORTANT: blender stores u as the faster changing index
    ctrlpts = ctrlpts.reshape((s.point_count_v, s.point_count_u, 4))
    if s.use_cyclic_u:
        ctrlpts = np.concatenate([ctrlpts, ctrlpts[:, 0:s.order_u - 1, :]], axis=1)
    if s.use_cyclic_v:
        ctrlpts = np.concatenate([ctrlpts, ctrlpts[0:s.order_v - 1, :, :]], axis=0)

    ctrlpts = ctrlpts.transpose(1,0,2).reshape((-1,4))
        
    surf.ctrlpts = ctrlpts[:, :-1]
    surf.weights = ctrlpts[:, -1]
    return surf

def geomdl_to_mesh(surf: NURBS.Surface, eval_delta, name="geomdl_mesh"):
    surf.delta = eval_delta
    points = np.array(surf.evalpts)

    edges, faces = compute_cylinder_topology(
    surf.sample_size_u, surf.sample_size_v, cyclic=False)

    mesh = bpy.data.meshes.new(name=name)
    mesh.from_pydata(points, edges, faces)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj

def map_param_to_valid_domain(knots: np.array, order: int, u: np.array, cyclic: bool):
    u_start, u_end = knots[[order - 1, -order]]
    if not cyclic and ((u_start > u).any() or (u > u_end).any()):
        raise ValueError("out of domain parameter")
    _, r = np.divmod(u - u_start, u_end - u_start)
    return r + u_start

# for cyclic u or v, wrap them around to valid domain.
# raise exception if not cyclic and out of domain
def map_uv_to_valid_domain(s: bpy.types.Spline, uv: np.array):
    knotsu = bnurbs.get_knotsu(s)
    knotsv = bnurbs.get_knotsv(s)
    u = map_param_to_valid_domain(knotsu, s.order_u, uv[:,0], s.use_cyclic_u)
    v = map_param_to_valid_domain(knotsv, s.order_v, uv[:,1], s.use_cyclic_v)
    return np.stack([u,v], axis=-1)

def geomdl_nurbs(ctrlpts, eval_delta, ws=None, kv_u=None, kv_v=None, name='loft_nurbs', cyclic_v=True):

    n, m, _ = ctrlpts.shape
    degree_u, degree_v = (3, 3)

    if isinstance(kv_u, str):
        kv_u = generate_knotvector(degree_u, n, mode=kv_u)
    if isinstance(kv_v, str):
        kv_v = generate_knotvector(kv_v, m, mode=kv_v)

    surf = NURBS.Surface(normalize_kv=False)

    surf.degree_u, surf.degree_v = (degree_u, degree_v)
    surf.ctrlpts_size_u, surf.ctrlpts_size_v = n, m + cyclic_v * degree_v

    if cyclic_v:  # wrap around p control points
        ctrlpts = np.concatenate([ctrlpts, ctrlpts[:, 0:degree_v, :]], axis=1)
        if ws is not None:
            ws = np.concatenate([ws, ws[:, 0:degree_v, :]], axis=1)

    surf.ctrlpts = ctrlpts.reshape(-1, 3)
    if ws is not None:
        surf.weights = ws

    surf.knotvector_u = generate_knotvector(
        surf.degree_u, n) if kv_u is None else list(kv_u)

    # uniform spacing is generally recommended, especially for cyclic v
    if kv_v is None:
        kv_v = np.array(generate_knotvector(surf.degree_v, m,
                        mode='uniform', clamped=not cyclic_v))

    if cyclic_v:  # wrap around p knot intervals
        kv_v = np.append(kv_v, kv_v[1:degree_v+1] + kv_v[-1] - kv_v[0])
    surf.knotvector_v = list(kv_v)

    surf.delta = eval_delta

    points = np.array(surf.evalpts)
    if cyclic_v:  # drop the last point (which is a duplicate) for each loop
        points = points.reshape(
            surf.sample_size_u, surf.sample_size_v, -1)[:, :-1, :].reshape(-1, 3)

    edges, faces, uvs = compute_cylinder_topology(surf.sample_size_u, surf.sample_size_v - cyclic_v, 
        cyclic=cyclic_v, uvs=True)
    return blender_mesh_from_pydata(points, edges, faces, uvs=uvs, name=name)


def nurbs(ctrlpts, method, face_size=0.01, debug=False, **kwargs):

    n, m, _ = ctrlpts.shape
    ulength = np.linalg.norm(np.diff(ctrlpts, axis=0),
                             axis=-1).sum(axis=0).max()
    vlength = np.linalg.norm(np.diff(ctrlpts, axis=1),
                             axis=-1).sum(axis=1).max()

    if method == 'geomdl':
        steps = face_size / max(ulength, vlength)
        obj = geomdl_nurbs(ctrlpts, steps, **kwargs)
    elif method == 'blender':
        resolution = np.clip(
            np.array([ulength, vlength])/face_size, 6, 40).astype(int)
        resolution = (6, 6)
        obj = blender_nurbs(ctrlpts, resolution=resolution)
    elif method == 'subdiv':
        upres_fac = max(ulength/n, vlength/m) / face_size
        level = math.ceil(np.log2(upres_fac))
        obj = subdiv_mesh_nurbs(ctrlpts, level=np.clip(level, 2, 7), **kwargs)
    else:
        raise ValueError(f'Unrecognized nurbs({method=})')

    if debug:
        handles = butil.spawn_point_cloud('handles', ctrlpts.reshape(-1, 3))
        handles.parent = obj

    return obj
