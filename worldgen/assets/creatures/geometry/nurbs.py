import math

import bpy
import bmesh

from geomdl import NURBS, knotvector
import numpy as np

from util import blender as butil
from util.math import randomspacing

def compute_cylinder_topology(n: int, m: int, uvs=False, cyclic=True, h_neighbors=None):

    # n: num vertices in vertical direction
    # m: num vertices in each loop
    # compute ring edges
    if h_neighbors is None:
        h_neighbors = np.stack([loop, np.roll(loop, -1)], axis=-1)
    ring_edges = ring_start_offsets[:, None, None] + h_neighbors[None]
    ring_edges = ring_edges.reshape(-1, 2)

    # compute bridge edges
    v_neighbors = np.stack([loop, loop + m], axis=-1)
    bridge_edges = ring_start_offsets[:, None, None] + v_neighbors[None]
    bridge_edges = bridge_edges.reshape(-1, 2)

    edges = np.concatenate([ring_edges, bridge_edges])

    # compute faces
    faces = ring_start_offsets[:, None, None] + face_neighbors[None]
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

        middle = np.repeat(np.arange(0, n), degree)
    elif mode == 'random_uniform':
    else:
        raise ValueError(f'Unrecognized {mode=} for generate_knotvector')

    knot = knot / knot.max()
    return list(knot)
def blender_mesh_from_pydata(points, edges, faces, uvs=None, name="pydata_mesh"):

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

    if isinstance(kv_u, str):
        kv_u = generate_knotvector(degree_u, n, mode=kv_u)
    if isinstance(kv_v, str):
        kv_v = generate_knotvector(kv_v, m, mode=kv_v)


    surf.delta = eval_delta

    points = np.array(surf.evalpts)
    edges, faces, uvs = compute_cylinder_topology(surf.sample_size_u, surf.sample_size_v - cyclic_v, 
        cyclic=cyclic_v, uvs=True)
    return blender_mesh_from_pydata(points, edges, faces, uvs=uvs, name=name)

def nurbs(ctrlpts, method, face_size=0.01, debug=False, **kwargs):

    n, m, _ = ctrlpts.shape

    if method == 'geomdl':
        steps = face_size / max(ulength, vlength)
        obj = geomdl_nurbs(ctrlpts, steps, **kwargs)
    elif method == 'blender':
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

