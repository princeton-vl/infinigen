# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick


import pdb
import warnings

import bpy, mathutils
from mathutils.bvhtree import BVHTree
from mathutils import geometry, Vector

from infinigen.core.util import blender as butil

from infinigen.assets.creatures.util.geometry.nurbs import compute_cylinder_topology, blender_mesh_from_pydata

import numpy as np

def invert_line(line, point, eps=1e-4):

    '''
    assumes point is on line = (p, v)
    returns t st `p + v * t = point`
    '''

    if line[0] is None or line[1] is None:
        raise ValueError()

    div = np.array(point - line[0]) / np.array(line[1])
    div = div[~np.isnan(div)]
    return div.mean()

def intersect_line_seg(line, seg):
    v1, v2 = seg
    line_start, line_dir = line
    res = geometry.intersect_line_line(line_start, line_start + line_dir, v1, v2)
    
    if res is None:
        return None, None
    
    lp, vp = res

    t = invert_line((v1, v2 - v1), vp)
    if t < 0 or t > 1:
        return None, None

    return lp, vp

def find_poly_line_bounds(mesh, poly_idx, line, eps=1e-5):

    '''
    assumes `mesh.polygons[poly_idx]` is valid, convex, and contains `line`
    returns t1, t2 such that `for all t1<t<t2: line(t) \in poly`
    '''

    poly = mesh.polygons[poly_idx]

    # find all line - polyedge intersections
    ts, dists = [], []
    for vi_1, vi_2 in poly.edge_keys:
        v1, v2 = mesh.vertices[vi_1].co, mesh.vertices[vi_2].co
        lp, vp = intersect_line_seg(line, (v1, v2))
        if lp is None:
            continue
        ts.append(invert_line(line, lp))    
        dists.append((lp - vp).length)
    ts = np.array(ts)
    dists = np.array(dists)

    mask = dists < eps
    if mask.sum() < 2:
        raise ValueError(f'find_poly_line_bounds(..., {eps=}) had {mask.sum()} intersections, maybe increase eps. {ts=}, {dists=}')
    ts = ts[mask]

    return float(ts.min()), float(ts.max())

def intersect_poly_poly(am: bpy.types.Mesh, bm: bpy.types.Mesh, ai: int, bi: int, return_normals=False):
    
    '''
    ai, bi = polygon index into am.polygons, bm.polygons
    '''

    # find line of intersection
    ap, bp = am.polygons[ai], bm.polygons[bi]
    ap_pos, bp_pos = am.vertices[ap.vertices[0]].co, bm.vertices[bp.vertices[0]].co
    line = geometry.intersect_plane_plane(ap_pos, ap.normal, bp_pos, bp.normal)
    if line == (None, None):
        return None
    
    a_tmin, a_tmax = find_poly_line_bounds(am, ai, line)
    b_tmin, b_tmax = find_poly_line_bounds(bm, bi, line)

    tmin = max(a_tmin, b_tmin)
    tmax = min(a_tmax, b_tmax)
    if tmin > tmax:
        return None
    
    p0 = np.array(line[0] + tmin * line[1])
    p1 = np.array(line[0] + tmax * line[1])
    
    if not return_normals:
        return p0, p1

    raise NotImplementedError
    
def normal_offset_verts(verts, pusher_bvh, snap_to_bvh, dist):
    offset_verts = np.empty_like(verts)
    for i, v in enumerate(verts):
        _, push_normal, _, _ = pusher_bvh.find_nearest(v)
        pushed = v + push_normal * dist
        snapped, _, _, _ = snap_to_bvh.find_nearest(pushed)
        offset_verts[i] = snapped
    return offset_verts

def compute_intersection_curve(a, b, a_bvh, b_bvh, simplify_thresh=1.5e-2):

    overlap = a_bvh.overlap(b_bvh)
    segs = [intersect_poly_poly(a.data, b.data, ai, bi) for ai, bi in overlap]
    segs = np.array([s for s in segs if s is not None])
    
    # join and merge by distance
    m = len(overlap)
    loop_verts = segs.reshape(2*m, 3)
    pair_edges = np.arange(2*m).reshape(-1, 2)
    obj = blender_mesh_from_pydata(loop_verts, pair_edges, [])

    butil.merge_by_distance(obj, simplify_thresh)

    return obj

def create_bevel_connection(
    a, b, a_bvh: BVHTree, b_bvh: BVHTree, 
    width: float, segments=9,
    close_caps=True, intersection_curve=None,
):
    
    inter = intersection_curve or compute_intersection_curve(a, b, a_bvh, b_bvh)

    verts = np.empty((len(inter.data.vertices), 3))   
    edges = np.empty((len(inter.data.edges), 2), dtype=int)
    inter.data.vertices.foreach_get("co", verts.ravel())
    inter.data.edges.foreach_get('vertices', edges.ravel())

    if intersection_curve is None:
        # only delete it if we made it ourselvse
        butil.delete(inter)

    if len(verts) < 3:
        raise ValueError(f'create_bevel_connection({a=}, {b=}) had only {len(verts)=} intersecting points')

    a_offset = normal_offset_verts(verts, a_bvh, b_bvh, width)
    b_offset = normal_offset_verts(verts, b_bvh, a_bvh, width)

    final_vert_parts = [a_offset, verts, b_offset ]
    if close_caps:
        close = lambda vs: np.ones_like(vs) * vs.mean(axis=0, keepdims=True)
        final_vert_parts = [close(a_offset)] + final_vert_parts + [close(b_offset)]

    final_verts = np.concatenate(final_vert_parts, axis=0)
    final_edges, final_faces = compute_cylinder_topology(len(final_vert_parts), len(verts), cyclic=True, h_neighbors=edges)
    final = blender_mesh_from_pydata(final_verts, final_edges.reshape(-1, 2), final_faces)

    def select_loop(li):
        in_loop = lambda vi: (li * len(verts) <= vi) and (vi < (li + 1) * len(verts))
        for vi, v in enumerate(final.data.vertices):
            v.select = in_loop(vi)
        for e in final.data.edges:
            e.select = in_loop(e.vertices[0]) and in_loop(e.vertices[1])

    with butil.ViewportMode(final, 'EDIT'):
        
        if close_caps:
            select_loop(0)
            bpy.ops.mesh.mark_sharp()
            select_loop(len(final_vert_parts) - 1)
            bpy.ops.mesh.mark_sharp()

        center_part_idx = next(i for i, v in enumerate(final_vert_parts) if v is verts)
        select_loop(center_part_idx)   
        bpy.ops.mesh.bevel(offset_type='PERCENT', offset_pct=98, segments=segments)

        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.quads_convert_to_tris(quad_method='BEAUTY', ngon_method='BEAUTY')
        bpy.ops.mesh.normals_make_consistent(inside=False)

    return final

def smooth_around_line(obj, line_obj, rad, iters=30, factor=0.9):
    
    '''
    Assumes: polyline is fairly densely sampled with points,
        obj and line_obj have same transform
    '''

    assert obj.matrix_world == line_obj.matrix_world

    kd = mathutils.kdtree.KDTree(len(line_obj.data.vertices))
    for i, v in enumerate(line_obj.data.vertices):
        kd.insert(v.co, i)
    kd.balance()

    ds = np.array([kd.find(v.co)[2] for v in obj.data.vertices])
    for i, v in enumerate(obj.data.vertices):
        v.select = (ds[i] < rad)

    with butil.ViewportMode(obj, mode='EDIT'):
        bpy.ops.mesh.vertices_smooth(repeat=iters, factor=0.9)

        






    
 
    
    

        

