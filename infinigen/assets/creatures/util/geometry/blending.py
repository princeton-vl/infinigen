# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Jia Deng


"""
Module for constructing blending surfaces
See exmaple_use() for an example
"""
from __future__ import annotations
from multiprocessing.sharedctypes import Value
from typing import Callable, Iterable, Literal, Tuple
from networkx.classes import ordered
import numpy as np
import bpy
import mathutils
from geomdl import NURBS

from infinigen.core.util.math import rotate_match_directions, normalize, project_to_unit_vector
from infinigen.core.util.math import wrap_around_cyclic_coord, new_domain_from_affine, affine_from_new_domain
from infinigen.core.util.math import FixedSeed
from infinigen.core.util import blender as butil

from infinigen.assets.creatures.util.geometry.nurbs import blender_mesh_from_pydata, compute_cylinder_topology
from infinigen.assets.creatures.util.geometry import nurbs
from infinigen.assets.creatures.util.geometry import nurbs, lofting, skin_ops

raise NotImplementedError ('blending.py not currently used, please re-add shapely as a dependency and delete this line')

from shapely.geometry import Polygon, Point, LineString
import shapely
import rtree

class CurveND:
    def __init__(self, eval_fn: Callable[[np.array], Tuple[np.array, np.array]], dim, domain=(0, 1)):
        self._eval_fn = eval_fn
        if domain[1] <= domain[0]:
            raise ValueError("invalid domain")
        self._domain = domain
        self._dim = dim

    @property
    def domain(self):
        return self._domain

    @property
    def eval_fn(self):
        return self._eval_fn

    @property
    def dim(self):
        return self._dim

    def evaluate_points_and_derivatives_at_t(self, t: np.array) -> Tuple[np.array, np.array]:
        if (t < self.domain[0]).any() or (t > self.domain[1]).any():
            raise ValueError("out of domain t value")

        points, derivatives = self._eval_fn(t)

        if points.shape != t.shape + (self.dim,):
            raise ValueError(
                f"points {points.shape} has wrong shape, {t.shape}")
        if derivatives.shape != t.shape + (self.dim,):
            raise ValueError("derivatives has wrong shape")

        return (points, derivatives)

    def evaluate_points_and_derivatives(self, resolution: int) -> Tuple[np.array, np.array]:
        t = np.linspace(self.domain[0], self.domain[1], resolution)
        return self.eval_points_and_derivatives_at_t(t)

    def affine_transform_domain(self, a=1, b=0) -> CurveND:
        """get a new curve (u(f(t)), v(f(t)) where f(t) = a * t + b"""
        new_domain = new_domain_from_affine(self.domain, a, b)

        def new_eval_fn(t):
            ft = a * t + b
            p, d = self.eval_fn(ft)
            return (p, a * d)
        return CurveND(new_eval_fn, self.dim, domain=new_domain)

    def affine_new_domain(self, new_domain=(0, 1)) -> CurveND:
        """ get an equivalent curve whose domain is new_domain.
        Here new_domain as (1, 0) is valid: the new curve will have domain (0,1) 
        but with a flipped axis
        """
        a, b = affine_from_new_domain(self.domain, new_domain)
        return self.affine_transform_domain(a, b)

    def sub_curve(self, interval=(0, 1)) -> CurveND:
        """ get a new curve restricted to interval [a,b] """
        return CurveND(self.eval_fn, self.dim, domain=interval)


class Curve2DFactory:
    @staticmethod
    def circle(center, start_pos, arc_angle=2 * np.pi) -> CurveND:
        center = np.array(center)
        start_pos = np.array(start_pos)
        if center.shape != (2,) or start_pos.shape != (2,):
            raise ValueError(
                f"wrong shapes for center {center.shape} or start_pos {start_pos.shape}")
        r = start_pos - center
        rad = np.linalg.norm(r)
        start_angle = np.arccos(r[0]/rad)
        if r[1] < 0:
            start_angle = 2 * np.pi - start_angle

        def eval_fn(t):
            uv = np.stack([np.cos(t * arc_angle + start_angle) * rad + center[0],
                           np.sin(t * arc_angle + start_angle) * rad + center[1]], axis=-1)
            duvdt = rad * arc_angle * np.stack([-np.sin(t * arc_angle + start_angle),
                                                np.cos(t * arc_angle + start_angle)], axis=-1)
            return (uv, duvdt)
        return CurveND(eval_fn, dim=2)

    @staticmethod
    def nurbs(ctrlpts: np.array, degree=3, knots: np.array = None, weights: np.array = None, make_cyclic=False) -> CurveND:
        """returns a 2D curve. extra dimensions in ctrlpts are ignored
        If make_cyclic is True, will create new control points and knots to wrap around 
        If knots is not specified, defaults to clamped uniform, unless make_cyclic is True, in which case defaults to unclamped uniform
        """
        if ctrlpts.shape[-1] < 2:
            raise ValueError(
                f"control points have wrong shape {ctrlpts.shape}")
        ctrlpts = ctrlpts[..., :2].reshape(-1, 2)

        if make_cyclic:
            ctrlpts = np.concatenate([ctrlpts, ctrlpts[0:degree, :]], axis=0)
            if knots is None:
                knots = np.arange(len(ctrlpts) + degree + 1)
            else:
                knots = knots.append(knots[0:degree])
            if weights is not None:
                weights = weights.append(weights[0:degree])
        else:
            if knots is None:
                knots = np.arange(len(ctrlpts) + degree + 1)
                knots[0:degree] = knots[degree]
                knots[-degree:] = knots[-degree-1]

        curve = NURBS.Curve(normalize_kv=False)
        curve.degree = degree
        curve.ctrlpts = ctrlpts
        curve.knotvector = knots
        if weights is not None:
            curve.weights = weights

        def eval_fn(t):
            d_tmp = np.empty(t.shape + (2, 2))
            for i in np.ndindex(t.shape):
                d_tmp[i] = np.array(curve.derivatives(t[i], order=1))
            uv = d_tmp[..., 0, :]
            duvdt = d_tmp[..., 1, :]
            return (uv, duvdt)

        return CurveND(eval_fn, dim=2, domain=curve.domain)


class UVMesh:
    def __init__(self, uvpoints, edges, faces, cyclic_v=False, pos_cross_edges=None, domain=None):
        if uvpoints.shape[-1] != 2:
            raise ValueError("wrong shape of uvpoints")

        uvpoints = uvpoints.reshape(-1, 2)
        self._uvpoints = [uv for uv in uvpoints]
        self._uvpoints_deleted = [False] * len(self._uvpoints)

        self._edges_of_point = {i: set() for i in range(len(self._uvpoints))}
        for e in edges:
            self._edges_of_point[e[0]].add(e[1])
            self._edges_of_point[e[1]].add(e[0])

        self._faces = [list(f) for f in faces]
        self._faces_deleted = [False] * len(faces)

        self._face_of_edge = {
            (f[j-1], f[j]): i for i, f in enumerate(self._faces) for j in range(0, len(f))}

        self._cyclic_v = cyclic_v
        self._pos_cross_edges = set(((e[0], e[1]) for e in pos_cross_edges))
        self._domain = domain

        self._vspan = domain[1][1] - domain[1][0]
        voffsets = [np.cumsum([0] + [self._cross_edge_dir((f[i-1], f[i]))
                              for i in range(1, len(f))]) for f in self._faces]

        self._faces_cross_direction = [
            vs[np.argmax(vs != 0)] for vs in voffsets]

        self._polygons = [Polygon([self._uvpoints[p] + np.array([0, c * self._vspan])
                                  for c, p in zip(cs, f)]) for cs, f in zip(voffsets, faces)]

        self._polygons_rshift = [shapely.affinity.translate(
            p, yoff=self._vspan) if self._faces_cross_direction[i] < 0 else None for i, p in enumerate(self._polygons)]
        self._polygons_lshift = [shapely.affinity.translate(
            p, yoff=-self._vspan) if self._faces_cross_direction[i] > 0 else None for i, p in enumerate(self._polygons)]

        self._rtree_idx = rtree.index.Index(
            ((i, x.bounds, None) for ps in [self._polygons, self._polygons_rshift, self._polygons_lshift]
             for i, x in enumerate(ps) if x is not None))

    def export_uvmesh(self):
        """return uvpoints, edges, faces"""
        mask = ~np.array(self._uvpoints_deleted)
        new_ids = np.cumsum(mask)-1
        uvpoints = np.array(self._uvpoints)[mask]
        edges = [(new_ids[pt1], new_ids[pt2]) for pt1 in range(len(self._uvpoints))
                 for pt2 in self._edges_of_point[pt1]
                 if not self._uvpoints_deleted[pt1] and not self._uvpoints_deleted[pt2]]
        faces = [[new_ids[pt] for pt in f_pts] for f_id, f_pts in enumerate(
            self._faces) if not self._faces_deleted[f_id]]
        return (uvpoints, edges, faces)

    @staticmethod
    def from_meshgrid(resolution_u: int, resolution_v: int, domain=((0, 1), (0, 1)), cyclic_v=False) -> UVMesh:
        if cyclic_v and resolution_v <= 3:
            raise ValueError("resoultion v is too low")

        d = domain
        u = np.linspace(d[0][0], d[0][1], resolution_u)
        v = np.linspace(d[1][0], d[1][1], resolution_v)
        uv = np.stack(np.meshgrid(u, v, indexing='ij'), axis=-1)

        # drop the duplicates
        if cyclic_v:
            uv = uv[:, :-1, :]
        edges, faces = compute_cylinder_topology(
            resolution_u, resolution_v - cyclic_v, cyclic_v)

        cross_edges = [e for e in edges if e[0] % (
            resolution_v - 1) == resolution_v - 2 and e[1] % (resolution_v-1) == 0] if cyclic_v else []

        return UVMesh(uv.reshape(-1, 2), edges, faces, cyclic_v, cross_edges, domain)

    def _enclosing_polygon(self, face_id, pt_coords):
        point = Point(pt_coords)
        for poly in [self._polygons[face_id], self._polygons_lshift[face_id], self._polygons_rshift[face_id]]:
            if poly is not None and poly.covers(point):
                return poly
        return None

    def _intersecting_polygons_of_line(self, face_id, line: LineString):
        if line is None:
            return []
        res = []
        for poly in [self._polygons[face_id], self._polygons_lshift[face_id], self._polygons_rshift[face_id]]:
            if poly is not None and poly.covers(line):
                res.append(poly)
        return res

    def _enclosing_faces_polys_edges_verts_of_point(self, coords):
        if self._cyclic_v:
            coords = self._wrap_around_v(coords)
        if not self._within_domain(coords):
            raise ValueError("coords must be within domain")
        candidate_faces = self._rtree_idx.intersection(coords)
        e_faces_polys = [(i, self._enclosing_polygon(i, coords))
                         for i in candidate_faces if not self._faces_deleted[i]]
        e_faces_polys = [(i, p) for i, p in e_faces_polys if p is not None]
        e_edges = [(self._faces[f][i], self._faces[f][(i+1) % len(self._faces[f])]) for f, po in e_faces_polys for i in range(len(self._faces[f]))
                   if shapely.geometry.LineString([po.exterior.coords[i], po.exterior.coords[i+1]]).contains(Point(coords))]
        e_edges = list(
            set(((e[0], e[1]) if e[0] < e[1] else ((e[1], e[0])) for e in e_edges)))
        e_verts = [pt for f, _ in e_faces_polys for pt in self._faces[f] if (
            np.array(coords) == np.array(self._uvpoints[pt])).all()]
        e_verts = list(set(e_verts))
        return (e_faces_polys, e_edges, e_verts)

    @staticmethod
    def _print_polygon(p: Polygon):
        print(list(p.exterior.coords))

    def _print_all(self):
        print("points")
        print([(i, p[0]) for i, p in enumerate(
            zip(self._uvpoints, self._uvpoints_deleted)) if not p[1]])

        print("edges")
        print(sorted(self._edges_of_point.items()))

        print("cross edges")
        print(sorted(self._pos_cross_edges))

        print("faces")
        print([(i, fd[0]) for i, fd in enumerate(
            zip(self._faces, self._faces_deleted)) if not fd[1]])

        print("face_of_edge")
        print(sorted(self._face_of_edge.items()))

        print("polys")
        print([list(p.exterior.coords) for p in self._polygons])

        print("polly lshift")
        print([list(p.exterior.coords)
              for p in self._polygons_lshift if p is not None])

        print("polly rshift")
        print([list(p.exterior.coords)
              for p in self._polygons_rshift if p is not None])

    def _cross_edge_dir(self, e):
        if e in self._pos_cross_edges:
            return 1
        if (e[1], e[0]) in self._pos_cross_edges:
            return -1
        return 0

    def _get_enclosing_polygon(self, new_pt_coords, enclosing_face):
        p_m = self._polygons[enclosing_face]
        p_l = self._polygons_lshift[enclosing_face]
        p_r = self._polygons_rshift[enclosing_face]
        p_e = None
        for p in [p_m, p_l, p_r]:
            if p is not None and p.covers(Point(new_pt_coords)):
                p_e = p
                break
        return p_e

    def _add_face(self, pts):
        f_id = len(self._faces)
        self._faces.append(pts)
        self._faces_deleted.append(False)
        self._face_of_edge.update(
            ((pts[i-1], pts[i]), f_id) for i in range(len(pts)))

        vs = np.cumsum([0] + [self._cross_edge_dir((pts[i-1], pts[i]))
                              for i in range(1, len(pts))])

        cross_dir = vs[np.argmax(vs != 0)]
        self._faces_cross_direction.append(cross_dir)

        poly = Polygon([self._uvpoints[p] + np.array([0, c * self._vspan])
                        for c, p in zip(vs, pts)])
        self._polygons.append(poly)

        poly_r = shapely.affinity.translate(
            poly, yoff=self._vspan) if cross_dir < 0 else None
        poly_l = shapely.affinity.translate(
            poly, yoff=-self._vspan) if cross_dir > 0 else None
        self._polygons_rshift.append(poly_r)
        self._polygons_lshift.append(poly_l)

        self._rtree_idx.insert(f_id, poly.bounds, None)
        if poly_r is not None:
            self._rtree_idx.insert(f_id, poly_r.bounds, None)
        if poly_l is not None:
            self._rtree_idx.insert(f_id, poly_l.bounds, None)

        return f_id

    def _add_point(self, new_pt_coords):
        new_pt = len(self._uvpoints)
        self._uvpoints.append(new_pt_coords)
        self._uvpoints_deleted.append(False)
        self._edges_of_point[new_pt] = set()
        return new_pt

    def _delete_edge(self, pt1, pt2):
        self._edges_of_point[pt1].remove(pt2)
        self._edges_of_point[pt2].remove(pt1)
        if self._cyclic_v:
            self._pos_cross_edges.discard((pt1, pt2))
            self._pos_cross_edges.discard((pt2, pt1))

    def _add_edge(self, pt1, pt2, cross_dir=0):
        self._edges_of_point[pt1].add(pt2)
        self._edges_of_point[pt2].add(pt1)
        if self._cyclic_v and cross_dir != 0:
            self._pos_cross_edges.add(
                (pt1, pt2) if cross_dir > 0 else (pt2, pt1))

    def _get_cross_dir_from_pt(self, pt, enclosing_face, enclosing_polygon):
        p_e = enclosing_polygon
        pt_idx = self._faces[enclosing_face].index(pt)
        pt = p_e.exterior.coords[pt_idx]
        pt_poly_co = p_e.exterior.coords[pt_idx]
        cross_dir = 1 if pt_poly_co[1] < self._domain[1][0] else (
            -1 if pt_poly_co[1] >= self._domain[1][1] else 0)
        return cross_dir

    def _delete_edge_and_merge_faces(self, pt1, pt2):
        f1 = self._face_of_edge.pop((pt1, pt2))
        f2 = self._face_of_edge.pop((pt2, pt1))

        self._faces_deleted[f1] = True
        self._faces_deleted[f2] = True

        f1_pts = np.array(self._faces[f1])
        f2_pts = np.array(self._faces[f2])

        if len(set(f1_pts).intersection(set(f2_pts))) > 2:
            raise ValueError(
                "cannot merge faces that share more than one edge")

        new_f_pts = list(np.roll(f1_pts, -np.argmax(f1_pts == pt2))
                         )[:-1] + list(np.roll(f2_pts, -np.argmax(f2_pts == pt1)))[:-1]

        self._delete_edge(pt1, pt2)
        return self._add_face(new_f_pts)

    def _delete_point_and_merge_faces(self, pt):
        """delete pt and its edges and faces"""
        neighbor_pts = [i for i in self._edges_of_point[pt]]
        faces = [self._face_of_edge[(pt, j)] for j in neighbor_pts]
        new_f_pts = neighbor_pts[0:1]
        face_count = 0
        while face_count < len(neighbor_pts):
            cur_pt = new_f_pts[-1]
            f = self._face_of_edge.get((pt, cur_pt), None)
            if f is None:
                break
            f_pts = self._faces[f]
            f_pts = list(np.roll(f_pts, -f_pts.index(pt)))
            new_f_pts.extend(f_pts[2:])
            face_count += 1
        while face_count < len(neighbor_pts):
            cur_pt = new_f_pts[0]
            f = self._face_of_edge.get((cur_pt, pt), None)
            if f is None:
                break
            f_pts = self._faces[f]
            f_pts = list(np.roll(f_pts, -f_pts.index(cur_pt)))
            new_f_pts = f_pts[2:] + new_f_pts
            face_count += 1
        if face_count < len(neighbor_pts):
            raise ValueError("non-manifold mesh")
        if new_f_pts[-1] == new_f_pts[0]:
            new_f_pts = new_f_pts[:-1]

        for f in faces:
            self._faces_deleted[f] = True
        for cur_pt in neighbor_pts:
            self._delete_edge(pt, cur_pt)

        self._uvpoints_deleted[pt] = True

        return self._add_face(new_f_pts)

    def _split_face_with_new_edge(self, face_id, pt1, pt2):
        # todo: make sure new edge is within face
        # check if cross edge
        self._faces_deleted[face_id] = True

        f_pts = self._faces[face_id]
        r_pts = list(np.roll(f_pts, -f_pts.index(pt1)))
        r_pt2_idx = r_pts.index(pt2)

        if r_pt2_idx == 1:
            raise ValueError("(pt1, pt2) is already an edge")

        cross_dir = np.cumsum([self._cross_edge_dir(
            (r_pts[i-1], r_pts[i]))for i in range(1, r_pt2_idx+1)])[-1]

        self._add_edge(pt1, pt2, cross_dir)
        f1 = self._add_face(r_pts[:r_pt2_idx+1])
        f2 = self._add_face(r_pts[r_pt2_idx:] + [pt1])
        return (f1, f2)

    def _split_face_with_new_point(self, face_id, pt_coords):
        new_pt = self._add_point(pt_coords)
        f_pts = self._faces[face_id]
        p_e = self._get_enclosing_polygon(pt_coords, face_id)
        if p_e is None:
            raise ValueError("no enclosing polygon")

        self._faces_deleted[face_id] = True
        for i in range(len(f_pts)):
            pt1 = f_pts[i-1]
            pt2 = f_pts[i]
            cross_dir1 = self._get_cross_dir_from_pt(pt1, face_id, p_e)
            cross_dir2 = self._get_cross_dir_from_pt(pt2, face_id, p_e)
            self._add_edge(pt1, new_pt, cross_dir1)
            self._add_edge(pt2, new_pt, cross_dir2)
            self._add_face([pt1, pt2, new_pt])
        return new_pt

    def _triangulate_face_from_pt(self, face_id, pt):
        f_pts = self._faces[face_id]
        if len(f_pts) <= 3:
            return 0

        poly = self._polygons[face_id]

        pt_idx = f_pts.index(pt)
        pt_coords = poly.exterior.coords[pt_idx]
        r_pts = np.roll(f_pts, -pt_idx)

        pt1_coords = poly.exterior.coords[f_pts.index(r_pts[1])]
        for pt2 in r_pts[2:-1]:
            pt2_coords = poly.exterior.coords[f_pts.index(pt2)]
            line = LineString([pt_coords, pt2_coords])
            if poly.covers(line) and \
                    not line.covers(Point(pt1_coords)):
                new_f0, new_f = self._split_face_with_new_edge(
                    face_id, pt, pt2)
                return 1 + self._triangulate_face_from_pt(new_f, pt)
        return 0

    def _triangulate_all_faces_of_point(self, pt):
        for f in self._faces_of_point(pt):
            self._triangulate_face(f)

    def _triangulate_face(self, face_id):
        poly = self._polygons[face_id]
        f_pts = self._faces[face_id]
        if len(f_pts) <= 3:
            return
        for i, pt1 in enumerate(f_pts):
            for j, pt2 in enumerate(f_pts):
                if i != j and pt2 not in self._edges_of_point[pt1]:
                    line = LineString(
                        [poly.exterior.coords[i], poly.exterior.coords[j]])
                    if poly.covers(line) and \
                            not poly.exterior.covers(line):
                        f1, f2 = self._split_face_with_new_edge(
                            face_id, pt1, pt2)
                        self._triangulate_face(f1)
                        self._triangulate_face(f2)
                        return

    def _split_edge_with_new_point(self, pt1, pt2, pt_coords, enclosing_face=None, enclosing_polygon=None):
        new_pt = self._add_point(pt_coords)

        f1 = self._face_of_edge.pop((pt1, pt2), None)
        f2 = self._face_of_edge.pop((pt2, pt1), None)

        if f1 is not None:
            self._faces_deleted[f1] = True
            if enclosing_face is None:
                enclosing_face = f1
                enclosing_polygon = self._get_enclosing_polygon(
                    pt_coords, enclosing_face)
        if f2 is not None:
            self._faces_deleted[f2] = True
            if enclosing_face is None:
                enclosing_face = f2
                enclosing_polygon = self._get_enclosing_polygon(
                    pt_coords, enclosing_face)

        self._delete_edge(pt1, pt2)

        cross_dir1 = self._get_cross_dir_from_pt(
            pt1, enclosing_face, enclosing_polygon)
        cross_dir2 = self._get_cross_dir_from_pt(
            pt2, enclosing_face, enclosing_polygon)
        self._add_edge(pt1, new_pt, cross_dir1)
        self._add_edge(pt2, new_pt, cross_dir2)

        # done adding edges and points. now add faces
        f1_pts = self._faces[f1] if f1 is not None else None
        f2_pts = self._faces[f2] if f2 is not None else None
        f1_pts.insert(f1_pts.index(pt1) + 1,
                      new_pt) if f1 is not None else None
        f2_pts.insert(f2_pts.index(pt2) + 1,
                      new_pt) if f2 is not None else None
        if f1 is not None:
            new_f1 = self._add_face(f1_pts)
            self._triangulate_face_from_pt(new_f1, new_pt)
        if f2 is not None:
            new_f2 = self._add_face(f2_pts)
            self._triangulate_face_from_pt(new_f2, new_pt)

        return new_pt

    def _wrap_around_v(self, uv):
        uv = np.array(uv)
        new_uv = uv.copy()
        new_uv[..., 1] = wrap_around_cyclic_coord(uv[..., 1], *self._domain[1])
        return new_uv

    def _within_domain(self, uv):
        uv = np.array(uv)
        return (self._domain[0][0] <= uv[..., 0]).all() and (uv[..., 0] < self._domain[0][1]).all() \
            and (self._domain[1][0] <= uv[..., 1]).all() and (uv[..., 1] < self._domain[1][1]).all()

    def _edges_of_face(self, f):
        return [(f[i-1], f[i]) for i in range(len(self._faces[f]))]

    def _faces_of_point(self, pt):
        return [self._face_of_edge[(pt, j)] for j in self._edges_of_point[pt] if (pt, j) in self._face_of_edge]

    def _poly_of_point_on_face(self, face_id, pt):
        for poly in [self._polygons[face_id], self._polygons_lshift[face_id], self._polygons_rshift[face_id]]:
            if poly is not None and \
                    self._within_domain(np.array(poly.exterior.coords[self._faces[face_id].index(pt)])):
                return poly

    def add_edge_and_remesh(self, pt1, pt2, cross_dir=0):
        if self._uvpoints_deleted[pt1] or self._uvpoints_deleted[pt2]:
            raise ValueError("pt1 or pt2 does not exist")
        if pt2 in self._edges_of_point[pt1]:
            if (pt1, pt2) in self._face_of_edge:
                self._triangulate_face(self._face_of_edge[(pt1, pt2)])
            if (pt2, pt1) in self._face_of_edge:
                self._triangulate_face(self._face_of_edge[(pt2, pt1)])
            return

        shared_f = set(self._faces_of_point(pt1)).intersection(
            set(self._faces_of_point(pt2)))
        if len(shared_f) > 0:
            if len(shared_f) > 1:
                raise ValueError("non-convex faces or redudant points")
            f1, f2 = self._split_face_with_new_edge(
                list(shared_f)[0], pt1, pt2)
            self._triangulate_face(f1)
            self._triangulate_face(f2)
            return

        pt1_coords = self._uvpoints[pt1]
        pt2_coords = self._uvpoints[pt2].copy()
        pt2_coords[1] += cross_dir * self._vspan
        line = LineString([pt1_coords, pt2_coords])

        for pt in self._edges_of_point[pt1]:
            pt_coords = self._uvpoints[pt].copy()
            pt_coords[1] += self._cross_edge_dir((pt1, pt)) * self._vspan
            if line.covers(Point(pt_coords)):
                new_f = self._delete_point_and_merge_faces(pt)
                self._triangulate_face_from_pt(new_f, pt1)
                self.add_edge_and_remesh(pt1, pt2, cross_dir)
                self._triangulate_all_faces_of_point(pt1)
                return

        for f in self._faces_of_point(pt1):
            f_pts = self._faces[f]

            poly = self._poly_of_point_on_face(f, pt1)

            # for numerical stability. get pt1_coords from poly
            line = LineString(
                [poly.exterior.coords[f_pts.index(pt1)], pt2_coords])

            if poly.crosses(line):
                for i in range(0, len(f_pts)):
                    q1 = f_pts[i]
                    q2 = f_pts[(i+1) % len(f_pts)]
                    q1_coords = poly.exterior.coords[i]
                    q2_coords = poly.exterior.coords[i+1]
                    if LineString([q1_coords, q2_coords]).crosses(line):
                        new_f = self._delete_edge_and_merge_faces(q1, q2)
                        self._triangulate_face_from_pt(new_f, pt1)
                        self.add_edge_and_remesh(pt1, pt2, cross_dir)
                        self._triangulate_all_faces_of_point(pt1)
                        return
                raise ValueError("should never be here")

        raise ValueError("should never be here")

    def add_point_and_remesh(self, pt_coords):
        pt_coords = np.array(pt_coords)
        if not self._within_domain(pt_coords):
            raise ValueError("pt_coords must be within domain")
        e_faces_polys, e_edges, e_verts = self._enclosing_faces_polys_edges_verts_of_point(
            pt_coords)

        new_pt = None
        if len(e_verts) > 0:
            if len(e_verts) > 1:
                raise ValueError("cannot have more than 1 e_vert")
            new_pt = e_verts[0]
        elif len(e_edges) > 0:
            if len(e_edges) > 1:
                raise ValueError("cannot have more than 1 e_edge")
            e = e_edges[0]
            new_pt = self._split_edge_with_new_point(e[0], e[1], pt_coords)
        else:
            if len(e_faces_polys) != 1:
                raise ValueError("must have an enclosing face")
            f, _ = e_faces_polys[0]
            new_pt = self._split_face_with_new_point(f, pt_coords)
        return new_pt

    def add_line_and_remesh(self, start_coords, end_coords):
        pt1 = self.add_point_and_remesh(self._wrap_around_v(start_coords))
        pt2 = self.add_point_and_remesh(self._wrap_around_v(end_coords))
        q1 = (start_coords[1] - self._domain[1][0]) // self._vspan
        q2 = (end_coords[1] - self._domain[1][0]) // self._vspan
        cross_dir = q2 - q1
        if cross_dir > 0:
            cross_dir = 1
        if cross_dir < 0:
            cross_dir = -1
        self.add_edge_and_remesh(pt1, pt2, cross_dir)
        return (pt1, pt2)

    def add_poly_curve_and_remesh(self, uvpoints: np.array, cyclic_curve=False, vloop=False):
        if uvpoints.shape[-1] != 2:
            raise ValueError("wrong shape of curve")
        uvpoints = uvpoints.reshape(-1, 2)
        pts = []
        for i in range(len(uvpoints)-1):
            pt1, pt2 = self.add_line_and_remesh(uvpoints[i], uvpoints[i+1])
            if i == 0:
                pts.append(pt1)
            else:
                if pt1 != pts[-1]:
                    raise ValueError("numerical issues!")
            pts.append(pt2)
        if cyclic_curve:
            if vloop:
                cross_dir = 1 if uvpoints[-1][1] >= uvpoints[0][1] else -1
                self.add_edge_and_remesh(pts[-1], pts[0], cross_dir)
            else:
                self.add_line_and_remesh(uvpoints[-1], uvpoints[0])
        return pts

    def add_curve_and_remesh(self, curve: CurveND, resolution: int, cyclic_curve=False, vloop=False,
                             cut_inside=False, cut_outside=False) -> Iterable[int]:
        t = np.linspace(*curve.domain, resolution)
        if cyclic_curve:
            t = t[:-1]
        uvpoints, _ = curve.evaluate_points_and_derivatives_at_t(t)
        pts = self.add_poly_curve_and_remesh(uvpoints, cyclic_curve, vloop)
        if cyclic_curve and (cut_inside or cut_outside):
            comps = self.connected_components(pts)
            if len(comps) < 2:
                raise ValueError("two few components")
            poly_points = list(uvpoints)
            if vloop:
                head = uvpoints[0].copy()
                tail1 = uvpoints[0].copy()
                tail2 = uvpoints[-1].copy()
                head[0] = self._domain[0][0] - 0.1
                tail1[1] += self._vspan
                tail2[0] = self._domain[0][0] - 0.1
                poly_points = [head] + poly_points + [tail1, tail2]
            poly = Polygon(poly_points)
            for comp in comps:
                pt = comp[0]
                pt_coords = self._uvpoints[pt]
                pt_coords1 = pt_coords.copy()
                pt_coords1[1] += self._vspan
                pt_coords2 = pt_coords.copy()
                pt_coords2[1] -= self._vspan
                if poly.covers(Point(pt_coords)) or poly.covers(Point(pt_coords1)) or poly.covers(Point(pt_coords2)):
                    if cut_inside:
                        self.remove_points(comp)
                else:
                    if cut_outside:
                        self.remove_points(comp)
        return pts

    def connected_components(self, boundary_pts: Iterable[int]) -> Iterable[Iterable[int]]:
        color_of_pts = - np.ones(len(self._uvpoints))
        color_of_pts[boundary_pts] = -2
        color_of_pts[self._uvpoints_deleted] = -2
        cur_color = 0
        for i in range(len(color_of_pts)):
            if color_of_pts[i] == -1:
                stack = [i]
                while len(stack) > 0:
                    pt = stack.pop()
                    color_of_pts[pt] = cur_color
                    for pt2 in self._edges_of_point[pt]:
                        if color_of_pts[pt2] == -1:
                            stack.append(pt2)
                cur_color += 1
        res = [[i for i, c in enumerate(color_of_pts) if c == color]
               for color in range(cur_color)]
        return res

    def remove_points(self, pts: Iterable[int]):
        for pt in pts:
            self._uvpoints_deleted[pt] = True
            for f in self._faces_of_point(pt):
                self._faces_deleted[f] = True
            for j in self._edges_of_point[pt]:
                self._edges_of_point[j].remove(pt)
            self._edges_of_point[pt] = set()


class Surface:
    """
    General parametric surface S(u,v)
    eval_fn: evaluation function that returns 3D points and derivatives (s(u,v), ds/du(u,v), ds/dv(u,v))
    See SurfaceFactory for examples of eval_fn
    """

    def __init__(self, eval_fn: Callable[[np.array], Tuple[np.array, np.array]],
                 domain=((0, 1), (0, 1)), cyclic_u=False, cyclic_v=False):
        self._eval_fn = eval_fn

        if (np.array(domain)[:, 1] <= np.array(domain)[:, 0]).any():
            raise ValueError("invalid domain")
        self._domain = tuple(domain)

        self._cyclic_u = cyclic_u
        self._cyclic_v = cyclic_v

    @property
    def eval_fn(self):
        return self._eval_fn

    @property
    def domain(self):
        return self._domain

    @property
    def cyclic_u(self):
        return self._cyclic_u

    @property
    def cyclic_v(self):
        return self._cyclic_v

    def affine_transform_domain(self, ua=1, ub=0, va=1, vb=0) -> Surface:
        """ get a reparameterized surface G(u,v) = S(ua*u + ub, va * v + vb) """
        new_domain_u = new_domain_from_affine(self.domain[0], ua, ub)
        new_domain_v = new_domain_from_affine(self.domain[1], va, vb)
        new_domain = (new_domain_u, new_domain_v)

        def new_eval_fn(uv):
            fuv = np.stack(
                [ua * uv[..., 0] + ub, va * uv[..., 1] + vb], axis=-1)
            p, dsdu, dsdv = self.eval_fn(fuv)
            return (p, dsdu * ua, dsdv * va)
        return Surface(new_eval_fn, new_domain, self.cyclic_u, self.cyclic_v)

    def affine_new_domain(self, new_domain=((0, 1), (0, 1))) -> CurveND:
        """ 
        get an equivalent Surface whose domain is new_domain 
        new_domain such as ((1, 0), (0,1)) is valid; the new surface will still
        have domain ((0,1),(0,1)) but with a flipped u axis
        """
        ua, ub = affine_from_new_domain(self.domain[0], new_domain[0])
        va, vb = affine_from_new_domain(self.domain[1], new_domain[1])
        return self.affine_transform_domain(ua, ub, va, vb)

    def create_mesh(self, resolution_u: int, resolution_v: int):
        points, _, _ = self.evaluate_points_and_derivatives(
            resolution_u, resolution_v)
        points = points.reshape(-1, 3)
        edges, faces = compute_cylinder_topology(
            resolution_u, resolution_v, self.cyclic_v)
        return blender_mesh_from_pydata(points, edges, faces)

    def create_mesh_from_uvmesh(self, uvmesh: UVMesh):
        uvpoints, edges, faces = uvmesh.export_uvmesh()
        points, _, _ = self.evaluate_points_and_derivatives_at_uv(uvpoints)
        return blender_mesh_from_pydata(points, edges, faces)

    def evaluate_points_and_derivatives_at_uv(self, uv: np.array) -> Tuple[np.array, np.array, np.array]:
        if uv.shape[-1] != 2:
            raise ValueError("wrong uv shape")

        def check_domain(u, domain, cyclic, name):
            if cyclic:
                return
            if (u < domain[0]).any() or (u > domain[1]).any():
                raise ValueError(
                    f"{name} out of domain, {domain}, {self.domain}")

        check_domain(uv[..., 0], self.domain[0], self.cyclic_u, "u")
        check_domain(uv[..., 1], self.domain[1], self.cyclic_v, "v")

        eval_uv = uv.copy()
        if self.cyclic_u:
            eval_uv[..., 0] = wrap_around_cyclic_coord(
                uv[..., 0], *self.domain[0])
        if self.cyclic_v:
            eval_uv[..., 1] = wrap_around_cyclic_coord(
                uv[..., 1], *self.domain[1])

        points, derivatives_u, derivatives_v = self._eval_fn(eval_uv)
        if points.shape != uv.shape[:-1] + (3,):
            raise ValueError("points has wrong shape", points.shape, uv.shape)
        if derivatives_u is not None and derivatives_u.shape != uv.shape[:-1] + (3,):
            raise ValueError("derivatives u has wrong shape")
        if derivatives_v is not None and derivatives_v.shape != uv.shape[:-1] + (3,):
            raise ValueError("derivatives v has wrong shape")
        return (points, derivatives_u, derivatives_v)

    def evaluate_points_and_derivatives(self, resolution_u: int, resolution_v: int) -> Tuple[np.array, np.array, np.array]:
        d = self.domain
        u = np.linspace(d[0][0], d[0][1], resolution_u)
        v = np.linspace(d[1][0], d[1][1], resolution_v)
        uv = np.stack(np.meshgrid(u, v, indexing='ij'), axis=-1)
        return self.evaluate_points_and_derivatives_at_uv(uv)


class RailCurve:
    """
    constructs a rail curve on a Surface surf S(u,v) and a Curve2D curve_uv (u(t),v(t))
    The (u(t), v(t)) should be within S(u,v)'s domain
    The constructed rail curve's domain is always (0,1). curve_uv can have arbitrary domain
    """

    def __init__(self, surf: Surface, curve_uv: CurveND):
        self._surf = surf
        self._curve_uv = curve_uv.affine_new_domain((0, 1))

    def evaluate_points_derivatives_normals(self, t: np.array) -> Tuple[np.array, np.array]:
        uv, duvdt = self._curve_uv.evaluate_points_and_derivatives_at_t(t)
        points, dsdu, dsdv = self._surf.evaluate_points_and_derivatives_at_uv(
            uv)
        dcdt = dsdu * duvdt[..., 0, None] + dsdv * duvdt[..., 1, None]
        z = np.cross(dsdu, dsdv)
        return (points, dcdt, z)


class SurfaceFactory:
    @staticmethod
    def from_blender_nurbs(s: bpy.types.Spline) -> Surface:
        surf = nurbs.blender_nurbs_to_geomdl(s)

        def eval_fn(uv):
            d_tmp = np.empty(uv.shape[:-1] + (2, 2, 3))
            for i in np.ndindex(uv.shape[:-1]):
                d_tmp[i] = np.array(surf.derivatives(*uv[i], 1))
            points = d_tmp[..., 0, 0, :]
            dsdu = d_tmp[..., 1, 0, :]
            dsdv = d_tmp[..., 0, 1, :]
            return (points, dsdu, dsdv)
        return Surface(eval_fn, domain=surf.domain, cyclic_u=s.use_cyclic_u, cyclic_v=s.use_cyclic_v)

    @staticmethod
    def plane(center, normal, domain=((-1, 1), (-1, 1))) -> Surface:
        center = np.array(center)
        normal = np.array(normal)

        def eval_fn(uv):
            points = np.concatenate(
                [uv, np.zeros(uv.shape[:-1] + (1,))], axis=-1)
            dsdu = np.zeros_like(points)
            dsdu[..., 0] = 1
            dsdv = np.zeros_like(points)
            dsdv[..., 1] = 1
            upward = np.array([0, 0, 1])
            rot_mat = np.squeeze(rotate_match_directions(
                upward[None], normal[None]))
            points = np.einsum('...ij,...j->...i', rot_mat,
                               points) + center[None]
            dsdu = np.einsum('...ij,...j->...i', rot_mat, dsdu)
            dsdv = np.einsum('...ij,...j->...i', rot_mat, dsdv)

            return (points, dsdu, dsdv)
        return Surface(eval_fn, domain)

    @staticmethod
    def blending(r1: RailCurve, r2: RailCurve, alpha=(0, 0), w=(0.1, 0.1), sweep_left=(False, False)) -> Surface:
        """
        Constructs a blending surface B(s,t) that spans two rail curves r1(t) and r2(t), where s,t in [0,1], 
        such that r1(t) = B(0,t) and r2(t) = s(1,t) for all t in [0,1], and that B(s,t) smoothly blends. 
        The blending surface basically smoothly sweeps one rail curve toward the other to span a new surface. 

        r1(t) and r2(t) share the same t, i.e. with fixed t, r1(t) and r2(t) are "corresponding" points.
        it is important that r1(t) and r2(t) travel in the same direction (e.g. counterclockwise), otherwise you will get 
        a twisted surface. You can flip the travel direction by CurveND.affine_new_domain

        Algorithm by Daniel Filip "Blending parametric surfaces", ACM Trans. on Graphics, 1989, with modification to 
             gaurantee that we sweep a rail curve consistently to one side (left or right, but not mixed) and avoid twisted surfaces

        alpha: parameter in [0,1] that controls the direction in which to sweep a rail curve. if alpha=0, sweep more in the orthogonal direction
               if alipha=1, sweep more in the direction of the corresponding point in the other rail curve. There is an alpha for each rail curve.

        w: parameter in [0,1] that controls the curvature of the blending surface at the rail curve. If w=1, it will approximately give a 
               large circle arc that connects the two rail curves. Large w results in slower transition.  One w for each rail curve

        sweep_left: given a rail curve, we have two choices of constructing a blend: we can sweep to the left side (i.e. leaving the right side visible)
                or right. By default we sweep right. We use right hand convention: z = (u cross v) points to outside (visible side of the base surface), 
                (rail_curve_tangent cross z) points to the right. 
        """
        alpha1, alpha2 = alpha
        w1, w2 = w

        def eval_fn(st):
            s = st[..., 0]
            t = st[..., 1]
            c1, dc1dt, z1 = r1.evaluate_points_derivatives_normals(t)
            c2, dc2dt, z2 = r2.evaluate_points_derivatives_normals(t)

            def _compute_blending_tangent(dcdt, k, z, alpha, w, flip):
                if flip:
                    z = -z

                # unit vector on tangent plane orthogonal to curve pointing to covered side of the curve after blending
                n = normalize(np.cross(dcdt, z), disallow_zero_norm=True)

                N = normalize(n + alpha * project_to_unit_vector(k, dcdt))
                k_norm = np.linalg.norm(k, axis=-1)
                g = k_norm + (k * N).sum(axis=-1)
                l = np.square(k_norm) * 2
                l[g > 0] /= g[g > 0]
                return l[..., None] * w * N

            k = c2 - c1
            T1 = _compute_blending_tangent(
                dc1dt, k, z1, alpha1, w1, sweep_left[0])
            T2 = _compute_blending_tangent(
                dc2dt, k, z2, alpha2, w2, sweep_left[1])

            H1 = s * s * (2 * s - 3) + 1
            H2 = 1 - H1
            H3 = s * np.square(s - 1)
            H4 = s * s * (s - 1)

            points = H1[..., None] * c1 + H2[..., None] * \
                c2 + H3[..., None] * T1 + H4[..., None] * T2
            return (points, None, None)

        return Surface(eval_fn)


def example_use():
    n = 4
    m = 10
    resolution_s = 50
    resolution_t = 50
    with FixedSeed(1103):
        def make_object():
            skin = skin_ops.random_skin(1, n, m)
            skeleton = np.hstack(
                (np.random.normal(0, 0.2, [5, 2]), np.linspace(0, 5, 5).reshape(-1, 1)))
            method = 'blender'
            obj = lofting.loft(skeleton, skin, method=method,
                               face_size=0.1, cyclic_v=True)
            return obj
        obj1 = make_object()
        obj2 = make_object()
        obj2.location = mathutils.Vector((5, 0, 0))

        with butil.SelectObjects([obj1, obj2]):
            bpy.ops.object.transform_apply(
                location=True, rotation=True, scale=True)

        # base surface 1, with domain normalized to ((0,1), (0,1))
        # domain normalization isn't required, but may be convenient for specifying the curve in uv space
        surf1 = SurfaceFactory.from_blender_nurbs(
            obj1.data.splines[0]).affine_new_domain(((0, 1), (0, 1)))
        r1 = RailCurve(
            surf1, Curve2DFactory.circle([0.5, 0.8], [0.5, 0.6]))

        # base surface 2, with domain normalized to ((0,1), (0,1)), but with a flipped u axis
        surf2 = SurfaceFactory.from_blender_nurbs(
            obj2.data.splines[0]).affine_new_domain(((1, 0), (0, 1)))
        r2 = RailCurve(
            surf2, Curve2DFactory.circle([0.5, 0.3], [0.5, 0.5]))

        b = SurfaceFactory.blending(r1, r2, alpha=(
            0, 0), w=(0.5, 0.5), sweep_left=(True, False))
        b.create_mesh(resolution_s, resolution_t)


def example_use2():
    n = 4
    m = 10
    resolution_s = 50
    resolution_t = 50
    with FixedSeed(1103):
        def make_object():
            skin = skin_ops.random_skin(1, n, m)
            skeleton = np.hstack(
                (np.random.normal(0, 0.2, [5, 2]), np.linspace(0, 5, 5).reshape(-1, 1)))
            method = 'blender'
            obj = lofting.loft(skeleton, skin, method=method,
                               face_size=0.1, cyclic_v=True)
            return obj
        obj1 = make_object()
        obj2 = make_object()
        obj2.location = mathutils.Vector((5, 0, 0))

        with butil.SelectObjects([obj1, obj2]):
            bpy.ops.object.transform_apply(
                location=True, rotation=True, scale=True)

        # base surface 1, with domain normalized to ((0,1), (0,1))
        # domain normalization isn't required, but may be convenient for specifying the curve in uv space
        surf1 = SurfaceFactory.from_blender_nurbs(
            obj1.data.splines[0]).affine_new_domain(((0, 1), (0, 1)))
        ctrlpts = np.array([[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [
                           1, -1]]) * np.array([-1, -1])[None] * 0.2 + np.array([0.5, 0.8])[None]
        r1 = RailCurve(
            surf1, Curve2DFactory.nurbs(ctrlpts, make_cyclic=True))

        # base surface 2, with domain normalized to ((0,1), (0,1)), but with a flipped u axis
        surf2 = SurfaceFactory.from_blender_nurbs(
            obj2.data.splines[0]).affine_new_domain(((1, 0), (0, 1)))
        r2 = RailCurve(
            surf2, Curve2DFactory.circle([0.5, 0.3], [0.5, 0.5]))

        b = SurfaceFactory.blending(r1, r2, alpha=(
            0, 0), w=(0.5, 0.5), sweep_left=(True, False))
        b.create_mesh(resolution_s, resolution_t)


def example_use3():
    n = 4
    m = 10
    resolution_s = 50
    resolution_t = 50
    with FixedSeed(1103):
        uv_mesh = UVMesh.from_meshgrid(
            resolution_s, resolution_t, cyclic_v=True)

        def make_object():
            skin = skin_ops.random_skin(1, n, m)
            skeleton = np.hstack(
                (np.random.normal(0, 0.2, [5, 2]), np.linspace(0, 5, 5).reshape(-1, 1)))
            method = 'blender'
            obj = lofting.loft(skeleton, skin, method=method,
                               face_size=0.1, cyclic_v=True)
            return obj
        obj1 = make_object()

        with butil.SelectObjects([obj1, ]):
            bpy.ops.object.transform_apply(
                location=True, rotation=True, scale=True)

        surf1 = SurfaceFactory.from_blender_nurbs(
            obj1.data.splines[0]).affine_new_domain(((0, 1), (0, 1)))
        butil.delete([obj1, ])

        #uv_mesh.add_line_and_remesh((0.2, -0.2), (0.29, 0.1))
        resolution_c = 50
        curve1 = Curve2DFactory.circle([0.5, 0.1], [0.5, 0.3])
        curve2 = Curve2DFactory.nurbs(
            np.array([[0.5, 0.1], [0.5, 1.1]]), degree=1)
        comps0 = uv_mesh.connected_components([])
        if False:
            uv_mesh.add_curve_and_remesh(
                curve2, resolution_c, cyclic_curve=True, vloop=True, cut_inside=True, cut_outside=False)
        else:
            uv_mesh.add_curve_and_remesh(
                curve1, resolution_c, cyclic_curve=True, vloop=False, cut_inside=True, cut_outside=False)
        obj2 = surf1.create_mesh_from_uvmesh(uv_mesh)


def example_use4():
    n = 4
    m = 10
    resolution_s = 50
    resolution_t = 50
    with FixedSeed(1103):
        def make_object():
            skin = skin_ops.random_skin(1, n, m)
            skeleton = np.hstack(
                (np.random.normal(0, 0.2, [5, 2]), np.linspace(0, 5, 5).reshape(-1, 1)))
            method = 'blender'
            obj = lofting.loft(skeleton, skin, method=method,
                               face_size=0.1, cyclic_v=True)
            return obj
        obj1 = make_object()
        obj2 = make_object()
        obj2.location = mathutils.Vector((5, 0, 0))

        with butil.SelectObjects([obj1, obj2]):
            bpy.ops.object.transform_apply(
                location=True, rotation=True, scale=True)

        # base surface 1, with domain normalized to ((0,1), (0,1))
        # domain normalization isn't required, but may be convenient for specifying the curve in uv space
        surf1 = SurfaceFactory.from_blender_nurbs(
            obj1.data.splines[0]).affine_new_domain(((0, 1), (0, 1)))
        ctrlpts = np.array([[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [
                           1, -1]]) * np.array([-1, -1])[None] * 0.2 + np.array([0.5, 0.8])[None]
        curve1 = Curve2DFactory.nurbs(ctrlpts, make_cyclic=True)
        r1 = RailCurve(surf1, curve1)

        # base surface 2, with domain normalized to ((0,1), (0,1)), but with a flipped u axis
        surf2 = SurfaceFactory.from_blender_nurbs(
            obj2.data.splines[0]).affine_new_domain(((1, 0), (0, 1)))
        curve2 = Curve2DFactory.circle([0.5, 0.3], [0.5, 0.5])
        r2 = RailCurve(surf2, curve2)

        b = SurfaceFactory.blending(r1, r2, alpha=(
            0, 0), w=(0.5, 0.5), sweep_left=(True, False))
        b.create_mesh(resolution_s, resolution_t)

        # replace obj1, obj2 with custom mesh
        butil.delete([obj1, obj2])
        uv_mesh1 = UVMesh.from_meshgrid(resolution_s, resolution_t)
        uv_mesh1.add_curve_and_remesh(curve1, resolution_t, cyclic_curve=True, cut_inside=True)
        uv_mesh2 = UVMesh.from_meshgrid(resolution_s, resolution_t)
        uv_mesh2.add_curve_and_remesh(curve2, resolution_t, cyclic_curve=True, cut_inside=True)

        obj1 = surf1.create_mesh_from_uvmesh(uv_mesh1)
        obj2 = surf2.create_mesh_from_uvmesh(uv_mesh2)

