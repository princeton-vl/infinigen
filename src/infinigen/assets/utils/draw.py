# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors: Lingjie Mei


from collections.abc import Sized

import bmesh
import bpy
import numpy as np
from numpy.random import uniform
from scipy.interpolate import interp1d

from infinigen.assets.utils.decorate import (
    read_co,
    remove_vertices,
    write_attribute,
    write_co,
)
from infinigen.assets.utils.mesh import polygon_angles
from infinigen.assets.utils.misc import make_circular, make_circular_angle
from infinigen.assets.utils.object import data2mesh, mesh2obj, separate_loose
from infinigen.core.nodes.node_info import Nodes
from infinigen.core.placement.detail import sharp_remesh_with_attrs
from infinigen.core.surface import read_attr_data
from infinigen.core.util import blender as butil


def shape_by_angles(obj, angles, scales=None, displacements=None, method="quadratic"):
    x, y, z = read_co(obj).T
    vert_angles = np.arctan2(y, x)
    if scales is not None:
        f = interp1d(angles, scales, method, bounds_error=False, fill_value=0)
        vert_scales = f(vert_angles)
        write_co(obj, vert_scales[:, np.newaxis] * read_co(obj))
    if displacements is not None:
        g = interp1d(angles, displacements, method, bounds_error=False, fill_value=0)
        vert_displacements = g(vert_angles)
        co = read_co(obj)
        co[:, -1] += vert_displacements * np.linalg.norm(co, axis=-1)
        write_co(obj, co)
    return obj


def shape_by_xs(obj, xs, displacements, method="quadratic"):
    co = read_co(obj)
    f = interp1d(xs, displacements, method, bounds_error=False, fill_value=0)
    vert_displacements = f(co[:, 0])
    co[:, -1] += vert_displacements
    write_co(obj, co)
    return obj


def surface_from_func(fn, div_x=16, div_y=16, size_x=2, size_y=2):
    x, y = np.meshgrid(
        np.linspace(-size_x / 2, size_x / 2, div_x + 1),
        np.linspace(-size_y / 2, size_y / 2, div_y + 1),
    )
    z = fn(x, y)
    vertices = np.stack([x.flatten(), y.flatten(), z.flatten()]).T
    faces = np.array([[0, div_y + 1, div_y + 2, 1]]) + np.expand_dims(
        (
            np.expand_dims(np.arange(div_y), 0)
            + np.expand_dims(np.arange(div_x) * (div_y + 1), 1)
        ).flatten(),
        -1,
    )

    mesh = bpy.data.meshes.new("z_function_surface")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    return mesh


def bezier_curve(anchors, vector_locations=(), resolution=None, to_mesh=True):
    n = [len(r) for r in anchors if isinstance(r, Sized)][0]
    anchors = np.array(
        [
            np.array(r, dtype=float) if isinstance(r, Sized) else np.full(n, r)
            for r in anchors
        ]
    )
    bpy.ops.curve.primitive_bezier_curve_add(location=(0, 0, 0))
    obj = bpy.context.active_object

    if n > 2:
        with butil.ViewportMode(obj, "EDIT"):
            bpy.ops.curve.subdivide(number_cuts=n - 2)
    points = obj.data.splines[0].bezier_points
    for i in range(n):
        points[i].co = anchors[:, i]
    for i in range(n):
        if i in vector_locations:
            points[i].handle_left_type = "VECTOR"
            points[i].handle_right_type = "VECTOR"
        else:
            points[i].handle_left_type = "AUTO"
            points[i].handle_right_type = "AUTO"
    obj.data.splines[0].resolution_u = resolution if resolution is not None else 12
    if not to_mesh:
        return obj
    return curve2mesh(obj)


def curve2mesh(obj):
    points = obj.data.splines[0].bezier_points
    cos = np.array([p.co for p in points])
    length = np.linalg.norm(cos[:-1] - cos[1:], axis=-1)
    min_length = 5e-3
    with butil.ViewportMode(obj, "EDIT"):
        for i in range(len(points)):
            if points[i].handle_left_type == "FREE":
                points[i].handle_left_type = "ALIGNED"
            if points[i].handle_right_type == "FREE":
                points[i].handle_right_type = "ALIGNED"
        for i in reversed(range(len(points) - 1)):
            points = list(obj.data.splines[0].bezier_points)
            number_cuts = min(int(length[i] / min_length) - 1, 64)
            if number_cuts < 0:
                continue
            bpy.ops.curve.select_all(action="DESELECT")
            points[i].select_control_point = True
            points[i + 1].select_control_point = True
            bpy.ops.curve.subdivide(number_cuts=number_cuts)
    obj.data.splines[0].resolution_u = 1
    with butil.SelectObjects(obj):
        bpy.ops.object.convert(target="MESH")
    obj = bpy.context.active_object
    butil.modify_mesh(obj, "WELD", merge_threshold=1e-3)
    return obj


def align_bezier(
    anchors, axes=None, scale=None, vector_locations=(), resolution=None, to_mesh=True
):
    obj = bezier_curve(anchors, vector_locations, resolution, False)
    points = obj.data.splines[0].bezier_points
    if scale is None:
        scale = np.ones(2 * len(points) - 2)
    if axes is None:
        axes = [None] * len(points)
    scale = [1, *scale, 1]
    for i, p in enumerate(points):
        a = axes[i]
        if a is None:
            continue
        a = np.array(a)
        p.handle_left_type = "FREE"
        p.handle_right_type = "FREE"
        proj_left = np.array(p.handle_left - p.co) @ a * a
        p.handle_left = (
            np.array(p.co)
            + proj_left
            / np.linalg.norm(proj_left)
            * np.linalg.norm(p.handle_left - p.co)
            * scale[2 * i]
        )
        proj_right = np.array(p.handle_right - p.co) @ a * a
        p.handle_right = (
            np.array(p.co)
            + proj_right
            / np.linalg.norm(proj_right)
            * np.linalg.norm(p.handle_right - p.co)
            * scale[2 * i + 1]
        )
    if not to_mesh:
        return obj
    return curve2mesh(obj)


def remesh_fill(obj, resolution=0.005):
    n = len(obj.data.vertices)
    butil.modify_mesh(obj, "SOLIDIFY", thickness=0.1)
    write_attribute(
        obj,
        lambda nw, position: nw.compare("GREATER_EQUAL", nw.new_node(Nodes.Index), n),
        "top",
    )
    sharp_remesh_with_attrs(obj, resolution)
    is_top = read_attr_data(obj, "top") > 1e-3
    remove_vertices(obj, lambda x, y, z: is_top)
    obj.data.attributes.remove(obj.data.attributes["top"])
    return obj


def spin(
    anchors,
    vector_locations=(),
    resolution=None,
    rotation_resolution=None,
    axis=(0, 0, 1),
    loop=False,
    dupli=False,
):
    obj = bezier_curve(anchors, vector_locations, resolution)
    co = read_co(obj)
    mean_radius = np.mean(
        np.linalg.norm(
            co - (co @ np.array(axis))[:, np.newaxis] * np.array(axis), axis=-1
        )
    )
    if rotation_resolution is None:
        rotation_resolution = min(int(2 * np.pi * mean_radius / 5e-3), 128)
    butil.modify_mesh(obj, "WELD", merge_threshold=1e-3)
    if loop:
        with butil.ViewportMode(obj, "EDIT"), butil.Suppress():
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.fill()
        remesh_fill(obj)
    with butil.ViewportMode(obj, "EDIT"), butil.Suppress():
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.spin(
            steps=rotation_resolution, angle=np.pi * 2, axis=axis, dupli=dupli
        )
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.remove_doubles(threshold=1e-3)
    return obj


def leaf(x_anchors, y_anchors, vector_locations=(), subdivision=64, face_size=None):
    curves = []
    for i in [-1, 1]:
        anchors = [x_anchors, i * np.array(y_anchors), 0]
        curves.append(bezier_curve(anchors, vector_locations, subdivision))
    obj = butil.join_objects(curves)
    butil.modify_mesh(obj, "WELD", merge_threshold=0.001)
    with butil.ViewportMode(obj, "EDIT"), butil.Suppress():
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.fill()
    remesh_fill(obj)
    if face_size is not None:
        butil.modify_mesh(obj, "WELD", merge_threshold=face_size / 2)
    with butil.ViewportMode(obj, "EDIT"), butil.Suppress():
        bpy.ops.mesh.region_to_loop()
        bpy.context.object.vertex_groups.new(name="boundary")
        bpy.ops.object.vertex_group_assign()
    obj = separate_loose(obj)
    return obj


def cut_plane(obj, cut_center, cut_normal, clear_outer=True):
    with butil.ViewportMode(obj, "EDIT"):
        bpy.ops.mesh.select_mode(type="FACE")
        bm = bmesh.from_edit_mesh(obj.data)
        bisect_plane = bmesh.ops.bisect_plane(
            bm,
            geom=bm.verts[:] + bm.edges[:] + bm.faces[:],
            plane_co=cut_center,
            plane_no=cut_normal,
            clear_outer=clear_outer,
            clear_inner=not clear_outer,
        )
        edges = [
            e for e in bisect_plane["geom_cut"] if isinstance(e, bmesh.types.BMEdge)
        ]
        face = bmesh.ops.edgeloop_fill(bm, edges=edges)["faces"][0]

        locations = np.array([v.co for v in face.verts])
        bmesh.ops.delete(bm, geom=[face], context="FACES_ONLY")
        bmesh.update_edit_mesh(obj.data)

    cut = mesh2obj(data2mesh(locations, [], [list(range(len(locations)))]))
    remesh_fill(cut)
    return obj, cut


def make_circular_interp(low, high, n, fn=uniform):
    xs = make_circular_angle(polygon_angles(n))
    ys = make_circular(fn(low, high, n))
    return interp1d(xs, ys, "quadratic")
