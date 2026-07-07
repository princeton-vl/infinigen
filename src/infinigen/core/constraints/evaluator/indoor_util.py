# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Karhan Kayan

import math
import random
from typing import Union

import bpy
import numpy as np
import trimesh
from shapely import LineString, Point

from infinigen.core.util import blender as butil


def meshes_from_names(scene, names):
    if isinstance(names, str):
        names = [names]
    return [scene.geometry[g] for _, g in (scene.graph[n] for n in names)]


def blender_objs_from_names(names):
    if isinstance(names, str):
        names = [names]
    return [bpy.data.objects[n] for n in names]


def name_from_mesh(scene, mesh):
    mesh_name = None
    for name, mesh in scene.geometry.items():
        if mesh == mesh:
            mesh_name = name
            break
    return mesh_name


def project_to_xy_path2d(mesh: trimesh.Trimesh) -> trimesh.path.Path2D:
    poly = trimesh.path.polygons.projected(mesh, (0, 0, 1), (0, 0, 0))
    d = trimesh.path.exchange.misc.polygon_to_path(poly)
    return trimesh.path.Path2D(entities=d["entities"], vertices=d["vertices"])


def project_to_xy_poly(mesh: trimesh.Trimesh):
    poly = trimesh.path.polygons.projected(mesh, (0, 0, 1), (0, 0, 0))
    return poly


def closest_edge_to_point(polygon, point):
    closest_distance = float("inf")
    closest_edge = None

    for i, coord in enumerate(polygon.exterior.coords[:-1]):
        start, end = coord, polygon.exterior.coords[i + 1]
        line = LineString([start, end])
        distance = line.distance(point)

        if distance < closest_distance:
            closest_distance = distance
            closest_edge = line

    return closest_edge


def compute_outward_normal(line, polygon):
    dx = line.xy[0][1] - line.xy[0][0]  # x1 - x0
    dy = line.xy[1][1] - line.xy[1][0]  # y1 - y0

    # Candidate normal vectors (perpendicular to edge)
    normal_vector_1 = np.array([dy, -dx])
    normal_vector_2 = -normal_vector_1

    # Normalize the vectors (optional but recommended for consistency)
    normal_vector_1 = normal_vector_1 / np.linalg.norm(normal_vector_1)
    normal_vector_2 = normal_vector_2 / np.linalg.norm(normal_vector_2)

    # Midpoint of the line segment
    mid_point = line.interpolate(0.5, normalized=True)

    # Move a tiny bit in the direction of the normals to check which points outside
    test_point_1 = mid_point.coords[0] + 0.01 * normal_vector_1
    mid_point.coords[0] + 0.01 * normal_vector_2

    # Return the normal for which the test point lies outside the polygon
    if polygon.contains(Point(test_point_1)):
        return normal_vector_2
    else:
        return normal_vector_1


def get_transformed_axis(scene, obj_name):
    obj = bpy.data.objects[obj_name]
    trimesh_mesh = meshes_from_names(scene, obj_name)[0]
    axis = trimesh_mesh.axis
    rot_mat = np.array(obj.matrix_world.to_3x3())
    return rot_mat @ np.array(axis)


def set_axis(scene, objs: Union[str, list[str]], canonical_axis):
    if isinstance(objs, str):
        objs = [objs]
    obj_meshes = meshes_from_names(scene, objs)
    for obj_name, obj in zip(objs, obj_meshes):
        obj.axis = canonical_axis
        obj.axis = get_transformed_axis(scene, obj_name)


def get_plane_from_3dmatrix(matrix):
    """Extract the plane_normal and plane_origin from a transformation matrix."""
    # The normal of the plane can be extracted from the 3x3 rotation part of the matrix
    plane_normal = matrix[:3, 2]
    plane_origin = matrix[:3, 3]
    return plane_normal, plane_origin


def project_points_onto_plane(points, plane_origin, plane_normal):
    """Project 3D points onto a plane."""
    d = np.dot(points - plane_origin, plane_normal)[:, None]
    return points - d * plane_normal


def to_2d_coordinates(points, plane_normal):
    """Convert 3D points to 2D using the plane defined by its normal."""
    # Compute two perpendicular vectors on the plane
    u = np.cross(plane_normal, [1, 0, 0])
    if np.linalg.norm(u) < 1e-10:
        u = np.cross(plane_normal, [0, 1, 0])
    u /= np.linalg.norm(u)
    v = np.cross(plane_normal, u)
    v /= np.linalg.norm(v)

    # Convert 3D points to 2D using dot products
    return np.column_stack([points.dot(u), points.dot(v)])


def ensure_correct_order(points):
    """
    Ensures the points are in counter-clockwise order.
    If not, it reverses them.
    """
    # Calculate signed area
    n = len(points)
    area = (
        sum(
            (points[i][0] * points[(i + 1) % n][1])
            - (points[(i + 1) % n][0] * points[i][1])
            for i in range(n)
        )
        / 2.0
    )
    # Return the points in reverse order if area is negative
    return points[::-1] if area < 0 else points


def sample_random_point(polygon):
    """
    Sample a random point from inside the given Shapely polygon.
    """
    minx, miny, maxx, maxy = polygon.bounds
    while True:
        p = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
        if polygon.contains(p):
            return p


def delete_obj(a, scene=None):
    if isinstance(a, str):
        a = [a]
    for obj_name in a:
        bpy.data.objects.remove(bpy.data.objects[obj_name], do_unlink=True)
        if scene:
            scene.graph.transforms.remove_node(obj_name)
            scene.delete_geometry(obj_name + "_mesh")


def is_planar(obj, tolerance=1e-6):
    if len(obj.data.polygons) != 1:
        return False

    polygon = obj.data.polygons[0]
    global_normal = butil.global_polygon_normal(obj, polygon)

    # Take the first vertex as a reference point on the plane
    ref_vertex = butil.global_vertex_coordinates(
        obj, obj.data.vertices[polygon.vertices[0]]
    )

    # Check if all vertices lie on the plane defined by the reference vertex and the global normal
    for vertex in obj.data.vertices:
        distance = (butil.global_vertex_coordinates(obj, vertex) - ref_vertex).dot(
            global_normal
        )
        if not math.isclose(distance, 0, abs_tol=tolerance):
            return False

    return True


def planes_parallel(plane_obj_a, plane_obj_b, tolerance=1e-6):
    if plane_obj_a.type != "MESH" or plane_obj_b.type != "MESH":
        raise ValueError("Both objects should be of type 'MESH'")

    # # Check if the objects are planar
    # if not is_planar(plane_obj_a) or not is_planar(plane_obj_b):
    #     raise ValueError("One or both objects are not planar")

    global_normal_a = butil.global_polygon_normal(
        plane_obj_a, plane_obj_a.data.polygons[0]
    )
    global_normal_b = butil.global_polygon_normal(
        plane_obj_b, plane_obj_b.data.polygons[0]
    )

    dot_product = global_normal_a.dot(global_normal_b)

    return math.isclose(dot_product, 1, abs_tol=tolerance) or math.isclose(
        dot_product, -1, abs_tol=tolerance
    )


def distance_to_plane(point, plane_point, plane_normal):
    """Compute the distance from a point to a plane defined by a point and a normal."""
    return abs((point - plane_point).dot(plane_normal))


def is_within_margin_from_plane(obj, obj_b, margin, tol=1e-6):
    """Check if all vertices of an object are within a given margin from a plane."""
    polygon_b = obj_b.data.polygons[0]
    plane_point_b = butil.global_vertex_coordinates(
        obj_b, obj_b.data.vertices[polygon_b.vertices[0]]
    )
    plane_normal_b = butil.global_polygon_normal(obj_b, polygon_b)
    for vertex in obj.data.vertices:
        global_vertex = butil.global_vertex_coordinates(obj, vertex)
        distance = distance_to_plane(global_vertex, plane_point_b, plane_normal_b)
        if not math.isclose(distance, margin, abs_tol=tol):
            return False
    return True


# def update_blender_representation(scene, trimesh_obj):

#     transform_matrix =


# def update_trimesh_representation(scnene, blender_obj):
#     pass
