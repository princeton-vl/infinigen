# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

import functools
import logging
import math
import random

# Authors: Karhan Kayan
from typing import Union

import bpy
import fcl
import gin
import numpy as np
import trimesh
from mathutils import Matrix, Vector
from shapely import LineString, MultiPolygon, Point, Polygon
from sklearn.decomposition import PCA
from trimesh import Scene

from infinigen.core import tagging
from infinigen.core.util import blender as butil

logger = logging.getLogger(__name__)


@gin.configurable
def bvh_caching_config(enabled=True):
    return enabled


@functools.cache
def group(scene, x):
    if isinstance(x, (list, set)):
        x = tuple(x)
    return subset(scene, x)


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


def closest_edge_to_point_poly(polygon, point):
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


def closest_edge_to_point_edge_list(edge_list: list[LineString], point):
    closest_distance = float("inf")
    closest_edge = None

    for line in edge_list:
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


def delete_obj(scene, a, delete_blender=True):
    if isinstance(a, str):
        a = [a]
    if delete_blender:
        obj_list = [bpy.data.objects[obj_name] for obj_name in a]
        butil.delete(obj_list)
    for obj_name in a:
        # bpy.data.objects.remove(bpy.data.objects[obj_name], do_unlink=True)
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


def subset(scene: Scene, incl):
    if isinstance(incl, str):
        incl = [incl]

    objs = []
    for n in scene.graph.nodes:
        T, g = scene.graph[n]
        if g is None:
            continue
        otags = scene.geometry[g].metadata["tags"]
        if any(t in incl for t in otags):
            objs.append(n)

    # assert len(objs) > 0, incl

    return objs


def add_object_cached(col, name, col_obj, fcl_obj):
    geom = fcl_obj
    o = col_obj
    # # Add collision object to set
    if name in col._objs:
        col._manager.unregisterObject(col._objs[name])
    col._objs[name] = {"obj": o, "geom": geom}
    # # store the name of the geometry
    col._names[id(geom)] = name

    col._manager.registerObject(o)
    col._manager.update()
    return o


def col_from_subset(scene, names, tags=None, bvh_cache=None):
    if isinstance(names, str):
        names = [names]

    if bvh_cache is not None and bvh_caching_config():
        tag_key = frozenset(tags) if tags is not None else None
        key = (frozenset(names), tag_key)
        res = bvh_cache.get(key)
        if res is not None:
            return res

    col = trimesh.collision.CollisionManager()

    for name in names:
        T, g = scene.graph[name]
        geom = scene.geometry[g]
        if tags is not None and len(tags) > 0:
            obj = blender_objs_from_names(name)[0]
            mask = tagging.tagged_face_mask(obj, tags)
            if not mask.any():
                logger.warning(f"{name=} had {mask.sum()=} for {tags=}")
                continue
            geom = geom.submesh(np.where(mask), append=True)
            T = trimesh.transformations.identity_matrix()
            t = fcl.Transform(T[:3, :3], T[:3, 3])
            geom.fcl_obj = col._get_fcl_obj(geom)
            geom.col_obj = fcl.CollisionObject(geom.fcl_obj, t)
            assert len(geom.faces) == mask.sum()
        # col.add_object(name, geom, T)
        add_object_cached(col, name, geom.col_obj, geom.fcl_obj)

    if len(col._objs) == 0:
        logger.debug(f"{names=} got no objs, returning None")
        col = None

    if bvh_cache is not None and bvh_caching_config():
        bvh_cache[key] = col

    return col


def plot_geometry(ax, geom, color="blue"):
    if isinstance(geom, Polygon):
        x, y = geom.exterior.xy
        ax.fill(x, y, alpha=0.5, fc=color, ec="black")
    elif isinstance(geom, MultiPolygon):
        for sub_geom in geom:
            x, y = sub_geom.exterior.xy
            ax.fill(x, y, alpha=0.5, fc=color, ec="black")
    elif isinstance(geom, LineString):
        x, y = geom.xy
        ax.plot(x, y, color=color)
    elif isinstance(geom, Point):
        ax.plot(geom.x, geom.y, "o", color=color)


def sync_trimesh(scene: trimesh.Scene, obj_name: str):
    bpy.context.view_layer.update()
    blender_obj = bpy.data.objects[obj_name]
    mesh = meshes_from_names(scene, obj_name)[0]
    T_old = mesh.current_transform
    T = np.array(blender_obj.matrix_world)
    mesh.apply_transform(T @ np.linalg.inv(T_old))
    mesh.current_transform = np.array(blender_obj.matrix_world)
    t = fcl.Transform(T[:3, :3], T[:3, 3])
    mesh.col_obj.setTransform(t)


def translate(scene: trimesh.Scene, a: str, translation):
    blender_obj = bpy.data.objects[a]
    blender_obj.location += Vector(translation)
    if scene:
        sync_trimesh(scene, a)


def rotate(scene: trimesh.Scene, a: str, axis, angle):
    blender_obj = bpy.data.objects[a]

    rotation_matrix = trimesh.transformations.rotation_matrix(angle, axis)
    transform_matrix = Matrix(rotation_matrix).to_4x4()
    loc, rot, scale = blender_obj.matrix_world.decompose()
    rot = rot.to_matrix().to_4x4()
    rot = transform_matrix @ rot
    rot = rot.to_quaternion()
    blender_obj.matrix_world = Matrix.LocRotScale(loc, rot, scale)

    if scene:
        sync_trimesh(scene, a)


def set_location(scene: trimesh.Scene, obj_name: str, location):
    blender_mesh = bpy.data.objects[obj_name]
    blender_mesh.location = location
    sync_trimesh(scene, obj_name)


def set_rotation(scene: trimesh.Scene, obj_name: str, rotation):
    blender_mesh = blender_objs_from_names(obj_name)[0]
    blender_mesh.rotation_euler = rotation
    sync_trimesh(scene, obj_name)


# for debugging. does not actually find centroid
def blender_centroid(a):
    return np.mean([a.matrix_world @ v.co for v in a.data.vertices], axis=0)


def order_objects_by_principal_axis(objects: list[bpy.types.Object]):
    locations = [obj.location for obj in objects]
    location_matrix = np.array(locations)
    pca = PCA(n_components=1)
    pca.fit(location_matrix)
    locations_projected = pca.transform(location_matrix)
    sorted_indices = np.argsort(locations_projected.ravel())
    return [objects[i] for i in sorted_indices]
