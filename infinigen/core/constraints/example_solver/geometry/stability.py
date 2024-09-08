# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Karhan Kayan

from __future__ import annotations

import logging

import bmesh
import bpy
import gin
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from mathutils import Vector
from shapely import MultiPolygon, Polygon
from shapely.affinity import rotate
from shapely.geometry import Point

from infinigen.core import tagging
from infinigen.core.constraints import constraint_language as cl
from infinigen.core.constraints.constraint_language import util as iu
from infinigen.core.constraints.example_solver import state_def

# from infinigen.core.util import blender as butil
from infinigen.core.constraints.example_solver.geometry import planes as planes
from infinigen.core.util import blender as butil

# import fcl


logger = logging.getLogger(__name__)


def project_and_align_z_with_x(polygons, z_direction):
    """
    Rotate polygons so that the Z-direction is aligned with the X-axis in 2D.

    Parameters:
    polygons (list[Polygon]): List of Shapely Polygons representing the projected 2D polygons.
    z_direction (np.array): The 2D direction vector where the Z-axis is projected.

    Returns:
    list[Polygon]: Rotated polygons with the Z-direction aligned with the X-axis.
    """
    # Calculate the angle between the Z-direction projection and the X-axis
    angle_rad = np.arctan2(z_direction[1], z_direction[0])
    angle_deg = np.degrees(angle_rad)

    # Rotate polygons to align Z-direction with X-axis
    rotated_polygons = [
        rotate(polygon, angle_deg, origin=(0, 0), use_radians=False)
        for polygon in polygons
    ]

    return rotated_polygons


def is_vertically_contained(poly_a, poly_b):
    """
    Check if polygon A is vertically contained within polygon B, ignoring X-axis spillover.

    Parameters:
    poly_a (Polygon): Polygon A.
    poly_b (Polygon): Polygon B.

    Returns:
    bool: True if A is vertically contained within B.
    """
    y_coords_a = [point[1] for point in poly_a.exterior.coords]
    y_coords_b = [point[1] for point in poly_b.exterior.coords]

    # Check vertical containment along the Y-axis
    min_a, max_a = min(y_coords_a), max(y_coords_a)
    min_b, max_b = min(y_coords_b), max(y_coords_b)

    return min_b <= min_a and max_a <= max_b


def project_vector(vector, origin, normal):
    transform = trimesh.geometry.plane_transform(origin, normal)
    transformed = trimesh.transformations.transform_points(
        [np.array([0, 0, 0]), vector], transform
    )[:, :2]
    transformed_vector = transformed[1] - transformed[0]
    return transformed_vector


@gin.configurable
def stable_against(
    state: state_def.State,
    obj_name: str,
    relation_state: state_def.RelationState,
    visualize=False,
    allow_overhangs=False,
):
    """
    check paralell, close to, and not overhanging.
    """

    relation = relation_state.relation
    assert isinstance(relation, cl.StableAgainst)

    logger.debug(f"stable against {obj_name=} {relation_state=} {relation.rev_normal=}")
    a_blender_obj = state.objs[obj_name].obj
    b_blender_obj = state.objs[relation_state.target_name].obj
    sa = state.objs[obj_name]
    sb = state.objs[relation_state.target_name]

    pa, pb = state.planes.get_rel_state_planes(state, obj_name, relation_state)

    poly_a = state.planes.planerep_to_poly(pa)
    poly_b = state.planes.planerep_to_poly(pb)

    normal_a = butil.global_polygon_normal(a_blender_obj, poly_a)
    normal_b = butil.global_polygon_normal(
        b_blender_obj, poly_b, rev_normal=relation.rev_normal
    )
    dot = np.array(normal_a).dot(normal_b)
    if not (np.isclose(np.abs(dot), 1, atol=1e-2) or np.isclose(dot, -1, atol=1e-2)):
        logger.debug(f"stable against failed, not parallel {dot=}")
        return False

    origin_b = butil.global_vertex_coordinates(
        b_blender_obj, b_blender_obj.data.vertices[poly_b.vertices[0]]
    )

    scene = state.trimesh_scene
    a_trimesh = iu.meshes_from_names(scene, sa.obj.name)[0]
    b_trimesh = iu.meshes_from_names(scene, sb.obj.name)[0]

    mask = tagging.tagged_face_mask(sb.obj, relation.parent_tags)
    mask = state.planes.tagged_plane_mask(sb.obj, mask, pb)
    assert mask.any()
    b_trimesh_mask = b_trimesh.submesh([np.where(mask)[0]], append=True)

    # Project mesh A onto the plane of mesh B
    projected_a = trimesh.path.polygons.projected(a_trimesh, normal_b, origin_b)
    projected_b = trimesh.path.polygons.projected(b_trimesh_mask, normal_b, origin_b)
    logger.debug(
        f"stable_against projecting along {normal_b} for parent_tags {relation.parent_tags}"
    )

    if projected_a is None or projected_b is None:
        raise ValueError(f"Invalid {projected_a=} {projected_b=}")

    if allow_overhangs:
        res = projected_a.overlaps(projected_b)
    elif relation.check_z:
        res = projected_a.within(projected_b.buffer(1e-2))
    else:
        z_proj = project_vector(np.array([0, 0, 1]), origin_b, normal_b)
        projected_a_rotated, projected_b_rotated = project_and_align_z_with_x(
            [projected_a, projected_b], z_proj
        )
        res = is_vertically_contained(projected_a_rotated, projected_b_rotated)

    if visualize:
        fig, ax = plt.subplots()
        iu.plot_geometry(ax, projected_a, "blue")
        iu.plot_geometry(ax, projected_b, "green")
        plt.title(f"{obj_name} stable against {relation_state.target_name}? {res=}")
        plt.show()

    logger.debug(f"stable_against {res=}")
    if not res:
        return False

    for vertex in poly_a.vertices:
        vertex_global = butil.global_vertex_coordinates(
            a_blender_obj, a_blender_obj.data.vertices[vertex]
        )
        distance = iu.distance_to_plane(vertex_global, origin_b, normal_b)
        if not np.isclose(distance, relation_state.relation.margin, atol=1e-2):
            logger.debug(f"stable against failed, not close to {distance=}")
            return False

    return True


@gin.configurable
def coplanar(
    state: state_def.State,
    obj_name: str,
    relation_state: state_def.RelationState,
):
    """
    check that the object's tagged surface is coplanar with the target object's tagged surface translated with margin.
    """

    relation = relation_state.relation
    assert isinstance(relation, cl.CoPlanar)

    logger.debug(f"coplanar {obj_name=} {relation_state=}")
    a_blender_obj = state.objs[obj_name].obj
    b_blender_obj = state.objs[relation_state.target_name].obj

    pa, pb = state.planes.get_rel_state_planes(state, obj_name, relation_state)

    poly_a = state.planes.planerep_to_poly(pa)
    poly_b = state.planes.planerep_to_poly(pb)

    normal_a = butil.global_polygon_normal(a_blender_obj, poly_a)
    normal_b = butil.global_polygon_normal(
        b_blender_obj, poly_b, rev_normal=relation.rev_normal
    )
    dot = np.array(normal_a).dot(normal_b)
    if not (np.isclose(np.abs(dot), 1, atol=1e-2) or np.isclose(dot, -1, atol=1e-2)):
        logger.debug(f"coplanar failed, not parallel {dot=}")
        return False

    origin_b = butil.global_vertex_coordinates(
        b_blender_obj, b_blender_obj.data.vertices[poly_b.vertices[0]]
    )

    for vertex in poly_a.vertices:
        vertex_global = butil.global_vertex_coordinates(
            a_blender_obj, a_blender_obj.data.vertices[vertex]
        )
        distance = iu.distance_to_plane(vertex_global, origin_b, normal_b)
        if not np.isclose(distance, relation_state.relation.margin, atol=1e-2):
            logger.debug(f"coplanar failed, not close to {distance=}")
            return False

    return True


def snap_against(scene, a, b, a_plane, b_plane, margin=0, rev_normal=False):
    """
    snap a against b with some margin.
    """
    logging.debug("snap_against", a, b, a_plane, b_plane, margin, rev_normal)

    a_obj = bpy.data.objects[a]
    b_obj = bpy.data.objects[b]

    a_poly_index = a_plane[1]
    a_poly = a_obj.data.polygons[a_poly_index]
    b_poly_index = b_plane[1]
    b_poly = b_obj.data.polygons[b_poly_index]
    plane_point_a = butil.global_vertex_coordinates(
        a_obj, a_obj.data.vertices[a_poly.vertices[0]]
    )
    plane_normal_a = butil.global_polygon_normal(a_obj, a_poly)
    plane_point_b = butil.global_vertex_coordinates(
        b_obj, b_obj.data.vertices[b_poly.vertices[0]]
    )
    plane_normal_b = butil.global_polygon_normal(b_obj, b_poly, rev_normal)
    plane_normal_b = -plane_normal_b

    norm_mag_a = np.linalg.norm(plane_normal_a)
    norm_mag_b = np.linalg.norm(plane_normal_b)
    assert np.isclose(norm_mag_a, 1), norm_mag_a
    assert np.isclose(norm_mag_b, 1), norm_mag_b

    rotation_axis = np.cross(plane_normal_a, plane_normal_b)
    if not np.isclose(np.linalg.norm(rotation_axis), 0, atol=1e-05):
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
    else:
        rotation_axis = np.array([0, 0, 1])

    dot = plane_normal_a.dot(plane_normal_b)
    rotation_angle = np.arccos(np.clip(dot, -1, 1))
    if np.isnan(rotation_angle):
        raise ValueError(f"Invalid {rotation_angle=}")
    iu.rotate(scene, a, rotation_axis, rotation_angle)

    a_obj = bpy.data.objects[a]
    a_poly = a_obj.data.polygons[a_poly_index]
    # Recalculate vertex_a and normal_a after rotation
    plane_point_a = butil.global_vertex_coordinates(
        a_obj, a_obj.data.vertices[a_poly.vertices[0]]
    )
    plane_normal_a = butil.global_polygon_normal(a_obj, a_poly)

    distance = (plane_point_a - plane_point_b).dot(plane_normal_b)

    # Move object a by the average distance minus the margin in the direction of the plane normal of b
    translation = -(distance + margin) * plane_normal_b.normalized()
    iu.translate(scene, a, translation)


def random_sample_point(
    state: state_def.State,
    obj: bpy.types.Object,
    face_mask: np.ndarray,
    plane: tuple[str, int],
) -> Vector:
    """
    Given a plane, return a random point on the plane.
    """

    if obj.type != "MESH":
        raise ValueError(f"Unexpected {obj.type=}")

    plane_mask = state.planes.tagged_plane_mask(obj, face_mask, plane)
    if not np.any(plane_mask):
        logging.warning(
            f"No faces in object {obj.name} are coplanar with plane {plane}."
        )

    # Create a bmesh from the object mesh
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.faces.ensure_lookup_table()

    faces = [bm.faces[i] for i in np.where(plane_mask)[0]]

    # Calculate the area for each face and create a cumulative distribution
    areas = np.array([f.calc_area() for f in faces])
    cumulative_areas = np.cumsum(areas)
    total_area = cumulative_areas[-1]

    # Generate a random number and find the corresponding face
    random_area_point = np.random.rand() * total_area
    face_index = np.searchsorted(cumulative_areas, random_area_point)
    selected_face = faces[face_index]

    verts = [v.co for v in selected_face.verts]

    # Use barycentric coordinates to sample a random point in the triangle
    # Random weights for each vertex
    weights = np.random.rand(3)
    weights /= np.sum(weights)
    random_point_local = (
        weights[0] * verts[0] + weights[1] * verts[1] + weights[2] * verts[2]
    )
    random_point_global = obj.matrix_world @ Vector(random_point_local)

    bm.free()

    return random_point_global


def move_obj_random_pt(
    state: state_def.State, a, b, face_mask: np.ndarray, plane: tuple[str, int]
):
    """
    move a to a random point on b
    """
    scene = state.trimesh_scene
    b_obj = iu.blender_objs_from_names(b)[0]

    random_point_global = random_sample_point(state, b_obj, face_mask, plane)
    iu.set_location(scene, a, random_point_global)


# def place_randomly(scene, a, b, visualize = False):
#     """
#     place a randomly on b.
#     """
#     a_blender_mesh  = blender_objs_from_names(a)[0]
#     a_trimesh = meshes_from_names(scene, a)[0]
#     b_blender_mesh = blender_objs_from_names(b)[0]
#     b_trimesh = meshes_from_names(scene, b)[0]
#     b_proj = project_to_xy_poly(b_trimesh)

#     xy_loc = sample_random_point(b_proj)
#     if visualize:
#         fig, ax = plt.subplots()
#         if isinstance(b_proj, Polygon):
#             x, y = b_proj.exterior.xy
#             ax.fill(x, y, alpha=0.5, fc='red', ec='black', label='Polygon b')
#         elif isinstance(b_proj, MultiPolygon):
#             for sub_poly in b_proj.geoms:
#                 x, y = sub_poly.exterior.xy
#                 ax.fill(x, y, alpha=0.5, fc='red', ec='black', label='Polygon b')
#         ax.plot(xy_loc.x, xy_loc.y, 'o', color='black', label='Random point')
#         plt.show()

#     set_location(scene, a, Vector((xy_loc.x, xy_loc.y, 0)))


def supported_by(scene, a, b, visualize=False):
    # check for collision first

    if isinstance(a, str):
        a = [a]

    a_meshes = iu.blender_objs_from_names(a)
    a_trimeshes = iu.meshes_from_names(scene, a)
    b_mesh = iu.blender_objs_from_names(b)[0]
    b_trimesh = iu.meshes_from_names(scene, b)[0]

    if visualize:
        fig, ax = plt.subplots()
        ax.set_aspect("equal", "box")
        b_poly = iu.project_to_xy_poly(b_trimesh)
        if isinstance(b_poly, Polygon):
            x, y = b_poly.exterior.xy
            ax.fill(x, y, alpha=0.5, fc="red", ec="black", label="Polygon b")
        elif isinstance(b_poly, MultiPolygon):
            for sub_poly in b_poly.geoms:
                x, y = sub_poly.exterior.xy
                ax.fill(x, y, alpha=0.5, fc="red", ec="black", label="Polygon b")

    for a_mesh, a_trimesh in zip(a_meshes, a_trimeshes):
        cloned_a = butil.deep_clone_obj(
            a_mesh, keep_modifiers=True, keep_materials=False
        )
        butil.modify_mesh(
            cloned_a, "BOOLEAN", apply=True, operation="INTERSECT", object=b_mesh
        )
        iu.preprocess_obj(cloned_a)
        intersection = iu.to_trimesh(cloned_a)
        intersection_poly = iu.project_to_xy_poly(intersection)
        intersection_convex = intersection_poly.convex_hull
        com_projected = a_trimesh.centroid[:2]
        if visualize:
            if isinstance(intersection_poly, Polygon):
                x, y = intersection_poly.exterior.xy
                ax.fill(x, y, alpha=0.5, fc="blue", ec="black", label="Polygon a")
            elif isinstance(intersection_poly, MultiPolygon):
                for sub_poly in intersection_poly.geoms:
                    x, y = sub_poly.exterior.xy
                    ax.fill(x, y, alpha=0.5, fc="blue", ec="black", label="Polygon a")
            ax.plot(
                com_projected[0], com_projected[1], "o", color="black", label="COM of a"
            )

        if not intersection_convex.contains(Point(com_projected)):
            if visualize:
                plt.show()
            return False
    if visualize:
        plt.show()
    return True
