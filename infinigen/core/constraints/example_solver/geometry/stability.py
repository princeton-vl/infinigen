# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Karhan Kayan

from __future__ import annotations
import logging
from dataclasses import dataclass
from copy import copy

import numpy as np

import bpy
import trimesh
from shapely.geometry import Point, LineString
from shapely.ops import unary_union, nearest_points
from shapely import Polygon
from shapely import MultiPolygon
import bmesh

import matplotlib.pyplot as plt
# import fcl

# from infinigen.core.util import blender as butil
from infinigen.core.util import blender as butil
from mathutils import Vector, Quaternion
import logging
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
    rotated_polygons = [rotate(polygon, angle_deg, origin=(0, 0), use_radians=False) for polygon in polygons]
    
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
    transformed = trimesh.transformations.transform_points([np.array([0,0,0]), vector], transform)[:, :2]
    transformed_vector = transformed[1] - transformed[0]
    return transformed_vector

    """
    check paralell, close to, and not overhanging. 
    """

    logger.debug(f'stable against {obj_name=} {relation_state=}')
    pa, pb = state.planes.get_rel_state_planes(state, obj_name, relation_state)

    poly_a = state.planes.planerep_to_poly(pa)
    poly_b = state.planes.planerep_to_poly(pb)

    if not (np.isclose(np.abs(dot), 1, atol=1e-2) or np.isclose(dot, -1, atol=1e-2)):
        logger.debug(f'stable against failed, not parallel {dot=}')
        return False


    mask = state.planes.tagged_plane_mask(sb.obj, mask, pb)
    # Project mesh A onto the plane of mesh B


    if projected_a is None or projected_b is None:
        raise ValueError(f'Invalid {projected_a=} {projected_b=}')

        res = projected_a.within(projected_b.buffer(1e-2))
        z_proj = project_vector(np.array([0, 0, 1]), origin_b, normal_b)
        projected_a_rotated, projected_b_rotated = project_and_align_z_with_x([projected_a, projected_b], z_proj)
        res = is_vertically_contained(projected_a_rotated, projected_b_rotated)
        return False
    
    for vertex in poly_a.vertices:
        if not np.isclose(distance, relation_state.relation.margin, atol=1e-2):
            logger.debug(f'stable against failed, not close to {distance=}')
            return False


    return True

def snap_against(scene, a, b, a_plane, b_plane, margin = 0):
    """
    snap a against b with some margin. 
    """
    logging.debug("snap_against", a, b, a_plane, b_plane, margin)

    a_obj = bpy.data.objects[a]
    b_obj = bpy.data.objects[b]

    a_poly_index = a_plane[1]
    a_poly = a_obj.data.polygons[a_poly_index]
    b_poly_index = b_plane[1]
    b_poly = b_obj.data.polygons[b_poly_index]
    plane_normal_b = -plane_normal_b


    

    rotation_axis = np.cross(plane_normal_a, plane_normal_b)
    if not np.isclose(np.linalg.norm(rotation_axis),0, atol = 1e-05): 
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
    else:
        rotation_axis = np.array([0,0,1])



    a_obj = bpy.data.objects[a]
    a_poly = a_obj.data.polygons[a_poly_index]
    # Recalculate vertex_a and normal_a after rotation
    
    distance = (plane_point_a - plane_point_b).dot(plane_normal_b)

    # Move object a by the average distance minus the margin in the direction of the plane normal of b



def random_sample_point(state: state_def.State, obj: bpy.types.Object, face_mask: np.ndarray, plane: tuple[str, int]) -> Vector:
    """
    Given a plane, return a random point on the plane.
    """
    plane_mask = state.planes.tagged_plane_mask(obj, face_mask, plane)
    if not np.any(plane_mask):
        logging.warning(
            f'No faces in object {obj.name} are coplanar with plane {plane}.'
        )

    # Create a bmesh from the object mesh
    bm = bmesh.new()
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
    random_point_local = weights[0] * verts[0] + weights[1] * verts[1] + weights[2] * verts[2]
    random_point_global = obj.matrix_world @ Vector(random_point_local)

    bm.free()

    return random_point_global

def move_obj_random_pt(state: state_def.State, a, b, face_mask: np.ndarray, plane: tuple[str, int]):
    """
    move a to a random point on b
    """
    scene = state.trimesh_scene

    random_point_global = random_sample_point(state, b_obj, face_mask, plane)


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

def supported_by(scene, a, b, visualize = False):

    #check for collision first 


    if isinstance(a, str):
        a = [a]


    if visualize:
        fig, ax = plt.subplots()
        ax.set_aspect('equal', 'box')
        if isinstance(b_poly, Polygon):
            x, y = b_poly.exterior.xy
            ax.fill(x, y, alpha=0.5, fc='red', ec='black', label='Polygon b')
        elif isinstance(b_poly, MultiPolygon):
            for sub_poly in b_poly.geoms:
                x, y = sub_poly.exterior.xy
                ax.fill(x, y, alpha=0.5, fc='red', ec='black', label='Polygon b')

    for a_mesh, a_trimesh in zip(a_meshes, a_trimeshes):
        cloned_a = butil.deep_clone_obj(
            a_mesh, keep_modifiers=True, keep_materials=False
        )        
        butil.modify_mesh(
            cloned_a, "BOOLEAN", apply=True, operation="INTERSECT", object=b_mesh
        )
        intersection_convex = intersection_poly.convex_hull
        com_projected = a_trimesh.centroid[:2]
        if visualize:
            if isinstance(intersection_poly, Polygon):
                x, y = intersection_poly.exterior.xy
                ax.fill(x, y, alpha=0.5, fc='blue', ec='black', label='Polygon a')
            elif isinstance(intersection_poly, MultiPolygon):
                for sub_poly in intersection_poly.geoms:
                    x, y = sub_poly.exterior.xy
                    ax.fill(x, y, alpha=0.5, fc='blue', ec='black', label='Polygon a')
            ax.plot(com_projected[0], com_projected[1], 'o', color='black', label='COM of a')
            
        if not intersection_convex.contains(Point(com_projected)):
            if visualize: 
                plt.show()
            return False 
    if visualize: 
        plt.show()
    return True