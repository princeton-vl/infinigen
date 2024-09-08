# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Karhan Kayan: primary author
# - Alexander Raistrick: initial version of collision/distance
# Acknowledgement: Some metrics draw inspiration from https://dl.acm.org/doi/10.1145/1964921.1964981 by Yu et al.

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Union

import bpy
import gin
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import trimesh
from mathutils import Vector
from scipy.optimize import linear_sum_assignment
from shapely import MultiPolygon, Polygon
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points, unary_union
from trimesh import Scene

import infinigen.core.constraints.constraint_language.util as iu
import infinigen.core.constraints.evaluator.node_impl.symmetry as symmetry
from infinigen.core import tagging
from infinigen.core import tags as t
from infinigen.core.constraints.example_solver import state_def
from infinigen.core.constraints.example_solver.geometry.parse_scene import add_to_scene
from infinigen.core.util import blender as butil
from infinigen.core.util.logging import lazydebug

# import fcl


# from infinigen.core.tagging import tag_object,tag_system
# from scipy.optimize import dual_annealing
# from tqdm import tqdm

logger = logging.getLogger(__name__)


def get_cardinal_planes_bbox(vertices: np.ndarray):
    """
    Get the mid dividing planes. Assumes vertices form a box
    """
    centroid = np.mean(vertices, axis=0)

    # Calculate the covariance matrix and principal components
    centered_vertices = vertices - centroid
    cov_matrix = np.cov(centered_vertices[:, :2].T)  # Covariance on XY plane
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Sort eigenvectors based on eigenvalues
    order = eigenvalues.argsort()[::-1]
    principal_axes = eigenvectors[:, order]

    # Determine the longer and shorter plane normals and normalize them
    if eigenvalues[order[0]] > eigenvalues[order[1]]:
        longer_plane_normal = np.array([principal_axes[0, 1], principal_axes[1, 1], 0])
        shorter_plane_normal = np.array([principal_axes[0, 0], principal_axes[1, 0], 0])
    else:
        longer_plane_normal = np.array([principal_axes[0, 0], principal_axes[1, 0], 0])
        shorter_plane_normal = np.array([principal_axes[0, 1], principal_axes[1, 1], 0])

    longer_plane_normal /= np.linalg.norm(longer_plane_normal)
    shorter_plane_normal /= np.linalg.norm(shorter_plane_normal)
    return [
        [Vector(centroid), Vector(longer_plane_normal)],
        [Vector(centroid), Vector(shorter_plane_normal)],
    ]


def get_axis(state: state_def.State, obj: bpy.types.Object, tag=t.Subpart.Front):
    a_front_planes = state.planes.get_tagged_planes(obj, tag)
    if len(a_front_planes) > 1:
        logging.warning(
            f"{obj.name=} had too many front planes ({len(a_front_planes)})"
        )
    a_front_plane = a_front_planes[0]
    a_front_plane_ind = a_front_plane[1]
    a_poly = obj.data.polygons[a_front_plane_ind]
    front_plane_pt = butil.global_vertex_coordinates(
        obj, obj.data.vertices[a_poly.vertices[0]]
    )
    front_plane_normal = butil.global_polygon_normal(obj, a_poly)
    return front_plane_pt, front_plane_normal


def preprocess_collision_query_cases(a, b, a_tags, b_tags):
    if isinstance(a, list):
        a = set(a)
    if isinstance(b, list):
        b = set(b)

    if a is not None and len(a) == 1:
        a = a.pop()
    if b is not None and len(b) == 1:
        b = b.pop()

    # eliminate symmetrical cases
    if a is None or (isinstance(b, set) and not isinstance(a, set)):
        a, b = b, a
        a_tags, b_tags = b_tags, a_tags

    # nobody wants to be told a 0 distance if they query how far a chair is from the set of all chairs
    if isinstance(b, str) and isinstance(a, set) and b in a:
        a.remove(b)

    if isinstance(a, set) and len(a) == 0:
        raise ValueError(f"query recieved empty input {a=}")
    if isinstance(a, set) and len(a) == 0:
        raise ValueError(f"query recieved empty input {b=}")

    # single-to-single is treated as many-to-single
    if isinstance(a, str):
        a = [a]

    assert a is not None

    if a_tags is None:
        a_tags = set()
    if b_tags is None:
        b_tags = set()

    return a, b, a_tags, b_tags


@dataclass
class ContactResult:
    hit: bool
    names: list[str]
    contacts: list


def any_touching(
    scene: Scene,
    a: Union[str, list[str]],
    b: Union[str, list[str]] = None,
    a_tags=None,
    b_tags=None,
    bvh_cache=None,
):
    """
    Computes one-to-one, many-to-one, one-to-many or many-to-many collisions

    In all cases, returns True if any one object from a and b touch
    """
    a, b, a_tags, b_tags = preprocess_collision_query_cases(a, b, a_tags, b_tags)

    col = iu.col_from_subset(scene, a, a_tags, bvh_cache)

    if b is None and len(a) == 1:
        # query makes no sense, asking for intra-set collision on one element
        hit, names, contacts = None, (a, b), []
    elif b is None:
        hit, names, contacts = col.in_collision_internal(
            return_data=True, return_names=True
        )
    elif isinstance(b, str):
        T, g = scene.graph[b]
        hit, names, contacts = col.in_collision_single(
            scene.geometry[g], transform=T, return_data=True, return_names=True
        )
    elif isinstance(b, list):
        col2 = iu.col_from_subset(scene, b, b_tags, bvh_cache)
        hit, names, contacts = col.in_collision_other(
            col2, return_names=True, return_data=True
        )
    else:
        raise ValueError(f"Unhandled case {a=} {b=}")

    names = list(names)
    if len(names) == 1:
        assert isinstance(b, str)
        names.append(b)
        logging.debug(f"added name {b} to make {names}")

    if len(names) == 0:
        names = [a, b]

    return ContactResult(hit=hit, names=names, contacts=contacts)


@dataclass
class DistanceResult:
    dist: float
    names: list[str]
    data: trimesh.collision.DistanceData


def min_dist(
    scene: Scene,
    a: str | list[str] | None,
    b: str | list[str] | None = None,
    a_tags: set = None,
    b_tags: set = None,
    bvh_cache: dict = None,
):
    """
    Computes one-to-one, many-to-one, one-to-many or many-to-many distance

    In all cases, returns the minimum distance between any object in a and b
    """
    # we get fcl error otherwise
    if len(a) == 1 and len(b) == 1 and a[0] == b[0]:
        return DistanceResult(dist=0, names=[a[0], b[0]], data=None)
    a, b, a_tags, b_tags = preprocess_collision_query_cases(a, b, a_tags, b_tags)
    col = iu.col_from_subset(scene, a, a_tags, bvh_cache)

    if b is None and len(a) == 1:
        dist, data = 1e9, None
    elif b is None:
        lazydebug(logger, lambda: f"min_dist_internal({a=}, {b=})")
        dist, data = col.min_distance_internal(return_data=True)
    elif isinstance(b, str):
        T, g = scene.graph[b]
        geom = scene.geometry[g]
        if b_tags is not None and len(b_tags) > 0:
            obj = iu.blender_objs_from_names(b)[0]
            mask = tagging.tagged_face_mask(obj, b_tags)
            if not mask.any():
                lazydebug(logger, lambda: f"{b=} had {mask.sum()=} for {b_tags=}")
            geom = geom.submesh(np.where(mask), append=True)
            assert len(geom.faces) == mask.sum()

        lazydebug(logger, lambda: f"min_dist_single({a=}, {b=})")
        dist, data = col.min_distance_single(geom, transform=T, return_data=True)

        if "__external" in data.names:
            data.names.remove("__external")
            data.names.add(b)
            data._points[b] = data._points["__external"]
            data._points.pop("__external")
            logging.debug(f"WARNING: swapped __external for {b} to make {data.names}")
    elif isinstance(b, (list, set)):
        logger.debug(f"min_dist_other({a=}, {b=})")
        col2 = iu.col_from_subset(scene, b, b_tags, bvh_cache)
        dist, data = col.min_distance_other(col2, return_data=True)
    else:
        raise ValueError(f"Unhandled case {a=} {b=}")

    if data is not None:
        assert "__external" not in data.names

    return DistanceResult(
        dist=dist, names=list(data.names) if data is not None else None, data=data
    )


def contains(scene: Scene, a: str, b: str, tol=1e-6) -> bool:
    """
    Check if a contains b
    """
    mesh_a = scene.geometry[a]
    mesh_b = scene.geometry[b]

    difference = mesh_a.difference(mesh_b)

    return abs(difference.volume - mesh_a.volume) < tol


def contains_all(
    scene: trimesh.Scene, a: Union[str, list[str]], b: Union[str, list[str]]
) -> bool:
    """
    Check if all objects in list 'a' contain all objects in list 'b' within the given scene.

    Parameters:
    - scene: The trimesh.Scene instance.
    - a: Name or list of names of objects to check for containment.
    - b: Name or list of names of objects that might be contained.

    Returns:
    - True if all objects in list 'a' contain all objects in list 'b', False otherwise.
    """

    if isinstance(a, str):
        a = [a]
    if isinstance(b, str):
        b = [b]

    for obj_a in a:
        if not all(contains(scene, obj_a, obj_b) for obj_b in b):
            return False

    return True


def contains_any(
    scene: trimesh.Scene, a: Union[str, list[str]], b: Union[str, list[str]]
) -> bool:
    """
    Check if any object in list 'a' contains any object in list 'b' within the given scene.

    Parameters:
    - scene: The trimesh.Scene instance.
    - a: Name or list of names of objects to check for containment.
    - b: Name or list of names of objects that might be contained.

    Returns:
    - True if any object in list 'a' contains any object in list 'b', False otherwise.
    """

    if isinstance(a, str):
        a = [a]
    if isinstance(b, str):
        b = [b]

    for obj_a in a:
        if any(contains(scene, obj_a, obj_b) for obj_b in b):
            return True

    return False


def has_line_of_sight(
    scene: trimesh.Scene,
    a: Union[str, list[str]],
    b: Union[str, list[str]],
    num_samples: int = 100,
) -> bool:
    """
    Check if any object in list 'a' in the scene has a line of sight to any object in list 'b'.

    Parameters:
    - scene: The trimesh.Scene instance.
    - a: Name or list of names of objects from which line of sight is checked.
    - b: Name or list of names of objects to which line of sight is checked.
    - num_samples: Number of points to sample from each object for ray casting.

    Returns:
    - True if any object in list 'a' has a line of sight to any object in list 'b', False otherwise.
    """

    # Ensure 'a' and 'b' are lists
    if isinstance(a, str):
        a = [a]
    if isinstance(b, str):
        b = [b]

    a = iu.meshes_from_names(scene, a)
    b = iu.meshes_from_names(scene, b)

    # Check line of sight for each object in 'a' against any object in 'b'
    for obj_a in a:
        # Sample points from the surface of object 'a'
        points_a = obj_a.sample(num_samples)

        combined_mesh = trimesh.util.concatenate(
            [mesh for name, mesh in scene.geometry.items() if mesh != obj_a]
        )

        for obj_b in b:
            # Sample points from the surface of object 'b'
            points_b = obj_b.sample(num_samples)

            # Create rays from points on 'a' to points on 'b'
            ray_origins = np.tile(points_a, (num_samples, 1))
            ray_directions = np.repeat(points_b, num_samples, axis=0) - ray_origins
            ray_directions /= np.linalg.norm(ray_directions, axis=1)[:, None]

            # Check for intersections with the combined mesh
            locations, index_ray, index_tri = (
                combined_mesh.ray_pyembree.intersects_location(
                    ray_origins, ray_directions, multiple_hits=False
                )
            )

            # Check if point is reached
            for i in range(index_ray.shape[0]):
                index = index_ray[i]
                hit_location = locations[i]

                # Check if any intersection is close to the point
                if np.linalg.norm(points_b[index // num_samples] - hit_location) < 1e-6:
                    return True

    return False


def freespace_2d(
    scene: trimesh.Scene, a: Union[str, list[str]], b: Union[str, list[str]]
) -> float:
    if isinstance(a, str):
        a = [a]
    if isinstance(b, str):
        b = [b]

    a_meshes = iu.meshes_from_names(scene, a)

    b_meshes = iu.meshes_from_names(scene, b)

    total_projected_area = sum(iu.project_to_xy_path2d(mesh).area for mesh in b_meshes)

    available_area = sum(iu.project_to_xy_path2d(mesh).area for mesh in a_meshes)

    percent_available = ((available_area - total_projected_area) / available_area) * 100

    return percent_available


def rasterize_space_with_obstacles(
    scene,
    a: Union[str, list[str]],
    b: Union[str, list[str]],
    start_location,
    end_location,
    cell_size=1.0,
    visualize=False,
):
    """
    Rasterize the union of multiple space polygons while considering obstacle polygons,
    then find and visualize the shortest path from start to end.

    Parameters:
    - space_polygons: list of shapely.geometry.polygon.Polygon objects representing the main spaces
    - obstacle_polygons: list of shapely.geometry.polygon.Polygon objects representing obstacles
    - start_location: tuple (x, y) representing the start location
    - end_location: tuple (x, y) representing the end location
    - cell_size: size of each cell in the grid
    - visualize: boolean, if True, visualize the union of spaces, obstacles, and the shortest path

    Returns:
    - graph: A networkx.Graph object representing the rasterized union of spaces minus the obstacles
    - path: list of nodes representing the shortest path from start to end
    """

    def is_close_to_any_node(neighbor, graph, threshold=1e-6):
        for node in graph.nodes():
            distance = np.linalg.norm(np.array(neighbor) - np.array(node))
            if distance < threshold:
                return node
        return None

    if isinstance(a, str):
        a = [a]
    if isinstance(b, str):
        b = [b]

    a_meshes = iu.meshes_from_names(scene, a)
    b_meshes = iu.meshes_from_names(scene, b)

    space_polygons = [iu.project_to_xy_poly(mesh) for mesh in a_meshes]
    obstacle_polygons = [iu.project_to_xy_poly(mesh) for mesh in b_meshes]

    # Get the union of all space polygons
    union_space = unary_union(space_polygons)

    # Get bounding box of the union space
    minx, miny, maxx, maxy = union_space.bounds

    # Create a grid over the bounding box
    x_coords = np.arange(minx, maxx, cell_size)
    y_coords = np.arange(miny, maxy, cell_size)

    graph = nx.Graph()

    # For visualization
    if visualize:
        fig, ax = plt.subplots()
        for space in space_polygons:
            if isinstance(space, Polygon):
                x, y = space.exterior.xy
                ax.fill(x, y, alpha=0.5)  # Fill the space
                ax.plot(x, y, color="black")  # Plot the space boundary
            elif isinstance(space, MultiPolygon):
                for sub_space in space.geoms:
                    x, y = sub_space.exterior.xy
                    ax.fill(x, y, alpha=0.5)
                    ax.plot(x, y, color="black")

        for obstacle in obstacle_polygons:
            if isinstance(obstacle, Polygon):
                x, y = obstacle.exterior.xy
                ax.fill(x, y, color="grey")  # Fill the obstacles
                ax.plot(x, y, color="black")  # Plot the obstacle boundary
            elif isinstance(obstacle, MultiPolygon):
                for sub_obstacle in obstacle.geoms:
                    x, y = sub_obstacle.exterior.xy
                    ax.fill(x, y, color="grey")
                    ax.plot(x, y, color="black")

    # For each cell in the grid, check if its center is inside the union space and outside all obstacle polygons
    for x in x_coords:
        for y in y_coords:
            cell_center = Point(x + cell_size / 2, y + cell_size / 2)
            if cell_center.within(union_space) and all(
                not cell_center.within(obstacle) for obstacle in obstacle_polygons
            ):
                graph.add_node((x + cell_size / 2, y + cell_size / 2))

                # For visualization
                if visualize:
                    ax.plot(
                        cell_center.x, cell_center.y, "bo", markersize=3
                    )  # Plot the point inside the union space and outside obstacles

    # Connect each node to its neighboring nodes
    for node in graph.nodes():
        x, y = node
        neighbors = [
            (x + cell_size, y),
            (x - cell_size, y),
            (x, y + cell_size),
            (x, y - cell_size),
        ]
        for neighbor in neighbors:
            closest_node = is_close_to_any_node(neighbor, graph)
            if closest_node is not None:
                graph.add_edge(node, closest_node)

    # Find the closest nodes to the start and end locations
    start_node = min(
        graph.nodes(),
        key=lambda node: np.linalg.norm(np.array(node) - np.array(start_location)),
    )
    end_node = min(
        graph.nodes(),
        key=lambda node: np.linalg.norm(np.array(node) - np.array(end_location)),
    )

    # Calculate the shortest path using Dijkstra's algorithm
    path = nx.shortest_path(graph, source=start_node, target=end_node, weight="weight")

    # Visualize the path
    if visualize:
        path_x = [x for x, y in path]
        path_y = [y for x, y in path]
        ax.plot(path_x, path_y, c="red", linewidth=2, label="Shortest Path")
        ax.scatter(
            [start_node[0], end_node[0]],
            [start_node[1], end_node[1]],
            c="green",
            s=100,
            label="Start & End",
        )
        plt.legend()
        plt.title("Shortest Path from Start to End")
        plt.show()

    return graph, path


def angle_alignment_cost_tagged(
    state: state_def.State,
    a: Union[str, list[str]],
    b: Union[str, list[str]],
    b_tags=None,
    visualize=False,
):
    """
    Return the dot product between the axes of a and the normal of the closest edge of b
    """
    if isinstance(a, str):
        a = [a]

    b_objs = iu.blender_objs_from_names(b)
    b_surfs = []
    for b_obj in b_objs:
        b_surf = tagging.extract_tagged_faces(b_obj, b_tags)
        b_surfs.append(b_surf)

    b_surf_names = []
    for i, b_surf in enumerate(b_surfs):
        add_to_scene(state.trimesh_scene, b_surf)
        b_surf_names.append(b_surf.name)

    res = angle_alignment_cost_base(state, a, b_surf_names, visualize)

    for b_surf_name in b_surf_names:
        iu.delete_obj(state.trimesh_scene, b_surf_name)

    return res


def angle_alignment_cost_base(
    state: state_def.State,
    a: Union[str, list[str]],
    b: Union[str, list[str]],
    visualize=False,
):
    """
    Return the dot product between the axes of a and the normal of the closest edge of b
    """
    # print(f'{a=}, {b=}')
    scene = state.trimesh_scene
    a_meshes = iu.meshes_from_names(scene, a)
    b_meshes = iu.meshes_from_names(scene, b)
    b_edges = []
    for b_name, b_mesh in zip(b, b_meshes):
        b_poly = iu.project_to_xy_poly(b_mesh)
        if (b_poly is not None) and (not b_poly.is_empty):
            if isinstance(b_poly, Polygon):
                for i, coord in enumerate(b_poly.exterior.coords[:-1]):
                    start, end = coord, b_poly.exterior.coords[i + 1]
                    if np.isclose(start, end).all():
                        continue
                    b_edges.append((LineString([start, end]), b_name))
            elif isinstance(b_poly, MultiPolygon):
                for sub_poly in b_poly.geoms:
                    for i, coord in enumerate(sub_poly.exterior.coords[:-1]):
                        start, end = coord, sub_poly.exterior.coords[i + 1]
                        if np.isclose(start, end).all():
                            continue
                        b_edges.append((LineString([start, end]), b_name))
        else:
            for edge3d in b_mesh.edges:
                start = b_mesh.vertices[edge3d[0]][:2]
                end = b_mesh.vertices[edge3d[1]][:2]
                if np.isclose(start, end).all():
                    continue
                b_edges.append((LineString([start, end]), b_name))

    a_blender_objs = iu.blender_objs_from_names(a)

    if visualize:
        fig, ax = plt.subplots()
        for edge, _ in b_edges:
            x, y = edge.xy
            ax.plot(x, y, color="red", linewidth=1, label="B Edges")

    score = 0

    for a_name, a_obj, a_mesh in zip(a, a_blender_objs, a_meshes):
        _, axis = get_axis(state, a_obj)
        axis = axis[:2]
        a_poly = iu.project_to_xy_poly(a_mesh)

        if a_poly is not None:
            if isinstance(a_poly, Polygon):
                a_centroid = a_poly.centroid
            elif isinstance(a_poly, MultiPolygon):
                a_centroid = a_poly.centroid
        else:
            a_centroid = Point(a_mesh.vertices[:, :2].mean(axis=0))

        filtered_b_edges = [edge for edge, b_name in b_edges if b_name != a_name]
        if len(filtered_b_edges) == 0:
            continue
        closest_line = iu.closest_edge_to_point_edge_list(filtered_b_edges, a_centroid)

        dx = closest_line.xy[0][1] - closest_line.xy[0][0]  # x1 - x0
        dy = closest_line.xy[1][1] - closest_line.xy[1][0]  # y1 - y0

        # Candidate normal vectors (perpendicular to edge)
        normal_vector_1 = np.array([dy, -dx])
        normal_vector_2 = -normal_vector_1

        # Normalize the vectors
        normal_vector_1 /= np.linalg.norm(normal_vector_1)
        normal_vector_2 /= np.linalg.norm(normal_vector_2)

        dot1 = np.dot(axis, normal_vector_1)
        dot2 = np.dot(axis, normal_vector_2)

        score1 = -dot1 / 2 + 0.5
        score2 = -dot2 / 2 + 0.5

        score += min(score1, score2)

        if visualize:
            if a_poly is not None:
                if isinstance(a_poly, Polygon):
                    x, y = a_poly.exterior.xy
                    ax.fill(x, y, alpha=0.5, fc="blue", ec="black", label="Polygon a")
                elif isinstance(a_poly, MultiPolygon):
                    for sub_poly in a_poly.geoms:
                        x, y = sub_poly.exterior.xy
                        ax.fill(
                            x, y, alpha=0.5, fc="blue", ec="black", label="Polygon a"
                        )
            else:
                x, y = a_mesh.vertices[:, 0], a_mesh.vertices[:, 1]
                ax.scatter(x, y, color="blue", label="Vertices a")

            ax.arrow(
                a_centroid.x,
                a_centroid.y,
                axis[0],
                axis[1],
                head_width=0.15,
                head_length=0.25,
                fc="green",
                ec="green",
                label="Axis of a",
            )
            x, y = closest_line.xy
            ax.plot(x, y, color="green", linewidth=2.5, label="Closest Edge")
            ax.plot(
                a_centroid.x, a_centroid.y, "o", color="black", label="Centroid of a"
            )
            mid_point = closest_line.interpolate(0.5, normalized=True)
            ax.arrow(
                mid_point.x,
                mid_point.y,
                normal_vector_1[0],
                normal_vector_1[1],
                head_width=0.15,
                head_length=0.25,
                fc="yellow",
                ec="yellow",
                label="Normal Vector 1",
            )
            ax.arrow(
                mid_point.x,
                mid_point.y,
                normal_vector_2[0],
                normal_vector_2[1],
                head_width=0.15,
                head_length=0.25,
                fc="orange",
                ec="orange",
                label="Normal Vector 2",
            )

    if visualize:
        ax.set_title("Polygons, Closest Edge and Normal")
        ax.set_aspect("equal")
        ax.grid(True)
        plt.show()

    return score


def angle_alignment_cost(
    state: state_def.State,
    a: Union[str, list[str]],
    b: Union[str, list[str]],
    b_tags=None,
    visualize=False,
):
    if b_tags is not None:
        return angle_alignment_cost_tagged(state, a, b, b_tags, visualize)
    return angle_alignment_cost_base(state, a, b, visualize)


@gin.configurable
def focus_score(
    state: state_def.State, a: Union[str, list[str]], b: str, visualize=False
):
    """
    The how much objects in a focus on b
    """
    scene = state.trimesh_scene
    if isinstance(a, str):
        a = [a]

    a_meshes = iu.meshes_from_names(scene, a)
    a_blender_objs = iu.blender_objs_from_names(a)
    b_mesh = iu.meshes_from_names(scene, b)[0]

    a_polys = [iu.project_to_xy_poly(mesh) for mesh in a_meshes]
    b_poly = iu.project_to_xy_poly(b_mesh)

    if visualize:
        # Plotting the polygons and normals
        fig, ax = plt.subplots()
        if isinstance(b_poly, Polygon):
            x, y = b_poly.exterior.xy
            ax.fill(x, y, alpha=0.5, fc="red", ec="black", label="Polygon b")
        elif isinstance(b_poly, MultiPolygon):
            for sub_poly in b_poly.geoms:
                x, y = sub_poly.exterior.xy
                ax.fill(x, y, alpha=0.5, fc="red", ec="black", label="Polygon b")

    score = 0
    for a_poly, a_mesh, a_obj in zip(a_polys, a_meshes, a_blender_objs):
        axis = get_axis(state, a_obj)[1][:2]
        a_centroid = a_poly.centroid
        b_centroid = b_poly.centroid

        # turn centroids to np array
        a_centroid = np.array([a_centroid.x, a_centroid.y])
        b_centroid = np.array([b_centroid.x, b_centroid.y])

        focus_vec = b_centroid - a_centroid
        focus_vec /= np.linalg.norm(focus_vec)

        if visualize:
            # Plotting the polygons
            if isinstance(a_poly, Polygon):
                x, y = a_poly.exterior.xy
                ax.fill(x, y, alpha=0.5, fc="blue", ec="black", label="Polygon a")
            elif isinstance(a_poly, MultiPolygon):
                for sub_poly in a_poly.geoms:
                    x, y = sub_poly.exterior.xy
                    ax.fill(x, y, alpha=0.5, fc="blue", ec="black", label="Polygon a")

            # plot axis
            ax.arrow(
                a_centroid[0],
                a_centroid[1],
                axis[0],
                axis[1],
                head_width=0.15,
                head_length=0.25,
                fc="green",
                ec="green",
                label="Axis of a",
            )

            # Highlight centroid of a
            ax.plot(
                a_centroid[0], a_centroid[1], "o", color="black", label="Centroid of a"
            )

            # Plot the outward normal vector
            ax.arrow(
                a_centroid[0],
                a_centroid[1],
                focus_vec[0],
                focus_vec[1],
                head_width=0.15,
                head_length=0.25,
                fc="yellow",
                ec="yellow",
                label="Focus vector",
            )

        score += -np.dot(axis, focus_vec) / 2 + 0.5

    if visualize:
        # Set axis properties
        ax.set_title("Polygons, Focus Vector")
        ax.set_aspect("equal")
        ax.grid(True)
        # ax.legend(loc="upper left")
        plt.show()

    return score  # / len(a)


def edge(scene, surface_name: str):
    surface = iu.meshes_from_names(scene, surface_name)[0]
    outline_3d = surface.outline()
    return outline_3d


def min_dist_2d(scene, a: Union[str, list[str]], b, visualize=False):
    """
    projects onto b and finds the min distance between a and b
    """
    if isinstance(a, str):
        a = [a]

    if visualize:
        fig, ax = plt.subplots()
    min_dist = np.inf

    b_path2d, to_3D = b.to_planar()
    plane_normal, plane_origin = iu.get_plane_from_3dmatrix(to_3D)

    a_meshes = iu.meshes_from_names(scene, a)

    a_projections = [
        trimesh.path.polygons.projected(mesh, plane_normal, plane_origin)
        for mesh in a_meshes
    ]
    # Measure the distance
    for a_proj in a_projections:
        source_geom = a_proj
        target_geom = b_path2d.polygons_closed[0].exterior
        dist = source_geom.distance(target_geom)
        if dist < min_dist:
            if visualize:
                pt_a, pt_b = nearest_points(source_geom, target_geom)
                ax.plot([pt_a.x, pt_b.x], [pt_a.y, pt_b.y], color="red")
                # plot source and target geoms
                iu.plot_geometry(ax, source_geom, "blue")
                iu.plot_geometry(ax, target_geom, "green")
            min_dist = dist

    if visualize:
        plt.show()
    return min_dist


def min_dist_boundary(scene: Scene, a: Union[str, list[str]], boundary):
    if isinstance(a, str):
        a = [a]
    if isinstance(boundary, trimesh.path.path.Path3D):
        pass
    elif isinstance(boundary, trimesh.path.path.Path2D):
        pass
    else:
        raise TypeError(f"Unhandled type {boundary=}")


class ConstraintViolated(Exception):
    pass


FATAL = True


def constraint_violated(message):
    if FATAL:
        raise ConstraintViolated(message)
    else:
        print(f"{ConstraintViolated.__name__}: {message}")


def constrain_contact(
    res: ContactResult,
    should_touch=True,
    max_depth=1e-2,
    # normal_dir=None,
    # normal_dot_min=None,
    # normal_dot_max=None
):
    if res.hit is None:
        return False  # arises from an internal-contact query on a set of one element

    if should_touch is not None and should_touch != res.hit:
        if should_touch:
            return False  # constraint_violated(f'At least one of {res.names} must touch eachother')
        else:
            return False  # constraint_violated(f'{res.names} must not touch')

    if res.hit and max_depth is not None:
        observed_depth = max(c.depth for c in res.contacts)
        if observed_depth > max_depth:
            return False  # constraint_violated(f'Contact between {res.names} penetrates by depth {observed_depth} > {max_depth}')
    return True


def constrain_dist(res: dict, min=None, max=None):
    if res.data is None:  # results from internal distance check on 1 object
        print("res data error")
        return

    if not (min is None or min < res.dist):
        return False

    if not (max is None or max > res.dist):
        return False
    return True


def constrain_dist_soft(res: dict, min=None, max=None):
    if res.data is None:  # results from internal distance check on 1 object
        print("res data error")
        return

    if res.dist < min:
        return min - res.dist

    if res.dist > max:
        return res.dist - max
    return 0


def touching_soft(scene, a, b):
    res = any_touching(scene, a, b)

    if res.hit is None:
        print("res hit error")
        return np.inf  # arises from an internal-contact query on a set of one element

    if res.hit:
        observed_depth = max(c.depth for c in res.contacts)
        return observed_depth
    else:
        res = min_dist(scene, a, b)
        if res.data is None:
            return np.inf
        else:
            return res.dist


def dist_soft_score(res: dict, min, max):
    if res.data is None:  # results from internal distance check on 1 object
        return 0

    if res.dist > max:
        return res.dist - max
    elif res.dist < min:
        return min - res.dist
    else:
        return 0


_accessibility_vis_seen_objs = set()  # used to make vis=True below less spammy


def accessibility_cost_cuboid_penetration(
    scene: trimesh.Scene,
    a: Union[str, list[str]],
    b: Union[str, list[str]],
    normal_dir: np.ndarray,
    dist: float,
    bvh_cache: dict = None,
    vis=False,
):
    """
    Extrude the bbox of a by dist in the direction of normal_dir, and check for collisions with b
    Return the maximum distance that any part of b penetrates this extrusion
    """

    if isinstance(a, str):
        a = [a]
    if isinstance(b, str):
        b = [b]

    if len(a) == 0 or len(b) == 0:
        return 0

    a_free_col = trimesh.collision.CollisionManager()

    # find which of +X, -X +Y, -Y, +Z, -Z is the normal_dir. Only these values are supported
    if (
        not np.isclose(np.linalg.norm(normal_dir), 1)
        or np.isclose(normal_dir, 0).sum() != 2
    ):
        raise ValueError(
            f"Invalid normal_dir {normal_dir=}, expected +X, -X, +Y, -Y, +Z, -Z"
        )
    normal_axis = np.argmax(np.abs(normal_dir))
    normal_sign = np.sign(normal_dir[normal_axis])

    visobjs = []
    for name in a:
        T, g = scene.graph[name]
        geom = scene.geometry[g]

        # create an extrusion of the bbox by dist in the direction of normal_dir
        bpy_obj = bpy.data.objects[name]

        freespace_exts = np.copy(np.array(bpy_obj.dimensions))
        freespace_exts[normal_axis] = dist
        freespace_box = trimesh.creation.box(freespace_exts)

        bbox = np.array(bpy_obj.bound_box)
        origin_to_bbox_center = bbox.mean(axis=0)
        extent_from_real_origin = bbox[0 if normal_sign < 0 else -1][normal_axis]

        offset_vec = normal_dir * (
            dist / 2 + extent_from_real_origin - origin_to_bbox_center[normal_axis]
        )
        total_offset_vec = origin_to_bbox_center + offset_vec

        freespace_box_transform = np.array(
            bpy_obj.matrix_world
        ) @ trimesh.transformations.translation_matrix(total_offset_vec)

        a_free_col.add_object(name, freespace_box, freespace_box_transform)

        visobjs.append(geom.apply_transform(T))

        visobjs.append(freespace_box.apply_transform(freespace_box_transform))

    b_col = iu.col_from_subset(scene, b, bvh_cache=bvh_cache)
    hit, contacts = b_col.in_collision_other(a_free_col, return_data=True)

    if vis:
        bobjs = iu.meshes_from_names(scene, b)
        print(
            f"{np.round(origin_to_bbox_center, 3)=} {extent_from_real_origin} {bpy_obj.dimensions}"
        )
        if not all(name in _accessibility_vis_seen_objs for name in a + b):
            trimesh.Scene(visobjs + bobjs).show()
        _accessibility_vis_seen_objs.update(a + b)

    if hit:
        return max(c.depth for c in contacts)
    else:
        return 0


@gin.configurable
def accessibility_cost(scene, a, b, normal, visualize=False, fast=True):
    """
    Computes how much objs b block front access to a. b obj blockages are not summed.
    the closest b obj to a is taken as the representative blockage
    """

    if isinstance(a, str):
        a = [a]
    if isinstance(b, str):
        b = [b]

    b = [b_name for b_name in b if b_name not in a]
    if len(b) == 0:
        return 0

    if visualize:
        fig, ax = plt.subplots()
    a_trimeshes = iu.meshes_from_names(scene, a)
    b_trimeshes = iu.meshes_from_names(scene, b)

    a_objs = iu.blender_objs_from_names(a)
    iu.blender_objs_from_names(b)

    score = 0
    for a_name, a_obj, a_trimesh in zip(a, a_objs, a_trimeshes):
        a_centroid = a_trimesh.centroid

        front_plane_pt = a_centroid
        front_plane_normal = np.array(a_obj.matrix_world.to_3x3() @ Vector(normal))

        a_centroid_proj = (
            a_centroid
            - np.dot(a_centroid - front_plane_pt, front_plane_normal)
            * front_plane_normal
        )

        if fast:
            # get the closest centroid in b and the mesh that it belongs to
            b_centroids = [b_trimesh.centroid for b_trimesh in b_trimeshes]
            distances = [np.linalg.norm(pt - a_centroid_proj) for pt in b_centroids]
            min_index = np.argmin(distances)
            b_closest_pt = b_centroids[min_index]
            b_chosen = b[min_index]
        else:
            # might need to change this to closest pt on the frontal plane
            res = min_dist(scene, a_name, b)
            b_chosen = res.names[1] if res.names[0] == a_name else res.names[0]
            b_closest_pt = res.data.point(b_chosen)

        centroid_to_b = b_closest_pt - a_centroid_proj

        dist = np.linalg.norm(centroid_to_b)
        bounds = iu.meshes_from_names(scene, b_chosen)[0].bounds
        diag_length = np.linalg.norm(bounds[1] - bounds[0])
        if np.dot(centroid_to_b, front_plane_normal) < 0:
            continue
        # cos theta/dist
        score += (np.dot(centroid_to_b, front_plane_normal) / dist**2) * diag_length
        if visualize:
            ax.plot(
                [a_centroid_proj[0], b_closest_pt[0]],
                [a_centroid_proj[1], b_closest_pt[1]],
                color="red",
            )
            # plot source and target geoms
            iu.plot_geometry(ax, a_trimesh, "blue")
            iu.plot_geometry(ax, iu.meshes_from_names(scene, b_chosen)[0], "green")
            # plot front plane
            # plot_geometry(ax, planes.extract_tagged_plane(a_obj, a_tag, a_front_plane), 'black')
            # plot centroid
            ax.plot(
                a_centroid_proj[0],
                a_centroid_proj[1],
                "o",
                color="black",
                label="Centroid of a",
            )

    if visualize:
        plt.show()
    return score


def center_stable_surface(scene, a, state):
    """
    center a objects on their assigned surfaces.
    """
    if isinstance(a, str):
        a = [a]

    score = 0
    a_trimeshes = iu.meshes_from_names(scene, [state.objs[ai].obj.name for ai in a])

    for name, mesh in zip(a, a_trimeshes):
        obj_state = state.objs[name]
        obj = obj_state.obj
        for i, relation_state in enumerate(obj_state.relations):
            relation = relation_state.relation
            parent_obj = state.objs[relation_state.target_name].obj
            obj_tags = relation.child_tags
            parent_tags = relation.parent_tags
            parent_all_planes = state.planes.get_tagged_planes(parent_obj, parent_tags)
            obj_all_planes = state.planes.get_tagged_planes(obj, obj_tags)
            parent_plane = parent_all_planes[relation_state.parent_plane_idx]
            obj_all_planes[relation_state.child_plane_idx]

            if relation_state.parent_plane_idx >= len(parent_all_planes):
                logging.warning(
                    f"{parent_obj.name=} had too few planes ({len(parent_all_planes)}) for {relation_state}"
                )
                return False
            if relation_state.child_plane_idx >= len(obj_all_planes):
                logging.warning(
                    f"{obj.name=} had too few planes ({len(obj_all_planes)}) for {relation_state}"
                )
                return False

            splitted_parent = state.planes.extract_tagged_plane(
                parent_obj, parent_tags, parent_plane
            )
            parent_trimesh = add_to_scene(
                state.trimesh_scene, splitted_parent, preprocess=True
            )
            # splitted_obj = planes.extract_tagged_plane(obj, obj_tags, obj_plane)
            # add_to_scene(state.trimesh_scene, splitted_obj, preprocess=True)
            obj_centroid = mesh.centroid
            parent_centroid = parent_trimesh.centroid
            score += np.linalg.norm(obj_centroid - parent_centroid)

            iu.delete_obj(scene, splitted_parent.name)

    return score


def reflectional_asymmetry_score(
    scene, a: Union[str, list[str]], b: str, use_long_plane=True
):
    """
    Computes the reflectional asymmetry score between a and b
    """
    if isinstance(a, str):
        a = [a]
    if b is None or len(b) == 0:
        return 0

    iu.meshes_from_names(scene, a)
    b_trimesh = iu.meshes_from_names(scene, b)[0]

    a_objs = iu.blender_objs_from_names(a)
    iu.blender_objs_from_names(b)[0]

    bbox = b_trimesh.bounding_box_oriented
    vertices = bbox.vertices

    mid_planes = get_cardinal_planes_bbox(vertices)
    if use_long_plane:
        plane_pt, plane_normal = mid_planes[0]
    else:
        plane_pt, plane_normal = mid_planes[1]

    return symmetry.calculate_reflectional_asymmetry(a_objs, plane_pt, plane_normal)


def coplanarity_cost_pair(scene, a: str, b: str):
    """
    Computes the coplanarity cost between a and b
    """
    a_trimesh = iu.meshes_from_names(scene, a)[0]
    b_trimesh = iu.meshes_from_names(scene, b)[0]

    iu.blender_objs_from_names(a)[0]
    iu.blender_objs_from_names(b)[0]

    a_trimesh_bbox = a_trimesh.bounding_box_oriented
    b_trimesh_bbox = b_trimesh.bounding_box_oriented

    object1_planes = []
    object2_planes = []

    # Helper function to check if a normal is close to any in the list
    def is_normal_new(normal, normals_list):
        normals_np = np.array(normals_list)
        if len(normals_list) > 0:
            return not np.any(np.all(np.isclose(normals_np, normal, atol=1e-3), axis=1))
        return True

    for i in range(len(a_trimesh_bbox.faces)):
        normal = a_trimesh_bbox.face_normals[i]
        if is_normal_new(normal, [n for _, n in object1_planes]):
            object1_planes.append(
                (a_trimesh_bbox.vertices[a_trimesh_bbox.faces[i]][0], normal)
            )

    for i in range(len(b_trimesh_bbox.faces)):
        normal = b_trimesh_bbox.face_normals[i]
        if is_normal_new(normal, [n for _, n in object2_planes]):
            object2_planes.append(
                (b_trimesh_bbox.vertices[b_trimesh_bbox.faces[i]][0], normal)
            )

    # Calculate angle cost matrix for bipartite matching
    angle_cost_matrix = np.zeros((len(object1_planes), len(object2_planes)))
    for j, plane1 in enumerate(object1_planes):
        for k, plane2 in enumerate(object2_planes):
            angle_cost = 1 - np.dot(plane1[1], plane2[1])
            angle_cost_matrix[j, k] = angle_cost

    # Perform linear sum assignment based on angle alignment
    row_ind, col_ind = linear_sum_assignment(angle_cost_matrix)

    # Calculate total costs (angle + distance) for the optimal matching
    total_costs = []
    for r, c in zip(row_ind, col_ind):
        distance_cost = iu.distance_to_plane(
            object1_planes[r][0], object2_planes[c][0], object2_planes[c][1]
        )
        total_cost = (
            angle_cost_matrix[r, c] + distance_cost
        )  # Sum angle and distance costs
        total_costs.append(total_cost)
    total_costs = sorted(total_costs)

    return sum(total_costs[:-2])


def coplanarity_cost(scene, a: Union[str, list[str]]):
    """
    Computes the coplanarity cost between a and b
    """
    if isinstance(a, str):
        a = [a]

    iu.meshes_from_names(scene, a)
    a_objs = iu.blender_objs_from_names(a)

    # Order objects by principal axis
    ordered_objects = iu.order_objects_by_principal_axis(a_objs)

    all_total_costs = []  # To store the sum of angle and distance costs for each optimal matching

    # Iterate over pairs of consecutive objects
    for i in range(len(ordered_objects) - 1):
        all_total_costs.append(
            coplanarity_cost_pair(
                scene, ordered_objects[i].name, ordered_objects[i + 1].name
            )
        )

    # Calculate the final cost as the sum of the remaining costs
    final_cost = sum(all_total_costs) / len(a_objs)

    return final_cost
