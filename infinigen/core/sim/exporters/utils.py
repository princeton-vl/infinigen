# Copyright (C) 2025, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors:
# - Abhishek Joshi: primary author

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import bpy
import mathutils
import numpy as np
import trimesh
from trimesh.collision import CollisionManager

from infinigen.core import surface, tagging
from infinigen.core.sim.exporters.base import PathItem
from infinigen.core.sim.kinematic_node import (
    JointType,
    KinematicNode,
    KinematicType,
    kinematic_node_factory,
)
from infinigen.core.util import blender as butil


def array_to_string(arr: np.ndarray):
    """
    Converts an array to string format to store in mjcf.
    """
    return " ".join(map(str, arr.flatten()))


def string_to_array(string: str):
    """
    Converts a mjcf vector string to numpy array format.
    """
    return np.array(
        [float(x) if x != "None" else None for x in string.strip().split(" ")]
    )


def parse_sim_blueprint(sim_blueprint: Dict) -> Tuple[str, KinematicNode]:
    """
    Parses the given sim blueprint to be used to generate the required file.
    """

    # gets the name of the model
    name = sim_blueprint["name"]

    # build the kinematic DAG and returns the root
    nodes = {}
    for idn, node_info in sim_blueprint["graph"].items():
        nodes[idn] = kinematic_node_factory(
            kinematic_type=KinematicType(node_info["kinematic_type"]),
            joint_type=JointType(node_info["joint_type"]),
            idn=idn if idn != "root" else "",
        )
    for idn, node_info in sim_blueprint["graph"].items():
        for attr_value, child in node_info["children"].items():
            nodes[idn].add_child(int(attr_value), nodes[child])
    kinematic_root = nodes["root"]

    metadata = sim_blueprint["metadata"]
    for _, v in metadata.items():
        del v["parent body label"]
        del v["child body label"]

    metadata["part_labels"] = sim_blueprint["labels"]

    return name, kinematic_root, metadata


def get_aabb_center(geometries: List[bpy.types.Object]):
    """
    Returns the bounding box center of a list of blender objects.
    """
    if isinstance(geometries, bpy.types.Object):
        geometries = [geometries]
    if len(geometries) == 0:
        raise ValueError("geometries is an empty")
    combined_mesh = combine_geometries(geometries)
    vertices = combined_mesh.vertices
    min_corner = np.min(vertices, axis=0)
    max_corner = np.max(vertices, axis=0)
    center = (min_corner + max_corner) / 2
    return center


def get_geometry_given_attribs(
    obj: bpy.types.Object,
    attribs: List[Tuple],
    center_at_origin: bool = False,
    extra_attribs: List[Tuple] = [],
) -> bpy.types.Object:
    """
    Gets the geometry corresponding to the attributes query.
    """

    # adding attributes that we always ignore when considering actual geometries
    attribs = attribs + extra_attribs

    vertex_mask = np.ones(len(obj.data.vertices), dtype=bool)
    for attr, val in attribs:
        # handles switch cases where certain joints may not be a
        # part of the final geometry
        if attr not in obj.data.attributes:
            continue
        data = surface.read_attr_data(obj, attr)
        vertex_mask = vertex_mask & (data == val)

    if not any(vertex_mask):
        logging.warning(
            f"A geometry with the following attributes was not found: {attribs}"
        )

    # extract the mesh based on the vertex mask
    obj_clone = butil.deep_clone_obj(obj, keep_modifiers=True, keep_materials=True)
    mesh_obj = tagging.extract_vertex_mask(obj_clone, vertex_mask)

    if center_at_origin:
        translation = mathutils.Vector(-get_aabb_center(mesh_obj))
        for v in mesh_obj.data.vertices:
            v.co += translation
    butil.delete(obj_clone)
    return mesh_obj


def attribs_to_tuples(attribs: List[PathItem]):
    """
    Converts a list of attributes to a list of tuples.
    """
    res = []
    for attrib in attribs:
        res.append((attrib.node.idn, attrib.value))
    return res


def combine_geometries(geometries: List[bpy.types.Object]) -> trimesh.Trimesh:
    """
    Combines a list of blender objects into one geometry.
    """
    combined_mesh = None
    for geometry in geometries:
        mesh = trimesh.Trimesh(
            vertices=[list(vertex.co) for vertex in geometry.data.vertices],
            faces=[
                list(triangle.vertices) for triangle in geometry.data.loop_triangles
            ],
        )
        if combined_mesh is None:
            combined_mesh = mesh
        else:
            combined_mesh = trimesh.util.concatenate([combined_mesh, mesh])
    return combined_mesh


def clean_name(name: str):
    """
    Cleans a string to make it compatible with simulation formats.
    """
    name = name.replace(" ", "_")
    return name.replace("&", "_and_")


def is_2d(obj: bpy.types.Object):
    """Check if an asset is very thin or 2D."""
    verts = np.array([obj.matrix_world @ v.co for v in obj.data.vertices])
    spread = verts.ptp(axis=0)  # max - min per axis
    return np.any(spread < 1e-4)


def get_coord_frame(obj: bpy.types.Object, prefix: str, idx: int, offset: np.ndarray):
    """
    Gets the coordinate frame of a child body.
    """
    xverts = get_geometry_given_attribs(obj, [(prefix + "_xaxis", 1)]).data.vertices

    yverts = get_geometry_given_attribs(obj, [(prefix + "_yaxis", 1)]).data.vertices

    zverts = get_geometry_given_attribs(obj, [(prefix + "_zaxis", 1)]).data.vertices

    vertices = np.vstack(
        [
            xverts[idx].co,
            yverts[idx].co,
            zverts[idx].co,
        ]
    )

    coord_frame = vertices - offset
    coord_frame = coord_frame.T
    assert np.allclose(coord_frame @ coord_frame.T, np.eye(3), atol=1e-2), (
        f"Coordinate frame not orthonormal:\n {coord_frame}"
    )

    return coord_frame


def get_joint_properties(obj: bpy.types.Object, prefix: str):
    """
    Returns properties of the joint as specified in the blend file to be
    used by exporters.
    """
    pos_vals = surface.read_attr_data(obj, prefix + "_poschild")
    pos_mask = np.any(pos_vals != 0.0, axis=1)
    if all(~pos_mask):
        position = np.zeros(3)
    else:
        position = pos_vals[pos_mask].mean(axis=0)

    axis_vals = surface.read_attr_data(obj, prefix + "_axis")
    axis_mask = np.any(axis_vals != 0.0, axis=1)
    if all(~axis_mask):
        axis = np.zeros(3)
    else:
        axis = axis_vals[axis_mask].mean(axis=0)

    min_vals = surface.read_attr_data(obj, prefix + "_min")
    min_mask = min_vals != 0.0
    if all(~min_mask):
        range_min = 0.0
    else:
        range_min = min_vals[min_mask].mean()

    max_vals = surface.read_attr_data(obj, prefix + "_max")
    max_mask = max_vals != 0.0
    if all(~max_mask):
        range_max = 0.0
    else:
        range_max = max_vals[max_mask].mean()

    return position, axis, range_min, range_max


def post_process_collisions(
    post_process_info: Dict, assets_dir: Path, exclude_links: set
):
    """
    Post processes collision meshes so that collision meshes do not intersect
    at initialization.
    """
    MAX_ITERATIONS = 100
    for i in range(MAX_ITERATIONS):
        if i == MAX_ITERATIONS - 1:
            logging.warning(
                "Reached maximum iterations for post processing collisions."
            )
        # get a list of the collision managers and meshes in the scene
        managers, meshes = get_collision_managers(
            post_process_info, assets_dir, offset=i == 0
        )

        # show_scene(meshes)
        # breakpoint()

        collides = False
        for i in range(len(managers)):
            for j in range(i + 1, len(managers)):
                ln1, m1 = managers[i]
                ln2, m2 = managers[j]
                if exclude_links is not None and (
                    (ln1, ln2) in exclude_links or (ln2, ln1) in exclude_links
                ):
                    continue
                has_collision, name_pairs, data = m1.in_collision_other(
                    m2, return_names=True, return_data=True
                )
                if has_collision:
                    collides = True
                    n1, n2 = next(iter(name_pairs))

                    logging.warning(
                        f"Links {ln1} and {ln2} collide at initialization. Updating collision geoms {n1} and {n2}"
                    )

                    # for mjcf and urdf
                    if "path" in post_process_info[ln1][n1]:
                        p1, p2 = (
                            assets_dir / post_process_info[ln1][n1]["path"],
                            assets_dir / post_process_info[ln2][n2]["path"],
                        )

                        mesh1 = post_process_info[ln1][n1]["mesh"]
                        mesh2 = post_process_info[ln2][n2]["mesh"]

                        # show_colliding_meshes(mesh1, mesh2, meshes, data[0].normal)

                        axis = data[0].normal
                        scaled_mesh1 = scale_along_axis(mesh1, axis, 0.99)
                        scaled_mesh2 = scale_along_axis(mesh2, axis, 0.99)

                        # show_colliding_meshes(scaled_mesh1, scaled_mesh2, meshes, data[0].normal)

                        post_process_info[ln1][n1]["mesh"] = scaled_mesh1
                        post_process_info[ln2][n2]["mesh"] = scaled_mesh2

                        scaled_mesh1_copy = scaled_mesh1.copy()
                        scaled_mesh2_copy = scaled_mesh2.copy()

                        scaled_mesh1_copy.apply_translation(
                            -post_process_info[ln1][n1]["offset"]
                        )
                        scaled_mesh2_copy.apply_translation(
                            -post_process_info[ln2][n2]["offset"]
                        )

                        scaled_mesh1_copy.export(p1)
                        scaled_mesh2_copy.export(p2)

                    # for usd
                    else:
                        mesh1 = post_process_info[ln1][n1]["mesh"]
                        mesh2 = post_process_info[ln2][n2]["mesh"]

                        axis = data[0].normal
                        scaled_mesh1 = scale_along_axis(mesh1, axis, 0.99)
                        scaled_mesh2 = scale_along_axis(mesh2, axis, 0.99)

                        # show_colliding_meshes(
                        #     scaled_mesh1, scaled_mesh2, meshes, data[0].normal
                        # )

                        post_process_info[ln1][n1]["mesh"] = scaled_mesh1
                        post_process_info[ln2][n2]["mesh"] = scaled_mesh2

                        colref1 = post_process_info[ln1][n1]["ref"]
                        colref2 = post_process_info[ln2][n2]["ref"]

                        scaled_mesh1_copy = scaled_mesh1.copy()
                        scaled_mesh2_copy = scaled_mesh2.copy()

                        scaled_mesh1_copy.apply_translation(
                            -post_process_info[ln1][n1]["offset"]
                        )
                        scaled_mesh2_copy.apply_translation(
                            -post_process_info[ln2][n2]["offset"]
                        )

                        colref1.GetPointsAttr().Set(scaled_mesh1_copy.vertices)
                        colref2.GetPointsAttr().Set(scaled_mesh2_copy.vertices)

        if not collides:
            break


def scale_along_axis(mesh, axis, fac):
    axis = np.abs(np.asarray(axis, dtype=float))
    axis /= np.linalg.norm(axis)

    scaled_mesh = mesh.copy()

    # center the mesh
    center = scaled_mesh.bounds.mean(axis=0)
    scaled_mesh.apply_translation(-center)

    factor = np.dot(scaled_mesh.vertices, axis)

    # offset = np.matmul(factor.reshape(-1, 1), axis.reshape(-1, 1).T) * dist
    offset = factor.reshape(-1, 1) * axis * (1 - fac)

    new_vertices = scaled_mesh.vertices - offset
    scaled_mesh.vertices = new_vertices

    # move the mesh back to the original position
    scaled_mesh.apply_translation(center)

    return scaled_mesh


def get_collision_managers(post_process_info, assets_dir, offset=False):
    # add a collision manager for each link and load all the associated colliders
    managers = []
    meshes = []
    for link_name, col_info in post_process_info.items():
        cm = CollisionManager()
        for name, info in col_info.items():
            if "mesh" in info:
                mesh = info["mesh"]
            else:
                path = assets_dir / info["path"]
                mesh = trimesh.load(path, force="mesh")
                info["mesh"] = mesh
            if offset:
                mesh.apply_translation(info["offset"])
            meshes.append(mesh)
            cm.add_object(name, mesh)
        managers.append((link_name, cm))
    return managers, meshes


def show_scene(meshes):
    for m in meshes:
        m.visual.vertex_colors = (30, 30, 30, 50)
    scene = trimesh.Scene(meshes)
    scene.show()


def show_colliding_meshes(mesh1, mesh2, meshes, collision_normal):
    scene = trimesh.Scene(meshes)

    frame = trimesh.creation.axis(origin_size=0.05, axis_length=0.2)
    scene.add_geometry(frame)

    for m in meshes:
        m.visual.vertex_colors = (30, 30, 30, 50)

    scene.add_geometry(mesh1)
    scene.add_geometry(mesh2)
    mesh1.visual.vertex_colors = (255, 0, 0, 255)
    mesh2.visual.vertex_colors = (0, 0, 255, 255)

    normal = draw_vector((0, 0, 0), collision_normal)
    scene.add_geometry(normal)

    scene.show()


def draw_vector(start, direction, length=1.0, color=[0, 0, 0, 1]):
    start = np.asarray(start)

    # cylinder for the shaft
    shaft = trimesh.creation.cylinder(radius=0.01, height=length * 0.9, sections=16)
    shaft.apply_translation([0, 0, length * 0.45])

    # cone for the arrowhead
    head = trimesh.creation.cone(radius=0.03, height=length * 0.1)
    head.apply_translation([0, 0, length * 0.95])

    # combine
    arrow = trimesh.util.concatenate([shaft, head])

    # orient it
    T = trimesh.geometry.align_vectors([0, 0, 1], direction)
    arrow.apply_transform(T)
    arrow.apply_translation(start)
    arrow.visual.face_colors = (np.array(color) * 255).astype(np.uint8)
    return arrow
