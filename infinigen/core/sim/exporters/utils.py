# Copyright (C) 2025, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors:
# - Abhishek Joshi: primary author

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple

import bpy
import numpy as np
import trimesh

from infinigen.core import surface, tagging
from infinigen.core.sim.kinematic_node import (
    JointType,
    KinematicNode,
    KinematicType,
    kinematic_node_factory,
)
from infinigen.core.util import blender as butil


def parse_sim_blueprint(sim_blueprint: Path) -> Tuple[str, KinematicNode]:
    """
    Parses the given sim blueprint to be used to generate the required file.
    """
    with open(sim_blueprint, "r") as f:
        blueprint = json.load(f)

    # gets the name of the model
    name = blueprint["name"]

    # build the kinematic DAG and returns the root
    nodes = {}
    for idn, node_info in blueprint["graph"].items():
        nodes[idn] = kinematic_node_factory(
            kinematic_type=KinematicType(node_info["kinematic_type"]),
            joint_type=JointType(node_info["joint_type"]),
            idn=idn if idn != "root" else "",
        )
    for idn, node_info in blueprint["graph"].items():
        for attr_value, child in node_info["children"].items():
            nodes[idn].add_child(int(attr_value), nodes[child])
    kinematic_root = nodes["root"]

    metadata = blueprint["metadata"]
    for _, v in metadata.items():
        del v["parent body label"]
        del v["child body label"]

    metadata["part_labels"] = blueprint["labels"]

    return name, kinematic_root, metadata


def check_all_geom(parent: ET.Element) -> bool:
    """
    Checks if all the children elements of the parent element are geoms.
    """
    return all(child.tag == "geom" for child in parent)


def extract_base_name(name: str) -> str:
    """
    Returns the base asset given the full name.
    """
    return name.split("_")[-1]


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


def get_aabb_center(geometries: List[bpy.types.Object]):
    if isinstance(geometries, list) and len(geometries) == 0:
        raise ValueError("geometries is an empty")
    if isinstance(geometries, bpy.types.Object):
        geometries = [geometries]
    combined_mesh = combine_geometries(geometries)
    vertices = combined_mesh.vertices
    min_corner = np.min(vertices, axis=0)
    max_corner = np.max(vertices, axis=0)
    center = (min_corner + max_corner) / 2
    return center


def get_asset_name(path: List[str]) -> str:
    res = ""
    for i, p in enumerate(path):
        if p == "-0":
            continue
        res += p
        res += "_" if i < len(path) - 1 else ""
    return res


def get_mesh_geometry(obj: bpy.types.Object, attrs: Dict) -> bpy.types.Object:
    """
    Returns the mesh geometry corresponding to the given attributes.
    """
    # get the vertex mask based on the attributes
    vertex_mask = np.ones(len(obj.data.vertices), dtype=bool)
    for attr, val in attrs.items():
        # handles switch cases where certain joints may not be a
        # part of the final geometry
        if attr not in obj.data.attributes:
            continue
        data = surface.read_attr_data(obj, attr)
        vertex_mask = vertex_mask & (data == int(val))

    # extract the mesh based on the vertex mask
    obj_clone = butil.deep_clone_obj(obj, keep_modifiers=True, keep_materials=True)
    mesh_obj = tagging.extract_vertex_mask(obj_clone, vertex_mask)
    butil.delete(obj_clone)
    return mesh_obj


def combine_geometries(geometries: List[bpy.types.Object]) -> trimesh.Trimesh:
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


def get_top_level_geoms(body: ET.Element) -> List[ET.Element]:
    """
    Get the top level geometries part of a given body
    """
    top_geoms = []
    for element in body:
        if element.tag == "geom":
            top_geoms.append(element)
        elif element.tag == "body":
            if element.find("joint") is not None:
                continue
            else:
                top_geoms.extend(get_top_level_geoms(element))

    return top_geoms


def export_individual_mesh(obj: bpy.types.Object, output_dir: Path) -> None:
    """
    Exports an individual mesh to obj format.
    TODO: account for other materials
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    with butil.SelectObjects(obj, active=1):
        bpy.ops.wm.obj_export(
            filepath=str(output_dir / f"{obj.name}.obj"),
            up_axis="Z",
            forward_axis="Y",
            export_selected_objects=True,
        )
