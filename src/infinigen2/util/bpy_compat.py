# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

from typing import Any

import bpy

__all__ = [
    "attribute_factor_output",
    "bpy_version",
    "eevee_engine_name",
    "new_interface_socket",
    "scene_compositor_node_tree",
    "set_displacement_method",
]


def bpy_version() -> tuple[int, int]:
    return tuple(bpy.app.version[:2])


def new_interface_socket(
    node_tree: bpy.types.NodeTree, name: str, in_out: str, socket_type: str
) -> Any:
    if bpy_version() < (4, 0):
        sockets = node_tree.inputs if in_out == "INPUT" else node_tree.outputs
        return sockets.new(socket_type, name)
    return node_tree.interface.new_socket(
        name=name, in_out=in_out, socket_type=socket_type
    )


def set_displacement_method(material: bpy.types.Material, method: str) -> None:
    if bpy_version() < (4, 1):
        material.cycles.displacement_method = method
        return
    material.displacement_method = method


def eevee_engine_name() -> str:
    if (4, 2) <= bpy_version() < (5, 0):
        return "BLENDER_EEVEE_NEXT"
    return "BLENDER_EEVEE"


def scene_compositor_node_tree(scene: bpy.types.Scene) -> bpy.types.NodeTree:
    if bpy_version() < (5, 0):
        scene.use_nodes = True
        return scene.node_tree
    if scene.compositing_node_group is None:
        scene.compositing_node_group = bpy.data.node_groups.new(
            f"{scene.name} compositor", "CompositorNodeTree"
        )
    return scene.compositing_node_group


def attribute_factor_output(
    node: bpy.types.ShaderNodeAttribute,
) -> bpy.types.NodeSocket:
    if bpy_version() < (5, 2):
        return node.outputs["Fac"]
    return node.outputs["Factor"]
