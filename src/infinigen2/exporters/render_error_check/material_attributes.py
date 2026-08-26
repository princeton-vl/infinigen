# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

import logging

import bpy
import procfunc as pf

from infinigen2 import context
from infinigen2.exporters.render_error_check.util import (
    context_meshes,
    evaluated_mesh,
    iter_all_nodes,
)

logger = logging.getLogger(__name__)


class MissingAttributeError(ValueError):
    pass


def _node_required_attribute(node: bpy.types.Node) -> str | None:
    if not any(o.is_linked for o in node.outputs):
        return None
    if node.bl_idname == "ShaderNodeAttribute":
        if node.attribute_type == "GEOMETRY" and node.attribute_name:
            return node.attribute_name
    if node.bl_idname == "ShaderNodeVertexColor" and node.layer_name:
        return node.layer_name
    return None


def _material_required_attributes(material: bpy.types.Material) -> set[str]:
    if not material.use_nodes or material.node_tree is None:
        return set()
    names = set()
    for node, _ in iter_all_nodes(material.node_tree):
        name = _node_required_attribute(node)
        if name is not None:
            names.add(name)
    return names


def _missing_for_material(
    obj_name: str, material: bpy.types.Material, present: set[str]
) -> list[str]:
    return [
        f"{obj_name}/{material.name}: samples attribute {name!r} absent on the "
        "mesh (Cycles reads zeros)"
        for name in _material_required_attributes(material)
        if name not in present
    ]


def missing_attribute_issues(obj: bpy.types.Object) -> list[str]:
    mesh = evaluated_mesh(obj)
    if mesh is None:
        return []
    present = {a.name for a in mesh.attributes}
    present |= {uv.name for uv in getattr(mesh, "uv_layers", [])}
    issues = []
    for slot in obj.material_slots:
        if slot.material is not None:
            issues += _missing_for_material(obj.name, slot.material, present)
    return issues


def assert_material_attributes_present(objects: list[pf.MeshObject] | None = None):
    issues = [
        issue
        for obj in context_meshes(objects)
        if obj.type == "MESH" and getattr(obj, "material_slots", None)
        for issue in missing_attribute_issues(obj)
    ]
    if issues:
        error = MissingAttributeError(
            f"materials sample missing mesh attributes: {issues}"
        )
        context.raise_or_warn(
            context.globals.error_mode_missing_attribute, error, logger
        )
