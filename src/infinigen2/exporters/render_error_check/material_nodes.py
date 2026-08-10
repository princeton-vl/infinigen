# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

import logging

import bpy
import procfunc as pf

from infinigen2 import context
from infinigen2.exporters.render_error_check.util import (
    context_materials,
    iter_all_nodes,
)

logger = logging.getLogger(__name__)

NORMAL_INPUT_CHECK = "material_normal_input"
TEXTURE_VECTOR_CHECK = "material_texture_vector"
FLOATING_INTERFACE_CHECK = "material_floating_interface"


class MaterialNodeError(ValueError):
    pass


# nodes whose linked Normal/Coat Normal input encodes bump lost under displacement
_NORMAL_INPUT_NODE_TYPES = frozenset(
    {
        "ShaderNodeAmbientOcclusion",
        "ShaderNodeBevel",
        "ShaderNodeBsdfAnisotropic",
        "ShaderNodeBsdfDiffuse",
        "ShaderNodeBsdfGlass",
        "ShaderNodeBsdfPrincipled",
        "ShaderNodeBsdfRefraction",
        "ShaderNodeBsdfSheen",
        "ShaderNodeBsdfToon",
        "ShaderNodeBsdfTranslucent",
        "ShaderNodeFresnel",
        "ShaderNodeLayerWeight",
        "ShaderNodeSubsurfaceScattering",
    }
)

_NORMAL_INPUT_SOCKETS = ("Normal", "Coat Normal")


def _node_issues(node: bpy.types.Node, nested: bool) -> list[tuple[str, str]]:
    name = node.bl_idname
    if name == "ShaderNodeNormalMap":
        msg = f"{name}: use the displacement output instead of normals"
        return [(NORMAL_INPUT_CHECK, msg)]
    if name in _NORMAL_INPUT_NODE_TYPES:
        return [
            (
                NORMAL_INPUT_CHECK,
                f"{name}: {sock!r} input set; use displacement instead",
            )
            for sock in _NORMAL_INPUT_SOCKETS
            if node.inputs.get(sock) is not None and node.inputs[sock].is_linked
        ]
    if name.startswith("ShaderNodeTex"):
        vec = node.inputs.get("Vector")
        if vec is None or not vec.enabled or vec.is_linked:
            return []
        msg = (
            f"{name}: Vector input unlinked, so Cycles samples Generated coords "
            "instead of the intended sample vector; pass an explicit vector"
        )
        return [(TEXTURE_VECTOR_CHECK, msg)]
    if nested and name.startswith("ShaderNodeOutput"):
        msg = f"{name}: floating output node; route through the interface"
        return [(FLOATING_INTERFACE_CHECK, msg)]
    if name.startswith(("FunctionNodeInput", "GeometryNodeInput")):
        msg = f"{name}: floating input node; route through the interface"
        return [(FLOATING_INTERFACE_CHECK, msg)]
    return []


def material_node_issues(material: bpy.types.Material) -> dict[str, list[str]]:
    if not material.use_nodes or material.node_tree is None:
        return {}
    issues: dict[str, list[str]] = {}
    for node, nested in iter_all_nodes(material.node_tree):
        for check, msg in _node_issues(node, nested):
            issues.setdefault(check, []).append(f"{material.name}: {msg}")
    return issues


def assert_material_nodes_valid(objects: list[pf.MeshObject] | None = None):
    grouped: dict[str, list[str]] = {}
    for material in context_materials(objects):
        for check, msgs in material_node_issues(material).items():
            grouped.setdefault(check, []).extend(msgs)

    for check, msgs in grouped.items():
        error = MaterialNodeError(
            f"materials contain invalid shader nodes [{check}]: {msgs}"
        )
        mode = getattr(context.globals, "error_mode_" + check)
        context.raise_or_warn(mode, error, logger)
