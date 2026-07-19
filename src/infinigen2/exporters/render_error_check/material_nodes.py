# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

import bpy
import procfunc as pf

from infinigen2.exporters.render_error_check.util import (
    context_materials,
    iter_all_nodes,
)


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


def _node_issues(node: bpy.types.Node, nested: bool) -> list[str]:
    name = node.bl_idname
    if name == "ShaderNodeNormalMap":
        return [f"{name}: use the displacement output instead of normals"]
    if name in _NORMAL_INPUT_NODE_TYPES:
        return [
            f"{name}: {sock!r} input set; use displacement instead"
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
        return [msg]
    if nested and name.startswith("ShaderNodeOutput"):
        return [f"{name}: floating output node; route through the interface"]
    if name.startswith(("FunctionNodeInput", "GeometryNodeInput")):
        return [f"{name}: floating input node; route through the interface"]
    return []


def material_node_issues(material: bpy.types.Material) -> list[str]:
    if not material.use_nodes or material.node_tree is None:
        return []
    issues = []
    for node, nested in iter_all_nodes(material.node_tree):
        issues += [f"{material.name}: {i}" for i in _node_issues(node, nested)]
    return issues


def assert_material_nodes_valid(objects: list[pf.MeshObject] | None = None):
    issues = [
        issue for m in context_materials(objects) for issue in material_node_issues(m)
    ]
    if issues:
        raise MaterialNodeError(f"materials contain invalid shader nodes: {issues}")
