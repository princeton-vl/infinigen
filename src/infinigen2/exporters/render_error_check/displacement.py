# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

import bpy
import procfunc as pf

from infinigen2.exporters.render_error_check.util import (
    active_material_output,
    backward_reachable,
    context_materials,
)
from infinigen2.exporters.util.blender_render import DisplacementMode

DISPLACEMENT_UNSAFE_NODE_TYPES = {
    "ShaderNodeUVMap": "uv_map",
    "ShaderNodeAttribute": "attribute",
    "ShaderNodeVertexColor": "vertex_color",
}


class DisplacementCoordError(ValueError):
    pass


def _material_displacement_unsafe_node(material: bpy.types.Material) -> str | None:
    if not material.use_nodes or material.node_tree is None:
        return None
    out = active_material_output(material.node_tree)
    if out is None or not out.inputs["Displacement"].is_linked:
        return None
    for node in backward_reachable(out.inputs["Displacement"]):
        if node.bl_idname in DISPLACEMENT_UNSAFE_NODE_TYPES:
            return node.bl_idname
    return None


def unsafe_displacement_materials(
    materials: list[bpy.types.Material] | None = None,
) -> dict[str, str]:
    if materials is None:
        materials = [m for m in bpy.data.materials if m.users]
    return {
        m.name: node
        for m in materials
        if (node := _material_displacement_unsafe_node(m))
    }


def assert_displacement_coords_safe(
    objects: list[pf.MeshObject] | None = None,
    displacement_mode: DisplacementMode = DisplacementMode.DISPLACEMENT_AND_BUMP,
):
    if displacement_mode not in [
        DisplacementMode.DISPLACEMENT,
        DisplacementMode.DISPLACEMENT_AND_BUMP,
    ]:
        return
    unsafe = unsafe_displacement_materials(context_materials(objects))
    if unsafe:
        raise DisplacementCoordError(
            "materials drive displacement from named-attribute nodes, which Cycles "
            "does not evaluate in the displacement pass (geometry renders flat); "
            f"use coord()/geometry() instead: {unsafe}"
        )
