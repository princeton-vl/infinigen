# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

import logging
from typing import NamedTuple

import bpy
import numpy as np
import procfunc as pf

from infinigen2 import context
from infinigen2.exporters.render_error_check.util import (
    context_meshes,
    iter_all_nodes,
)

logger = logging.getLogger(__name__)

ACTIVE_RENDER = "<active-render>"

DEGENERATE_UV_AREA = 1e-9


def _node_sampled_layer(node: bpy.types.Node) -> str | None:
    if node.bl_idname == "ShaderNodeTexCoord":
        uv_out = node.outputs.get("UV")
        if uv_out is not None and uv_out.is_linked:
            return ACTIVE_RENDER
        return None
    any_linked = any(o.is_linked for o in node.outputs)
    if node.bl_idname == "ShaderNodeUVMap":
        if any_linked and node.uv_map:
            return node.uv_map
        return None
    if node.bl_idname == "ShaderNodeAttribute":
        if any_linked and node.attribute_type == "GEOMETRY" and node.attribute_name:
            return node.attribute_name
    return None


def _material_sampled_uv_layers(material: bpy.types.Material) -> set:
    if not material.use_nodes or material.node_tree is None:
        return set()
    layers = set()
    for node, _ in iter_all_nodes(material.node_tree):
        layer = _node_sampled_layer(node)
        if layer is not None:
            layers.add(layer)
    return layers


class UVLayerInfo(NamedTuple):
    active_render: str | None  # name of the active-render UV layer, or None
    names: tuple[str, ...]  # all UV layer names on the mesh
    degenerate: frozenset  # names of layers whose UV bbox is ~zero-area


def _uv_layer_area(uv_layer: bpy.types.MeshUVLoopLayer) -> float:
    n = len(uv_layer.data)
    if n == 0:
        return 0.0
    arr = np.empty(n * 2)
    uv_layer.data.foreach_get("uv", arr)
    arr = arr.reshape(-1, 2)
    span = arr.max(axis=0) - arr.min(axis=0)
    return float(span[0] * span[1])


def _object_uv_layers(obj: bpy.types.Object) -> UVLayerInfo:
    uv_layers = getattr(obj.data, "uv_layers", None)
    if not uv_layers:
        return UVLayerInfo(active_render=None, names=(), degenerate=frozenset())
    names = tuple(layer.name for layer in uv_layers)
    active_render = next((l.name for l in uv_layers if l.active_render), None)
    degenerate = frozenset(
        layer.name for layer in uv_layers if _uv_layer_area(layer) <= DEGENERATE_UV_AREA
    )
    return UVLayerInfo(active_render=active_render, names=names, degenerate=degenerate)


class UVCoordError(ValueError):
    pass


def _resolve_material(
    obj: bpy.types.Object,
    material: bpy.types.Material | None,
    mat_index: int | None,
) -> bpy.types.Material | None:
    if material is not None:
        return material
    if mat_index is not None:
        mats = obj.data.materials
        if mat_index >= len(mats) or mats[mat_index] is None:
            return None
        return mats[mat_index]
    return obj.active_material


def _active_render_issue(obj_name: str, mat_name: str, info: UVLayerInfo) -> str | None:
    if info.active_render is None:
        return (
            f"{obj_name}/{mat_name}: material samples coord().uv but the mesh "
            "has no active-render UV layer"
        )
    if info.active_render in info.degenerate:
        return (
            f"{obj_name}/{mat_name}: active-render UV layer "
            f"{info.active_render!r} is degenerate (UV bbox ~zero area)"
        )
    return None


def _named_layer_issue(
    obj_name: str,
    mat_name: str,
    layer: str,
    info: UVLayerInfo,
    attr_names: set[str],
) -> str | None:
    if layer not in info.names and layer not in attr_names:
        return (
            f"{obj_name}/{mat_name}: material samples named layer {layer!r} but the "
            "mesh has no such UV layer (nor any attribute by that name)"
        )
    if layer in info.names and layer in info.degenerate:
        return (
            f"{obj_name}/{mat_name}: UV layer {layer!r} is degenerate "
            "(UV bbox ~zero area)"
        )
    return None


def _layer_issue(
    obj_name: str,
    mat_name: str,
    layer: str,
    info: UVLayerInfo,
    attr_names: set[str],
) -> str | None:
    if layer == ACTIVE_RENDER:
        return _active_render_issue(obj_name, mat_name, info)
    return _named_layer_issue(obj_name, mat_name, layer, info, attr_names)


def check_material_uv_coords(
    obj: bpy.types.Object,
    material: bpy.types.Material | None = None,
    mat_index: int | None = None,
) -> list[str]:
    polygons = getattr(obj.data, "polygons", None)
    if polygons is None or len(polygons) == 0:
        return []  # no faces -> no rendered surface, so UVs cannot matter
    mat = _resolve_material(obj, material, mat_index)
    if mat is None:
        return []
    required = _material_sampled_uv_layers(mat)
    if not required:
        return []
    info = _object_uv_layers(obj)
    attr_names = {a.name for a in obj.data.attributes}
    issues = (
        _layer_issue(obj.name, mat.name, layer, info, attr_names) for layer in required
    )
    return [i for i in issues if i is not None]


def assert_uv_coords_satisfied(objects: list[pf.MeshObject] | None = None):
    issues = [
        issue
        for obj in context_meshes(objects)
        if obj.type == "MESH" and getattr(obj, "material_slots", None)
        for mat_index in range(len(obj.material_slots))
        for issue in check_material_uv_coords(obj, mat_index=mat_index)
    ]
    if issues:
        error = UVCoordError(
            f"materials sample invalid UV coordinates (renders flat): {issues}"
        )
        context.raise_or_warn(context.globals.error_mode_uv_coords, error, logger)
