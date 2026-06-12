import os
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import NamedTuple

import bpy
import numpy as np

SHADER_NODE_COUNT_FAIL = 1000


class ShaderTooComplexError(RuntimeError):
    pass


def _iter_all_nodes(
    node_tree: bpy.types.NodeTree, seen: set[int] | None = None
) -> Iterator[bpy.types.Node]:
    """Yield every node in `node_tree` and each nested node-group tree, visiting
    each distinct group tree once. Order is unspecified; use for collect-all
    passes, not for counting (which needs per-instance inlining)."""
    if seen is None:
        seen = set()
    for node in node_tree.nodes:
        yield node
        if node.type == "GROUP" and node.node_tree and id(node.node_tree) not in seen:
            seen.add(id(node.node_tree))
            yield from _iter_all_nodes(node.node_tree, seen)


def _flattened_node_count(node_tree: bpy.types.NodeTree, memo: dict[int, int]) -> int:
    """Nodes as Cycles compiles them: every group instance is inlined. Counting
    is per-instance — a group reused twice adds its flattened count twice. `memo`
    keyed by id(tree) makes reuse and diamond DAGs cheap and correct."""
    cached = memo.get(id(node_tree))
    if cached is not None:
        return cached
    n = 0
    for node in node_tree.nodes:
        if node.type == "GROUP" and node.node_tree:
            n += _flattened_node_count(node.node_tree, memo)
        else:
            n += 1
    memo[id(node_tree)] = n
    return n


def count_material_nodes(material: bpy.types.Material) -> int:
    """Flattened node count for one material, as Cycles compiles it (every group
    instance inlined). The caller applies SHADER_NODE_COUNT_FAIL and raises
    ShaderTooComplexError as it sees fit."""
    if not material.use_nodes or material.node_tree is None:
        return 0
    return _flattened_node_count(material.node_tree, {})


DISPLACEMENT_UNSAFE_NODE_TYPES = {
    "ShaderNodeUVMap": "uv_map",
    "ShaderNodeAttribute": "attribute",
    "ShaderNodeVertexColor": "vertex_color",
}


class DisplacementCoordError(ValueError):
    pass


def _displacement_visit(
    sock: bpy.types.NodeSocket,
    groups: tuple[bpy.types.Node, ...],
    visited: set,
    stack: list,
) -> str | None:
    """Expand one socket of the backward displacement walk: push its upstream
    sockets onto `stack` and return an unsafe bl_idname if one is reached.

    Descends precisely into node groups: a group output is followed to the
    matching NodeGroupOutput socket, and a NodeGroupInput is mapped back to the
    instance's own input socket, so only wiring that actually feeds displacement
    is examined. Visited is keyed on stable DNA pointers (as_pointer): Python's
    id() is unusable here, as bpy hands out a fresh wrapper per access."""
    for link in sock.links:
        node, from_sock = link.from_node, link.from_socket
        key = (from_sock.as_pointer(), tuple(g.as_pointer() for g in groups))
        if key in visited:
            continue
        visited.add(key)
        if node.bl_idname in DISPLACEMENT_UNSAFE_NODE_TYPES:
            return node.bl_idname
        if node.bl_idname == "NodeGroupInput":
            if groups:
                inp = groups[-1].inputs.get(from_sock.name)
                if inp is not None and inp.is_linked:
                    stack.append((inp, groups[:-1]))
        elif node.type == "GROUP" and node.node_tree:
            inner_out = next(
                (n for n in node.node_tree.nodes if n.bl_idname == "NodeGroupOutput"),
                None,
            )
            if inner_out is not None:
                inner = inner_out.inputs.get(from_sock.name)
                if inner is not None and inner.is_linked:
                    stack.append((inner, groups + (node,)))
        else:
            for inp in node.inputs:
                if inp.is_linked:
                    stack.append((inp, groups))
    return None


def _material_displacement_unsafe_node(material: bpy.types.Material) -> str | None:
    """Return the bl_idname of an unsafe named-attribute node driving the
    material's Displacement output, or None. Walks backward from the Displacement
    input over linked sockets."""
    if not material.use_nodes or material.node_tree is None:
        return None
    tree = material.node_tree
    outputs = [n for n in tree.nodes if n.bl_idname == "ShaderNodeOutputMaterial"]
    out = next(
        (n for n in outputs if n.is_active_output), outputs[0] if outputs else None
    )
    if out is None or not out.inputs["Displacement"].is_linked:
        return None

    visited = set()
    stack = [(out.inputs["Displacement"], ())]
    while stack:
        sock, groups = stack.pop()
        hit = _displacement_visit(sock, groups, visited, stack)
        if hit is not None:
            return hit
    return None


def unsafe_displacement_materials(
    materials: list[bpy.types.Material] | None = None,
) -> dict[str, str]:
    """Map material name -> offending named-attribute node bl_idname for every
    material whose Displacement output is driven by one. Defaults to all in-use
    scene materials. No raise; the caller escalates."""
    if materials is None:
        materials = [m for m in bpy.data.materials if m.users]
    return {
        m.name: node
        for m in materials
        if (node := _material_displacement_unsafe_node(m))
    }


CYCLES_ERROR_PATTERNS = (
    "SVM stack",
    "shader graph",
    "shader compile",
    "Error: ",
)


class CyclesShaderError(RuntimeError):
    pass


@contextmanager
def detect_cycles_errors(replay: bool = True) -> Iterator[None]:
    sys.stdout.flush()
    sys.stderr.flush()

    old1, old2 = os.dup(1), os.dup(2)
    r, w = os.pipe()
    captured = bytearray()

    def pump():
        while chunk := os.read(r, 4096):
            if replay:
                os.write(old1, chunk)
            captured.extend(chunk)

    t = threading.Thread(target=pump, daemon=True)
    t.start()

    try:
        os.dup2(w, 1)
        os.dup2(w, 2)
        os.close(w)
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(old1, 1)
        os.dup2(old2, 2)
        t.join(timeout=5)
        os.close(old1)
        os.close(old2)
        os.close(r)

    text = captured.decode(errors="replace")
    hits = [
        ln for ln in text.splitlines() if any(p in ln for p in CYCLES_ERROR_PATTERNS)
    ]
    if hits:
        raise CyclesShaderError("\n".join(hits))


ACTIVE_RENDER = "<active-render>"

DEGENERATE_UV_AREA = 1e-9


def _node_sampled_layer(node: bpy.types.Node) -> str | None:
    """The UV layer a single node requires, or None.

    A linked ShaderNodeTexCoord UV output requires the active-render layer; a
    ShaderNodeUVMap requires its named layer; a GEOMETRY ShaderNodeAttribute
    requires its named attribute (only flagged later if it names a UV layer).
    "Sampled" means the relevant output socket is linked."""
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
    """UV layers a material samples across its surface AND displacement graphs.
    Elements are layer names and/or the ACTIVE_RENDER sentinel."""
    if not material.use_nodes or material.node_tree is None:
        return set()
    layers = set()
    for node in _iter_all_nodes(material.node_tree):
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
    """Describe the realized mesh's UV layers: the active-render layer name (or
    None), all layer names, and which layers are degenerate (UV bbox ~zero)."""
    uv_layers = getattr(obj.data, "uv_layers", None)
    if not uv_layers:
        return UVLayerInfo(active_render=None, names=(), degenerate=frozenset())
    names = tuple(layer.name for layer in uv_layers)
    active_render = next((l.name for l in uv_layers if l.active_render), None)
    degenerate = frozenset(
        layer.name for layer in uv_layers if _uv_layer_area(layer) <= DEGENERATE_UV_AREA
    )
    return UVLayerInfo(active_render=active_render, names=names, degenerate=degenerate)


MISSING = "missing"  # hard: required UV layer absent on the mesh (the real bug)
DEGENERATE = "degenerate"  # softer: layer present but collapsed to ~zero area


class UVCoordIssue(NamedTuple):
    object_name: str
    material_name: str
    layer: str  # required layer name, or ACTIVE_RENDER
    severity: str  # MISSING or DEGENERATE
    reason: str


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


def _layer_issue(
    obj_name: str,
    mat_name: str,
    layer: str,
    info: UVLayerInfo,
    attr_names: set[str],
) -> UVCoordIssue | None:
    """One UVCoordIssue for a required UV layer, or None when it is satisfied."""
    if layer == ACTIVE_RENDER:
        if info.active_render is None:
            return UVCoordIssue(
                obj_name,
                mat_name,
                ACTIVE_RENDER,
                MISSING,
                "material samples coord().uv (active-render UV layer) but the mesh "
                "has no active-render UV layer",
            )
        if info.active_render in info.degenerate:
            return UVCoordIssue(
                obj_name,
                mat_name,
                info.active_render,
                DEGENERATE,
                f"active-render UV layer {info.active_render!r} is degenerate "
                "(UV bbox ~zero area); UV-driven textures render flat",
            )
        return None
    if layer in info.names:
        if layer in info.degenerate:
            return UVCoordIssue(
                obj_name,
                mat_name,
                layer,
                DEGENERATE,
                f"UV layer {layer!r} is degenerate (UV bbox ~zero area)",
            )
        return None
    if layer not in attr_names:
        return UVCoordIssue(
            obj_name,
            mat_name,
            layer,
            MISSING,
            f"material samples named layer {layer!r} but the mesh has no such UV "
            "layer (nor any attribute by that name)",
        )
    return None


def check_material_uv_coords(
    obj: bpy.types.Object,
    material: bpy.types.Material | None = None,
    mat_index: int | None = None,
) -> list[UVCoordIssue]:
    """Diagnose whether `material`'s UV sampling is satisfied by `obj`'s realized
    UV layers. Pass EITHER a `material` or a `mat_index` into obj.data.materials;
    defaults to obj.active_material. Returns a list of issue messages (empty == ok);
    never raises — callers decide whether to escalate."""
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
