# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

from collections.abc import Iterator

import bpy
import procfunc as pf


def iter_all_nodes(
    node_tree: bpy.types.NodeTree,
    nested: bool = False,
    seen: set[int] | None = None,
) -> Iterator[tuple[bpy.types.Node, bool]]:
    if seen is None:
        seen = set()
    for node in node_tree.nodes:
        yield node, nested
        if node.type == "GROUP" and node.node_tree and id(node.node_tree) not in seen:
            seen.add(id(node.node_tree))
            yield from iter_all_nodes(node.node_tree, True, seen)


def flattened_node_count(node_tree: bpy.types.NodeTree, memo: dict[int, int]) -> int:
    cached = memo.get(id(node_tree))
    if cached is not None:
        return cached
    n = 0
    for node in node_tree.nodes:
        if node.type == "GROUP" and node.node_tree:
            n += flattened_node_count(node.node_tree, memo)
        else:
            n += 1
    memo[id(node_tree)] = n
    return n


def active_material_output(tree: bpy.types.NodeTree) -> bpy.types.Node | None:
    outputs = [n for n in tree.nodes if n.bl_idname == "ShaderNodeOutputMaterial"]
    return next(
        (n for n in outputs if n.is_active_output), outputs[0] if outputs else None
    )


def _push_group_input(
    from_sock: bpy.types.NodeSocket, groups: tuple, stack: list
) -> None:
    if not groups:
        return
    inp = groups[-1].inputs.get(from_sock.name)
    if inp is not None and inp.is_linked:
        stack.append((inp, groups[:-1]))


def _push_group_output(
    node: bpy.types.Node, from_sock: bpy.types.NodeSocket, groups: tuple, stack: list
) -> None:
    inner_out = next(
        (n for n in node.node_tree.nodes if n.bl_idname == "NodeGroupOutput"), None
    )
    if inner_out is None:
        return
    inner = inner_out.inputs.get(from_sock.name)
    if inner is not None and inner.is_linked:
        stack.append((inner, groups + (node,)))


def _push_node_inputs(node: bpy.types.Node, groups: tuple, stack: list) -> None:
    for inp in node.inputs:
        if inp.is_linked:
            stack.append((inp, groups))


def _push_upstream(
    node: bpy.types.Node, from_sock: bpy.types.NodeSocket, groups: tuple, stack: list
) -> None:
    if node.bl_idname == "NodeGroupInput":
        _push_group_input(from_sock, groups, stack)
    elif node.type == "GROUP" and node.node_tree:
        _push_group_output(node, from_sock, groups, stack)
    else:
        _push_node_inputs(node, groups, stack)


def _expand_socket(
    sock: bpy.types.NodeSocket, groups: tuple, visited: set, stack: list
) -> list:
    found = []
    for link in sock.links:
        node, from_sock = link.from_node, link.from_socket
        # as_pointer, not id: bpy hands out a fresh wrapper per access
        key = (from_sock.as_pointer(), tuple(g.as_pointer() for g in groups))
        if key in visited:
            continue
        visited.add(key)
        found.append(node)
        _push_upstream(node, from_sock, groups, stack)
    return found


def backward_reachable(socket: bpy.types.NodeSocket) -> Iterator[bpy.types.Node]:
    visited: set = set()
    stack: list = [(socket, ())]
    while stack:
        sock, groups = stack.pop()
        yield from _expand_socket(sock, groups, visited, stack)


def context_meshes(objects: list[pf.MeshObject] | None) -> list[bpy.types.Object]:
    if objects is None:
        return list(bpy.data.objects)
    return [obj.item() for obj in objects]


def _collect_object_materials(obj: pf.MeshObject, mats: dict) -> None:
    for slot in obj.item().material_slots:
        if slot.material is None:
            continue
        mats[slot.material.name] = slot.material


def context_materials(objects: list[pf.MeshObject] | None) -> list[bpy.types.Material]:
    if objects is None:
        return [m for m in bpy.data.materials if m.users]
    mats = {}
    for obj in objects:
        _collect_object_materials(obj, mats)
    return list(mats.values())
