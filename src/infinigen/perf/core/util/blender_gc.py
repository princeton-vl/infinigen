# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Alexander Raistrick, Zeyu Ma, Lahav Lipson, Hei Law, Lingjie Mei, Karhan Kayan: original Infinigen Blender utilities (https://github.com/princeton-vl/infinigen/blob/224598d4a2ff95b4fc928e72c222598b7122c698/src/infinigen/core/util/blender.py)
# - David Yan, Jia Deng: optimized implementation

"""Counterparts of `butil` garbage collection, collection membership, parenting and
instance counting."""

import bpy

from infinigen.core.util.blender import (
    SelectObjects,
    get_collection,
    select_none,
    traverse_children,
)

_GC_BATCHABLE = {
    "BlendDataMeshes",
    "BlendDataTextures",
    "BlendDataMaterials",
    "BlendDataImages",
}


def _garbage_collect_node_groups(t, orig, keep_in_use, verbose):
    """Remove node groups in one batch, with the default loop's result: user counts are
    tracked as groups go, so a nested group goes in the same pass only if it comes after
    its last user in list order, as in that loop."""
    groups = list(t)
    users = {ng: ng.users for ng in groups}
    doomed = []
    for ng in groups:
        if keep_in_use and users[ng] > 0:
            continue
        if ng.name in orig:
            continue
        if "(no gc)" in ng.name:
            continue
        if verbose:
            print(f"Garbage collecting {ng} from {t}")
        doomed.append(ng)
        for node in ng.nodes:
            sub = getattr(node, "node_tree", None)
            if sub is not None and sub in users:
                users[sub] -= 1
    if doomed:
        bpy.data.batch_remove(doomed)


def garbage_collect(targets, keep_in_use=True, keep_names=None, verbose=False):
    if keep_names is None:
        keep_names = [[]] * len(targets)

    # Targets are processed in order, so a mesh removed here can orphan a
    # material that a later target removes. Each remove() rescans all IDs to
    # unlink users, so types that cannot reference their own type (meshes,
    # textures, materials, images) are removed in one batch per target. Other
    # types keep one-at-a-time removal: removing one datablock can drop another
    # of the same type to zero users, which the default loop removes in the same
    # pass and a batch computed up front would keep.
    for t, orig in zip(targets, keep_names):
        if t.bl_rna.identifier == "BlendDataNodeTrees":
            _garbage_collect_node_groups(t, orig, keep_in_use, verbose)
            continue
        batch = t.bl_rna.identifier in _GC_BATCHABLE
        doomed = []
        for o in t:
            if keep_in_use and o.users > 0:
                continue
            if o.name in orig:
                continue
            if "(no gc)" in o.name:
                continue
            if verbose:
                print(f"Garbage collecting {o} from {t}")
            if batch:
                doomed.append(o)
            else:
                t.remove(o)
        if doomed:
            bpy.data.batch_remove(doomed)


def count_instance():
    depsgraph = bpy.context.evaluated_depsgraph_get()
    # object_instances holds one non-instance entry per evaluated object plus
    # the generated instances; the lengths avoid a Python loop over them
    return len(depsgraph.object_instances) - len(depsgraph.objects)


def group_in_collection(objs, name: str, reuse=True, **kwargs):
    """Collect `objs` (each None, an object or a list) and their children in `name`."""

    collection = get_collection(name, reuse=reuse)
    grouped = []

    for obj in objs:
        if obj is None:
            continue
        if not isinstance(obj, list):
            obj = [obj]
        for child in obj:
            traverse_children(child, grouped.append)

    put_in_collection(grouped, collection, **kwargs)
    return collection


def parent_to(
    a, b, type="OBJECT", keep_transform=False, no_inverse=False, no_transform=False
):
    if a.name == b.name:
        raise ValueError(f"parent_to expects two distinct objects, got {a=} {b=}")

    if no_inverse and type == "OBJECT":
        matrix_world = a.matrix_world.copy() if keep_transform else None
        a.parent = b
        a.matrix_parent_inverse.identity()
        if keep_transform:
            a.matrix_world = matrix_world
        if no_transform:
            a.location = (0, 0, 0)
            a.rotation_euler = (0, 0, 0)
        if a.parent is not b:
            raise ValueError(
                f"parent_to({a=}, {b=}) failed, after assignment we saw {a.parent=}"
            )
        return

    select_none()
    with SelectObjects([a, b], active=1):
        if no_inverse:
            bpy.ops.object.parent_no_inverse_set(keep_transform=keep_transform)
        else:
            bpy.ops.object.parent_set(type=type, keep_transform=keep_transform)

    if no_transform:
        a.location = (0, 0, 0)
        a.rotation_euler = (0, 0, 0)

    if a.parent is not b:
        raise ValueError(
            f"parent_to({a=}, {b=}) failed, after execution we saw {a.parent=}"
        )


def put_in_collection(objs, collection, exclusive=True):
    if isinstance(collection, str):
        collection = get_collection(collection)
    if isinstance(objs, bpy.types.Object):
        objs = [objs]
    else:
        objs = list(objs)
    for o in objs:
        if exclusive:
            for current in tuple(o.users_collection):
                if current != collection:
                    current.objects.unlink(o)
        if collection not in o.users_collection:
            collection.objects.link(o)
    return collection


def unlink(obj):
    if not isinstance(obj, list):
        obj = [obj]
    for o in obj:
        for c in tuple(o.users_collection):
            c.objects.unlink(o)
