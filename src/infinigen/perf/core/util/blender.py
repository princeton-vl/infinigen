# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Alexander Raistrick, Zeyu Ma, Lahav Lipson, Hei Law, Lingjie Mei, Karhan Kayan: original Infinigen Blender utilities (https://github.com/princeton-vl/infinigen/blob/224598d4a2ff95b4fc928e72c222598b7122c698/src/infinigen/core/util/blender.py)
# - David Yan, Jia Deng: optimized implementation

"""`butil` counterparts and Blender helpers for the optimized packages.

Inside an isolated build, `select` and `SelectObjects` expose an object missing from
the scratch view layer instead of refusing it, and `to_mesh` makes it evaluable first.
"""

import logging

import bpy

from infinigen.core.util import blender as butil
from infinigen.core.util.logging import Suppress
from infinigen.perf import options
from infinigen.perf.core.util.isolation import IsolatedScene

logger = logging.getLogger(__name__)


def save_blend(path, autopack=False, verbose=False):
    """`butil.save_blend`, compressed unless `compress_output_blend` is off."""
    if verbose:
        print(f"Saving .blend to {path} ({'with' if autopack else 'without'} textures)")

    with Suppress():
        if autopack:
            bpy.ops.file.autopack_toggle()
        bpy.ops.wm.save_as_mainfile(
            filepath=str(path), compress=options.compress_output_blend()
        )
        if autopack:
            bpy.ops.file.autopack_toggle()


def apply_modifiers_via_evaluated_mesh(obj: bpy.types.Object) -> None:
    """Materialize a mesh modifier stack without the object operator."""

    if obj.type != "MESH":
        raise TypeError(f"Expected mesh object, got {obj.type}")
    if not obj.modifiers:
        return

    old_mesh = obj.data
    old_name = old_mesh.name
    IsolatedScene.ensure_evaluable(obj)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(depsgraph)
    new_mesh = bpy.data.meshes.new_from_object(evaluated, depsgraph=depsgraph)

    # new_from_object has copied the evaluated result, so clearing the stack
    # does not affect the new datablock
    obj.modifiers.clear()
    obj.data = new_mesh
    bpy.data.meshes.remove(old_mesh)
    new_mesh.name = old_name
    butil.purge_empty_materials(obj)


class SelectObjects:
    """Counterpart of `butil.SelectObjects`; holds only the method that differs."""

    def _check_selectable(self):
        for o in self.objects:
            if o.name not in bpy.context.scene.objects or IsolatedScene.is_parked(o):
                IsolatedScene.ensure_selectable(o)
        unlinked = [o for o in self.objects if o.name not in bpy.context.scene.objects]
        if len(unlinked) > 0:
            raise ValueError(
                f"{butil.SelectObjects.__name__} had objects {unlinked=} which are not in bpy.context.scene.objects and cannot be selected"
            )

        hidden = [o for o in self.objects if o.hide_viewport]
        if len(hidden) > 0:
            raise ValueError(
                f"{butil.SelectObjects.__name__} had objects {hidden=} which are hidden and cannot be selected"
            )


def select(objs: bpy.types.Object | list[bpy.types.Object]):
    butil.select_none()
    if not isinstance(objs, list):
        objs = [objs]
    for o in objs:
        if o.name not in bpy.context.scene.objects:
            if not IsolatedScene.ensure_selectable(o):
                raise ValueError(f"Object {o.name=} not in scene and cant be selected")
        elif IsolatedScene.is_parked(o):
            IsolatedScene.ensure_selectable(o)
        o.select_set(True)


def to_mesh(object, context=bpy.context):
    IsolatedScene.ensure_evaluable(object)
    deg = context.evaluated_depsgraph_get()
    me = bpy.data.meshes.new_from_object(object.evaluated_get(deg), depsgraph=deg)

    new_obj = bpy.data.objects.new(object.name + "_mesh", me)
    context.collection.objects.link(new_obj)

    for o in context.selected_objects:
        o.select_set(False)

    new_obj.matrix_world = object.matrix_world
    new_obj.select_set(True)
    context.view_layer.objects.active = new_obj

    return new_obj
