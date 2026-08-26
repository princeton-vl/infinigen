# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

import logging

import procfunc as pf

from infinigen2 import context

logger = logging.getLogger(__name__)


class HiddenRenderObjectError(RuntimeError):
    pass


def hidden_render_objects(objects: list[pf.MeshObject]) -> dict[str, str]:
    hidden = {}
    for mo in objects:
        obj = mo.item()
        if obj.hide_render:
            hidden[obj.name] = "hide_render"
        elif not obj.visible_camera:
            hidden[obj.name] = "visible_camera=False"
    return hidden


def assert_render_objects_visible(objects: list[pf.MeshObject] | None = None):
    if not objects:
        return
    hidden = hidden_render_objects(objects)
    if hidden:
        error = HiddenRenderObjectError(
            "objects passed to the render are hidden from the camera and will "
            f"silently not appear: {hidden}"
        )
        context.raise_or_warn(
            context.globals.error_mode_hidden_render_object, error, logger
        )
