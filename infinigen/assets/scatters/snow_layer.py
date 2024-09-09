# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Zeyu Ma


import bpy

from infinigen.core.init import require_blender_addon
from infinigen.core.tagging import tag_object
from infinigen.core.util import blender as butil

require_blender_addon("real_snow", fail="warn")


class Snowlayer:
    def __init__(self):
        bpy.ops.preferences.addon_enable(module="real_snow")
        pass

    def __call__(self, obj, **kwargs):
        bpy.context.scene.snow.height = 0.1
        with butil.SelectObjects(obj):
            bpy.ops.snow.create()
            snow = bpy.context.active_object
        tag_object(snow, "snow")

    def apply(self, obj, **kwargs):
        require_blender_addon("real_snow")

        bpy.context.scene.snow.height = 0.1
        with butil.SelectObjects(obj):
            bpy.ops.snow.create()
            snow = bpy.context.active_object
        tag_object(snow, "snow")


def apply(obj, selection=None):
    snowlayer = Snowlayer()
    snowlayer.apply(obj)
    # snowlayer(obj)
    return snowlayer
