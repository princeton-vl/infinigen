# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Zeyu Ma


import logging

import bpy

from infinigen.core.init import require_blender_addon
from infinigen.core.tagging import tag_object
from infinigen.core.util import blender as butil
from infinigen.infinigen_gpl.surfaces import snow as bundled_snow

logger = logging.getLogger(__name__)


class Snowlayer:
    def __init__(self):
        self.use_addon = require_blender_addon("real_snow", fail="warn")
        self.use_bundled_fallback = not self.use_addon
        if self.use_bundled_fallback:
            logger.info(
                "Using bundled GPL snow fallback because Blender addon 'real_snow' is unavailable."
            )

    def apply(self, obj, **kwargs):
        if self.use_bundled_fallback:
            bundled_snow.apply(obj, **kwargs)
            return
        bpy.context.scene.snow.height = 0.1
        with butil.SelectObjects(obj):
            bpy.ops.snow.create()
            snow = bpy.context.active_object
        tag_object(snow, "snow")


def apply(obj, selection=None):
    snowlayer = Snowlayer()
    snowlayer.apply(obj, selection=selection)
    # snowlayer(obj)
    return snowlayer
