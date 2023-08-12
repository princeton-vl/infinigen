# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Zeyu Ma


import bpy
import mathutils

from numpy.random import uniform, normal

from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core import surface
from infinigen.core.util import blender as butil
from infinigen.core.nodes import node_utils

from infinigen.assets.utils.tag import tag_object, tag_nodegroup

class Snowlayer:
    def __init__(self):
        pass
    
    def apply(self, obj, **kwargs):
        bpy.context.scene.snow.height = 0.1
        with butil.SelectObjects(obj):
            bpy.ops.snow.create()
            snow = bpy.context.active_object
        tag_object(snow, "snow")
        tag_object(snow, "boulder")
