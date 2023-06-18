# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Zeyu Ma
# Date Signed: June 9 2023

import bpy
import mathutils

from numpy.random import uniform, normal

from nodes.node_wrangler import Nodes, NodeWrangler
from surfaces import surface
from util import blender as butil
from nodes import node_utils

from surfaces.templates import snow
from assets.utils.tag import tag_object, tag_nodegroup

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
