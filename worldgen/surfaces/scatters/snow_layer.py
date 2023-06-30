import bpy
import mathutils

from numpy.random import uniform, normal

from nodes.node_wrangler import Nodes, NodeWrangler
from surfaces import surface
from util import blender as butil
from nodes import node_utils

from surfaces.templates import snow

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
