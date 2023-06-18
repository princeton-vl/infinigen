# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lahav Lipson
# Date Signed: May 2 2023

'''
To use this file, do the following:

cd worldgen
blender dev_scene.blend

Then in the blender gui:
- click the 'Scripting' tab
- click the folder button to open this file as a new script
- append some meshes from another blend file, or run some other code to create some objects 
- implement a geonode / material setup using the node wrangler in the functions below
- left/right click the object you want to apply a material to
- click the play button to run the script

'''

import bpy
import gin
import mathutils
from numpy.random import uniform, normal
from nodes.node_wrangler import Nodes, NodeWrangler
from surfaces import surface
from surfaces.templates import dirt, cobble_stone, cracked_ground, sand, mountain
import generate # Necessary to parse gin config files

gin.clear_config()
gin.enter_interactive_mode()
gin.parse_config_files_and_bindings(["config/base.gin"], [])

def right_half(nw: NodeWrangler, **kwargs):
    x = nw.new_node(Nodes.SeparateXYZ, [nw.new_node(Nodes.InputPosition)])
    return nw.new_node(Nodes.Math, [(x, "Y"), 0], attrs={"operation": "GREATER_THAN"})

def left_half(nw: NodeWrangler, **kwargs):
    x = nw.new_node(Nodes.SeparateXYZ, [nw.new_node(Nodes.InputPosition)])
    return nw.new_node(Nodes.Math, [(x, "Y"), 0], attrs={"operation": "LESS_THAN"})

obj = bpy.context.active_object
cracked_ground.apply(obj, selection=surface.write_attribute(obj, right_half))
dirt.apply(obj, selection=surface.write_attribute(obj, left_half))