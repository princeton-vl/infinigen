# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick
# Date Signed: May 30, 2023

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
import mathutils
from numpy.random import uniform, normal
from nodes.node_wrangler import Nodes, NodeWrangler
from surfaces import surface

def shader_dev(nw):
    principled_bsdf = nw.new_node(Nodes.PrincipledBSDF)
    
    material_output = nw.new_node(Nodes.MaterialOutput,
        input_kwargs={'Surface': principled_bsdf})

def geo_dev(nw):
    group_input = nw.new_node(Nodes.GroupInput)
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': group_input.outputs["Geometry"]})

def apply(obj):
    surface.add_geomod(obj, geo_dev, apply=False)
    surface.add_material(obj, shader_dev, reuse=False)
apply(bpy.context.active_object)