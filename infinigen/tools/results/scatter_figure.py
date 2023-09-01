# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Hongyu Wen


import bpy
import mathutils
from pathlib import Path
import sys, importlib
import numpy as np
import os
import argparse
import numpy as np
from numpy.random import uniform as U
    
import gin
from infinigen.core.surface import registry

from infinigen.assets.scatters import (
    grass, 
    chopped_trees, 
    pine_needle, 
    flowerplant, 
    fern, 
    pinecone, 
    urchin, 
    seaweed,
    seashells
)
from infinigen.assets.materials import dirt, sand, mud
from infinigen.core.placement import density
import math
from infinigen.core.util import blender as butil 
from infinigen.assets.lighting import sky_lighting
from infinigen.core import init

gin.clear_config()
gin.enter_interactive_mode()

gin.parse_config_files_and_bindings(['config/base.gin'], [])
registry.initialize_from_gin()

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--save', type=str, default='stable')
parser.add_argument('-m', '--mode', type=str, default='grass')
parser.add_argument('-d', '--debug', type=bool, default=False)
parser.add_argument('-v', '--view', type=int, default=0.5)
parser.add_argument('-ix', '--index_x', type=int, default=0)
parser.add_argument('-iy', '--index_y', type=int, default=0)
args = init.parse_args_blender()


def apply_scatters(obj, mode, index):
    all_objects = [obj]
    path = f"outputs/scatter_figure/{mode}/{mode}_{index[0]}_{index[1]}.png"
    if os.path.exists(path) and args.save == 'stable':
        butil.delete(all_objects)
        return
   
    if mode == 'grass':
        mud.apply(obj)
        selection = density.placement_mask(normal_dir=(0, 0, 1), scale=3, 
                return_scalar=True, select_thresh=U(0, 0.2))
        go, _ = grass.apply(obj, selection=selection, density=15)
        all_objects.append(go)
        # if U() < 0.3:
        #     fo, _ = flowerplant.apply(obj, 
        #         selection=density.get_placement_distribution(normal_dir=(0, 0, 1), scale=U(1, 3), select_thresh=0.1, return_scalar=True), 
        #         density=U(0.3, 2))
        #     all_objects.append(fo)
    # elif mode == 'fern':
    #     dirt.apply(obj)
    #     fern.apply(obj, density=3.5, selection=density.get_placement_distribution(normal_dir=(0, 0, 1), scale=0.1, return_scalar=True))
    elif mode == 'seafloor':
        mud.apply(obj)
        dirt.apply(obj)
        uo, _ = urchin.apply(obj, selection=density.placement_mask(scale=U(1, 3), select_thresh=U(0.8, 1.2)), density=U(0.8, 1.2), n=int(U(3, 10)))
        mo, _ = seashells.apply(obj, selection=density.placement_mask(scale=U(1, 3), select_thresh=U(0.8, 1.2)), density=U(1.5, 2.5), n=int(U(5, 15)))
        so, _ = seaweed.apply(obj, selection=density.placement_mask(scale=U(1, 3), select_thresh=U(0.8, 1.2), normal_thresh=0.4), density=U(1, 2), n=int(U(3, 10)))
        all_objects.append(uo)
        all_objects.append(mo)
        all_objects.append(so)
    elif mode == 'fallen_trees':
        po, _ = pine_needle.apply(obj,
                selection=density.placement_mask(scale=U(0.2, 1), select_thresh=U(0.4, 0.6), return_scalar=True),
                density=U(1000, 3000))
        pio, _ = pinecone.apply(obj,
            selection=density.placement_mask(scale=U(0.1, 0.4), select_thresh=U(0.4, 0.6)),
            density=U(0.4, 0.6))
        selection = density.placement_mask(scale=U(0.1, 0.4), select_thresh=U(0.4, 0.6), density=U(0.4, 0.6))
        co, _ = chopped_trees.apply(obj, selection=selection)
        all_objects.append(po)
        all_objects.append(pio)
        all_objects.append(co)
    else:
        assert False
    
    bpy.context.scene.render.film_transparent = True
    bpy.context.scene.render.image_settings.color_mode = 'RGBA'
    # bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.render.filepath = path
    bpy.ops.render.render(scene=bpy.context.scene.name, write_still=True) 
    # butil.delete(all_objects)
    
s = 3
margin = 0
n = 100
rowsize = 10
planeres = 300

butil.clear_scene()
sky_lighting.add_lighting()


bpy.ops.object.camera_add(location=mathutils.Vector(args.view * np.array([-10,-10,10])), rotation=(np.deg2rad(70), 0, np.deg2rad(-45)))
cam = bpy.context.active_object
bpy.context.scene.camera = cam
bpy.context.scene.render.resolution_x = 2048
bpy.context.scene.render.resolution_y = 1024

seen_list = [
    (0, 0), (0, 1), (0, 2), (0, 3), (0, 4),
    (1, 0), (1, 1), (1, 2), (1, 3), (1, 4),
    (2, 0), (2, 1), (2, 2), (2, 3), (2, 4),
    (3, 0), (3, 1), (3, 2), (3, 3), (3, 4), 
    (4, 0), (4, 1), (4, 2), (4, 3), (4, 4), 
    (5, 1), (5, 2), (5, 3), (5, 4), (5, 5), 
    (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), 
]

try_list = [(1, 1)]

if (args.debug):
    enum_list = try_list
else:
    enum_list = seen_list

prime1 = 1009
prime2 = 17
mode = args.mode
x = args.index_x
y = args.index_y
np.random.seed(x * prime1 + y + prime2)
# x, y = i // rowsize, i % rowsize
pos = (s + margin) * mathutils.Vector((x, y, 0))
bpy.ops.mesh.primitive_grid_add(size=s, location=pos, x_subdivisions=planeres, y_subdivisions=planeres)
plane = bpy.context.active_object
apply_scatters(plane, mode=mode, index=(x, y))
    # butil.delete(plane)

# path = f"outputs/scatter_figure/grass.png"
# bpy.context.scene.render.filepath = path
# bpy.ops.render.render(scene=bpy.context.scene.name, write_still=True) 