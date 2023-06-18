# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick
# Date Signed: May 30, 2023

'''
1. Create a blender file at worldgen/dev_scene.blend
2. Click the "Scripting" tab
3. Copy this script into a new script
4. Set the 'mode' to one of the following options:
   - mode='print' will print the result script to your terminal
   - mode='make_script' will create a new script in your blender UI, which you can open
        and run to apply the node code to an object of your choice
   - mode='write_file' will write the script to a new file called 'worldgen/generated_surface_script.py.
        Make sure not to rename / move the script before committing it to git.     
5. Select an object which has some materials and/or geometry nodes on it
6. Click the play button at the top of this script to run it!
7. You should see one python function printed for each material/geonodes on your object
'''

import os
import sys
import importlib
from pathlib import Path
import pdb

import bpy
import mathutils

sys.path.append(os.getcwd())

from util.dev import rreload
from nodes.node_transpiler import transpiler
from nodes import node_wrangler, node_info

rreload(transpiler, 'worldgen')

mode = 'write_file'
target = 'object'

dependencies = [
    'assets.creatures.nodegroups.attach',
    'assets.creatures.nodegroups.curve',
    'assets.creatures.nodegroups.geometry',
    'assets.creatures.nodegroups.hair',
    'assets.creatures.nodegroups.math',
    'assets.creatures.nodegroups.shader',
]

if target == 'object':
    res = transpiler.transpile_object(bpy.context.active_object, dependencies)
elif target == 'world':
    res = transpiler.transpile_world(dependencies)
else:
    raise ValueError(f'Unrecognized {target=}')

if mode == 'print':
    print('START')
    print('\n')
    print(res)
    print('END')
elif mode == 'make_script':
    res_debug = (
        "import bpy\n" +
        res +
        "\napply(bpy.context.active_object)"
    )
    script = bpy.data.texts.new('generated_surface_script')
    script.from_string(res_debug)
elif mode == 'write_file':
    
    filename = 'generated_surface_script.py'
    print(f'Writing generated script to {filename}')
    with Path(filename).open('w') as f:
        f.write(res)
else:
    raise ValueError(f'Unrecognized {mode=}')


