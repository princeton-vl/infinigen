# Reference: Princeton University.



'''
Functionality: Convert a Blender material object to a folder containing its script and supp files

Input: 
    1. Blender Material Object
    2. Output folder path

Output: 
    1. A folder contains the Blender-python script and supplementary files like image texture

Problems:
    1. Cannot save supp files
    2. Cannot modify sub-properties inside a node, such as color mode of normal map node
    3. Only support one selected object 
    
'''

import os
import sys
import importlib
from pathlib import Path
import pdb

import bpy
import mathutils

from infinigen.core.nodes.node_transpiler import transpiler
from infinigen.core.nodes import node_wrangler, node_info

output_path = 'generated_surface_script.py'
target = 'object'

dependencies = [
    # if your transpile target is using nodegroups taken from some python file,
    # add those filepaths here so the transpiler imports from them rather than creating a duplicate definition.
]


# Separate the two kinds of targets and transpile their nodes(both geometry and material)
if target == 'object':
    res = transpiler.transpile_object(bpy.context.active_object, dependencies)
elif target == 'world':
    res = transpiler.transpile_world(dependencies)
else:
    raise ValueError(f'Unrecognized {target=}')

print(f'Writing generated script to {output_path}')
with Path(output_path).open('w') as f:
    f.write(res)


