# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick
# Date Signed: May 30, 2023

'''
1. Copy this script into the blender scripting UI
2. Select a Nurbs Cylinder object you have modified into some shape
3. Run the script
'''

import pdb

import bpy
import mathutils
import numpy as np

from assets.creatures.geometry import lofting, nurbs, skin_ops
from assets.creatures.util.creature_parser import parse_nurbs_data
from util import blender as butil
    
vis = False
    
for obj in bpy.context.selected_objects:
    handles = parse_nurbs_data(obj)[..., :3]
    
    # blender uses V = long axis of a cylinder by default, this is not our convention
    handles = handles.transpose(0, 1, 2)
    
    # blender has U = 0 face right, ours faces down
    handles = np.roll(handles, 2, axis=1)
    
    
    handles = handles[:, ::-1]

    if vis:
        new_obj = nurbs.nurbs(handles, method='blender', face_size=0.05)
        new_obj.location = obj.location + mathutils.Vector((0, 0.5, 0))

    path = f'{obj.name}.npy'
    np.save(path, handles)
    print('Saved', path)