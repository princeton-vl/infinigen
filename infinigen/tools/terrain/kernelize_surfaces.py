# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Zeyu Ma


import os
import sys

import subprocess
from pathlib import Path

import bpy
from infinigen import __version__
from infinigen.assets.materials import chunkyrock, cobble_stone, cracked_ground, dirt, ice, mountain, mud, sand, sandstone, snow, soil, stone
from infinigen.terrain.surface_kernel.kernelizer import Kernelizer
from infinigen.core.util.blender import clear_scene

if __name__ == "__main__":
    parser = Kernelizer()
    for surface in [chunkyrock, cobble_stone, cracked_ground, dirt, ice, mountain, mud, sand, sandstone, snow, soil, stone]:
        clear_scene()
        bpy.ops.mesh.primitive_cube_add()
        obj = bpy.context.active_object
        surface.apply(obj, selection=None, is_rock=False)
        code, _, _ = parser(obj.modifiers[surface.mod_name])
        folder = Path("terrain/source/common/surfaces")
        folder.mkdir(exist_ok=1)
        dst = folder/f"{surface.name}.h"
        with open(dst, "w") as f:
            f.write(f'''// Code generated using version {__version__} of infinigen/tools/kernelize_surfaces.py; refer to infinigen/assets/materials/{surface.name}.py which has the copyright and authors''')
            f.write(code)
            f.write("\n")
        # optional: clang-format needed to format output code
        subprocess.call(f"clang-format -style=\"{{IndentWidth: 4}}\"  -i '{dst}'", shell=True)