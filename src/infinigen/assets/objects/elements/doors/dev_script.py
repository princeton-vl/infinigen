# Copyright (C) 2025, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors:
# - Yiming Zuo: primary author

"""
# INSTRUCTIONS FOR USE IN BLENDER
# Create a blender file at repo_root/worldgen/dev_scene.blend
# Click the 'Scripting' Tab, then + to make a new empty script
# Put python code in quotations below into that script, and run it.

# CODE TO BE PUT IN BLENDER EDITOR:
import bpy
from pathlib import Path
import sys, importlib
import numpy as np

pwd = Path(bpy.data.filepath).parent
if not str(pwd) in sys.path:
    sys.path.append(str(pwd))

import gin
gin.clear_config()
gin.enter_interactive_mode()

from assets.creatures.tools import dev_script

np.random.seed(1)

from types import ModuleType
from importlib import reload
import os, sys

from util.dev import rreload
from assets.creatures import generate
from util import blender

rreload(generate, 'worldgen')

from surfaces.surface import registry
gin.parse_config_files_and_bindings(['config/base.gin'], [])
registry.initialize_from_gin()

blender.clear_scene(keep=['Camera', 'UV', 'Plane', 'Reference', 'Example', 'Dev', 'duck'])

get_factory = lambda _: generate.CreatureFactory(np.random.randint(1e5))
dev_script.main(get_factory=get_factory, species=3, n=1, spacing=2,
    join=False,
    remesh=False,
    rigging=False,
    pose=False,
    constraints=False,
    animation=False,
    materials=False,
    skin_sim=False,
    particles=False
)


"""

import bpy
import mathutils
import numpy as np
from tqdm import tqdm


def main(get_factory, species=9, n=16, spacing=4, one_row=False, **kwargs):
    import timeit

    start = timeit.default_timer()

    if one_row:
        spec_row = species
        inst_row = n
    else:
        spec_row = int(np.sqrt(species))
        inst_row = int(np.sqrt(n))
    spec_spacing = (inst_row + 1) * spacing

    # spec_row = inst_row = 1

    cam = bpy.context.scene.camera
    pbar = tqdm(total=species * n)
    for spec in range(species):
        all_objs = []
        factory = get_factory(spec)
        base_pos = spec_spacing * mathutils.Vector(
            (spec % spec_row, spec // spec_row, 0)
        )
        for i in range(n):
            loc = base_pos + spacing * mathutils.Vector(
                (i % inst_row, i // inst_row, 0)
            )
            distance = (cam.location - loc).length if cam else 0.1
            obj = factory(
                i=i, loc=loc, rot=mathutils.Euler(), distance=distance, **kwargs
            )
            all_objs.append(obj)
            pbar.update(1)
        factory.finalize_assets(all_objs)
    time = timeit.default_timer() - start
    print(f"{time:.2f}s for {n} objects, {time / (n):.2f} per object")
