# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Karhan Kayan

import os
import shutil
from pathlib import Path

import bpy

addon_src = os.environ.get("FLIP_FLUIDS_ADDON_SRC")
if addon_src is not None:
    addons_dir = Path(bpy.utils.user_resource("SCRIPTS", path="addons", create=True))
    dest = addons_dir / Path(addon_src).name
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(addon_src, dest)
    print(f"Installed FLIP Fluids addon into {dest}")

bpy.ops.preferences.addon_enable(module="flip_fluids_addon")
bpy.ops.flip_fluid_operators.complete_installation()
