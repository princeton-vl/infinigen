# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors: Alexander Raistrick


"""
Copy this file into blender's scripting window and run it whenever you open a new blender instance.

It will configure the sys.path and load gin. This is necessary before any other procgen files can be imported/used within blender.

Once this is done, you can do things like `from infinigen.assets.objects.creatures.util.genomes.carnivore import CarnivoreFactory` then `CarnivoreFactory(0).spawn_asset(0)` directly in the blender commandline
"""

# ruff: noqa

import logging
import os
import sys
from pathlib import Path

import bpy

pwd = os.getcwd()
sys.path.append(str(Path(__file__).parent.parent.parent))

import gin

gin.clear_config()
gin.enter_interactive_mode()

from infinigen.core import init, surface
from infinigen_examples import generate_nature

init.apply_gin_configs(
    Path(pwd) / "infinigen_examples/configs_nature", ["base.gin"], skip_unknown=True
)
surface.registry.initialize_from_gin()

logging.basicConfig(
    format="[%(asctime)s.%(msecs)03d] [%(name)s] [%(levelname)s] | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.WARNING,
)
logging.getLogger("infinigen").setLevel(logging.DEBUG)
