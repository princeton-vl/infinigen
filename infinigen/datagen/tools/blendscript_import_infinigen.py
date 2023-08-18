

'''
Copy this file into blender's scripting window and run it whenever you open a new blender instance.

It will configure the sys.path and load gin. This is necessary before any other procgen files can be imported/used within blender.

Once this is done, you can do things like `from assets.creatures.genomes.carnivore import CarnivoreFactory` then `CarnivoreFactory(0).spawn_asset(0)` directly in the blender commandline
'''

import bpy
from pathlib import Path
import sys
import os

# ruff: noqa
pwd = os.getcwd()
sys.path.append(pwd)

import generate # so gin can find all its targets
import gin
gin.clear_config()
gin.enter_interactive_mode()

gin.parse_config_files_and_bindings(['config/base.gin'], [])
from surfaces.surface import registry
registry.initialize_from_gin()