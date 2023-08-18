from pathlib import Path
import importlib

import pytest
import bpy
import gin

from infinigen.core.util import blender as butil

from utils import (
    setup_gin, 
    load_txt_list, 
)

setup_gin()

@pytest.mark.parametrize('factory_name', load_txt_list('test_nature_materials_basic.txt'))
def test_material_runs(factory_name, **kwargs):
    butil.clear_scene()
    with gin.unlock_config():
        mat = importlib.import_module(f'infinigen.assets.materials.{factory_name}')
    bpy.ops.mesh.primitive_ico_sphere_add(radius=.8, subdivisions=5)
    asset = bpy.context.active_object
    mat.apply(asset)