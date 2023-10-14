from pathlib import Path
import importlib

import pytest
import bpy
import gin

from infinigen.core.util import blender as butil

from utils import (
    setup_gin, 
    load_txt_list, 
    import_item
)

setup_gin()

@pytest.mark.ci
@pytest.mark.parametrize('pathspec', load_txt_list('test_materials_basic.txt'))
def test_material_runs(pathspec, **kwargs):
    
    butil.clear_scene()
    bpy.ops.mesh.primitive_ico_sphere_add(radius=.8, subdivisions=5)
    asset = bpy.context.active_object

    mat = import_item(pathspec)
    mat.apply(asset)