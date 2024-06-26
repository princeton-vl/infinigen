# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors: Alexander Raistrick

from pathlib import Path
import importlib

import pytest
import bpy
import gin

from infinigen.core.util import blender as butil

from infinigen_examples.util.test_utils import (setup_gin, load_txt_list, import_item)


def check_material_runs(pathspec):
    butil.clear_scene()
    bpy.ops.mesh.primitive_ico_sphere_add(radius=.8, subdivisions=5)
    asset = bpy.context.active_object

    mat = import_item(pathspec)
    if type(mat) is type:
        mat = mat(0)
    mat.apply(asset)

    # should not crash for input LIST of objects
    bpy.ops.mesh.primitive_ico_sphere_add(radius=.8, subdivisions=5)
    asset2 = bpy.context.active_object
    mat.apply([asset, asset2]) 




@pytest.mark.nature
@pytest.mark.parametrize('pathspec', load_txt_list('tests/assets/list_nature_materials.txt'))
def test_nature_material_runs(pathspec, **kwargs):
    setup_gin('infinigen_examples/configs_nature')
    check_material_runs(pathspec)


@pytest.mark.parametrize('pathspec', load_txt_list('tests/assets/list_indoor_materials.txt'))
def test_indoor_material_runs(pathspec, **kwargs):
    setup_gin('infinigen_examples/configs_indoor')
    check_material_runs(pathspec)
