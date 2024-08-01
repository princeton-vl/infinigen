# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors: Alexander Raistrick


import bpy
import pytest

from infinigen.core.util import blender as butil
from infinigen.core.util.test_utils import import_item, load_txt_list, setup_gin


def check_material_runs(pathspec):
    butil.clear_scene()
    bpy.ops.mesh.primitive_ico_sphere_add(radius=0.8, subdivisions=5)
    asset = bpy.context.active_object

    mat = import_item(pathspec)
    if type(mat) is type:
        mat = mat(0)
    mat.apply(asset)

    # should not crash for input LIST of objects
    bpy.ops.mesh.primitive_ico_sphere_add(radius=0.8, subdivisions=5)
    asset2 = bpy.context.active_object
    mat.apply([asset, asset2])


@pytest.mark.nature
@pytest.mark.parametrize(
    "pathspec", load_txt_list("tests/assets/list_nature_materials.txt")
)
def test_nature_material_runs(pathspec, **kwargs):
    setup_gin("infinigen_examples/configs_nature", ["base_nature.gin"])
    check_material_runs(pathspec)


@pytest.mark.parametrize(
    "pathspec", load_txt_list("tests/assets/list_indoor_materials.txt")
)
def test_indoor_material_runs(pathspec, **kwargs):
    setup_gin(
        ["infinigen_examples/configs_indoor", "infinigen_examples/configs_nature"],
        ["base_indoors.gin"],
    )
    check_material_runs(pathspec)
