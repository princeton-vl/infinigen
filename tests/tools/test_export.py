# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors: Alexander Raistrick

import copy

import pytest

import bpy

from infinigen.tools import export

from infinigen.assets.mollusk import MolluskFactory
from infinigen.core.util import blender as butil

TEST_IMAGE_RES = 32

@pytest.mark.parametrize("format", TEST_FORMATS)
def test_export_one_obj(format, tmp_path):

    butil.clear_scene()
    asset = MolluskFactory(0).spawn_asset(0)
    assert file.suffix == f".{format}"

    asset_polys = len(asset.data.polygons)
    butil.clear_scene()
    new_obj = butil.import_mesh(file)


    
    # TODO David Yan add other guarantees (count objects, count/names of materials, any others)

@pytest.mark.parametrize("format", TEST_FORMATS)
def test_export_curr_scene(format, tmp_path):

    butil.clear_scene()
    asset1 = MolluskFactory(0).spawn_asset(0)
    asset2 = MolluskFactory(0).spawn_asset(1)
    asset2.parent = asset1
    asset2.location.x += 10

    file = export.export_curr_scene(tmp_path, format, image_res=TEST_IMAGE_RES)
    assert file.suffix == f".{format}"

    butil.clear_scene()
    butil.import_mesh(file)
        
    # TODO David Yan add other guarantees (count objects, count/names of materials, any others)

# TODO test all export.py features, including individual export, transparent mats, instances