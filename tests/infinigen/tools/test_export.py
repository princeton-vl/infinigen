# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors: Alexander Raistrick


import bpy
import pytest

from infinigen.assets.objects.mollusk import MolluskFactory
from infinigen.core.util import blender as butil
from infinigen.tools import export

TEST_FORMATS = ["obj", "usdc", "fbx", "ply", "usdc"]
TEST_IMAGE_RES = 32


@pytest.mark.parametrize("format", TEST_FORMATS)
def test_export_one_obj(format, tmp_path):
    butil.clear_scene()
    asset = MolluskFactory(0).spawn_asset(0)
    file = export.export_single_obj(asset, tmp_path, format, image_res=TEST_IMAGE_RES)

    assert file.suffix == f".{format}"

    asset_polys = len(asset.data.polygons)
    num_objs = len(bpy.data.objects)
    butil.clear_scene()
    new_obj = butil.import_mesh(file)

    if format == "usdc":
        assert num_objs + 1 == len(
            bpy.data.objects
        )  # usdc import generates extra "world" prim
    else:
        assert num_objs == len(bpy.data.objects)

    assert len(new_obj.data.polygons) == asset_polys

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

    num_objs = len(bpy.data.objects)
    poly_count1 = len(asset1.data.polygons)
    poly_count2 = len(asset2.data.polygons)

    butil.clear_scene()
    butil.import_mesh(file)
    total_polys = 0
    for obj in bpy.data.objects:
        if obj.name == "World":
            continue
        if obj.type == "EMPTY":
            continue
        total_polys += len(obj.data.polygons)

    assert total_polys == poly_count1 + poly_count2

    if format == "usdc":
        assert num_objs + 1 == len(
            bpy.data.objects
        )  # usdc import generates extra "world" prim
    elif format == "ply":
        assert len(bpy.data.objects) == 1
    else:
        assert num_objs == len(bpy.data.objects)

    # TODO David Yan add other guarantees (count objects, count/names of materials, any others)


# TODO test all export.py features, including individual export, transparent mats, instances
