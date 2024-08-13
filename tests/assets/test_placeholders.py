# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: David Yan


import numpy as np
import pytest

from infinigen.core import tags as t
from infinigen.core.constraints import usage_lookup
from infinigen.core.util import blender as butil
from infinigen_examples.constraints.semantics import home_asset_usage


def get_real_placeholder_facs():
    used_as = home_asset_usage()
    usage_lookup.initialize_from_dict(used_as)
    pholder_facs = usage_lookup.factories_for_usage({t.Semantics.RealPlaceholder})
    oversize_facs = usage_lookup.factories_for_usage({t.Semantics.OversizePlaceholder})
    pholder_facs.difference_update(oversize_facs)
    return sorted(list(pholder_facs), key=lambda x: x.__name__)


def get_oversize_placeholder_facs():
    used_as = home_asset_usage()
    usage_lookup.initialize_from_dict(used_as)
    pholder_facs = usage_lookup.factories_for_usage({t.Semantics.OversizePlaceholder})
    return sorted(list(pholder_facs), key=lambda _: _.__name__)


def get_asset_facs():
    used_as = home_asset_usage()
    usage_lookup.initialize_from_dict(used_as)
    asset_facs = usage_lookup.factories_for_usage({t.Semantics.PlaceholderBBox})
    return sorted(list(asset_facs), key=lambda x: x.__name__)


@pytest.mark.skip  # TODO re-enable. Too many assets fail this
@pytest.mark.parametrize("fac", get_real_placeholder_facs())
def test_real_placeholders(fac):
    butil.clear_scene()
    placeholder = fac(0).spawn_placeholder(0, loc=(0, 0, 0), rot=(0, 0, 0))
    asset = fac(0).spawn_asset(0)
    assert np.abs(placeholder.dimensions.x - asset.dimensions.x) <= 0.05 * np.abs(
        asset.dimensions.x
    ), "X dimension of placeholder not within 5 percent of mesh"
    assert np.abs(placeholder.dimensions.y - asset.dimensions.y) <= 0.05 * np.abs(
        asset.dimensions.y
    ), "Y dimension of placeholder not within 5 percent of mesh"
    assert np.abs(placeholder.dimensions.z - asset.dimensions.z) <= 0.05 * np.abs(
        asset.dimensions.z
    ), "Z dimension of placeholder not within 5 percent of mesh"
    asset_min_corner = np.array(
        asset.bound_box[0]
    )  # loXloYloZ https://blender.stackexchange.com/questions/32283/what-are-all-values-in-bound-box
    asset_max_corner = np.array(asset.bound_box[6])  # hiXhiYhiZ
    ph_min_corner = np.array(placeholder.bound_box[0])
    ph_max_corner = np.array(placeholder.bound_box[6])
    for i in range(3):
        assert (
            asset_min_corner[i] <= ph_max_corner[i]
            and asset_min_corner[i] >= ph_min_corner[i]
        ), "Asset not completely contained within placeholder"
        assert (
            asset_max_corner[i] <= ph_max_corner[i]
            and asset_max_corner[i] >= ph_min_corner[i]
        ), "Asset not completely contained within placeholder"


@pytest.mark.parametrize("fac", get_asset_facs())
def test_generated_placeholders(fac):
    butil.clear_scene()
    fac(0).spawn_placeholder(0, loc=(0, 0, 0), rot=(0, 0, 0))
    fac(0).spawn_asset(0)
    return
