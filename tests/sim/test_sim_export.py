# Copyright (C) 2025, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors:
# - Abhishek Joshi: primary author

import pytest

from infinigen.core.sim import sim_factory as sf

ASSETS = ["door", "lamp", "dishwasher", "multifridge", "multidoublefridge", "toaster"]
FILE_FORMATS = ["mjcf", "urdf", "usd"]


@pytest.mark.skip(reason="This test is temporarily disabled.")
@pytest.mark.parametrize("asset", ASSETS)
@pytest.mark.parametrize("format", FILE_FORMATS)
def test_sim_export(tmp_path, asset, format):
    sf.spawn_simready(
        name=asset,
        seed=1001,
        export_dir=tmp_path,
        exporter=format,
        visual_only=True,
        image_res=16,
    )
