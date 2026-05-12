# Copyright (C) 2025, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors:
# - Abhishek Joshi: primary author
"""
Tests unit tests to ensure that joints are properly constructed in sim. Ensure no errors are produced
during export and that the assets look visually the same as their Blender counterparts.
"""

from pathlib import Path

import pytest

pytest.importorskip(
    "coacd", reason="coacd is not installed, skipping all tests in this file"
)

from infinigen.core.nodes.node_transpiler.transpiler import clean_and_capitalize
from infinigen.core.sim import kinematic_compiler
from infinigen.core.sim.exporters.factory import sim_exporter_factory
from infinigen.core.sim.exporters.usd_exporter import UnsupportedAxisError
from infinigen.core.sim.utils import load_class_from_path
from infinigen.core.util import blender as butil

NUM_TEST_ASSETS = 12
FILE_FORMATS = ["mjcf", "urdf", "usd"]


@pytest.mark.parametrize("test_asset_id", range(NUM_TEST_ASSETS))
@pytest.mark.parametrize("format", FILE_FORMATS)
def test_sim_export(test_asset_id, format):
    # temporary path
    joint_test_path = Path("tests/sim/joint_tests")

    asset_name = f"test_{test_asset_id}"
    seed = 42

    path = joint_test_path / f"{asset_name}.py"
    cls_name = clean_and_capitalize(asset_name) + "Factory"
    factory = load_class_from_path(path, cls_name)

    # generate an instance for each file format
    asset = factory(seed)
    obj = asset.spawn_asset(i=0)
    sim_blueprint = kinematic_compiler.compile(obj)
    butil.apply_modifiers(obj)

    sim_blueprint["name"] = asset_name

    export_dir = joint_test_path / format
    export_func = sim_exporter_factory(exporter=format, legacy=False)

    try:
        export_path, semantic_mapping = export_func(
            blend_obj=obj,
            sim_blueprint=sim_blueprint,
            seed=asset.factory_seed,
            sample_joint_params_fn=factory.sample_joint_parameters,
            export_dir=export_dir,
            image_res=4,
            visual_only=True,
        )
    except UnsupportedAxisError as e:
        pytest.skip(str(e))


if __name__ == "__main__":
    for i in range(NUM_TEST_ASSETS):
        for format in FILE_FORMATS:
            test_sim_export(i, format)
