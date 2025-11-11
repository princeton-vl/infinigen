# Copyright (C) 2025, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors:
# - Abhishek Joshi: primary author

import json
import logging
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import pytest

pytest.importorskip(
    "coacd", reason="coacd is not installed, skipping all tests in this file"
)

import infinigen.core.util.blender as butil
from infinigen.assets.sim_objects.mapping import OBJECT_CLASS_MAP
from infinigen.core.sim import kinematic_compiler
from infinigen.core.sim.exporters.factory import sim_exporter_factory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EXPORTERS = ["mjcf", "urdf", "usd"]


def verify_mjcf_output(mjcf):
    worldbody = mjcf.find("worldbody")
    geoms = worldbody.findall(".//geom")
    joints = worldbody.findall(".//joint")

    geom_missing, joints_missing = [], []

    for geom in geoms:
        if geom.get("name").startswith("geom"):
            geom_missing.append(f"{ET.tostring(geom, encoding='unicode')}")
    for joint in joints:
        if joint.get("name").startswith("joint"):
            joints_missing.append(f"{ET.tostring(joint, encoding='unicode')}")

    assert len(geom_missing) == 0, (
        f"{len(geom_missing)} geoms are missing names: {geom_missing}"
    )
    assert len(joints_missing) == 0, (
        f"{len(joints_missing)} joints are missing names: {joints_missing}"
    )


def verify_urdf_output(urdf):
    pytest.skip("URDF metadata verification coming soon.")


def verify_usd_output(usd):
    pytest.skip("USD metadata verification coming soon.")


def pytest_generate_tests(metafunc):
    if "asset_name" in metafunc.fixturenames:
        assets = metafunc.config.getoption("--asset")
        if "all" in assets:
            assets = list(OBJECT_CLASS_MAP.keys())
        params = []
        for name in assets:
            if name not in OBJECT_CLASS_MAP:
                pytest.fail(f"Asset '{name}' not found in OBJECT_CLASS_MAP.")
            for seed in range(int(metafunc.config.getoption("--nr"))):
                for exporter in EXPORTERS:
                    params.append((name, seed, exporter))
        metafunc.parametrize("asset_name,seed,exporter", params)


class TestSimExports:
    def test_sim_export(self, tmp_path, asset_name, seed, exporter, cached_assets):
        filepath = Path(__file__).parent.resolve() / "skipped_tests.json"
        with open(filepath, "r") as f:
            skipped_tests = json.load(f)

        obj = butil.deep_clone_obj(
            cached_assets[(asset_name, seed)], keep_modifiers=True, keep_materials=True
        )
        obj.hide_render = False
        obj.hide_viewport = False
        sim_blueprint = kinematic_compiler.compile(obj)
        butil.apply_modifiers(obj)

        asset_class = OBJECT_CLASS_MAP[asset_name]
        sim_blueprint["name"] = f"{asset_name}_{seed}"

        export_dir = tmp_path / exporter

        # TODO (ajoshi): fix urdf and usd exporters
        if exporter == "urdf" or exporter == "usd":
            pytest.skip(f"{exporter} metadata verification coming soon.")

        export_func = sim_exporter_factory(exporter=exporter, legacy=False)
        asset_str, metadata = export_func(
            blend_obj=obj,
            sim_blueprint=sim_blueprint,
            seed=seed,
            sample_joint_params_fn=asset_class.sample_joint_parameters,
            export_dir=export_dir,
            image_res=4,
            get_raw_output=True,
            visual_only=False,
        )

        if exporter == "mjcf":
            # Step 1: Verify names
            if asset_name == "stovetop":
                pytest.skip(
                    "TODO: stove names in some instances need to be fixed. Skipping for now as it doesn't break the asset."
                )
            verify_mjcf_output(asset_str)

            # Step 2: Verify compilation (this should fail for cases like Mj.minValue)
            os.chdir(export_dir / (asset_name + "_" + str(seed)) / str(seed))

            raw_xml = ET.tostring(asset_str, encoding="unicode")
            spec = mujoco.MjSpec.from_string(raw_xml)
            try:
                spec.compile()
            except Exception as e:
                pytest.fail(
                    f"MJCF compilation failed for {asset_name} with seed {seed}."
                )

        elif exporter == "urdf":
            verify_urdf_output(None)
        else:
            verify_usd_output(None)
