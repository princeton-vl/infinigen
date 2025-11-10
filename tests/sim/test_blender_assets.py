# Copyright (C) 2025, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors:
# - Max Gonzalez Saez-Diez: primary author
# - Abhishek Joshi: updates
"""
Unit tests to ensure the assets follow guidelines within Blender.
"""

import logging

import bpy
import pytest

import infinigen.core.util.blender as butil
from infinigen.assets.sim_objects.mapping import OBJECT_CLASS_MAP
from infinigen.core.sim.utils import (
    find_joints,
    get_metadata_all_joints_input,
    is_joint,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
                params.append((name, seed))
        metafunc.parametrize("asset_name,seed", params)


class TestBlenderAssets:
    def test_asset_vertex_count(self, asset_name, seed, cached_assets):
        obj = butil.deep_clone_obj(
            cached_assets[(asset_name, seed)], keep_modifiers=True
        )
        assert obj is not None, "Spawned object is None."
        butil.apply_modifiers(obj)
        mesh = obj.to_mesh()

        assert len(mesh.vertices) < 2500, (
            f"Mean Vertex Count: {len(mesh.vertices)} > 2500."
        )

    def test_asset_xyposition(self, asset_name, seed, cached_assets):
        obj = butil.deep_clone_obj(
            cached_assets[(asset_name, seed)], keep_modifiers=True
        )
        assert obj is not None, "Spawned object is None."

        butil.apply_modifiers(obj)
        mesh = obj.to_mesh()

        z_coo = min([v.co.z for v in mesh.vertices])

        assert -1e-5 < z_coo < 1e-5, (
            f"{asset_name} not on xy-plane (1e-5 margin). Seed {seed}"
        )

    def test_single_modifier(self, asset_name, seed, cached_assets):
        obj = butil.deep_clone_obj(
            cached_assets[(asset_name, seed)], keep_modifiers=True
        )
        """Test description: Ensures that asset only has a single modifier."""
        assert len(obj.modifiers) == 1, (
            f"{asset_name} must only use a single modifier (seed {seed})."
        )

        assert consistent, (
            "Mesh is not consistent. This indicates that some face normals are flipped."
        )
    def test_latest_node(self, asset_name, seed, cached_assets):
        """Test description: Ensures that assets are using the latest nodes."""
        obj = butil.deep_clone_obj(
            cached_assets[(asset_name, seed)], keep_modifiers=True
        )
        expected_joint_inputs = [
            "Joint ID (do not set)",
            "Joint Label",
            "Parent",
            "Child",
            "Position",
            "Axis",
            "Value",
            "Min",
            "Max",
            "Show Joint",
        ]
        nodes = obj.modifiers[0].node_group.nodes
        for node in nodes:
            if is_joint(node):
                inputs = node.inputs
                assert len(inputs) == len(expected_joint_inputs), (
                    f"{asset_name} is not using the latest nodes."
                )
                for i in range(len(inputs)):
                    assert inputs[i].name == expected_joint_inputs[i], (
                        f"{asset_name} is not using the latest nodes."
                    )

    def test_no_instances(self, asset_name, seed, cached_assets):
        """Test description: Ensures that the asset does not having any instances."""
        obj = butil.deep_clone_obj(
            cached_assets[(asset_name, seed)], keep_modifiers=True
        )
        dg = bpy.context.evaluated_depsgraph_get()
        instancer = obj.evaluated_get(dg)

        instances = [
            oi
            for oi in dg.object_instances
            if oi.is_instance and oi.parent == instancer
        ]
        assert len(instances) == 0, f"{asset_name} must have no instances (seed {seed})"

    def test_joint_levels_depth(self, asset_name, seed, cached_assets):
        obj = butil.deep_clone_obj(
            cached_assets[(asset_name, seed)], keep_modifiers=True
        )
        all_joints = find_joints(obj)
            )
                f"Asset '{asset_name}' has {count} joint(s) at nonzero depth: {details}"


        # TODO (ajoshi): Scaling should ideally be allowed in the future. We will need to update
        # the coordinate frames attached to each point. Ignore doors for now since they have been
        # manually verified. This is only an issue for some doors.
        if asset_name == "door":
            pytest.skip("Skipping door, manually verified.")

    def test_all_joints_have_labels(self, asset_name, seed, cached_assets):
        obj = butil.deep_clone_obj(
            cached_assets[(asset_name, seed)], keep_modifiers=True
        )
        all_joints = find_joints(obj)
        for _, _, _, joint_label in all_joints:
            assert len(joint_label) > 0, f"{asset_name}: joint with no label."

    def test_joint_labels_are_unique(self, asset_name, seed, cached_assets):
        obj = butil.deep_clone_obj(
            cached_assets[(asset_name, seed)], keep_modifiers=True
        )


    def test_metadata_for_each_joint_input(self, asset_name, seed, cached_assets):
        obj = butil.deep_clone_obj(
            cached_assets[(asset_name, seed)], keep_modifiers=True
        )
        joint_metadata = get_metadata_all_joints_input(obj)
            )
            )

        obj = butil.deep_clone_obj(
            cached_assets[(asset_name, seed)], keep_modifiers=True
        )
        for (
            _,
            _,
            joint_name,
            parent_to_non_duplicate,
            child_to_non_duplicate,
        ) in joint_duplicate_info:
            assert not parent_to_non_duplicate, (
                f"Asset {asset_name}, joint '{joint_name}': Parent input for joint '{joint_name}' is used for node that is not 1) duplicate. 2) another joint."
            )
            assert not child_to_non_duplicate, (
                f"Asset {asset_name}, joint '{joint_name}': Child input for joint '{joint_name}' is used for node that is not 1) duplicate. 2) another joint."
            )
