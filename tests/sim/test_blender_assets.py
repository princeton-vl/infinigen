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

import bmesh
import bpy
import pytest
import trimesh

import infinigen.core.sim.exporters.utils as exputils
import infinigen.core.util.blender as butil
from infinigen.assets.sim_objects.mapping import OBJECT_CLASS_MAP
from infinigen.core.sim.utils import (
    check_if_asset_scaled_after_joint,
    find_joints,
    get_metadata_all_joints_input,
    is_joint,
    verify_joint_parent_child_output_used_correctly,
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
        """Test description: Checks for vertex count for each asset. We set a strict limit of 2500 vertices per asset."""
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
        """Test description: Ensures the asset lies on the XY plane. When spawning the asset, the lowest point of the asset should be at z=0, i.e. the asset should be resting on the ground plane."""
        obj = butil.deep_clone_obj(
            cached_assets[(asset_name, seed)], keep_modifiers=True
        )
        assert obj is not None, "Spawned object is None."

        butil.apply_modifiers(obj)
        obj = exputils.get_geometry_given_attribs(obj, attribs=[("axis_group", 0)])
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

    def test_same_face_normals(self, asset_name, seed, cached_assets):
        """Test description: Ensures that all face normals are pointing in the same direction."""
        obj = butil.deep_clone_obj(
            cached_assets[(asset_name, seed)], keep_modifiers=True
        )
        assert obj is not None, "Spawned object is None."

        butil.apply_modifiers(obj)

        mesh_data = obj.data
        bm = bmesh.new()
        bm.from_mesh(mesh_data)
        bmesh.ops.triangulate(bm, faces=bm.faces[:])

        vertices = [v.co.copy() for v in bm.verts]
        faces = [[v.index for v in f.verts] for f in bm.faces]

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        consistent = mesh.is_winding_consistent
        bm.free()

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
        """Test description: Ensure that all joints are at level 0 (no nested node groups). We are setting the strict requirement that all joints must be at level 0 and are not allowed to be nested inside node groups."""
        obj = butil.deep_clone_obj(
            cached_assets[(asset_name, seed)], keep_modifiers=True
        )
        all_joints = find_joints(obj)
        bad_joints = [
            (ng.name, joint_label, level)
            for _, ng, level, joint_label in all_joints
            if level != 0
        ]

        if bad_joints:
            count = len(bad_joints)
            details = " / ".join(
                [f"{name} ({label}): {level}" for name, label, level in bad_joints]
            )
            msg = (
                f"Asset '{asset_name}' has {count} joint(s) at nonzero depth: {details}"
            )
            assert False, msg

    def test_asset_no_scale(self, asset_name, seed, cached_assets):
        """Test description: Ensures that node graph does not scale asset after joint"""
        # Approach: loop through all nodes and find joints.
        # for each joint go forward until you hit output node and check if any scale nodes are present in between

        # TODO (ajoshi): Scaling should ideally be allowed in the future. We will need to update
        # the coordinate frames attached to each point. Ignore doors for now since they have been
        # manually verified. This is only an issue for some doors.
        if asset_name == "door":
            pytest.skip("Skipping door, manually verified.")

        obj = butil.deep_clone_obj(
            cached_assets[(asset_name, seed)], keep_modifiers=True
        )
        joints_scale_in_between = check_if_asset_scaled_after_joint(obj)

    def test_all_joints_have_labels(self, asset_name, seed, cached_assets):
        """Test description: Ensure that all joints have non-empty labels."""
        obj = butil.deep_clone_obj(
            cached_assets[(asset_name, seed)], keep_modifiers=True
        )
        all_joints = find_joints(obj)
        for _, _, _, joint_label in all_joints:
            assert len(joint_label) > 0, f"{asset_name}: joint with no label."

    def test_joint_labels_are_unique(self, asset_name, seed, cached_assets):
        """Test description: Ensure that all joint labels are unique."""
        obj = butil.deep_clone_obj(
            cached_assets[(asset_name, seed)], keep_modifiers=True
        )
        all_joints = find_joints(obj)

        joints_labels = [
            (ng, joint_label, level)
            for _, ng, level, joint_label in all_joints
            if len(joint_label) > 0
        ]

        # Identify duplicate labels
        labels = [j for _, j, _ in joints_labels]
        duplicates = {label for label in labels if labels.count(label) > 1}

        if duplicates:
            dup_info = [
                f" Label '{label}' found in node groups: "
                f"{', '.join([ng.name for ng, j, _ in joints_labels if j == label])}"
                for label in duplicates
            ]
            message = (
                f"Asset {asset_name}: Some joint labels are not unique:"
                + " ".join(dup_info)
            )
            assert False, message

    def test_metadata_for_each_joint_input(self, asset_name, seed, cached_assets):
        """Test description: Ensure that each joint input has metadata for both parent and child bodies. This requires a metadata node RIGHT BEFORE each joint node in the node tree. EXCEPTIONS: 1) no metadata is required when connecting parent/child output of a joint node to the input of a duplicate node. 2) no metadata is required when connecting parent/child output of a joint node to the parent/child of another joint node to form a multi-jointed body (i.e. sliding + hinge joint -> screw joint)."""
        obj = butil.deep_clone_obj(
            cached_assets[(asset_name, seed)], keep_modifiers=True
        )
        joint_metadata = get_metadata_all_joints_input(obj)
        missing_parent = []
        missing_child = []

        for (
            _,
            ng,
            joint_name,
            parent_has_metadata,
            child_has_metadata,
        ) in joint_metadata:
            if not parent_has_metadata:
                missing_parent.append((ng, joint_name))
            if not child_has_metadata:
                missing_child.append((ng, joint_name))

        errors = []

        if missing_parent:
            errors.append(
                f"{len(missing_parent)} joint(s) with missing parent metadata: {', '.join([f'{ng.name}:{joint_name}' for ng, joint_name in missing_parent])}"
            )

        if missing_child:
            errors.append(
                f"{len(missing_child)} joint(s) with missing child metadata: {', '.join([f'{ng.name}:{joint_name}' for ng, joint_name in missing_child])}"
            )

        if errors:
            error_message = f"Asset {asset_name}: Metadata issues found. " + " / ".join(
                errors
            )
            assert False, error_message

    def test_joint_parent_child_output_used_correctly(
        self, asset_name, seed, cached_assets
    ):
        """Test description: Checks whether the parent and child output of the joint node are used appropriately. Generally there are only TWO cases in which using the parent/child output socket of a joint node is allowed: 1) for duplicate nodes. Example: drawer. 2) multi-jointed bodies. For example, this test should pass whenever a sliding joint and a hinge joint are used one after the other to form a screw joint. In such a case, both the parent and child output of the first joint are used as input to parent/child of the second joint. This is allowed. Every other use of the parent/child output of a joint node is not allowed and should make this test fail."""
        obj = butil.deep_clone_obj(
            cached_assets[(asset_name, seed)], keep_modifiers=True
        )
        joint_duplicate_info = verify_joint_parent_child_output_used_correctly(obj)
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
