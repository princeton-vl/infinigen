# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Karhan Kayan

import numpy as np
import procfunc as pf
import pytest
import trimesh

from .test_exporters import configure_cube_scene


@pytest.mark.slow
def test_export_blender_file_and_mesh(
    tmp_path,
):
    objects, camera = configure_cube_scene()

    blend_path = tmp_path / "scene.blend"
    pf.ops.file.save_blend(output_path=blend_path)
    assert blend_path.exists()

    mesh_path = tmp_path / "mesh.ply"
    pf.ops.file.save_mesh(output_path=mesh_path, objects=list(objects))
    assert mesh_path.exists()

    mesh = trimesh.load_mesh(mesh_path, process=False)

    # Basic sanity checks on mesh integrity
    assert mesh.vertices.shape[0] > 0
    assert mesh.faces.shape[0] > 0

    extents = mesh.bounds[1] - mesh.bounds[0]
    # Cube is roughly 4 units in each dimension
    assert np.all(extents > 3.9), f"Mesh extents too small: {extents}"
    assert np.all(extents < 4.1), f"Mesh extents too large: {extents}"


@pytest.mark.slow
@pytest.mark.xfail(reason="OBJECTS_FILE not yet implemented")
def test_export_objects_file(tmp_path):
    pytest.skip("OBJECTS_FILE not yet implemented")
