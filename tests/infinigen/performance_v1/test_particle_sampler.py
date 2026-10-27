# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - David Yan, Jia Deng: primary authors

import numpy as np
import pytest

pytest.importorskip(
    "infinigen.infinigen_gpl.perf.particle_sampler._particle_sampler",
    reason="needs INFINIGEN_INSTALL_PERF=True",
)

from infinigen.infinigen_gpl.perf.particle_sampler import _particle_sampler


def test_cube_matches_blender_420_particle_distribution():
    vertices = np.asarray(
        [
            (-1, -1, -1),
            (-1, -1, 1),
            (-1, 1, -1),
            (-1, 1, 1),
            (1, -1, -1),
            (1, -1, 1),
            (1, 1, -1),
            (1, 1, 1),
        ],
        dtype=np.float32,
    )
    normals = vertices / np.float32(np.sqrt(3.0))
    # The two quads ending in vertex zero are rotated as Blender's legacy
    # MFace conversion rotates its zero-sentinel corner.
    faces = np.asarray(
        [
            (0, 1, 3, 2),
            (2, 3, 7, 6),
            (6, 7, 5, 4),
            (1, 0, 4, 5),
            (4, 0, 2, 6),
            (7, 3, 1, 5),
        ],
        dtype=np.int32,
    )
    smooth = np.zeros(6, dtype=np.uint8)
    matrix = np.asarray(
        [
            (1.25, 0, 0, 3),
            (0, 0.75, 0, -2),
            (0, 0, 2, 5),
            (0, 0, 0, 1),
        ],
        dtype=np.float32,
    )
    expected = np.asarray(
        [
            (3.282311201095581, -2.151622772216797, 3.543461799621582),
            (2.709165096282959, -1.459726095199585, 6.546154022216797),
            (3.7608187198638916, -1.4701957702636719, 3.733895778656006),
            (2.2609543800354004, -1.868190050125122, 4.364869594573975),
            (2.0725455284118652, -1.3108887672424316, 6.3324480056762695),
            (3.112466335296631, -1.7056143283843994, 6.365372180938721),
            (2.9158120155334473, -2.264854907989502, 3.030518054962158),
            (2.9534873962402344, -2.49284029006958, 4.103352069854736),
        ],
        dtype=np.float32,
    )

    actual = _particle_sampler.sample_mesh(
        vertices, normals, faces, smooth, matrix, 8, 17, volume=True
    )

    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


def test_sampling_is_repeatable():
    vertices = np.asarray(
        [(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)],
        dtype=np.float32,
    )
    normals = np.tile(np.asarray((0, 0, 1), dtype=np.float32), (4, 1))
    faces = np.asarray([(0, 1, 2, 3)], dtype=np.int32)
    smooth = np.zeros(1, dtype=np.uint8)
    matrix = np.eye(4, dtype=np.float32)

    first = _particle_sampler.sample_mesh(
        vertices, normals, faces, smooth, matrix, 128, 91, volume=False
    )
    second = _particle_sampler.sample_mesh(
        vertices, normals, faces, smooth, matrix, 128, 91, volume=False
    )

    np.testing.assert_array_equal(first.view(np.uint32), second.view(np.uint32))
