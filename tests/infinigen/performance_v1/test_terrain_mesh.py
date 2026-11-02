# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - David Yan, Jia Deng: primary authors

import numpy as np

from infinigen.terrain.utils.mesh import Mesh


def _legacy_cat(meshes):
    verts = np.zeros((0, 3))
    faces = np.zeros((0, 3), dtype=int)
    lenv = 0
    vertex_attributes = {}
    for mesh in meshes:
        verts = np.concatenate((verts, mesh.vertices), 0)
        faces = np.concatenate((faces, mesh.faces + lenv), 0)

        for attr in mesh.vertex_attributes:
            if mesh.vertex_attributes[attr].ndim == 1:
                mesh.vertex_attributes[attr] = mesh.vertex_attributes[attr].reshape(
                    (-1, 1)
                )
            mesh_va = mesh.vertex_attributes[attr]
            if attr not in vertex_attributes:
                va = np.zeros(
                    (lenv, mesh.vertex_attributes[attr].shape[1]),
                    dtype=mesh.vertex_attributes[attr].dtype,
                )
            else:
                va = vertex_attributes[attr]
            vertex_attributes[attr] = np.concatenate((va, mesh_va))
        lenv += len(mesh.vertices)

        for attr in vertex_attributes:
            if len(vertex_attributes[attr]) != lenv:
                fillup = np.zeros(
                    (
                        lenv - len(vertex_attributes[attr]),
                        vertex_attributes[attr].shape[1],
                    ),
                    dtype=vertex_attributes[attr].dtype,
                )
                vertex_attributes[attr] = np.concatenate(
                    (vertex_attributes[attr], fillup)
                )
    return Mesh(vertices=verts, faces=faces, vertex_attributes=vertex_attributes)


def _make_meshes():
    specs = [
        (
            np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32),
            np.array([[0, 1, 0]], dtype=np.int32),
            {
                "beta": np.array([2, 3], dtype=np.int16),
                "alpha": np.array([[0.25], [-0.0]], dtype=np.float32),
            },
        ),
        (
            np.array([[0, 1, 0], [1, 1, 0], [2, 1, 0]], dtype=np.float64),
            np.array([[0, 1, 2]], dtype=np.int64),
            {
                "alpha": np.array([[4.0], [5.0], [6.0]], dtype=np.float64),
                "gamma": np.array([7.0, np.nan, -0.0], dtype=np.float32),
            },
        ),
        (
            np.array([[9, 9, 9]], dtype=np.float32),
            np.zeros((0, 3), dtype=np.int16),
            {
                "beta": np.array([11], dtype=np.int32),
                "gamma": np.array([12.0], dtype=np.float64),
            },
        ),
    ]
    return [
        Mesh(vertices=vertices, faces=faces, vertex_attributes=attributes)
        for vertices, faces, attributes in specs
    ]


def _assert_array_identical(actual, expected):
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert actual.tobytes() == expected.tobytes()


def _assert_mesh_identical(actual, expected):
    _assert_array_identical(actual.vertices, expected.vertices)
    _assert_array_identical(actual.faces, expected.faces)
    assert list(actual.vertex_attributes) == list(expected.vertex_attributes)
    for name in expected.vertex_attributes:
        _assert_array_identical(
            actual.vertex_attributes[name], expected.vertex_attributes[name]
        )


def test_mesh_cat_matches_legacy_bytes_and_side_effects():
    legacy_inputs = _make_meshes()
    actual_inputs = _make_meshes()

    expected = _legacy_cat(legacy_inputs)
    actual = Mesh.cat(actual_inputs)

    _assert_mesh_identical(actual, expected)
    assert list(actual.vertex_attributes) == ["beta", "alpha", "gamma"]
    assert actual_inputs[0].vertex_attributes["beta"].shape == (2, 1)
    assert actual_inputs[1].vertex_attributes["gamma"].shape == (3, 1)
    assert actual_inputs[2].vertex_attributes["beta"].shape == (1, 1)


def test_mesh_cat_accepts_generators():
    expected = _legacy_cat(iter(_make_meshes()))
    actual = Mesh.cat(iter(_make_meshes()))
    _assert_mesh_identical(actual, expected)


def test_mesh_cat_empty_matches_legacy():
    expected = _legacy_cat([])
    actual = Mesh.cat([])
    _assert_mesh_identical(actual, expected)
