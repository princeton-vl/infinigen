# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - David Yan, Jia Deng: primary authors

import unittest

import numpy as np
import pytest

from infinigen.perf.assets.utils.heightfield import mesh_route

pytest.importorskip(
    "infinigen.perf.assets.utils.heightfield._adaptive_triangulation",
    reason="needs INFINIGEN_INSTALL_PERF=True",
)


class FakeObject(dict):
    def as_pointer(self):
        return id(self)


class HeightfieldMeshTest(unittest.TestCase):
    def test_face_size_budget_uses_world_surface_area(self):
        y, x = np.mgrid[0:1:11j, 0:2:21j]
        positions = np.stack((x, y, np.zeros_like(x)), axis=-1).astype(np.float32)
        complete = np.ones((10, 20), dtype=bool)
        with mesh_route.leaf_collection_decimation(0.999, face_size=0.1):
            with mesh_route.leaf_asset_scale(0.5):
                budget = mesh_route._face_size_budget(positions, complete, 400)
        # World area is 2 * .5**2 = .5, covered by two right triangles
        # per face_size-square cell: 2 * .5 / .1**2 = 100.
        self.assertEqual(budget, 100)
        self.assertIsNone(mesh_route.requested_leaf_face_size())

    def test_face_size_budget_clamps_and_validates(self):
        positions = np.zeros((2, 2, 3), dtype=np.float32)
        positions[0, 1, 0] = positions[1, 1, 0] = 1
        positions[1, 0, 1] = positions[1, 1, 1] = 1
        complete = np.ones((1, 1), dtype=bool)
        with mesh_route.leaf_collection_decimation(0.0, face_size=1e-6):
            self.assertEqual(mesh_route._face_size_budget(positions, complete, 2), 2)
        with self.assertRaisesRegex(ValueError, "face_size"):
            with mesh_route.leaf_collection_decimation(0.97, face_size=0):
                pass
        with self.assertRaisesRegex(ValueError, "asset scale"):
            with mesh_route.leaf_asset_scale(float("nan")):
                pass

    def test_passive_residual_correction_recovers_affine_fields(self):
        from infinigen.perf.assets.utils.heightfield.reconstruction import (
            correct_field_residuals,
        )

        y, x = np.mgrid[0:1:17j, 0:1:25j]
        raster = np.stack((x, y, np.full_like(x, 0.25)), axis=-1).astype(np.float32)
        chart = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        base = np.tile([0.5, 0.5, 0.25], (4, 1)).astype(np.float32)
        original = base.copy()
        corrected, strength = correct_field_residuals(
            raster, np.ones(x.shape, bool), chart, faces, base
        )
        np.testing.assert_allclose(corrected[:, :2], chart, atol=1e-6)
        np.testing.assert_array_equal(corrected[:, 2], original[:, 2])
        np.testing.assert_array_equal(base, original)
        self.assertTrue(np.all((strength >= 0) & (strength <= 1)))

    def test_sample_masked_raster_preserves_values_and_ignores_padding(self):
        from infinigen.perf.assets.utils.heightfield.mesh import sample_masked_raster

        raster = np.array([[0, 0, 0], [0, 2, 4], [0, 6, 8]], dtype=np.float32)
        mask = np.array([[0, 0, 0], [0, 1, 1], [0, 1, 1]], dtype=bool)
        samples = sample_masked_raster(
            raster,
            np.array([[0.75, 0.75], [0.50, 0.50]], dtype=np.float32),
            mask=mask,
        )
        np.testing.assert_allclose(samples, [5.0, 2.0])

    def test_geometry_simplifier_uses_height_without_asset_attributes(self):
        try:
            from infinigen.perf.assets.utils.heightfield import mesh as adaptive_mesh
        except ImportError as error:
            self.skipTest(f"native heightfield extension is not built: {error}")

        size = 65
        y, x = np.mgrid[-1 : 1 : complex(size), -1 : 1 : complex(size)]
        # A narrow ridge is the only detail signal; there is no vein or
        # material attribute.
        z = 0.08 * np.exp(-(((x - 0.2 * y) / 0.025) ** 2))
        vertices = np.column_stack((x.ravel(), y.ravel(), z.ravel())).astype(np.float32)
        ids = np.arange(size * size, dtype=np.uint32).reshape(size, size)
        quads = np.column_stack(
            (
                ids[:-1, :-1].ravel(),
                ids[:-1, 1:].ravel(),
                ids[1:, 1:].ravel(),
                ids[1:, :-1].ravel(),
            )
        )
        faces = np.empty((2 * len(quads), 3), dtype=np.uint32)
        faces[0::2] = quads[:, (0, 1, 2)]
        faces[1::2] = quads[:, (0, 2, 3)]
        (
            compact_vertices,
            compact_faces,
            source_indices,
            error,
        ) = adaptive_mesh.simplify_geometry(vertices, faces, 900)
        self.assertLessEqual(len(compact_faces), 910)
        self.assertGreater(len(compact_faces), 700)
        self.assertEqual(len(compact_vertices), len(source_indices))
        self.assertTrue(np.all(np.isfinite(compact_vertices)))
        self.assertTrue(np.isfinite(error))
        # Curved ridge samples must survive a geometry-only collapse.
        self.assertGreater(np.count_nonzero(compact_vertices[:, 2] > 0.04), 10)

        original_vertices = vertices.copy()
        (
            moved_vertices,
            moved_faces,
            moved_source_indices,
            _,
        ) = adaptive_mesh.simplify_geometry(
            vertices,
            faces,
            900,
            update_positions=True,
            target_error=1.0,
            regularize="none",
        )
        self.assertEqual(moved_faces.shape[1], 3)
        self.assertTrue(np.array_equal(vertices, original_vertices))
        self.assertGreater(
            np.max(np.abs(moved_vertices - vertices[moved_source_indices])), 0.0
        )

        passive = np.column_stack(
            (
                x.ravel(),
                y.ravel(),
                np.sin(37.0 * x.ravel()) * np.cos(29.0 * y.ravel()),
            )
        ).astype(np.float32)
        (
            passive_vertices,
            passive_faces,
            passive_source_indices,
            passive_output,
            _,
        ) = adaptive_mesh.simplify_geometry_with_fields(
            vertices, faces, passive, 900, target_error=1.0
        )
        # Passive fields are updated, but do not alter the geometry-only
        # collapse sequence or optimized positions.
        self.assertTrue(np.array_equal(passive_faces, moved_faces))
        self.assertTrue(np.array_equal(passive_source_indices, moved_source_indices))
        self.assertTrue(np.allclose(passive_vertices, moved_vertices))
        self.assertEqual(passive_output.shape, (len(passive_vertices), 3))
        self.assertTrue(np.all(np.isfinite(passive_output)))
        # Linear coordinate fields are interpolated at the point the geometry
        # moves to.
        np.testing.assert_allclose(
            passive_output[:, :2], passive_vertices[:, :2], atol=2e-6, rtol=0
        )
        self.assertGreater(
            np.max(np.abs(passive_output - passive[passive_source_indices])), 0.0
        )

        scaled_vertices, scaled_faces, _, _ = adaptive_mesh.simplify_geometry(
            vertices, faces, 900, metric_scale=(1.0, 1.0, 8.0)
        )
        self.assertEqual(scaled_faces.shape[1], 3)
        self.assertLessEqual(len(scaled_faces), 910)
        self.assertFalse(np.array_equal(scaled_faces, compact_faces))

        semantic = np.exp(-(((x + 0.45) / 0.012) ** 2)).astype(np.float32)
        feature_vertices, feature_faces, _, _ = adaptive_mesh.simplify_geometry(
            vertices,
            faces,
            900,
            features=semantic.ravel(),
            feature_weights=np.array([10.0], dtype=np.float32),
        )
        self.assertEqual(feature_faces.shape[1], 3)
        self.assertFalse(np.array_equal(feature_faces, compact_faces))

        with self.assertRaisesRegex(ValueError, "regularize"):
            adaptive_mesh.simplify_geometry(vertices, faces, 900, regularize="bad")
        with self.assertRaisesRegex(ValueError, "metric_scale"):
            adaptive_mesh.simplify_geometry(
                vertices, faces, 900, metric_scale=(1.0, 0.0, 1.0)
            )
        with self.assertRaisesRegex(ValueError, "provided together"):
            adaptive_mesh.simplify_geometry(vertices, faces, 900, features=semantic)
        with self.assertRaisesRegex(ValueError, "feature_weights"):
            adaptive_mesh.simplify_geometry(
                vertices,
                faces,
                900,
                features=semantic.ravel(),
                feature_weights=np.array([1.0, 2.0], dtype=np.float32),
            )
        with self.assertRaisesRegex(ValueError, "exactly one"):
            adaptive_mesh.simplify_geometry(vertices, faces)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            adaptive_mesh.simplify_geometry(vertices, faces, 900, ratio=0.1)
        ratio_vertices, ratio_faces, _, _ = adaptive_mesh.simplify_geometry(
            vertices, faces, ratio=900 / len(faces)
        )
        self.assertEqual(len(ratio_vertices), len(compact_vertices))
        self.assertTrue(np.array_equal(ratio_faces, compact_faces))

    def test_flat_grid_preconditioner_preserves_boundary_and_nonflat_regions(self):
        try:
            from infinigen.perf.assets.utils.heightfield import mesh as adaptive_mesh
        except ImportError as error:
            self.skipTest(f"native heightfield extension is not built: {error}")

        size = 17
        y, x = np.mgrid[-1 : 1 : complex(size), -1 : 1 : complex(size)]
        z = (np.float32(0.125) * x - np.float32(0.25) * y).astype(np.float32)
        vertices = np.column_stack((x.ravel(), y.ravel(), z.ravel())).astype(np.float32)
        ids = np.arange(size * size, dtype=np.int32).reshape(size, size)
        complete = np.ones((size - 1, size - 1), dtype=bool)
        faces, flat_cells, flat_blocks = adaptive_mesh.precondition_flat_grid_regions(
            vertices, ids, complete, tolerance=1e-7, minimum_block=4
        )
        self.assertEqual(flat_cells, (size - 1) ** 2)
        self.assertEqual(flat_blocks, 1)
        self.assertLess(len(faces), 2 * (size - 1) ** 2)
        edges = np.sort(
            np.concatenate((faces[:, :2], faces[:, 1:], faces[:, ::2]), axis=0), axis=1
        )
        _, counts = np.unique(edges, axis=0, return_counts=True)
        self.assertTrue(np.all((counts == 1) | (counts == 2)))
        self.assertEqual(np.count_nonzero(counts == 1), 4 * (size - 1))

        bumped = vertices.copy()
        bumped[ids[size // 2, size // 2], 2] += np.float32(0.1)
        bumped_faces, bumped_cells, _ = adaptive_mesh.precondition_flat_grid_regions(
            bumped, ids, complete, tolerance=1e-7, minimum_block=4
        )
        self.assertLess(bumped_cells, flat_cells)
        self.assertGreater(len(bumped_faces), len(faces))
        self.assertIn(ids[size // 2, size // 2], bumped_faces)

    def test_final_mesh_is_marked_without_properties(self):
        obj = FakeObject()
        mesh_route.mark_final_mesh(obj)
        self.assertTrue(mesh_route.is_final_mesh(obj))
        self.assertEqual(dict(obj), {})

    def test_collection_decimation_scales_native_face_budget(self):
        budget = mesh_route._scale_default_face_budget
        self.assertEqual(budget(9844), 9844)
        with mesh_route.leaf_collection_decimation(0.97):
            self.assertEqual(budget(9844), 9844)
        with mesh_route.leaf_collection_decimation(0.99):
            self.assertEqual(budget(9844), 3281)
        with mesh_route.leaf_collection_decimation(0.999):
            self.assertEqual(budget(9844), 328)
        self.assertEqual(budget(9844), 9844)

    def test_collection_decimation_rejects_invalid_rate(self):
        with self.assertRaisesRegex(ValueError, "decimate_rate"):
            with mesh_route.leaf_collection_decimation(1.01):
                pass
        with self.assertRaisesRegex(ValueError, "decimate_rate"):
            with mesh_route.leaf_collection_decimation(-0.01):
                pass

    def test_dense_raster_defaults_match_legacy_sampling(self):
        self.assertEqual(mesh_route.raster_shape("broadleaf"), (1025, 1025))
        self.assertEqual(mesh_route.raster_shape("maple"), (2049, 2049))
        self.assertEqual(
            mesh_route.LEGACY_DENSE_TRIANGLE_RATIOS,
            {"broadleaf": 0.03, "v2": 0.03, "maple": 0.0075, "ginko": 0.03},
        )


if __name__ == "__main__":
    unittest.main()
