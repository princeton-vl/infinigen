# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - David Yan, Jia Deng: primary authors

"""Input validation and determinism of the built `_native_fields` extension."""

import importlib.util
import unittest
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip(
    "infinigen.perf.assets.utils._native_fields",
    reason="needs INFINIGEN_INSTALL_PERF=True",
)


directory = Path(__file__).resolve().parents[3] / "src/infinigen/perf/assets/utils"
binary = next(directory.glob("_native_fields.*.so"))
spec = importlib.util.spec_from_file_location("_native_fields", binary)
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)


class NativeFieldAPI(unittest.TestCase):
    def setUp(self):
        self.params = np.array(
            [0.4, 0.8, 0.3, 0.7, 5, 12, 40, 0.1, 60, 162, 0.005], np.float32
        )
        self.tables = np.tile(np.linspace(0, 1, 257, dtype=np.float32), (8, 1))
        self.ranges = np.tile(np.array([0, 1], np.float32), (8, 1))
        self.xy = np.array([[0, 0], [0.1, 0.2], [-0.1, -0.2]], np.float32)

    def call(self, xy=None, **kwargs):
        return native.evaluate_broadleaf(
            self.xy if xy is None else xy,
            self.params,
            self.tables,
            self.ranges,
            **kwargs,
        )

    def test_repeated_calls_and_input_immutability(self):
        before = self.xy.copy()
        a = self.call()
        b = self.call()
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(self.xy, before)
        self.assertTrue(np.isfinite(a).all())
        self.assertEqual(a.shape, (3, 8))

    def test_mask(self):
        full = self.call()
        masked = self.call(mask=[1, 0, 1])
        np.testing.assert_array_equal(masked[[0, 2]], full[[0, 2]])
        np.testing.assert_array_equal(masked[1], 0)

    def test_empty(self):
        self.assertEqual(self.call(np.empty((0, 2), np.float32)).shape, (0, 8))

    def test_bad_mask(self):
        with self.assertRaises(ValueError):
            self.call(mask=[1])

    def test_nonfinite(self):
        with self.assertRaises(ValueError):
            self.call([[float("nan"), 0]])

    def test_bad_coordinates(self):
        with self.assertRaises(ValueError):
            self.call([[0, 0, 0]])

    def test_bad_curve_domain(self):
        self.ranges[0] = [1, 1]
        with self.assertRaises(ValueError):
            self.call()

    def test_generic_operations(self):
        for op in range(4):
            a = native.evaluate_operation([[0, 0, 0], [0.1, 0.2, 0.3]], op)
            self.assertTrue(np.isfinite(a).all())
            np.testing.assert_array_equal(
                a, native.evaluate_operation([[0, 0, 0], [0.1, 0.2, 0.3]], op)
            )
        self.assertEqual(native.evaluate_operation([[0, 0, 0]], 0)[0], 0.5)
        with self.assertRaises(ValueError):
            native.evaluate_operation([[0, 0, 0]], 4)

    def test_planar_shapes_and_determinism(self):
        for kind, nc, np_ in ((0, 5, 8), (1, 4, 6), (2, 7, 8)):
            params = np.ones(np_, np.float32)
            a = native.evaluate_planar_leaf(
                self.xy, kind, params, self.tables[:nc], self.ranges[:nc]
            )
            b = native.evaluate_planar_leaf(
                self.xy, kind, params, self.tables[:nc], self.ranges[:nc]
            )
            np.testing.assert_array_equal(a, b)
            self.assertEqual(a.shape, (3, 6))
            self.assertTrue(np.isfinite(a).all())
            empty = native.evaluate_planar_leaf(
                np.empty((0, 2)), kind, params, self.tables[:nc], self.ranges[:nc]
            )
            self.assertEqual(empty.shape, (0, 6))

    def test_planar_invalid_inputs(self):
        for kind in (-1, 3):
            with self.assertRaises(ValueError):
                native.evaluate_planar_leaf(
                    self.xy, kind, self.params, self.tables, self.ranges
                )
        with self.assertRaises(ValueError):
            native.evaluate_planar_leaf(
                self.xy, 0, [1] * 7, self.tables[:5], self.ranges[:5]
            )
        with self.assertRaises(ValueError):
            native.evaluate_planar_leaf(
                [[float("inf"), 0]], 0, [1] * 8, self.tables[:5], self.ranges[:5]
            )

    def test_planar_coverage_contract(self):
        x = np.linspace(-1, 1, 65, dtype=np.float32)
        y = np.linspace(-1, 1, 49, dtype=np.float32)
        gx, gy = np.meshgrid(x, y)
        xy = np.column_stack((gx.ravel(), gy.ravel()))
        for kind, nc, np_ in ((0, 5, 8), (1, 4, 6), (2, 7, 8)):
            params = np.ones(np_, np.float32)
            mask = native.planar_coverage(
                x, y, kind, params, self.tables[:nc], self.ranges[:nc]
            )
            full = native.evaluate_planar_leaf(
                xy, kind, params, self.tables[:nc], self.ranges[:nc]
            )
            masked = native.evaluate_planar_leaf(
                xy, kind, params, self.tables[:nc], self.ranges[:nc], coverage_mask=mask
            )
            self.assertFalse(np.any((full[:, 0] != 0) & (mask.ravel() == 0)))
            np.testing.assert_array_equal(
                full[full[:, 0] != 0], masked[full[:, 0] != 0]
            )
            with self.assertRaises(ValueError):
                native.evaluate_planar_leaf(
                    xy,
                    kind,
                    params,
                    self.tables[:nc],
                    self.ranges[:nc],
                    coverage_mask=[1],
                )
        with self.assertRaises(ValueError):
            native.planar_coverage(
                [np.nan], [0], 0, [1] * 8, self.tables[:5], self.ranges[:5]
            )
        with self.assertRaises(ValueError):
            native.planar_coverage(
                [0], [0], 4, [1] * 8, self.tables[:5], self.ranges[:5]
            )
        with self.assertRaises(ValueError):
            native.planar_coverage(
                [0], [0], 0, [1] * 8, self.tables[:5], self.ranges[:5], reach=-1
            )

    def test_coverage_window_is_asset_agnostic(self):
        spec = importlib.util.spec_from_file_location(
            "coverage_utility", directory / "heightfield" / "coverage.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        mask = np.zeros((17, 31), bool)
        mask[3:7, 8:19] = True
        self.assertEqual(module.coverage_slices(mask), (slice(2, 8), slice(7, 20)))
        with self.assertRaises(ValueError):
            module.coverage_slices(np.zeros((4, 5), bool))
        with self.assertRaises(ValueError):
            module.coverage_slices(mask, padding=-1)
        coarse = np.zeros((2, 3), bool)
        coarse[0, 1] = True
        full = module.upsample_mask(coarse, (5, 7))
        expected = np.zeros((5, 7), bool)
        expected[:3, 3:5] = True
        np.testing.assert_array_equal(full, expected)
        with self.assertRaises(ValueError):
            module.upsample_mask(np.zeros((1, 3), bool), (5, 7))

    def test_deferred_bend_matches_dense_formula(self):
        gx, gy = np.meshgrid(
            np.linspace(-1, 1, 65, dtype=np.float32),
            np.linspace(-1, 1, 65, dtype=np.float32),
        )
        xy = np.column_stack((gx.ravel(), gy.ravel()))
        for kind, nc, np_ in ((0, 5, 8), (1, 4, 6), (2, 7, 8)):
            params = np.ones(np_, np.float32)
            full = native.evaluate_planar_leaf(
                xy, kind, params, self.tables[:nc], self.ranges[:nc]
            )
            raw, bounds = native.evaluate_planar_leaf(
                xy, kind, params, self.tables[:nc], self.ranges[:nc], defer_bend=True
            )
            selected = full[:, 0] != 0
            np.testing.assert_array_equal(raw[:, 0], full[:, 0])
            np.testing.assert_array_equal(raw[:, 2:5], full[:, 2:5])
            xyz = np.column_stack((xy[selected], raw[selected, 1]))
            actual = native.bend_planar_leaf(
                xyz, kind, params, self.tables[:nc], self.ranges[:nc], bounds
            )
            expected = np.column_stack(
                (xy[selected, 0], full[selected, 5], full[selected, 1])
            )
            np.testing.assert_array_equal(actual, expected)
            with self.assertRaises(ValueError):
                native.bend_planar_leaf(
                    xyz, kind, params, self.tables[:nc], self.ranges[:nc], [0] * 4
                )


if __name__ == "__main__":
    unittest.main()
