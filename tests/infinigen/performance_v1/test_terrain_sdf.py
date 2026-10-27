# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - David Yan, Jia Deng: primary authors

import pytest

from infinigen._perf_selection import selected  # noqa: E402

if not selected("TERRAIN"):
    pytest.skip(
        "exercises the optimized terrain implementation; "
        "pass --optimized=terrain (or --optimized) to run",
        allow_module_level=True,
    )


import numpy as np
import pytest

from infinigen.perf.terrain.mesher.spherical_mesher import kernel_min_caller
from infinigen.terrain.mesher.spherical_mesher import kernel_caller
from infinigen.terrain.utils import Vars


def _kernel(values):
    def call(XYZ, sdf_only=1):
        assert sdf_only == 1
        assert len(XYZ) == len(values)
        return {Vars.SDF: values.copy()}

    return call


def _assert_bitwise_equal(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    unsigned_dtype = np.dtype(f"u{actual.dtype.itemsize}")
    assert np.array_equal(actual.view(unsigned_dtype), expected.view(unsigned_dtype))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_kernel_min_caller_matches_stacked_reduction_bitwise(dtype):
    XYZ = np.array(
        [
            [0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, -2.0, 0.0],
            [0.25, 0.25, 0.25],
            [-0.25, -0.25, -0.25],
        ],
        dtype=np.float64,
    )
    bounds = (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)
    values = [
        np.array([3.0, np.nan, -4.0, 5.0, -0.0, 7.0], dtype=dtype),
        np.array([2.0, 8.0, np.nan, -5.0, 0.0, 7.0], dtype=dtype),
        np.array([2.0, -8.0, 4.0, np.nan, -0.0, 9.0], dtype=dtype),
    ]
    kernels = [_kernel(value) for value in values]

    expected = kernel_caller(kernels, XYZ, bounds).min(axis=-1)
    actual = kernel_min_caller(kernels, XYZ, bounds)

    _assert_bitwise_equal(actual, expected)


def test_kernel_min_caller_preserves_stack_dtype_promotion():
    XYZ = np.zeros((4, 3), dtype=np.float64)
    kernels = [
        _kernel(np.array([3.0, -0.0, 5.0, np.nan], dtype=np.float32)),
        _kernel(np.array([2.0, 0.0, 6.0, 1.0], dtype=np.float64)),
    ]

    expected = kernel_caller(kernels, XYZ).min(axis=-1)
    actual = kernel_min_caller(kernels, XYZ)

    _assert_bitwise_equal(actual, expected)


def test_kernel_min_caller_handles_empty_positions_and_one_kernel():
    XYZ = np.empty((0, 3), dtype=np.float64)
    kernels = [_kernel(np.empty(0, dtype=np.float64))]

    expected = kernel_caller(kernels, XYZ).min(axis=-1)
    actual = kernel_min_caller(kernels, XYZ)

    _assert_bitwise_equal(actual, expected)


def test_kernel_min_caller_matches_empty_kernel_error():
    with pytest.raises(ValueError, match="need at least one array to stack"):
        kernel_min_caller([], np.zeros((2, 3), dtype=np.float64))


def _ocmesher():
    """`OcMesher` from the optional submodule, or None when it is not checked out."""
    try:
        from infinigen.OcMesher.ocmesher.core import OcMesher
    except ImportError:
        return None
    return OcMesher


_OCMESHER = _ocmesher()

needs_ocmesher = pytest.mark.skipif(
    _OCMESHER is None,
    reason="needs the optional OcMesher submodule",
)


@pytest.mark.skipif(
    not hasattr(_OCMESHER, "kernel_min_caller"),
    reason=(
        "kernel_min_caller exists only in the OcMesher commit that was never "
        "pushed upstream; the release pins the public one"
    ),
)
def test_ocmesher_kernel_min_caller_matches_stacked_reduction_bitwise():
    from infinigen.OcMesher.ocmesher.core import OcMesher

    mesher = object.__new__(OcMesher)
    mesher.sdf_np_float_type = np.float32
    mesher.enclosed = True
    mesher.bounds = (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)
    XYZ = np.array(
        [[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        dtype=np.float64,
    )
    values = [
        np.array([2.0, np.nan, -0.0], dtype=np.float64),
        np.array([1.0, -3.0, 0.0], dtype=np.float32),
    ]
    kernels = [lambda positions, value=value: value.copy() for value in values]

    expected = mesher.kernel_caller(kernels, XYZ).min(axis=-1)
    actual = mesher.kernel_min_caller(kernels, XYZ)

    _assert_bitwise_equal(actual, expected)


def _legacy_ocmesher_kernel_caller(mesher, kernels, XYZ_all):
    if len(XYZ_all) == 0:
        return np.zeros((0, len(kernels)), dtype=mesher.sdf_np_float_type)
    chunk_sdfs = []
    for start in range(0, len(XYZ_all), 10_000_000):
        positions = XYZ_all[start : start + 10_000_000]
        kernel_sdfs = []
        if mesher.enclosed:
            out_of_bounds = np.zeros(len(positions), dtype=bool)
            for coordinate in range(3):
                out_of_bounds |= (
                    positions[:, coordinate] <= mesher.bounds[coordinate * 2]
                )
                out_of_bounds |= (
                    positions[:, coordinate] >= mesher.bounds[coordinate * 2 + 1]
                )
        for kernel in kernels:
            sdf = kernel(positions)
            if mesher.enclosed:
                sdf[out_of_bounds] = 1
            kernel_sdfs.append(sdf)
        chunk_sdfs.append(np.stack(kernel_sdfs, -1).astype(mesher.sdf_np_float_type))
    return np.concatenate(chunk_sdfs, 0)


@needs_ocmesher
def test_ocmesher_full_kernel_caller_matches_legacy_matrix_bitwise():
    from infinigen.OcMesher.ocmesher.core import OcMesher

    mesher = object.__new__(OcMesher)
    mesher.sdf_np_float_type = np.float32
    mesher.enclosed = True
    mesher.bounds = (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    values = [
        np.array([2.0, np.nan, -0.0, 8.0], dtype=np.float64),
        np.array([1.0, -3.0, 0.0, np.nan], dtype=np.float32),
        np.array([-0.0, 4.0, 5.0, 6.0], dtype=np.float64),
    ]

    def make_kernels():
        return [lambda XYZ, value=value: value[: len(XYZ)].copy() for value in values]

    expected = _legacy_ocmesher_kernel_caller(mesher, make_kernels(), positions)
    actual = mesher.kernel_caller(make_kernels(), positions)

    _assert_bitwise_equal(actual, expected)


@needs_ocmesher
def test_ocmesher_full_kernel_caller_preserves_edge_cases():
    from infinigen.OcMesher.ocmesher.core import OcMesher

    mesher = object.__new__(OcMesher)
    mesher.sdf_np_float_type = np.float32
    mesher.enclosed = False

    empty = np.empty((0, 3), dtype=np.float64)
    actual = mesher.kernel_caller([lambda XYZ: np.empty(0)], empty)
    _assert_bitwise_equal(actual, np.zeros((0, 1), dtype=mesher.sdf_np_float_type))

    with pytest.raises(ValueError, match="need at least one array to stack"):
        mesher.kernel_caller([], np.zeros((2, 3), dtype=np.float64))


def _legacy_spherical_kernel_caller(kernels, XYZ, bounds=None):
    out_of_bounds = None
    if bounds is not None:
        out_of_bounds = np.zeros(len(XYZ), dtype=bool)
        for coordinate in range(3):
            out_of_bounds |= XYZ[:, coordinate] <= bounds[coordinate * 2]
            out_of_bounds |= XYZ[:, coordinate] >= bounds[coordinate * 2 + 1]
    sdfs = []
    for kernel in kernels:
        sdf = kernel(XYZ, sdf_only=1)[Vars.SDF]
        if out_of_bounds is not None:
            sdf[out_of_bounds] = 1e6
        sdfs.append(sdf)
    return np.stack(sdfs, -1)


def test_spherical_full_kernel_caller_matches_legacy_matrix_bitwise():
    XYZ = np.array(
        [
            [0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    bounds = (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)
    values = [
        np.array([2.0, np.nan, -0.0, 8.0], dtype=np.float32),
        np.array([1.0, -3.0, 0.0, np.nan], dtype=np.float64),
        np.array([-0.0, 4.0, 5.0, 6.0], dtype=np.float32),
    ]
    expected = _legacy_spherical_kernel_caller(
        [_kernel(value) for value in values], XYZ, bounds
    )
    actual = kernel_caller([_kernel(value) for value in values], XYZ, bounds)

    _assert_bitwise_equal(actual, expected)
    assert actual.flags.c_contiguous


def test_spherical_full_kernel_caller_preserves_edge_cases():
    empty = np.empty((0, 3), dtype=np.float64)
    actual = kernel_caller([_kernel(np.empty(0, dtype=np.float32))], empty)
    _assert_bitwise_equal(actual, np.zeros((0, 1), dtype=np.float32))

    with pytest.raises(ValueError, match="need at least one array to stack"):
        kernel_caller([], np.zeros((2, 3), dtype=np.float64))
