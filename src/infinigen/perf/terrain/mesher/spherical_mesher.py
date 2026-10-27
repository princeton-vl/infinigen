# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Zeyu Ma: original Infinigen spherical mesher (https://github.com/princeton-vl/infinigen/blob/224598d4a2ff95b4fc928e72c222598b7122c698/src/infinigen/terrain/mesher/spherical_mesher.py)
# - David Yan, Jia Deng: optimized implementation

import numpy as np

from infinigen.terrain.mesher import spherical_mesher as _default
from infinigen.terrain.mesher.cube_spherical_mesher import CubeSphericalMesher
from infinigen.terrain.mesher.frontview_spherical_mesher import FrontviewSphericalMesher
from infinigen.terrain.mesher.spherical_mesher import SphericalMesher, magnifier
from infinigen.terrain.utils import Vars


def kernel_caller(kernels, XYZ, bounds=None):
    kernels = list(kernels)
    out_bound = None
    if bounds is not None:
        out_bound = np.zeros(len(XYZ), dtype=bool)
        for i in range(3):
            out_bound |= XYZ[:, i] <= bounds[i * 2]
            out_bound |= XYZ[:, i] >= bounds[i * 2 + 1]

    sdfs = None
    for column, kernel in enumerate(kernels):
        ret = kernel(XYZ, sdf_only=1)
        sdf = ret[Vars.SDF]
        if out_bound is not None:
            sdf[out_bound] = (
                1e6  # because of skimage mc only provides coords, which is has precision limit
            )
        if sdfs is None:
            sdfs = np.empty((len(XYZ), len(kernels)), dtype=sdf.dtype)
        else:
            result_dtype = np.result_type(sdfs.dtype, sdf.dtype)
            if sdfs.dtype != result_dtype:
                promoted = np.empty(sdfs.shape, dtype=result_dtype)
                promoted[:, :column] = sdfs[:, :column]
                sdfs = promoted
        sdfs[:, column] = sdf
    if sdfs is None:
        raise ValueError("need at least one array to stack")
    return sdfs


def kernel_min_caller(kernels, XYZ, bounds=None):
    """Minimum of the `kernels`' SDFs at `XYZ` in O(N) memory instead of O(N * K)."""
    out_bound = None
    if bounds is not None:
        out_bound = np.zeros(len(XYZ), dtype=bool)
        for i in range(3):
            out_bound |= XYZ[:, i] <= bounds[i * 2]
            out_bound |= XYZ[:, i] >= bounds[i * 2 + 1]

    minimum = None
    for kernel in kernels:
        ret = kernel(XYZ, sdf_only=1)
        sdf = ret[Vars.SDF]
        if out_bound is not None:
            sdf[out_bound] = magnifier
        if minimum is None:
            minimum = sdf.copy()
        else:
            result_dtype = np.result_type(minimum.dtype, sdf.dtype)
            if minimum.dtype != result_dtype:
                minimum = minimum.astype(result_dtype)
            np.minimum(minimum, sdf, out=minimum)
    if minimum is None:
        raise ValueError("need at least one array to stack")
    return minimum


class OpaqueSphericalMesher(_default.OpaqueSphericalMesher):
    def __init__(
        self,
        cameras,
        bounds,
        base_90d_resolution=None,
        pixels_per_cube=1.84,
        test_downscale=5,
        upscale1=2,
        upscale2=4,
        r_lengthen=1,
    ):
        SphericalMesher.__init__(self, cameras, bounds)
        inview_upscale_coarse = upscale1
        inview_upscale_fine = upscale1 * upscale2
        outview_upscale = 1
        assert bool(base_90d_resolution is None) ^ bool(pixels_per_cube is None)
        if base_90d_resolution is None:
            base_90d_resolution = int(
                1
                / (
                    pixels_per_cube
                    * inview_upscale_fine
                    * self.fov[0]
                    / np.pi
                    * 2
                    / self.H
                )
            )
        base_90d_resolution = base_90d_resolution // test_downscale * test_downscale

        print(
            f"In view visible mesh angle resolution 90d/{base_90d_resolution * inview_upscale_fine}, about {base_90d_resolution * inview_upscale_fine * self.fov[0] / np.pi * 2 / self.H: .2f} marching cube per pixel"
        )
        print(
            f"In view invisible mesh angle resolution 90d/{base_90d_resolution * inview_upscale_coarse}, about {base_90d_resolution * inview_upscale_coarse * self.fov[0] / np.pi * 2 / self.H: .2f} marching cube per pixel"
        )
        print(
            f"Out view mesh angle resolution 90d/{base_90d_resolution * outview_upscale}, about {base_90d_resolution * outview_upscale * self.fov[0] / np.pi * 2 / self.H: .2f} marching cube per pixel"
        )

        fov = self.fov
        base_angle_resolution = np.pi / 2 / base_90d_resolution
        base_R = int(
            (np.log(self.r_max) - np.log(self.r_min))
            / (np.pi / 2 / base_90d_resolution)
            / r_lengthen
        )
        N0 = int(np.floor((1 - fov[0] * 2 / np.pi) / 2 * base_90d_resolution))
        N1 = int(np.floor((1 - fov[1] * 2 / np.pi) / 2 * base_90d_resolution))
        rounded_fov = (
            2 * (np.pi / 4 - N0 * base_angle_resolution),
            2 * (np.pi / 4 - N1 * base_angle_resolution),
        )
        H = (base_90d_resolution - N0 * 2) * upscale1
        W = (base_90d_resolution - N1 * 2) * upscale1
        R = base_R * upscale1
        print(f"In view invisible mesh marching cube resolution {H}x{W}x{R}")
        self.frontview_mesher = FrontviewSphericalMesher(
            self.cam_pose,
            rounded_fov[0],
            rounded_fov[1],
            self.r_min,
            self.r_max,
            H,
            W,
            R,
            upscale2,
            test_downscale=test_downscale,
            complete_depth_test=self.complete_depth_test,
        )
        self.frontview_mesher.kernel_caller = lambda k, xyz: _default.kernel_caller(
            k, xyz, self.bounds
        )
        self.frontview_mesher.kernel_min_caller = lambda k, xyz: kernel_min_caller(
            k, xyz, self.bounds
        )
        self.background_mesher = CubeSphericalMesher(
            self.cam_pose,
            self.r_min,
            self.r_max,
            base_90d_resolution * outview_upscale,
            base_R * outview_upscale,
            test_downscale=test_downscale,
            H_fov=rounded_fov[0],
            W_fov=rounded_fov[1],
            N0=N0,
            N1=N1,
        )
        self.background_mesher.kernel_caller = lambda k, xyz: _default.kernel_caller(
            k, xyz, self.bounds
        )
        self.background_mesher.kernel_min_caller = lambda k, xyz: kernel_min_caller(
            k, xyz, self.bounds
        )


class TransparentSphericalMesher(_default.TransparentSphericalMesher):
    def __init__(
        self,
        cameras,
        bounds,
        base_90d_resolution=None,
        pixels_per_cube=1.84,
        test_downscale=5,
        inv_scale=8,
        r_lengthen=3,
        camera_annotation_frames=None,
    ):
        SphericalMesher.__init__(self, cameras, bounds)
        self.cameras = cameras
        self.camera_annotation_frames = camera_annotation_frames
        assert bool(base_90d_resolution is None) ^ bool(pixels_per_cube is None)
        if base_90d_resolution is None:
            base_90d_resolution = int(
                1 / (pixels_per_cube * inv_scale * self.fov[0] / np.pi * 2 / self.H)
            )
        base_90d_resolution = base_90d_resolution // test_downscale * test_downscale

        base_R = int(
            (np.log(self.r_max) - np.log(self.r_min))
            / (np.pi / 2 / base_90d_resolution)
            / r_lengthen
        )
        print(
            f"In view mesh angle resolution 90d/{base_90d_resolution * inv_scale}, about {base_90d_resolution * inv_scale * self.fov[0] / np.pi * 2 / self.H: .2f} marching cube per pixel"
        )
        print(
            f"Out view mesh angle resolution 90d/{base_90d_resolution}, about {base_90d_resolution * self.fov[0] / np.pi * 2 / self.H: .2f} marching cube per pixel"
        )

        fov = self.fov
        base_angle_resolution = np.pi / 2 / base_90d_resolution
        N0 = int(np.floor((1 - fov[0] * 2 / np.pi) / 2 * base_90d_resolution))
        N1 = int(np.floor((1 - fov[1] * 2 / np.pi) / 2 * base_90d_resolution))
        rounded_fov = (
            2 * (np.pi / 4 - N0 * base_angle_resolution),
            2 * (np.pi / 4 - N1 * base_angle_resolution),
        )
        self.mesher = CubeSphericalMesher(
            self.cam_pose,
            self.r_min,
            self.r_max,
            base_90d_resolution,
            base_R,
            test_downscale=test_downscale,
            inview_upscale=inv_scale,
            H_fov=rounded_fov[0],
            W_fov=rounded_fov[1],
            N0=N0,
            N1=N1,
            complete_depth_test=self.complete_depth_test,
        )
        self.mesher.kernel_caller = lambda k, xyz: _default.kernel_caller(
            k, xyz, self.bounds
        )

        self.mesher.kernel_min_caller = lambda k, xyz: kernel_min_caller(
            k, xyz, self.bounds
        )
