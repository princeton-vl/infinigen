# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Zeyu Ma


import numpy as np
import gin

from infinigen.core.util.logging import Timer
from infinigen.terrain.utils import Mesh, get_caminfo, write_attributes, Vars
from .cube_spherical_mesher import CubeSphericalMesher
from .frontview_spherical_mesher import FrontviewSphericalMesher

magnifier = 1e6

# bounding_box for trailer use
@gin.configurable
def kernel_caller(kernels, XYZ, r_bound_max=None, cam_loc=None, bounding_box=None):
    sdfs = []
    for kernel in kernels:
        ret = kernel(XYZ, sdf_only=1)
        sdf = ret[Vars.SDF]
        if r_bound_max is not None:
            R = np.linalg.norm(XYZ - cam_loc, axis=-1)
            mask = R / r_bound_max - 1
            sdf = np.maximum(sdf, mask * magnifier)
        if bounding_box is not None:
            x_min, x_max, y_min, y_max, z_min, z_max = bounding_box
            out_bound = (XYZ[:, 0] < x_min) | (XYZ[:, 0] > x_max) \
                | (XYZ[:, 1] < y_min) | (XYZ[:, 1] > y_max) \
                | (XYZ[:, 2] < z_min) | (XYZ[:, 2] > z_max)
            sdf[out_bound] = 1e6
        sdfs.append(sdf)
    ret = np.stack(sdfs, -1)
    return ret

@gin.configurable
class SphericalMesher:
    def __init__(self,
        cameras,
        r_min=1,
        r_max=1e3,
        complete_depth_test=True,
    ):
        self.cam_pose, _, self.fov, self.H, self.W, _ = get_caminfo(cameras)
        assert self.fov[0] < np.pi / 2 and self.fov[1] < np.pi / 2, "the algorithm does not support larger-than-90-degree fov yet"
        self.r_min = r_min
        self.r_max = r_max
        self.complete_depth_test = complete_depth_test

@gin.configurable
class OpaqueSphericalMesher(SphericalMesher):
    def __init__(self,
        cameras,
        base_90d_resolution=None,
        pixels_per_cube=1.84,
        test_downscale=5,
        upscale1=2,
        upscale2=4,
        r_lengthen=1,
    ):
        SphericalMesher.__init__(self, cameras)
        inview_upscale_coarse = upscale1
        inview_upscale_fine = upscale1 *  upscale2
        outview_upscale = 1
        assert bool(base_90d_resolution is None) ^ bool(pixels_per_cube is None)
        if base_90d_resolution is None: base_90d_resolution = int(1 / (pixels_per_cube * inview_upscale_fine * self.fov[0] / np.pi * 2 / self.H))
        base_90d_resolution = base_90d_resolution // test_downscale * test_downscale

        print(f"In view visible mesh angle resolution 90d/{base_90d_resolution * inview_upscale_fine}, about {base_90d_resolution * inview_upscale_fine * self.fov[0] / np.pi * 2 / self.H: .2f} marching cube per pixel")
        print(f"In view invisible mesh angle resolution 90d/{base_90d_resolution * inview_upscale_coarse}, about {base_90d_resolution * inview_upscale_coarse * self.fov[0] / np.pi * 2 / self.H: .2f} marching cube per pixel")
        print(f"Out view mesh angle resolution 90d/{base_90d_resolution * outview_upscale}, about {base_90d_resolution * outview_upscale * self.fov[0] / np.pi * 2 / self.H: .2f} marching cube per pixel")

        fov = self.fov
        base_angle_resolution = np.pi / 2 / base_90d_resolution
        base_R =  int((np.log(self.r_max) - np.log(self.r_min)) / (np.pi/2 / base_90d_resolution) / r_lengthen)
        N0 = int(np.floor((1 - fov[0] * 2 / np.pi) / 2 * base_90d_resolution))
        N1 = int(np.floor((1 - fov[1] * 2 / np.pi) / 2 * base_90d_resolution))
        rounded_fov = (2 * (np.pi / 4 - N0 * base_angle_resolution), 2 * (np.pi / 4 - N1 * base_angle_resolution))
        H = (base_90d_resolution - N0 * 2) * upscale1
        W = (base_90d_resolution - N1 * 2) * upscale1
        R = base_R * upscale1
        print(f"In view invisible mesh marching cube resolution {H}x{W}x{R}")
        self.frontview_mesher = FrontviewSphericalMesher(
            self.cam_pose,
            rounded_fov[0], rounded_fov[1],
            self.r_min, self.r_max,
            H, W, R,
            upscale2,
            test_downscale=test_downscale,
            complete_depth_test=self.complete_depth_test,
        )
        self.frontview_mesher.kernel_caller = kernel_caller
        self.background_mesher = CubeSphericalMesher(
            self.cam_pose,
            self.r_min, self.r_max,
            base_90d_resolution * outview_upscale,
            base_R * outview_upscale,
            test_downscale=test_downscale,
            H_fov=rounded_fov[0], W_fov=rounded_fov[1],
            N0=N0, N1=N1,
        )
        self.background_mesher.kernel_caller = kernel_caller

    def __call__(self, kernels):
        with Timer("OpaqueSphericalMesher: frontview_mesher"):
            mesh1 = self.frontview_mesher(kernels)
        with Timer("OpaqueSphericalMesher: background_mesher"):
            mesh2 = self.background_mesher(kernels)
        mesh = Mesh.cat([mesh1, mesh2])
        return mesh



@gin.configurable
class TransparentSphericalMesher(SphericalMesher):
    def __init__(self,
        cameras,
        base_90d_resolution=None,
        pixels_per_cube=1.84,
        test_downscale=5,
        upscale=8,
        r_lengthen=3,
        coarse_multiplier=1,
        camera_annotation_frames=None,
    ):
        SphericalMesher.__init__(self, cameras)
        self.cameras = cameras
        self.camera_annotation_frames = camera_annotation_frames
        assert bool(base_90d_resolution is None) ^ bool(pixels_per_cube is None)
        if base_90d_resolution is None: base_90d_resolution = int(1 / (pixels_per_cube * upscale * self.fov[0] / np.pi * 2 / self.H))
        base_90d_resolution = base_90d_resolution  // coarse_multiplier // test_downscale * test_downscale

        base_R = int((np.log(self.r_max) - np.log(self.r_min)) / (np.pi/2 / base_90d_resolution) / r_lengthen)
        print(f"In view mesh angle resolution 90d/{base_90d_resolution * upscale}, about {base_90d_resolution * upscale * self.fov[0] / np.pi * 2 / self.H: .2f} marching cube per pixel")
        print(f"Out view mesh angle resolution 90d/{base_90d_resolution}, about {base_90d_resolution * self.fov[0] / np.pi * 2 / self.H: .2f} marching cube per pixel")

        fov = self.fov
        base_angle_resolution = np.pi / 2 / base_90d_resolution
        N0 = int(np.floor((1 - fov[0] * 2 / np.pi) / 2 * base_90d_resolution))
        N1 = int(np.floor((1 - fov[1] * 2 / np.pi) / 2 * base_90d_resolution))
        rounded_fov = (2 * (np.pi / 4 - N0 * base_angle_resolution), 2 * (np.pi / 4 - N1 * base_angle_resolution))
        self.mesher = CubeSphericalMesher(
            self.cam_pose,
            self.r_min, self.r_max,
            base_90d_resolution,
            base_R,
            test_downscale=test_downscale,
            inview_upscale=upscale,
            H_fov=rounded_fov[0], W_fov=rounded_fov[1],
            N0=N0, N1=N1,
            complete_depth_test=self.complete_depth_test,
        )
        r_bound_max = np.exp(np.log(self.r_min) + (np.log(self.r_max) - np.log(self.r_min)) * (base_R - 0.5) / base_R)
        self.mesher.kernel_caller = lambda k, xyz: kernel_caller(k, xyz, r_bound_max, self.cam_pose[:3, 3])


    def __call__(self, kernels):
        with Timer("TransparentSphericalMesher"):
            mesh = self.mesher(kernels)
            if self.camera_annotation_frames is not None:
                s, e = self.camera_annotation_frames
                mesh.camera_annotation(self.cameras, s, e)
            return mesh
