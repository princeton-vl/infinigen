# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Zeyu Ma


from ctypes import POINTER, c_double, c_int32

import gin
import numpy as np
from numpy import ascontiguousarray as AC
from ._marching_cubes_lewiner import marching_cubes
from infinigen.terrain.utils import ASDOUBLE, ASINT, Mesh, write_attributes, register_func, load_cdll
from infinigen.terrain.utils import Timer as tTimer
from infinigen.core.util.organization import Tags
from tqdm import tqdm

@gin.configurable("CubeSphericalMesherTimer")
class Timer(tTimer):
    def __init__(self, desc, verbose=False):
        super().__init__(desc, verbose)


@gin.configurable
class CubeSphericalMesher:

    def __init__(
        self, cam_pose,
        r_min, r_max,
        base_90d_resolution,
        base_R,
        H_fov, W_fov,
        N0, N1,
        complete_depth_test=False,
        inview_upscale=-1,
        test_downscale=5,
        bisection_iters=10,
        complete_depth_test_relax=4,
        device="cpu",
        verbose=False,
    ):
        self.cam_pose = cam_pose
        self.r_min, self.r_max = r_min, r_max
        self.H_fov, self.W_fov = H_fov, W_fov
        self.upscale = inview_upscale
        self.complete_depth_test = complete_depth_test
        self.complete_depth_test_relax = complete_depth_test_relax
        assert inview_upscale == 1 or inview_upscale == -1 or inview_upscale % 2 == 0
        self.L = base_90d_resolution
        self.R = base_R
        self.N0 = N0
        self.N1 = N1
        self.test_downscale = test_downscale
        self.bisection_iters = bisection_iters
        self.verbose = verbose

        dll = load_cdll(f"terrain/lib/{device}/meshing/cube_spherical_mesher.so")
        register_func(self, dll, "init_and_get_emptytest_queries", [
            POINTER(c_double), c_double, c_double, c_int32, c_int32, POINTER(c_double), c_int32, c_double, c_double,
            c_int32, c_int32, c_int32
        ])
        register_func(self, dll, "initial_update", [POINTER(c_double)], c_int32)
        register_func(self, dll, "get_coarse_queries", [POINTER(c_double), POINTER(c_int32)])
        register_func(self, dll, "update", [
            c_int32, POINTER(c_double), POINTER(c_int32), POINTER(c_double), c_int32, POINTER(c_int32), c_int32
        ])
        register_func(self, dll, "get_cnt", restype=c_int32)
        register_func(self, dll, "get_mesh_cnt", [POINTER(c_int32)])
        register_func(self, dll, "bisection_get_positions", [POINTER(c_double)])
        register_func(self, dll, "bisection_update", [POINTER(c_double)])
        register_func(self, dll, "finefront_init", restype=c_int32)
        register_func(self, dll, "finefront_get_queries", [POINTER(c_double)])
        register_func(self, dll, "finefront_update", [
            c_int32, POINTER(c_double), POINTER(c_double), c_int32, POINTER(c_int32), c_int32
        ])
        register_func(self, dll, "finefront_get_cnt", restype=c_int32)
        register_func(self, dll, "finefront_cleanup")
        register_func(self, dll, "complete_depth_test_get_queries", [c_int32, c_int32, POINTER(c_double)])
        register_func(self, dll, "complete_depth_test_update", [c_int32, c_int32, POINTER(c_double)])
        register_func(self, dll, "get_stitching_queries", [POINTER(c_double), POINTER(c_int32)])
        register_func(self, dll, "stitch_update", [POINTER(c_double), POINTER(c_int32), POINTER(c_double), c_int32, POINTER(c_int32), c_int32])
        register_func(self, dll, "get_final_mesh", [POINTER(c_double), POINTER(c_int32), POINTER(c_int32)])

    def __call__(self, kernels):
        H = self.L - 2 * self.N0
        W = self.L - 2 * self.N1
        R = self.R
        with Timer("init_and_get_emptytest_queries"):
            test_L = (self.L - 1) // self.test_downscale  + 1
            test_R = (self.R - 1) // self.test_downscale  + 1
            positions = AC(np.zeros((6 * (test_L + 1) ** 2 * (test_R + 1), 3), dtype=np.float64))
            self.init_and_get_emptytest_queries(
                ASDOUBLE(AC((self.cam_pose.reshape(-1).astype(np.float64)))),
                self.r_min, self.r_max, self.L, self.R,
                ASDOUBLE(positions),
                self.test_downscale, self.H_fov, self.W_fov, self.upscale,
                self.N0, self.N1,
            )
        
        with Timer(f"compute emptytest sdf of #{len(positions)} (6x{test_L + 1}^2x{test_R + 1})"):
            sdf = AC(self.kernel_caller(kernels, positions).min(axis=-1).astype(np.float64))

        with Timer("initial_update"):
            cnt = self.initial_update(ASDOUBLE(sdf))
        
        iter = 0
        while cnt > 0:
            if self.verbose: print(f"{iter=}")
            iter += 1
            with Timer(f"get_coarse_queries of {cnt} blocks"):
                positions = AC(np.zeros(((self.test_downscale + 1) ** 3 * cnt, 3), dtype=np.float64))
                position_bounds = AC(np.zeros((cnt * 3,), dtype=np.int32))
                self.get_coarse_queries(ASDOUBLE(positions), ASINT(position_bounds))
            with Timer("compute coarse sdf"):
                sdf = AC(self.kernel_caller(kernels, positions).min(axis=-1).astype(np.float64))
                del positions
            with Timer("run marching cube"):
                S = self.test_downscale + 1
                block_size = (self.test_downscale+1) ** 3
                if self.verbose: range_cnt = tqdm(range(cnt))
                else: range_cnt = range(cnt)
                for i in range_cnt:
                    S1, S2, S3 = position_bounds[i * 3: (i+1) * 3]
                    part_sdf = sdf[i * block_size: (i+1) * block_size].reshape(S, S, S)[:S1 + 1, :S2 + 1, :S3 + 1]
                    verts_i_int, verts_i_frac, faces_i, _, _ = marching_cubes(part_sdf, 0)
                    verts_i = verts_i_int + verts_i_frac
                    self.update(
                        i, ASDOUBLE(sdf), ASINT(position_bounds),
                        ASDOUBLE(AC(verts_i.astype(np.float64))), len(verts_i),
                        ASINT(AC(faces_i.astype(np.int32))), len(faces_i),
                    )
            with Timer("collect new cnt"):
                cnt = self.get_cnt()
            
        if self.upscale != -1 and self.upscale != 1:
            U = self.upscale
            S = U + 1
            cnt = self.finefront_init()
            iter = 0
            while True:
                if self.verbose: print(f"{iter=}")
                iter += 1
                if cnt == 0 and self.complete_depth_test:
                    with Timer("complete_depth_test"):
                        self.complete_depth_test = False # one time use
                        if self.verbose: batch_range = tqdm(range(0, W * self.upscale, self.complete_depth_test_relax))
                        else: batch_range = range(0, W * self.upscale, self.complete_depth_test_relax)
                        for b in batch_range:
                            positions = AC(np.zeros((((H * self.upscale - 1) // self.complete_depth_test_relax + 1) * ((R * self.upscale - 1) // self.complete_depth_test_relax + 1), 3), dtype=np.float64))
                            self.complete_depth_test_get_queries(self.complete_depth_test_relax, b, ASDOUBLE(positions))
                            sdf = AC(self.kernel_caller(kernels, positions).min(axis=-1).astype(np.float64))
                            self.complete_depth_test_update(self.complete_depth_test_relax, b, ASDOUBLE(sdf))
                        cnt = self.finefront_get_cnt()
                if cnt == 0: break
                with Timer(f"get_finefront_queries of {cnt} blocks"):
                    positions = AC(np.zeros(((self.upscale + 1) ** 3 * cnt, 3), dtype=np.float64))
                    self.finefront_get_queries(ASDOUBLE(positions))
                with Timer("compute finefront sdf"):
                    sdf = AC(self.kernel_caller(kernels, positions).min(axis=-1).astype(np.float64))
                    del positions
                with Timer("run marching cube"):
                    block_size = S ** 3
                    if self.verbose: range_cnt = tqdm(range(cnt))
                    else: range_cnt = range(cnt)
                    for i in range_cnt:
                        part_sdf = sdf[i * block_size: (i+1) * block_size].reshape(S, S, S)
                        if (part_sdf > 0).any() and (part_sdf <= 0).any():
                            verts_i_int, verts_i_frac, faces_i, _, _ = marching_cubes(part_sdf, 0)
                            verts_i = verts_i_int + verts_i_frac
                            self.finefront_update(
                                i, ASDOUBLE(sdf),
                                ASDOUBLE(AC(verts_i.astype(np.float64))), len(verts_i),
                                ASINT(AC(faces_i.astype(np.int32))), len(faces_i),
                            )
                with Timer("collect new cnt"):
                    cnt = self.finefront_get_cnt()

            self.finefront_cleanup()
            
            with Timer(f"get_stitching_queries of {((2 * (W + H) * R * (U * U + 2 * U + 1) + 4 * R * (U + 2)) * 8, 3)=}"):
                positions = AC(np.zeros(((2 * (W + H) * R * (U * U + 2 * U + 1) + 4 * R * (U + 2)) * 8, 3), dtype=np.float64))
                self.get_stitching_queries(ASDOUBLE(positions), POINTER(c_int32)())

            with Timer("compute stitching sdf"):
                sdf = AC(self.kernel_caller(kernels, positions).min(axis=-1).astype(np.float64))
                del positions
            sdf = sdf.reshape((-1, 2, 2, 2))

            with Timer("marching stiches"):
                sdf_repeated1 = sdf[:, 1:, :, :]
                sdf_repeated2 = np.concatenate((sdf[1:, :1, :, :], np.zeros((1, 1, 2, 2))))
                sdf = AC(np.concatenate((sdf, sdf_repeated1, sdf_repeated2), 1).reshape((-1, 2, 2)))
                if (sdf > 0).any() and (sdf <= 0).any():
                    verts_int, verts_frac, faces, _, _ = marching_cubes(sdf, 0)
                    self.stitch_update(
                        ASDOUBLE(sdf),
                        ASINT(AC(verts_int.astype(np.int32))),
                        ASDOUBLE(AC(verts_frac.astype(np.float64))), len(verts_int),
                        ASINT(AC(faces.astype(np.int32))), len(faces),
                    )

        with Timer("merge identifiers and get coarse vert counts"):
            NM = AC(np.zeros(2, dtype=np.int32))
            self.get_mesh_cnt(ASINT(NM))
        
        if self.verbose: print(f"mesh has {NM[0]} vertices and {NM[1]} faces")
        
        with Timer("bisection"):
            positions = AC(np.zeros((NM[0] * 3,), dtype=np.float64))
            if self.verbose: range_it = tqdm(range(self.bisection_iters))
            else: range_it = range(self.bisection_iters)
            for it in range_it:
                self.bisection_get_positions(ASDOUBLE(positions))
                sdf = np.ascontiguousarray(self.kernel_caller(kernels, positions.reshape((-1, 3))).min(axis=-1).astype(np.float64))
                self.bisection_update(ASDOUBLE(sdf))
        with Timer("get final results"):
            vertices = AC(np.zeros((NM[0], 3), dtype=np.float64))
            outview_annotation = AC(np.zeros(NM[0], dtype=np.int32))
            faces = AC(np.zeros((NM[1], 3), dtype=np.int32))
            self.get_final_mesh(ASDOUBLE(vertices), ASINT(faces), ASINT(outview_annotation))
            mesh = Mesh(vertices=vertices, faces=faces)

        
        with Timer("compute attributes"):
            write_attributes(kernels, mesh)
        mesh.vertex_attributes[Tags.OutOfView] = outview_annotation
        return mesh