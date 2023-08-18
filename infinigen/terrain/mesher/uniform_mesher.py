# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Zeyu Ma


from ctypes import POINTER, c_double, c_int32

import gin
import numpy as np
from numpy import ascontiguousarray as AC
from infinigen.terrain.utils import ASDOUBLE, ASINT, Mesh
from infinigen.terrain.utils import Timer as tTimer
from infinigen.terrain.utils import Vars, load_cdll, register_func, write_attributes

from ._marching_cubes_lewiner import marching_cubes


@gin.configurable("UniformMesherTimer")
class Timer(tTimer):
    def __init__(self, desc, verbose=False):
        super().__init__(desc, verbose)


@gin.configurable
class UniformMesher:
    def __init__(self,
        dimensions=(-75, 75, -75, 75, -25, 55),
        subdivisions=(64, -1, -1), # -1 means automatic
        upscale=3,
        enclosed=False,
        bisection_iters=10,
        device="cpu",
        verbose=False,
    ):
        self.enclosed = enclosed
        self.upscale = upscale
        self.dimensions = dimensions
        # Lx, Ly, Lz = dimensions[1] - dimensions[0], dimensions[3] - dimensions[2], dimensions[5] - dimensions[4]
        assert(np.sum(subdivisions == -1) in [0, 2])
        for i, s in enumerate(subdivisions):
            if s != -1:
                coarse_voxel_size = (dimensions[i * 2 + 1] - dimensions[i * 2]) / s
        
        if subdivisions[0] != -1:
            self.x_N = subdivisions[0]
        else:
            self.x_N = int((dimensions[1] - dimensions[0]) / coarse_voxel_size)
        if subdivisions[1] != -1:
            self.y_N = subdivisions[1]
        else:
            self.y_N = int((dimensions[3] - dimensions[2]) / coarse_voxel_size)
        if subdivisions[2] != -1:
            self.z_N = subdivisions[2]
        else:
            self.z_N = int((dimensions[5] - dimensions[4]) / coarse_voxel_size)

        self.x_min, self.x_max = dimensions[0], dimensions[1]
        self.y_min, self.y_max = dimensions[2], dimensions[3]
        self.z_min, self.z_max = dimensions[4], dimensions[5]
        self.closing_margin = coarse_voxel_size / upscale / 2
        self.verbose = verbose
        self.bisection_iters = bisection_iters

        dll = load_cdll(f"terrain/lib/{device}/meshing/uniform_mesher.so")
        register_func(self, dll, "init_and_get_coarse_queries", [
            c_double, c_double, c_int32, c_double, c_double, c_int32,
            c_double, c_double, c_int32, c_int32, POINTER(c_double),
        ])
        register_func(self, dll, "initial_update", [POINTER(c_double)], c_int32)
        register_func(self, dll, "get_fine_queries", [POINTER(c_double)])
        register_func(self, dll, "update", [
            c_int32, POINTER(c_double), POINTER(c_int32), POINTER(c_double), c_int32, POINTER(c_int32), c_int32,
        ])
        register_func(self, dll, "get_cnt", restype=c_int32)
        register_func(self, dll, "get_coarse_mesh_cnt", [POINTER(c_int32)])
        register_func(self, dll, "bisection_get_positions", [POINTER(c_double)])
        register_func(self, dll, "bisection_update", [POINTER(c_double)])
        register_func(self, dll, "get_final_mesh", [POINTER(c_double), POINTER(c_int32)])



    def kernel_caller(self, kernels, XYZ):
        sdfs = []
        for kernel in kernels:
            ret = kernel(XYZ, sdf_only=1)
            sdf = ret[Vars.SDF]
            if self.enclosed:
                out_bound = (XYZ[:, 0] < self.x_min + self.closing_margin) | (XYZ[:, 0] > self.x_max - self.closing_margin) \
                    | (XYZ[:, 1] < self.y_min + self.closing_margin) | (XYZ[:, 1] > self.y_max - self.closing_margin) \
                    | (XYZ[:, 2] < self.z_min + self.closing_margin) | (XYZ[:, 2] > self.z_max - self.closing_margin)
                sdf[out_bound] = 1e11
            sdfs.append(sdf)
        return np.stack(sdfs, -1)

    def __call__(self, kernels):
        with Timer("get_coarse_queries"):
            positions = AC(np.zeros(((self.x_N + 1) * (self.y_N + 1) * (self.z_N + 1), 3), dtype=np.float64))
            self.init_and_get_coarse_queries(
                self.x_min, self.x_max, self.x_N,
                self.y_min, self.y_max, self.y_N,
                self.z_min, self.z_max, self.z_N,
                self.upscale,
                ASDOUBLE(positions),
            )

        with Timer("compute sdf"):
            sdf = AC(self.kernel_caller(kernels, positions).min(axis=-1).astype(np.float64))

        with Timer("initial_update"):
            cnt = self.initial_update(ASDOUBLE(sdf))
    
        S = self.upscale + 1
        block_size = (self.upscale+1) ** 3
        while True:
            if cnt == 0: break
            with Timer(f"get_fine_queries of {cnt} blocks"):
                positions = AC(np.zeros(((self.upscale + 1) ** 3 * cnt, 3), dtype=np.float64))
                self.get_fine_queries(ASDOUBLE(positions))
            with Timer("compute fine sdf and run marching cube"):
                sdf = np.ascontiguousarray(self.kernel_caller(kernels, positions.reshape((-1, 3))).min(axis=-1).astype(np.float64))
                for i in range(cnt):
                    verts_int, verts_frac, faces, _, _ = marching_cubes(sdf[i * block_size: (i+1) * block_size].reshape(S, S, S), 0)
                    self.update(
                        i, ASDOUBLE(sdf),
                        ASINT(AC(verts_int.astype(np.int32))),
                        ASDOUBLE(AC(verts_frac.astype(np.float64))), len(verts_frac),
                        ASINT(AC(faces.astype(np.int32))), len(faces),
                    )

            with Timer("update"):
                cnt = self.get_cnt()
        
        with Timer("merge identifiers and get coarse vert counts"):
            NM = AC(np.zeros(2, dtype=np.int32))
            self.get_coarse_mesh_cnt(ASINT(NM))
            N = NM[0]
            M = NM[1]
        
        if N == 0: return Mesh()
        
        if self.verbose: print(f"Coarse mesh has {N} vertices and {M} faces")
            
        with Timer("bisection on in view coarse mesh"):
            positions = AC(np.zeros((N * 3,), dtype=np.float64))
            range_it = range(self.bisection_iters)
            for it in range_it:
                self.bisection_get_positions(ASDOUBLE(positions))
                sdf = np.ascontiguousarray(self.kernel_caller(kernels, positions.reshape((-1, 3))).min(axis=-1).astype(np.float64))
                self.bisection_update(ASDOUBLE(sdf))

        with Timer("get final results"):
            vertices = AC(np.zeros((NM[0], 3), dtype=np.float64))
            faces = AC(np.zeros((NM[1], 3), dtype=np.int32))
            self.get_final_mesh(ASDOUBLE(vertices), ASINT(faces))
            mesh = Mesh(vertices=vertices, faces=faces)

        with Timer("compute attributes"):
            write_attributes(kernels, mesh)
        return mesh
