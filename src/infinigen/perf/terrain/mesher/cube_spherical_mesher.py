# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Zeyu Ma: original Infinigen cube spherical mesher (https://github.com/princeton-vl/infinigen/blob/224598d4a2ff95b4fc928e72c222598b7122c698/src/infinigen/terrain/mesher/cube_spherical_mesher.py)
# - David Yan, Jia Deng: optimized implementation

from ctypes import POINTER, c_int32

import numpy as np
from numpy import ascontiguousarray as AC
from tqdm import tqdm

from infinigen.core.util.organization import Tags
from infinigen.perf.terrain.mesher._marching_cubes_lewiner import (
    marching_cubes_zero_level_blocks,
)
from infinigen.terrain.mesher import cube_spherical_mesher as _default
from infinigen.terrain.mesher._marching_cubes_lewiner import marching_cubes
from infinigen.terrain.mesher.cube_spherical_mesher import Timer
from infinigen.terrain.utils import ASDOUBLE, ASINT, Mesh, write_attributes


class CubeSphericalMesher(_default.CubeSphericalMesher):
    def __call__(self, kernels):
        H = self.L - 2 * self.N0
        W = self.L - 2 * self.N1
        R = self.R
        with Timer("init_and_get_emptytest_queries"):
            test_L = (self.L - 1) // self.test_downscale + 1
            test_R = (self.R - 1) // self.test_downscale + 1
            positions = AC(
                np.zeros((6 * (test_L + 1) ** 2 * (test_R + 1), 3), dtype=np.float64)
            )
            self.init_and_get_emptytest_queries(
                ASDOUBLE(AC((self.cam_pose.reshape(-1).astype(np.float64)))),
                self.r_min,
                self.r_max,
                self.L,
                self.R,
                ASDOUBLE(positions),
                self.test_downscale,
                self.H_fov,
                self.W_fov,
                self.upscale,
                self.N0,
                self.N1,
            )

        with Timer(
            f"compute emptytest sdf of #{len(positions)} (6x{test_L + 1}^2x{test_R + 1})"
        ):
            sdf = AC(self.kernel_min_caller(kernels, positions).astype(np.float64))

        with Timer("initial_update"):
            cnt = self.initial_update(ASDOUBLE(sdf))

        iter = 0
        while cnt > 0:
            if self.verbose:
                print(f"{iter=}")
            iter += 1
            with Timer(f"get_coarse_queries of {cnt} blocks"):
                positions = AC(
                    np.zeros(
                        ((self.test_downscale + 1) ** 3 * cnt, 3), dtype=np.float64
                    )
                )
                position_bounds = AC(np.zeros((cnt * 3,), dtype=np.int32))
                self.get_coarse_queries(ASDOUBLE(positions), ASINT(position_bounds))
            with Timer("compute coarse sdf"):
                sdf = AC(self.kernel_min_caller(kernels, positions).astype(np.float64))
                del positions
            with Timer("run marching cube"):
                S = self.test_downscale + 1
                _, _, verts_all, faces_all, ns, ms = marching_cubes_zero_level_blocks(
                    sdf.astype(np.float32).reshape(cnt, S, S, S),
                    position_bounds,
                    np.arange(cnt),
                )
                sdf_p, bounds_p = ASDOUBLE(sdf), ASINT(position_bounds)
                n0 = m0 = 0
                for i, n, m in zip(range(cnt), ns.tolist(), ms.tolist()):
                    self.update(
                        i,
                        sdf_p,
                        bounds_p,
                        ASDOUBLE(verts_all[n0 : n0 + n]),
                        n,
                        ASINT(faces_all[m0 : m0 + m]),
                        m,
                    )
                    n0 += n
                    m0 += m
            with Timer("collect new cnt"):
                cnt = self.get_cnt()

        if self.upscale != -1 and self.upscale != 1:
            U = self.upscale
            S = U + 1
            cnt = self.finefront_init()
            iter = 0
            while True:
                if self.verbose:
                    print(f"{iter=}")
                iter += 1
                if cnt == 0 and self.complete_depth_test:
                    with Timer("complete_depth_test"):
                        self.complete_depth_test = False  # one time use
                        relax = self.complete_depth_test_relax
                        batches = list(range(0, W * self.upscale, relax))
                        per_batch = ((H * self.upscale - 1) // relax + 1) * (
                            (R * self.upscale - 1) // relax + 1
                        )
                        # Each batch's queries are a fixed grid and its update
                        # reads only that batch's values, so all batches can be
                        # evaluated in one kernel call.
                        positions = AC(
                            np.zeros((per_batch * len(batches), 3), dtype=np.float64)
                        )
                        for bi, b in enumerate(batches):
                            self.complete_depth_test_get_queries(
                                relax,
                                b,
                                ASDOUBLE(
                                    positions[bi * per_batch : (bi + 1) * per_batch]
                                ),
                            )
                        sdf = AC(
                            self.kernel_min_caller(kernels, positions).astype(
                                np.float64
                            )
                        )
                        for bi, b in enumerate(batches):
                            self.complete_depth_test_update(
                                relax,
                                b,
                                ASDOUBLE(sdf[bi * per_batch : (bi + 1) * per_batch]),
                            )
                        cnt = self.finefront_get_cnt()
                if cnt == 0:
                    break
                with Timer(f"get_finefront_queries of {cnt} blocks"):
                    positions = AC(
                        np.zeros(((self.upscale + 1) ** 3 * cnt, 3), dtype=np.float64)
                    )
                    self.finefront_get_queries(ASDOUBLE(positions))
                with Timer("compute finefront sdf"):
                    sdf = AC(
                        self.kernel_min_caller(kernels, positions).astype(np.float64)
                    )
                    del positions
                with Timer("run marching cube"):
                    blocks = sdf.reshape(cnt, S, S, S)
                    straddles = (blocks > 0).any(axis=(1, 2, 3)) & (blocks <= 0).any(
                        axis=(1, 2, 3)
                    )
                    indices = np.nonzero(straddles)[0]
                    if len(indices):
                        _, _, verts_all, faces_all, ns, ms = (
                            marching_cubes_zero_level_blocks(
                                blocks.astype(np.float32), None, indices
                            )
                        )
                        sdf_p = ASDOUBLE(sdf)
                        n0 = m0 = 0
                        for i, n, m in zip(indices.tolist(), ns.tolist(), ms.tolist()):
                            self.finefront_update(
                                i,
                                sdf_p,
                                ASDOUBLE(verts_all[n0 : n0 + n]),
                                n,
                                ASINT(faces_all[m0 : m0 + m]),
                                m,
                            )
                            n0 += n
                            m0 += m
                with Timer("collect new cnt"):
                    cnt = self.finefront_get_cnt()

            self.finefront_cleanup()

            with Timer(
                f"get_stitching_queries of {((2 * (W + H) * R * (U * U + 2 * U + 1) + 4 * R * (U + 2)) * 8, 3)=}"
            ):
                positions = AC(
                    np.zeros(
                        (
                            (2 * (W + H) * R * (U * U + 2 * U + 1) + 4 * R * (U + 2))
                            * 8,
                            3,
                        ),
                        dtype=np.float64,
                    )
                )
                self.get_stitching_queries(ASDOUBLE(positions), POINTER(c_int32)())

            with Timer("compute stitching sdf"):
                sdf = AC(self.kernel_min_caller(kernels, positions).astype(np.float64))
                del positions
            sdf = sdf.reshape((-1, 2, 2, 2))

            with Timer("marching stiches"):
                sdf_repeated1 = sdf[:, 1:, :, :]
                sdf_repeated2 = np.concatenate(
                    (sdf[1:, :1, :, :], np.zeros((1, 1, 2, 2)))
                )
                sdf = AC(
                    np.concatenate((sdf, sdf_repeated1, sdf_repeated2), 1).reshape(
                        (-1, 2, 2)
                    )
                )
                if (sdf > 0).any() and (sdf <= 0).any():
                    verts_int, verts_frac, faces, _, _ = marching_cubes(sdf, 0)
                    self.stitch_update(
                        ASDOUBLE(sdf),
                        ASINT(AC(verts_int.astype(np.int32))),
                        ASDOUBLE(AC(verts_frac.astype(np.float64))),
                        len(verts_int),
                        ASINT(AC(faces.astype(np.int32))),
                        len(faces),
                    )

        with Timer("merge identifiers and get coarse vert counts"):
            NM = AC(np.zeros(2, dtype=np.int32))
            self.get_mesh_cnt(ASINT(NM))

        if self.verbose:
            print(f"mesh has {NM[0]} vertices and {NM[1]} faces")

        with Timer("bisection"):
            positions = AC(np.zeros((NM[0] * 3,), dtype=np.float64))
            if self.verbose:
                range_it = tqdm(range(self.bisection_iters))
            else:
                range_it = range(self.bisection_iters)
            for it in range_it:
                self.bisection_get_positions(ASDOUBLE(positions))
                sdf = np.ascontiguousarray(
                    self.kernel_min_caller(kernels, positions.reshape((-1, 3))).astype(
                        np.float64
                    )
                )
                self.bisection_update(ASDOUBLE(sdf))
        with Timer("get final results"):
            vertices = AC(np.zeros((NM[0], 3), dtype=np.float64))
            outview_annotation = AC(np.zeros(NM[0], dtype=np.int32))
            faces = AC(np.zeros((NM[1], 3), dtype=np.int32))
            self.get_final_mesh(
                ASDOUBLE(vertices), ASINT(faces), ASINT(outview_annotation)
            )
            mesh = Mesh(vertices=vertices, faces=faces)

        with Timer("compute attributes"):
            write_attributes(kernels, mesh)
        mesh.vertex_attributes[Tags.OutOfView] = outview_annotation
        return mesh
