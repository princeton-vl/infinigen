# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Zeyu Ma: original Infinigen front-view spherical mesher (https://github.com/princeton-vl/infinigen/blob/224598d4a2ff95b4fc928e72c222598b7122c698/src/infinigen/terrain/mesher/frontview_spherical_mesher.py)
# - David Yan, Jia Deng: optimized implementation

from ctypes import POINTER, c_double, c_int32

import numpy as np
from numpy import ascontiguousarray as AC
from tqdm import tqdm

from infinigen.perf.terrain.mesher._marching_cubes_lewiner import (
    marching_cubes_zero_level_blocks,
)
from infinigen.terrain.mesher import frontview_spherical_mesher as _default
from infinigen.terrain.mesher.frontview_spherical_mesher import Timer
from infinigen.terrain.utils import (
    ASDOUBLE,
    ASINT,
    Mesh,
    load_cdll,
    register_func,
    write_attributes,
)


class FrontviewSphericalMesher(_default.FrontviewSphericalMesher):
    def __init__(
        self,
        cam_pose,
        H_fov,
        W_fov,
        r_min,
        r_max,
        H,
        W,
        R,
        upscale,
        complete_depth_test,
        test_downscale=5,
        bisection_iters_coarse=7,
        bisection_iters_fine=15,
        relax1=0,
        complete_depth_test_relax=4,
        device="cpu",
        verbose=0,
    ):
        self.cam_pose = cam_pose
        self.H_fov, self.W_fov = H_fov, W_fov
        self.r_min, self.r_max = r_min, r_max
        self.H, self.W, self.R = H, W, R
        self.upscale = upscale
        self.relax1 = relax1
        self.test_downscale = test_downscale
        self.bisection_iters_coarse = bisection_iters_coarse
        self.bisection_iters_fine = bisection_iters_fine
        self.complete_depth_test = complete_depth_test
        self.complete_depth_test_relax = complete_depth_test_relax
        self.batch_complete_depth_test = True
        self.verbose = verbose

        dll = load_cdll(f"terrain/lib/{device}/meshing/frontview_spherical_mesher.so")
        register_func(
            self,
            dll,
            "init_and_get_emptytest_queries",
            [
                POINTER(c_double),
                c_double,
                c_double,
                c_double,
                c_double,
                c_int32,
                c_int32,
                c_int32,
                POINTER(c_double),
                c_int32,
                c_int32,
            ],
        )
        register_func(self, dll, "initial_update", [POINTER(c_double)], c_int32)
        register_func(
            self, dll, "get_coarse_queries", [POINTER(c_double), POINTER(c_int32)]
        )
        register_func(
            self,
            dll,
            "update",
            [
                c_int32,
                POINTER(c_double),
                POINTER(c_int32),
                POINTER(c_int32),
                POINTER(c_double),
                c_int32,
                POINTER(c_int32),
                c_int32,
            ],
        )
        register_func(self, dll, "get_cnt", restype=c_int32)
        register_func(self, dll, "get_coarse_mesh_cnt", [POINTER(c_int32)])
        register_func(
            self, dll, "bisection_get_positions", [c_int32, POINTER(c_double)]
        )
        register_func(self, dll, "bisection_update", [c_int32, POINTER(c_double)])
        register_func(self, dll, "init_fine", [c_int32, c_int32], c_int32)
        register_func(self, dll, "get_fine_queries", [POINTER(c_double)])
        register_func(
            self,
            dll,
            "update_fine_small",
            [
                c_int32,
                c_int32,
                POINTER(c_double),
                POINTER(c_int32),
                POINTER(c_double),
                c_int32,
                POINTER(c_int32),
                c_int32,
            ],
        )
        register_func(self, dll, "update_fine", restype=c_int32)
        register_func(
            self, dll, "complete_depth_test_get_query_cnt", [c_int32, c_int32], c_int32
        )
        register_func(
            self,
            dll,
            "complete_depth_test_get_queries",
            [c_int32, c_int32, POINTER(c_double)],
        )
        register_func(
            self,
            dll,
            "complete_depth_test_update",
            [c_int32, c_int32, POINTER(c_double)],
        )
        register_func(self, dll, "complete_depth_test_get_cnt", restype=c_int32)

        register_func(
            self,
            dll,
            "get_final_mesh_statistics",
            [POINTER(c_int32), POINTER(c_int32), POINTER(c_int32)],
        )
        register_func(
            self, dll, "get_final_mesh", [POINTER(c_double), POINTER(c_int32)]
        )

    def __call__(self, kernels):
        n_elements = len(kernels)

        with Timer("init_and_get_emptytest_queries"):
            test_H = (self.H - 1) // self.test_downscale + 1
            test_W = (self.W - 1) // self.test_downscale + 1
            test_R = (self.R - 1) // self.test_downscale + 1
            positions = AC(
                np.zeros(
                    ((test_H + 1) * (test_W + 1) * (test_R + 1), 3), dtype=np.float64
                )
            )
            self.init_and_get_emptytest_queries(
                ASDOUBLE(AC((self.cam_pose.reshape(-1).astype(np.float64)))),
                self.H_fov,
                self.W_fov,
                self.r_min,
                self.r_max,
                self.H,
                self.W,
                self.R,
                ASDOUBLE(positions),
                self.test_downscale,
                self.upscale,
            )

        with Timer(
            f"compute emptytest sdf of #{(test_H + 1) * (test_W + 1) * (test_R + 1)}"
        ):
            sdf = AC(self.kernel_min_caller(kernels, positions).astype(np.float64))

        with Timer("initial_update"):
            cnt = self.initial_update(ASDOUBLE(sdf))
        S = self.test_downscale + 1
        block_size = S**3
        it = 0
        while True:
            if self.verbose:
                print(f"{it=}")
            it += 1
            if cnt == 0:
                break
            with Timer(f"get_coarse_queries of {cnt} blocks"):
                positions = AC(np.zeros((S**3 * cnt, 3), dtype=np.float64))
                position_bounds = AC(np.zeros((cnt * 3,), dtype=np.int32))
                self.get_coarse_queries(ASDOUBLE(positions), ASINT(position_bounds))
            with Timer("compute coarse sdf"):
                sdf = AC(self.kernel_min_caller(kernels, positions).astype(np.float64))
            with Timer("run marching cube"):
                vi_all, vf_all, _, faces_all, ns, ms = marching_cubes_zero_level_blocks(
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
                        ASINT(vi_all[n0 : n0 + n]),
                        ASDOUBLE(vf_all[n0 : n0 + n]),
                        n,
                        ASINT(faces_all[m0 : m0 + m]),
                        m,
                    )
                    n0 += n
                    m0 += m
            with Timer("collect new cnt"):
                cnt = self.get_cnt()

        with Timer("merge identifiers and get coarse vert counts"):
            NM = AC(np.zeros(2, dtype=np.int32))
            self.get_coarse_mesh_cnt(ASINT(NM))
            N = NM[0]
            M = NM[1]

        if self.verbose:
            print(
                f"Entire in view coarse mesh (without visibility face removal) has {N} vertices and {M} faces"
            )

        with Timer("bisection on in view coarse mesh"):
            positions = AC(np.zeros((N * 3,), dtype=np.float64))
            if self.verbose:
                range_it = tqdm(range(self.bisection_iters_coarse))
            else:
                range_it = range(self.bisection_iters_coarse)
            for it in range_it:
                self.bisection_get_positions(-1, ASDOUBLE(positions))
                sdf = np.ascontiguousarray(
                    self.kernel_min_caller(kernels, positions.reshape((-1, 3))).astype(
                        np.float64
                    )
                )
                self.bisection_update(-1, ASDOUBLE(sdf))

        with Timer("visibility test for coarse mesh and init_fine_solids"):
            cnt = self.init_fine(self.relax1, n_elements)

        S = self.upscale + 1
        block_size = S**3
        it = 0
        while True:
            if self.verbose:
                print(f"{it=}")
            it += 1
            if cnt == 0 and self.complete_depth_test:
                with Timer("complete_depth_test"):
                    self.complete_depth_test = False  # one time use
                    relax = self.complete_depth_test_relax
                    batches = range(0, self.W * self.upscale, relax)
                    if self.batch_complete_depth_test:
                        # Batch 0's query count builds the shared depth table
                        # and the other batches only read it, so all queries
                        # can be evaluated in one kernel call.
                        batches = list(batches)
                        counts = [
                            self.complete_depth_test_get_query_cnt(relax, b)
                            for b in batches
                        ]
                        offsets = np.concatenate(([0], np.cumsum(counts)))
                        positions = AC(np.zeros((offsets[-1], 3), dtype=np.float64))
                        for b, start, end in zip(batches, offsets[:-1], offsets[1:]):
                            if end > start:
                                self.complete_depth_test_get_queries(
                                    relax, b, ASDOUBLE(positions[start:end])
                                )
                        sdf = AC(
                            self.kernel_min_caller(kernels, positions).astype(
                                np.float64
                            )
                        )
                        for b, start, end in zip(batches, offsets[:-1], offsets[1:]):
                            if end > start:
                                self.complete_depth_test_update(
                                    relax, b, ASDOUBLE(sdf[start:end])
                                )
                    else:
                        for b in tqdm(batches) if self.verbose else batches:
                            cnt = self.complete_depth_test_get_query_cnt(relax, b)
                            positions = AC(np.zeros((cnt, 3), dtype=np.float64))
                            self.complete_depth_test_get_queries(
                                relax, b, ASDOUBLE(positions)
                            )
                            sdf = AC(
                                self.kernel_min_caller(kernels, positions).astype(
                                    np.float64
                                )
                            )
                            self.complete_depth_test_update(relax, b, ASDOUBLE(sdf))
                    cnt = self.complete_depth_test_get_cnt()
            if cnt == 0:
                break
            with Timer(f"get_fine_positions of #{cnt}"):
                positions = AC(
                    np.zeros((cnt, (self.upscale + 1) ** 3, 3), dtype=np.float64)
                )
                self.get_fine_queries(ASDOUBLE(positions))
            with Timer("compute fine sdf"):
                sdf = np.ascontiguousarray(
                    self.kernel_caller(kernels, positions.reshape((-1, 3))).astype(
                        np.float64
                    )
                )
                del positions

            with Timer("run marching cube"):
                blocks = sdf.reshape(cnt, S, S, S, n_elements)
                # the (block, element) pairs with values both below and above 0
                straddles = ~(blocks >= 0).all(axis=(1, 2, 3)) & ~(blocks <= 0).all(
                    axis=(1, 2, 3)
                )
                block_i, elem_i = np.nonzero(straddles)
                if len(block_i):
                    vi_all, vf_all, _, faces_all, ns, ms = (
                        marching_cubes_zero_level_blocks(
                            blocks.astype(np.float32), None, block_i, elem_i
                        )
                    )
                    sdf_p = ASDOUBLE(sdf)
                    n0 = m0 = 0
                    for i, e, n, m in zip(
                        block_i.tolist(), elem_i.tolist(), ns.tolist(), ms.tolist()
                    ):
                        self.update_fine_small(
                            i,
                            e,
                            sdf_p,
                            ASINT(vi_all[n0 : n0 + n]),
                            ASDOUBLE(vf_all[n0 : n0 + n]),
                            n,
                            ASINT(faces_all[m0 : m0 + m]),
                            m,
                        )
                        n0 += n
                        m0 += m
            with Timer("update"):
                cnt = self.update_fine()

        with Timer("merge identifiers and get fine vert counts"):
            Ns = AC(np.zeros(n_elements, dtype=np.int32))
            Ms = AC(np.zeros(n_elements, dtype=np.int32))
            NM[:] = 0
            self.get_final_mesh_statistics(ASINT(NM), ASINT(Ns), ASINT(Ms))
            if self.verbose:
                print(
                    f"Invisible cleaned coarse mesh has {NM[0]} vertices and {NM[1]} faces"
                )
            for e in range(n_elements):
                if self.verbose:
                    print(
                        f"In view fine mesh (element {e}) has {Ns[e]} vertices and {Ms[e]} faces"
                    )

        with Timer("fine bisection"):
            for e in range(n_elements):
                positions = AC(np.zeros((Ns[e] * 3,), dtype=np.float64))
                if self.verbose:
                    range_it = tqdm(range(self.bisection_iters_fine))
                else:
                    range_it = range(self.bisection_iters_fine)
                for it in range_it:
                    self.bisection_get_positions(e, ASDOUBLE(positions))
                    sdf = np.ascontiguousarray(
                        self.kernel_caller(
                            [kernels[e]], positions.reshape((-1, 3))
                        ).astype(np.float64)
                    )
                    self.bisection_update(e, ASDOUBLE(sdf))

        with Timer("get final results"):
            vertices = AC(np.zeros((NM[0] + np.sum(Ns)) * 3, dtype=np.float64))
            faces = AC(np.zeros((NM[1] + np.sum(Ms)) * 3, dtype=np.int32))
            self.get_final_mesh(ASDOUBLE(vertices), ASINT(faces))
            mesh = Mesh(
                vertices=vertices[: NM[0] * 3].reshape((-1, 3)),
                faces=faces[: NM[1] * 3].reshape((-1, 3)),
            )
            meshes = []
            cnt_N = NM[0]
            cnt_M = NM[1]
            for e in range(n_elements):
                meshes.append(
                    Mesh(
                        vertices=vertices[cnt_N * 3 : (cnt_N + Ns[e]) * 3].reshape(
                            (-1, 3)
                        ),
                        faces=faces[cnt_M * 3 : (cnt_M + Ms[e]) * 3].reshape((-1, 3)),
                    )
                )
                cnt_N += Ns[e]
                cnt_M += Ms[e]

        with Timer("compute attributes"):
            write_attributes(kernels, mesh, meshes)

        with Timer("concat meshes"):
            catted_mesh = Mesh.cat([mesh, *meshes])

        return catted_mesh
