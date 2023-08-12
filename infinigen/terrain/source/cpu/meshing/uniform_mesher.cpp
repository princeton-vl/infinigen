// Copyright (c) Princeton University.
// This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

// Authors: Zeyu Ma


#include <stdio.h>
#include <assert.h>
#include <omp.h>
#include <string.h>
#include <math.h>
#include "utils.cpp"
using namespace std;


namespace pretest {
    int *flag, *newfound;
    int newfound_cap, cnt;
    int U, S;
    void init(int x_N, int y_N, int z_N, int upscale) {
        U = upscale;
        S = U + 1;
        cnt = 0;
        newfound_cap = 10000;
        HM(flag, x_N * y_N * z_N);
        HM(newfound, newfound_cap * 3);
    }
    void cleanup() {
        safefree(flag);
        safefree(newfound);
    }
}
namespace specs {
    double mins[3], maxs[3];
    int N_coarse[3];

    void init(
        double x_min, double x_max, int x_N,
        double y_min, double y_max, int y_N,
        double z_min, double z_max, int z_N
    ) {
        mins[0] = x_min; mins[1] = y_min; mins[2] = z_min;
        maxs[0] = x_max; maxs[1] = y_max; maxs[2] = z_max;
        N_coarse[0] = x_N; N_coarse[1] = y_N; N_coarse[2] = z_N;
    }
}

namespace uniform_mesh {
    int *faces, *lr_vertices;
    double *bis_vertices, *vertices;
    int N_cap, M_cap, N, M;
    void init() {
        N_cap = M_cap = 10000;
        N = M = 0;
        HM(faces, M_cap * 3);
        HM(lr_vertices, N_cap * 6);
    }
    void finalize() {
        fHM(vertices, N * 3);
        for (int i = 0; i < N * 3; i++) {
            vertices[i] = (bis_vertices[i * 2] + bis_vertices[i * 2 + 1]) / 2;
        }
        safefree(bis_vertices);
    }
    void cleanup() {
        safefree(faces);
        safefree(vertices);
    }
}

extern "C" {

    void init_and_get_coarse_queries(
        double x_min, double x_max, int x_N,
        double y_min, double y_max, int y_N,
        double z_min, double z_max, int z_N,
        int upscale,
        double *positions
    ) {
        specs::init(
            x_min, x_max, x_N,
            y_min, y_max, y_N,
            z_min, z_max, z_N
        );
        pretest::init(x_N, y_N, z_N, upscale);
        uniform_mesh::init();
        int t0 = (y_N + 1) * (z_N + 1), t1 = z_N + 1;
        for (int i = 0; i <= x_N; i++)
            for (int j = 0; j <= y_N; j++)
                for (int k = 0; k <= z_N; k++) {
                    positions[3 * (i * t0 + j * t1 + k)] = x_min + (x_max - x_min) * i / x_N;
                    positions[3 * (i * t0 + j * t1 + k) + 1] = y_min + (y_max - y_min) * j / y_N;
                    positions[3 * (i * t0 + j * t1 + k) + 2] = z_min + (z_max - z_min) * k / z_N;
                }
    }

    int initial_update(
        double *sdf
    ) {
        using namespace pretest;
        int x_N = specs::N_coarse[0], y_N = specs::N_coarse[1], z_N = specs::N_coarse[2];
        int t0 = y_N * z_N, t1 = z_N;
        cnt = 0;
        #pragma omp parallel for
        for (int i = 0; i < x_N; i++)
        for (int j = 0; j < y_N; j++)
        for (int k = 0; k < z_N; k++) {
            bool positive = 0, negative = 0, zero=0;
            for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++) {
                double sdf_c = sdf[(i + ii) * (y_N + 1) * (z_N + 1) + (j + jj) * (z_N + 1) + k + kk];
                positive |= sdf_c > 0;
                negative |= sdf_c < 0;
                zero |= sdf_c == 0;
            }
            if ((flag[i * t0 + j * t1 + k] = has_iso_surface(positive, negative, zero)))
            #pragma omp critical
            {
                int mycnt = cnt++;
                if (mycnt >= newfound_cap) {
                    newfound_cap *= 2;
                    HR(newfound, newfound_cap * 3);
                }
                newfound[3 * mycnt] = i;
                newfound[3 * mycnt + 1] = j;
                newfound[3 * mycnt + 2] = k;
            }
        }
        return cnt;
    }

    void get_fine_queries(
        double *positions
    ) {
        using namespace pretest;
        int x_N = specs::N_coarse[0], y_N = specs::N_coarse[1], z_N = specs::N_coarse[2];
        for (int c = 0; c < cnt; c++) {
            int position_offset = c * S * S * S;
            int i = newfound[c * 3], j = newfound[c * 3 + 1], k = newfound[c * 3 + 2];
            for (int ii = 0; ii < S; ii++)
            for (int jj = 0; jj < S; jj++)
            for (int kk = 0; kk < S; kk++) {
                int ijk[3] = {i, j, k};
                int iijjkk[3] = {ii, jj, kk};
                for (int p = 0; p < 3; p++) {
                    positions[3 * (position_offset + ii*S*S + jj*S + kk) + p] = \
                    specs::mins[p] + (specs::maxs[p] - specs::mins[p]) * (ijk[p] + 1.0 * iijjkk[p] / U) / specs::N_coarse[p];
                }
                
            }
            flag[i * y_N * z_N + j * z_N + k] = 3;
        }
    }

    // 1 last step
    // 2 new created
    // 3 ever explored

    void update(
        int c,
        double *sdf,
        int *verts_int,
        double *verts_frac, int N,
        int *faces, int M
    ) {
        using namespace pretest;
        int x_N = specs::N_coarse[0], y_N = specs::N_coarse[1], z_N = specs::N_coarse[2];
        int boundary_x_min[6] = {0, 0, 0, 0, 0, U};
        int boundary_x_max[6] = {U, U, U, U, 0, U};
        int boundary_y_min[6] = {0, 0, 0, U, 0, 0};
        int boundary_y_max[6] = {U, U, 0, U, U, U};
        int boundary_z_min[6] = {0, U, 0, 0, 0, 0};
        int boundary_z_max[6] = {0, U, U, U, U, U};
        int dx[6] = {0, 0, 0, 0, -1, 1};
        int dy[6] = {0, 0, -1, 1, 0, 0};
        int dz[6] = {-1, 1, 0, 0, 0, 0};

        int i = newfound[c * 3], j = newfound[c * 3 + 1], k = newfound[c * 3 + 2];
        for (int b = 0; b < 6; b++) {
            bool positive = 0, negative = 0, zero=0;
            for (int ii = boundary_x_min[b]; ii <= boundary_x_max[b]; ii++)
            for (int jj = boundary_y_min[b]; jj <= boundary_y_max[b]; jj++)
            for (int kk = boundary_z_min[b]; kk <= boundary_z_max[b]; kk++) {
                double sdf_c = sdf[c*S*S*S + ii*S*S + jj*S + kk];
                positive |= sdf_c > 0;
                negative |= sdf_c < 0;
                zero |= sdf_c == 0;
            }
            if (has_iso_surface(positive, negative, zero)) {
                int ni = i + dx[b], nj = j + dy[b], nk = k + dz[b];
                if (ni >= 0 && nj >= 0 && nk >= 0 && ni < x_N && nj < y_N && nk < z_N) {
                    int flag0 = flag[ni * y_N * z_N + nj * z_N + nk];
                    if (flag0 == 0) {
                        flag[ni * (y_N) * (z_N) + nj * (z_N) + nk] = 2;
                    }
                }
            }
        }
        while (uniform_mesh::M_cap <= uniform_mesh::M + M) {
            uniform_mesh::M_cap *= 2;
            HR(uniform_mesh::faces, uniform_mesh::M_cap * 3);
        }
        // dont omp
        for (int i = 0; i < 3 * M; i++)
            uniform_mesh::faces[3 * uniform_mesh::M + i] = faces[i] + uniform_mesh::N;
        while (uniform_mesh::N_cap <= uniform_mesh::N + N) {
            uniform_mesh::N_cap *= 2;
            HR(uniform_mesh::lr_vertices, uniform_mesh::N_cap * 6);
        }
        double *current_sdf = sdf + c * S * S * S;
        int t1 = S * S; int t2 = S;
        // dont omp
        for (int i = 0; i < N; i++) {
            int low = 0, high = 1;
            double sdf_floor = current_sdf[
                (verts_int[i * 3] + int_floor(verts_frac[i * 3])) * t1 \
                + (verts_int[i * 3 + 1] + int_floor(verts_frac[i * 3 + 1])) * t2 \
                + (verts_int[i * 3 + 2] + int_floor(verts_frac[i * 3 + 2]))
            ];
            double sdf_ceil = current_sdf[
                (verts_int[i * 3] + int_ceil(verts_frac[i * 3])) * t1 \
                + (verts_int[i * 3 + 1] + int_ceil(verts_frac[i * 3 + 1])) * t2 \
                + (verts_int[i * 3 + 2] + int_ceil(verts_frac[i * 3 + 2]))
            ];
            if (sdf_floor > sdf_ceil) {
                high = 0; low = 1;
            }
            for (int j = 0; j < 3; j++) {
                assert(verts_frac[i * 3 + j] >= 0 && verts_frac[i * 3 + j] <= 1);
                uniform_mesh::lr_vertices[6 * uniform_mesh::N + i * 6 + 2 * j + low] = verts_int[i * 3 + j] + int_floor(verts_frac[i * 3 + j]) + newfound[c * 3 + j] * U;
                uniform_mesh::lr_vertices[6 * uniform_mesh::N + i * 6 + 2 * j + high] = verts_int[i * 3 + j] + int_ceil(verts_frac[i * 3 + j]) + newfound[c * 3 + j] * U;
            }
        }
        uniform_mesh::N += N;
        uniform_mesh::M += M;
    }

    int get_cnt() {
        using namespace pretest;
        int x_N = specs::N_coarse[0], y_N = specs::N_coarse[1], z_N = specs::N_coarse[2];
        cnt = 0;
        #pragma omp parallel for
        for (int i = 0; i < x_N; i++)
        for (int j = 0; j < y_N; j++)
        for (int k = 0; k < z_N; k++)
            if (flag[i * (y_N) * (z_N) + j * (z_N) + k] == 2) {
                flag[i * (y_N) * (z_N) + j * (z_N) + k] = 1;
                #pragma omp critical
                {
                    int mycnt = cnt++;
                    if (mycnt >= newfound_cap) {
                        newfound_cap *= 2;
                        HR(newfound, newfound_cap * 3);
                    }
                    newfound[3 * mycnt] = i;
                    newfound[3 * mycnt + 1] = j;
                    newfound[3 * mycnt + 2] = k;
                }
            }
        return cnt;
    }

    void get_coarse_mesh_cnt(int *NM) {
        pretest::cleanup();
        using namespace uniform_mesh;
        merge_verts(
            lr_vertices, N, 6,
            faces, M
        );
        clean_faces(N, faces, M);
        clean_verts(lr_vertices, 6, N, faces, M);
        NM[0] = N;
        NM[1] = M;
        fHM(bis_vertices, N * 6);
        for (int i = 0; i < 6 * N; i++) {
            bis_vertices[i] = (double)lr_vertices[i];
        }
        safefree(lr_vertices);
    }


    void bisection_get_positions(
        double *positions
    ) {
        int N = uniform_mesh::N;
        double *identifiers = uniform_mesh::bis_vertices;
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < 3; j++)
                positions[i * 3 + j] = specs::mins[j] + (specs::maxs[j] - specs::mins[j]) * \
                (identifiers[i * 6 + j * 2] + identifiers[i * 6 + 1 + j * 2]) / 2 / specs::N_coarse[j] / pretest::U;
        }
    }

    void bisection_update(
        double *sdfs
    ) {
        int N = uniform_mesh::N;
        double *identifiers = uniform_mesh::bis_vertices;
        int cnt = 0;
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < 3; j++) {
                double middle = (identifiers[i * 6 + 2 * j] + identifiers[i * 6 + 2 * j + 1]) / 2;
                if (sdfs[cnt] > 0) identifiers[i * 6 + 2 * j + 1] = middle;
                else identifiers[i * 6 + 2 * j] = middle;
            }
            cnt++;
        }
    }

    void get_final_mesh(double *vertices, int *faces) {
        uniform_mesh::finalize();
        for (int i = 0; i < uniform_mesh::N; i++) {
            for (int j = 0; j < 3; j++)
                vertices[i * 3 + j] = specs::mins[j] + (specs::maxs[j] - specs::mins[j]) * \
                uniform_mesh::vertices[i * 3 + j] / specs::N_coarse[j] / pretest::U;
        }
        memcpy(faces, uniform_mesh::faces, 3 * uniform_mesh::M * sizeof(int));
        uniform_mesh::cleanup();
    }

}