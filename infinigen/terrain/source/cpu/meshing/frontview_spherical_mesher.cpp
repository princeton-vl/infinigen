// Copyright (c) Princeton University.
// This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

// Authors: Zeyu Ma


#include <stdio.h>
#include <assert.h>
#include <omp.h>
#include <string.h>
#include <math.h>
#include <algorithm>
#include "utils.cpp"
#include "visibility_test.cpp"
using namespace std;

namespace specs {
    double cam[12], H_fov, W_fov, r_min, r_max;
    int H, W, R, upscale;
}

namespace coarse_pretest {
    int H, W, R, U, S, cnt;
    int *flag, *newfound;
    int newfound_cap;

    void init(
        int H_, int W_, int R_, int U_
    ) {
        H = H_; W = W_; R = R_; U = U_; S = U + 1;
        newfound_cap = 10000;
        cnt = 0;
        HM(flag, H * W * R);
        HM(newfound, newfound_cap * 3);
    }

    void cleanup() {
        safefree(flag);
        safefree(newfound);
    }
}

namespace coarse {
    int *faces, *lr_vertices, *visibility;
    double *bis_vertices, *vertices;
    int N, M;
    int N_cap, M_cap;
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

namespace coarse_face_map {
    int *head, *nxt, *id, *depth, cnt;
    void init() {
        HM(head, specs::H * specs::W);
        HM(nxt, face_map::cnt);
        HM(id, face_map::cnt);
        HM(depth, face_map::cnt);
        memcpy(head, face_map::head, specs::H * specs::W * sizeof(int));
        memcpy(nxt, face_map::nxt, face_map::cnt * sizeof(int));
        memcpy(depth, face_map::depth, face_map::cnt * sizeof(int));
        memcpy(id, face_map::id, face_map::cnt * sizeof(int));
        cnt = face_map::cnt;
    }
    void cleanup() {
        safefree(head);
        safefree(nxt);
        safefree(id);
        safefree(depth);
    }
}

namespace fine {
    int *faces, *lr_vertices, *visibility;
    double *bis_vertices, *vertices;
    short *ecat;
    int N, M, updated_M;
    int N_cap, M_cap;
    int ne;

    void finalize() {
        for (int i = 0; i < N * 3; i++) {
            vertices[i] = (bis_vertices[i * 2] + bis_vertices[i * 2 + 1]) / 2;
        }
        safefree(bis_vertices);
    }

    void init() {
        N_cap = M_cap = 10000;
        N = M = updated_M = 0;
        HM(faces, M_cap * 3);
        HM(visibility, M_cap);
        HM(lr_vertices, N_cap * 7); // +1 bc ele category
        fHM(vertices, N_cap * 3);
    }

    void cleanup() {
        safefree(ecat);
        safefree(faces);
    }
}

namespace change_map {
    int *change_map, cap, *new_changes, *change_depth, cnt;
    void init() {
        HM(change_map, specs::H * specs::W);
        SET0(change_map, specs::H * specs::W);
        cap = 10000;
        HM(new_changes, cap * 2);
        HM(change_depth, cap);
        cnt = 0;
    }
    void cleanup() {
        safefree(change_map);
        safefree(new_changes);
        safefree(change_depth);
    }
    void clear() {
        for (int i = 0; i < cnt; i++) {
            change_map[new_changes[i * 2] * specs::W + new_changes[i * 2 + 1]] = 0;
        }
        cnt = 0;
    }
    void update(int h, int w) {
        int &l = change_map[h * specs::W + w];
        if (l == 0) {
            l = 1;
            if (cnt >= cap) {
                cap *= 2;
                HR(new_changes, cap * 2);
                HR(change_depth, cap);
            }
            change_depth[cnt] = -1;
            new_changes[cnt * 2] = h;
            new_changes[cnt++ * 2 + 1] = w;
        }
    }

}

namespace fine_pretest {
    const int hash_cap = 100663319;
    int *head, *nxt, *coords_R, cnt, newfound_cnt, boundary_newfound_cnt, U, S;
    short *coords_HWL;
    int cap;
    int *newfound, *boundary_newfound;
    int newfound_cap, boundary_newfound_cap;

    void init() {
        U = specs::upscale;
        S = U + 1;
        cnt = newfound_cnt = boundary_newfound_cnt = 0;
        cap = newfound_cap = boundary_newfound_cap = 10000;
        HM(head, hash_cap);
        HM(coords_R, cap);
        sHM(coords_HWL, cap * 3);
        HM(nxt, cap);
        SETN(head, hash_cap);
        HM(newfound, newfound_cap * 3);
        HM(boundary_newfound, boundary_newfound_cap * 3);
    }
    void cleanup() {
        safefree(head);
        safefree(nxt);
        safefree(coords_R);
        safefree(coords_HWL);
        safefree(newfound);
        safefree(boundary_newfound);
    }
    void add_newfound(int u, int v, int w) {
        int mycnt = newfound_cnt++;
        if (mycnt >= newfound_cap) {
            newfound_cap *= 2;
            HR(newfound, newfound_cap * 3);
        }
        newfound[3 * mycnt] = u;
        newfound[3 * mycnt + 1] = v;
        newfound[3 * mycnt + 2] = w;
    }
    void add_boundary_newfound(int u, int v, int w) {
        int mycnt = boundary_newfound_cnt++;
        if (mycnt >= boundary_newfound_cap) {
            boundary_newfound_cap *= 2;
            HR(boundary_newfound, boundary_newfound_cap * 3);
        }
        boundary_newfound[3 * mycnt] = u;
        boundary_newfound[3 * mycnt + 1] = v;
        boundary_newfound[3 * mycnt + 2] = w;
    }
    void update(
        int u, int v, int w, int l, int trace=0
    ) {
        int hash_value = myhash(myhash(myhash(u) + v) + w) % hash_cap;
        for (int i = head[hash_value]; i != -1; i = nxt[i]) {
            if (coords_HWL[i * 3] == u && coords_HWL[i * 3 + 1] == v && coords_R[i] == w) {
                if (trace == 1 && coords_HWL[i * 3 + 2] != l) {
                    add_newfound(u, v, w);
                } else if (trace == 2 && coords_HWL[i * 3 + 2] != l) {
                    add_boundary_newfound(u, v, w);
                }
                coords_HWL[i * 3 + 2] = l;
                return;
            }
        }
        int mycnt, old_heads;
        if (trace == 1) add_newfound(u, v, w);
        else if (trace == 2) add_boundary_newfound(u, v, w);
        mycnt = cnt++;
        if (mycnt >= cap) {
            cap *= 2;
            sHR(coords_HWL, cap * 3);
            HR(coords_R, cap);
            HR(nxt, cap);
        }
        coords_HWL[mycnt * 3] = u;
        coords_HWL[mycnt * 3 + 1] = v;
        coords_HWL[mycnt * 3 + 2] = l;
        coords_R[mycnt] = w;
        old_heads = head[hash_value];
        head[hash_value] = mycnt;
        nxt[mycnt] = old_heads;
    }

    int query(
        int u, int v, int w
    ) {
        int hash_value = myhash(myhash(myhash(u) + v) + w) % hash_cap;
        for (int i = head[hash_value]; i != -1; i = nxt[i]) {
            if (coords_HWL[i * 3] == u && coords_HWL[i * 3 + 1] == v && coords_R[i] == w) {
                return coords_HWL[i * 3 + 2];
            }
        }
        return 0;
    }

    int get_cnt(int lazy=0) {
        if (lazy) {
            for (int i = 0; i < newfound_cnt; i++) {
                int u = newfound[3 * i], v = newfound[3 * i + 1], w = newfound[3 * i + 2];
                #pragma omp critical
                update(u, v, w, 3);
            }
            return newfound_cnt;
        }
        newfound_cnt = 0;
        #pragma omp parallel for
        for (int i = 0; i < hash_cap; i++) {
            for (int j = head[i]; j != -1; j = nxt[j])
                if (coords_HWL[j * 3 + 2] == 1)
                    #pragma omp critical
                    {
                        add_newfound(coords_HWL[j * 3], coords_HWL[j * 3 + 1], coords_R[j]);
                        coords_HWL[j * 3 + 2] = 3;
                    }
        }
        return newfound_cnt;
    }
}

namespace depth_test {
    int *deepest;
    void init() {
        HM(deepest, specs::H * specs::upscale * specs::W * specs::upscale);
    }
    void cleanup() {
        safefree(deepest);
    }
}

void normalized_cam_to_world(
    double t, double p, double r,
    double *cam,
    double H_fov, double W_fov, double r_min, double r_max,
    double *X
) {
    t = 2 * tan(H_fov / 2) * (-0.5 + t);
    p = 2 * tan(W_fov / 2) * (-0.5 + p);
    r = exp(log(r_min) + (log(r_max) - log(r_min)) * r);
    double x = r / sqrt(1 + t * t + p * p);
    double y = p * x, z = -t * x;
    double tmp = x; x = -y; y = -z; z = tmp;
    X[0] = cam[0] * x + cam[1] * y + cam[2] * z + cam[3];
    X[1] = cam[4] * x + cam[5] * y + cam[6] * z + cam[7];
    X[2] = cam[8] * x + cam[9] * y + cam[10] * z + cam[11];
}


extern "C" {


    void init_and_get_emptytest_queries(
        double *cam, double H_fov, double W_fov, double r_min, double r_max, int H_, int W_, int R_,
        double *positions,
        int test_downscale,
        int upscale
    ) {
        specs::H_fov = H_fov;
        specs::W_fov = W_fov;
        for (int i = 0; i < 12; i++) specs::cam[i] = cam[i];
        specs::r_min = r_min;
        specs::r_max = r_max;
        specs::R = R_;
        specs::H = H_;
        specs::W = W_;
        specs::upscale = upscale;
        coarse_pretest::init(
            (H_ - 1) / test_downscale + 1,
            (W_ - 1) / test_downscale + 1,
            (R_ - 1) / test_downscale + 1,
            test_downscale
        );
        coarse::init();
        {
            using namespace coarse_pretest;
            #pragma omp parallel for
            for (int i = 0; i <= H; i++) {
                int iH = min(i * U, specs::H);
                for (int j = 0; j <= W; j++) {
                    int iW = min(j * U, specs::W);
                    for (int k = 0; k <= R; k++) {
                        int iR = min(k * U, specs::R);
                        int t1 = (W + 1) * (R + 1), t2 = R + 1;
                        normalized_cam_to_world(
                            iH * 1.0 / specs::H, iW * 1.0 / specs::W, iR * 1.0 / specs::R,
                            specs::cam,
                            specs::H_fov, specs::W_fov, specs::r_min, specs::r_max,
                            positions + 3 * (i * t1 + j * t2 + k)
                        );
                    }
                }
            }
        }
    }

    int initial_update(
        double *sdf
    ) {
        using namespace coarse_pretest;
        cnt = 0;
        #pragma omp parallel for
        for (int i = 0; i < H; i++)
        for (int j = 0; j < W; j++)
        for (int k = 0; k < R; k++) {
            bool positive = 0, negative = 0, zero = 0;
            for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++) {
                int t1 = (W + 1) * (R + 1), t2 = R + 1;
                double sdf_c = sdf[(i + ii) * t1 + (j + jj) * t2 + k + kk];
                positive |= sdf_c > 0;
                negative |= sdf_c < 0;
                zero |= sdf_c == 0;
            }
            int t1 = W * R, t2 = R;
            if ((flag[i * t1 + j * t2 + k] = has_iso_surface(positive, negative, zero))) {
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
        }
        return cnt;
    }

    void get_coarse_queries(
        double *positions,
        int *position_bounds
    ) {
        using namespace coarse_pretest;
        #pragma omp parallel for
        for (int c = 0; c < cnt; c++) {
            int i = newfound[c * 3], j = newfound[c * 3 + 1], k = newfound[c * 3 + 2];
            for (int ii = 0; ii < S && i * U + ii <= specs::H; ii++)
            for (int jj = 0; jj < S && j * U + jj <= specs::W; jj++)
            for (int kk = 0; kk < S && k * U + kk <= specs::R; kk++) {
                int iH = i * U + ii;
                int iW = j * U + jj;
                int iR = k * U + kk;
                normalized_cam_to_world(
                    iH * 1.0 / specs::H, iW * 1.0 / specs::W, iR * 1.0 / specs::R,
                    specs::cam,
                    specs::H_fov, specs::W_fov, specs::r_min, specs::r_max,
                    positions + 3 * (c * S*S*S + ii*S*S + jj*S + kk)
                );
            }
            position_bounds[c * 3] = min(S - 1, specs::H - i * U);
            position_bounds[c * 3 + 1] = min(S - 1, specs::W - j * U);
            position_bounds[c * 3 + 2] = min(S - 1, specs::R - k * U);
        }
    }

    // 1 last step
    // 2 new created
    // 3 ever explored

    void update(
        int c,
        double *sdf,
        int *position_bounds,
        int *verts_int,
        double *verts_frac, int N,
        int *faces, int M
    ) {
        using namespace coarse_pretest;
        int boundary_x_min[6] = {0, 0, 0, 0, 0, U};
        int boundary_x_max[6] = {U, U, U, U, 0, U};
        int boundary_y_min[6] = {0, 0, 0, U, 0, 0};
        int boundary_y_max[6] = {U, U, 0, U, U, U};
        int boundary_z_min[6] = {0, U, 0, 0, 0, 0};
        int boundary_z_max[6] = {0, U, U, U, U, U};
        int dx[6] = {0, 0, 0, 0, -1, 1};
        int dy[6] = {0, 0, -1, 1, 0, 0};
        int dz[6] = {-1, 1, 0, 0, 0, 0};

        int t1 = W * R, t2 = R;
        int i = newfound[c * 3], j = newfound[c * 3 + 1], k = newfound[c * 3 + 2];
        flag[i * t1 + j * t2 + k] = 3;
        for (int b = 0; b < 6; b++) {
            int ni = i + dx[b], nj = j + dy[b], nk = k + dz[b];
            if (!(ni >= 0 && nj >= 0 && nk >= 0 && ni < H && nj < W && nk < R)) continue;
            if (flag[ni * t1 + nj * t2 + nk] != 0) continue;
            bool positive = 0, negative = 0, zero = 0;
            int S1 = position_bounds[c * 3], S2 = position_bounds[c * 3 + 1], S3 = position_bounds[c * 3 + 2];
            for (int ii = min(S1, boundary_x_min[b]); ii <= min(S1, boundary_x_max[b]) && (!(positive && negative)); ii++)
            for (int jj = min(S2, boundary_y_min[b]); jj <= min(S2, boundary_y_max[b]) && (!(positive && negative)); jj++)
            for (int kk = min(S3, boundary_z_min[b]); kk <= min(S3, boundary_z_max[b]) && (!(positive && negative)); kk++) {
                double sdf_c = sdf[c*S*S*S + ii*S*S + jj*S + kk];
                positive |= sdf_c > 0;
                negative |= sdf_c < 0;
                zero |= sdf_c == 0;
            }
            if (has_iso_surface(positive, negative, zero)) {
                flag[ni * t1 + nj * t2 + nk] = 2;
            }
        }
        while (coarse::M_cap <= coarse::M + M) {
            coarse::M_cap *= 2;
            HR(coarse::faces, coarse::M_cap * 3);
        }
        // dont omp
        for (int i = 0; i < 3 * M; i++)
            coarse::faces[3 * coarse::M + i] = faces[i] + coarse::N;
        while (coarse::N_cap <= coarse::N + N) {
            coarse::N_cap *= 2;
            HR(coarse::lr_vertices, coarse::N_cap * 6);
        }
        double *current_sdf = sdf + c * S * S * S;
        t1 = S * S; t2 = S;

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
                coarse::lr_vertices[6 * coarse::N + i * 6 + 2 * j + low] = verts_int[i * 3 + j] + int_floor(verts_frac[i * 3 + j]) + newfound[c * 3 + j] * U;
                coarse::lr_vertices[6 * coarse::N + i * 6 + 2 * j + high] = verts_int[i * 3 + j] + int_ceil(verts_frac[i * 3 + j]) + newfound[c * 3 + j] * U;
            }
        }
        coarse::N += N;
        coarse::M += M;
    }

    int get_cnt() {
        using namespace coarse_pretest;
        int t1 = W * R, t2 = R;
        cnt = 0;
        #pragma omp parallel for
        for (int i = 0; i < H; i++)
        for (int j = 0; j < W; j++)
        for (int k = 0; k < R; k++)
            if (flag[i * t1 + j * t2 + k] == 2) {
                flag[i * t1 + j * t2 + k] = 1;
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
        coarse_pretest::cleanup();
        using namespace coarse;
        merge_verts(
            lr_vertices, N, 6,
            faces, M
        );
        NM[0] = N;
        NM[1] = M;
        fHM(bis_vertices, N * 6);
        for (int i = 0; i < 6 * N; i++) {
            bis_vertices[i] = (double)lr_vertices[i];
        }
        safefree(lr_vertices);
    }


    void bisection_get_positions(
        int is_fine,
        double *positions
    ) {
        int N, H, W, R;
        double *identifiers;
        if (is_fine == -1) {
            N = coarse::N;
            H = specs::H; W = specs::W; R = specs::R;
            identifiers = coarse::bis_vertices;
        }
        else {
            N = fine::N;
            H = specs::H * specs::upscale; W = specs::W * specs::upscale; R = specs::R * specs::upscale;
            identifiers = fine::bis_vertices;
        }
        int cnt = 0;
        for (int i = 0; i < N; i++) {
            if (is_fine != -1 && fine::ecat[i] != is_fine) continue;
            double iH = (identifiers[i * 6] + identifiers[i * 6 + 1]) / 2;
            double iW = (identifiers[i * 6 + 2] + identifiers[i * 6 + 3]) / 2;
            double iR = (identifiers[i * 6 + 4] + identifiers[i * 6 + 5]) / 2;
            normalized_cam_to_world(
                iH / H, iW / W, iR / R,
                specs::cam,
                specs::H_fov, specs::W_fov, specs::r_min, specs::r_max,
                positions + 3 * cnt++
            );
        }
    }

    void bisection_update(
        int is_fine,
        double *sdfs
    ) {
        int N;
        double *identifiers;
        if (is_fine == -1) {
            N = coarse::N;
            identifiers = coarse::bis_vertices;
        }
        else {
            N = fine::N;
            identifiers = fine::bis_vertices;
        }
        int cnt = 0;
        for (int i = 0; i < N; i++) {
            if (is_fine != -1 && fine::ecat[i] != is_fine) continue;
            for (int j = 0; j < 3; j++) {
                double middle = (identifiers[i * 6 + 2 * j] + identifiers[i * 6 + 2 * j + 1]) / 2;
                if (sdfs[cnt] > 0) identifiers[i * 6 + 2 * j + 1] = middle;
                else identifiers[i * 6 + 2 * j] = middle;
            }
            cnt++;
        }
    }

    // 1 last time
    // 2 newly created
    // 3 ever explored
    // 4 not visible

    int init_fine(int relax_iters, int ne) {
        coarse::finalize();
        fine_pretest::init();
        change_map::init();
        fine::ne = ne;
        initialize_visibility_engine(specs::H, specs::W, specs::R);
        HM(coarse::visibility, coarse::M);
        SET0(coarse::visibility, coarse::M);
        HM(face_map::backface, coarse::M);
        SET0(face_map::backface, coarse::M);
        visibility_engine_update(coarse::vertices, coarse::visibility, coarse::faces, 0, coarse::M, 0);
        safefree(face_map::backface);
        face_map::backface = NULL;
        coarse_face_map::init();
        extend::expand_visibility(coarse::faces, coarse::M, coarse::visibility, relax_iters);
        cleanup_visibility_engine();
        initialize_visibility_engine(specs::H * specs::upscale, specs::W * specs::upscale, specs::R * specs::upscale);
        // don't omp
        for (int it = 0; it < specs::H; it++)
        for (int ip = 0; ip < specs::W; ip++) {
            for (int j = coarse_face_map::head[it * specs::W + ip]; j != -1; j = coarse_face_map::nxt[j]) {
                int ir = min(specs::R - 1, coarse_face_map::depth[j] / 6);
                if (coarse::visibility[coarse_face_map::id[j]])
                    fine_pretest::update(it, ip, ir, 1);
                else if (fine_pretest::query(it, ip, ir) == 0)
                    fine_pretest::update(it, ip, ir, 4);
            }
        }
        safefree(coarse::visibility);
        fine::init();
        return fine_pretest::get_cnt();
    }



    void get_fine_queries(
        double *positions
    ) {
        using namespace fine_pretest;
        change_map::clear();

        #pragma omp parallel for
        for (int c = 0; c < newfound_cnt; c++) {
            int h = newfound[c * 3], w = newfound[c * 3 + 1], r = newfound[c * 3 + 2];
            for (int ii = 0; ii <= U; ii++)
            for (int jj = 0; jj <= U; jj++)
            for (int kk = 0; kk <= U; kk++) {
                normalized_cam_to_world(
                    (h + 1.0 * ii / U) / specs::H,
                    (w + 1.0 * jj / U) / specs::W,
                    (r + 1.0 * kk / U) / specs::R,
                    specs::cam,
                    specs::H_fov, specs::W_fov, specs::r_min, specs::r_max,
                    positions + c * S*S*S * 3 + (ii*S*S + jj*S + kk) * 3
                );
            }
            #pragma omp critical
            change_map::update(h, w);
        }
        boundary_newfound_cnt = 0;

    }


    void update_fine_small(
        int c, int e,
        double *sdf,
        int *verts_int,
        double *verts_frac, int N,
        int *faces, int M
    ) {
        using namespace fine_pretest;
        int boundary_x_min[6] = {0, 0, 0, 0, 0, U};
        int boundary_x_max[6] = {U, U, U, U, 0, U};
        int boundary_y_min[6] = {0, 0, 0, U, 0, 0};
        int boundary_y_max[6] = {U, U, 0, U, U, U};
        int boundary_z_min[6] = {0, U, 0, 0, 0, 0};
        int boundary_z_max[6] = {0, U, U, U, U, U};
        int dx[6] = {0, 0, 0, 0, -1, 1};
        int dy[6] = {0, 0, -1, 1, 0, 0};
        int dz[6] = {-1, 1, 0, 0, 0, 0};

        int h = newfound[c * 3], w = newfound[c * 3 + 1], r = newfound[c * 3 + 2];
        int current_news[3] = {h, w, r};
        for (int b = 0; b < 6; b++) {
            bool positive = 0, negative = 0, zero = 0;
            for (int ii = boundary_x_min[b]; ii <= boundary_x_max[b] && (!(positive && negative)); ii++)
            for (int jj = boundary_y_min[b]; jj <= boundary_y_max[b] && (!(positive && negative)); jj++)
            for (int kk = boundary_z_min[b]; kk <= boundary_z_max[b] && (!(positive && negative)); kk++) {
                double sdf_c = sdf[(c*S*S*S + ii*S*S + jj*S + kk) * fine::ne + e];
                positive |= sdf_c > 0;
                negative |= sdf_c < 0;
                zero |= sdf_c == 0;
            }
            if (has_iso_surface(positive, negative, zero)) {
                int ni = h + dx[b], nj = w + dy[b], nk = r + dz[b];
                if (ni >= 0 && nj >= 0 && nk >= 0 && ni < specs::H && nj < specs::W && nk < specs::R) {
                    int flag = query(ni, nj, nk);
                    if (flag == 0) {
                        update(ni, nj, nk, 2, 2);
                    }
                }
            }
        }

        while (fine::M_cap <= fine::M + M) {
            fine::M_cap *= 2;
            HR(fine::faces, fine::M_cap * 3);
            HR(fine::visibility, fine::M_cap);
            SET0(fine::visibility + fine::M_cap / 2, fine::M_cap / 2);
        }
        // dont omp
        for (int i = 0; i < 3 * M; i++)
            fine::faces[3 * fine::M + i] = faces[i] + fine::N;
        while (fine::N_cap <= fine::N + N) {
            fine::N_cap *= 2;
            HR(fine::lr_vertices, fine::N_cap * 7);
            fHR(fine::vertices, fine::N_cap * 3);
        }
        double *current_sdf = sdf + c * S * S * S * fine::ne;
        int t1 = S * S, t2 = S;

        // dont omp
        for (int i = 0; i < N; i++) {
            int low = 0, high = 1;
            double sdf_floor = current_sdf[fine::ne * (
                (verts_int[i * 3] + int_floor(verts_frac[i * 3])) * t1 \
                + (verts_int[i * 3 + 1] + int_floor(verts_frac[i * 3 + 1])) * t2 \
                + (verts_int[i * 3 + 2] + int_floor(verts_frac[i * 3 + 2]))
            ) + e];
            double sdf_ceil = current_sdf[fine::ne * (
                (verts_int[i * 3] + int_ceil(verts_frac[i * 3])) * t1 \
                + (verts_int[i * 3 + 1] + int_ceil(verts_frac[i * 3 + 1])) * t2 \
                + (verts_int[i * 3 + 2] + int_ceil(verts_frac[i * 3 + 2]))
            ) + e];
            if (sdf_floor > sdf_ceil) {
                high = 0; low = 1;
            }
            fine::lr_vertices[7 * fine::N + i * 7 + 6] = e;
            for (int j = 0; j < 3; j++) {
                fine::lr_vertices[7 * fine::N + i * 7 + 2 * j + low] = verts_int[i * 3 + j] + int_floor(verts_frac[i * 3 + j]) + current_news[j] * U;
                fine::lr_vertices[7 * fine::N + i * 7 + 2 * j + high] = verts_int[i * 3 + j] + int_ceil(verts_frac[i * 3 + j]) + current_news[j] * U;
                fine::vertices[3 * fine::N + i * 3 + j] = verts_int[i * 3 + j] + current_news[j] * U + verts_frac[i * 3 + j];
            }
        }
        fine::N += N;
        fine::M += M;
    }

    int update_fine() {
        visibility_engine_update(fine::vertices, fine::visibility, fine::faces, fine::updated_M, fine::M);
        fine::updated_M = fine::M;
        using namespace fine_pretest;
        newfound_cnt = 0;
        for (int i = 0; i < boundary_newfound_cnt; i++) 
        {
            int h = boundary_newfound[i * 3], w = boundary_newfound[i * 3 + 1], r = boundary_newfound[i * 3 + 2];
            int h_min = h * specs::upscale, h_max = h_min + specs::upscale;
            int w_min = w * specs::upscale, w_max = w_min + specs::upscale;
            bool v = visibility_engine_block_query(h_min, h_max, w_min, w_max,  r * specs::upscale * 6);
            update(h, w, r, v, v);
        }
        #pragma omp parallel for
        for (int i = 0; i < change_map::cnt; i++) {
            int it = change_map::new_changes[i * 2], ip = change_map::new_changes[i * 2 + 1];
            for (int j = coarse_face_map::head[it * specs::W + ip]; j != -1; j = coarse_face_map::nxt[j]) {
                int ir = min(specs::R - 1, coarse_face_map::depth[j] / 6);
                if (query(it, ip, ir) != 4) {
                    continue;
                }
                bool flag = 0;
                for (int ii = 0; ii < specs::upscale && !flag; ii++)
                for (int jj = 0; jj < specs::upscale && !flag; jj++) {
                    if (convex_map::super_head[(it * specs::upscale + ii) * face_map::W + (ip * specs::upscale + jj)] != -1) {
                        flag = 1;
                        break;
                    }
                }
                if (!flag) flag = update_coarse_visibility(
                    coarse::vertices,
                    coarse::faces + coarse_face_map::id[j] * 3,
                    specs::upscale, it * specs::upscale, (it+1) * specs::upscale, ip * specs::upscale, (ip+1) * specs::upscale
                );
                if (flag) {
                    change_map::change_depth[i] = ir;
                    break;
                }
            }
        }
        #pragma omp parallel for
        for (int i = 0; i < change_map::cnt; i++) {
            if (change_map::change_depth[i] != -1) {
                #pragma omp critical
                update(change_map::new_changes[2 * i], change_map::new_changes[2 * i + 1], change_map::change_depth[i], 1, 1);
            }
        }
        return fine_pretest::get_cnt(1);
    }

    int complete_depth_test_get_query_cnt(int relax, int batch) {
        if (batch == 0) {
            using namespace convex_map;
            for (int idx = 0; idx < H * W; idx++) {
                int i = face_map::head[idx];
                if (i != -1) {
                    int f = face_map::id[i];
                    assert(fine::visibility[f]);
                }
            }
            depth_test::init();
            deepest_query(depth_test::deepest);
        }
        int s = 0;
        for (int i = 0; i < face_map::H; i+= relax)
            if (depth_test::deepest[i * face_map::W + batch] > 0) {
                s += (depth_test::deepest[i * face_map::W + batch] - 1) / relax + 1;
            }
        return s;
    }

    void complete_depth_test_get_queries(int relax, int batch, double *positions) {
        int cnt = 0;
        for (int i = 0; i < face_map::H; i+= relax)
            for (int k = 0; k < depth_test::deepest[i * face_map::W + batch]; k += relax) {
                normalized_cam_to_world(
                    i * 1.0 / face_map::H, batch * 1.0 / face_map::W, k * 1.0 / depth_table::R,
                    specs::cam,
                    specs::H_fov, specs::W_fov, specs::r_min, specs::r_max,
                    positions + 3 * cnt++
                );
            }
    }

    void complete_depth_test_update(int relax, int batch, double *sdf) {
        int cnt = 0;
        for (int i = 0; i < face_map::H; i += relax)
            for (int k = 0; k < depth_test::deepest[i * face_map::W + batch]; k += relax) {
                if (sdf[cnt] < 0) {
                    if (fine_pretest::query(i / specs::upscale, batch / specs::upscale, k / specs::upscale) == 0)
                        fine_pretest::update(i / specs::upscale, batch / specs::upscale, k / specs::upscale, 1, 1);
                }
                cnt++;
            }
    }

    int complete_depth_test_get_cnt() {
        return fine_pretest::get_cnt(1);
    }


    void get_final_mesh_statistics(int *NM, int *Ns, int *Ms) {
        cleanup_visibility_engine();
        change_map::cleanup();
        coarse_face_map::cleanup();
        safefree(fine::visibility);
        {
            using namespace coarse;
            int cnt = 0;
            for (int i = 0; i < M; i++) {
                double t = 0, p = 0, r = 0;
                for (int j = 0; j < 3; j++) {
                    t += vertices[faces[i * 3 + j] * 3];
                    p += vertices[faces[i * 3 + j] * 3 + 1];
                    r += vertices[faces[i * 3 + j] * 3 + 2];
                }
                t /= 3; p /= 3; r /= 3;
                int it = min(specs::H - 1, int_floor(t));
                int ip = min(specs::W - 1, int_floor(p));
                int ir = min(specs::R - 1, int_floor(r));
                if (fine_pretest::query(it, ip, ir) == 4) {
                    memcpy(faces + cnt++ * 3, faces + i * 3, 3 * sizeof(int));
                }
            }
            M = cnt;
            clean_faces(N, faces, M);
            clean_verts_double(vertices, N, faces, M);
            NM[0] = N;
            NM[1] = M;
        }
        fine_pretest::cleanup();
        {
            using namespace fine;
            merge_verts(
                lr_vertices, N, 7,
                faces, M
            );
            clean_faces(N, faces, M);
            clean_verts(lr_vertices, 7, N, faces, M);
            fHM(bis_vertices, N * 6);
            sHM(ecat, N);
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < 6; j++) {
                    bis_vertices[i * 6 + j] = (double)lr_vertices[i * 7 + j];
                }
                Ns[ecat[i] = lr_vertices[i * 7 + 6]]++;
            }
            safefree(lr_vertices);
            for (int i = 0; i < M; i++) {
                Ms[ecat[faces[i * 3]]]++;
            }
        }
    }

    void get_final_mesh(double *vertices, int *faces) {
        fine::finalize();
        for (int i = 0; i < coarse::N; i++) {
            normalized_cam_to_world(
                coarse::vertices[i * 3] / specs::H, coarse::vertices[i * 3 + 1] / specs::W, coarse::vertices[i * 3 + 2] / specs::R,
                specs::cam,
                specs::H_fov, specs::W_fov, specs::r_min, specs::r_max,
                vertices + 3 * i
            );
        }
        memcpy(faces, coarse::faces, 3 * coarse::M * sizeof(int));
        coarse::cleanup();
        int *current_order;
        HM(current_order, fine::N);
        for (int i = 0; i < fine::N; i++) {
            current_order[i] = i;
        }
        int *N_ac;
        HM(N_ac, fine::ne);
        N_ac[0] = 0;
        int current_cnt = 0;
        for (int e = 0; e < fine::ne - 1; e++) {
            int i = current_cnt, j = fine::N - 1;
            while (i <= j) {
                while (i <= j && fine::ecat[current_order[i]] == e) i++;
                while (i <= j && fine::ecat[current_order[j]] != e) j--;
                if (i < j) {
                    swap(current_order[i], current_order[j]);
                    i++; j--;
                }
            }
            N_ac[e + 1] = current_cnt = i;
        }
        for (int i = 0; i < fine::N; i++) {
            int ii = current_order[i];
            normalized_cam_to_world(
                fine::vertices[ii * 3] / specs::upscale / specs::H,
                fine::vertices[ii * 3 + 1] / specs::upscale / specs::W,
                fine::vertices[ii * 3 + 2] / specs::upscale / specs::R,
                specs::cam,
                specs::H_fov, specs::W_fov, specs::r_min, specs::r_max,
                vertices + 3 * (coarse::N + i)
            );
        }
        safefree(fine::vertices);
        int *inverse_order;
        HM(inverse_order, fine::N);
        for (int i = 0; i < fine::N; i++) {
            inverse_order[current_order[i]] = i;
        }
        safefree(current_order);
        int cnt = 0;
        for (int e = 0; e < fine::ne; e++) {
            for (int i = 0; i < fine::M; i++) {
                int cat = fine::ecat[fine::faces[i * 3]];
                if (cat == e) {
                    for (int j = 0; j < 3; j++) {
                        faces[3 * (coarse::M + cnt) + j] = inverse_order[fine::faces[i * 3 + j]] - N_ac[e];
                    }
                    cnt++;
                }
            }
        }
        fine::cleanup();
        safefree(N_ac);
    }

}