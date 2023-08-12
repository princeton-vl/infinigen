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
using namespace std;

namespace specs {
    double cam[12], H_fov, W_fov, r_min, r_max;
    int L, R, upscale, N0, N1;
}

namespace finefront {
    int H, W, U, newfound_cap, cnt, R, S;
    int *newfound;
    int *all_identifiers=NULL;
    short *flag;
    void init() {
        R = specs::R;
        using namespace specs;
        U = upscale;
        S = U + 1;
        H = (L - 2 * N0);
        W = (L - 2 * N1);
        sHM(flag, H * W * R);
        sSET0(flag, H * W * R);
        newfound_cap = 10000;
        cnt = 0;
        HM(newfound, newfound_cap * 3);
    }
    void cleanup() {
        safefree(newfound);
        safefree(flag);
    }
}

namespace convention {
    int L, R;
    int vertices[24];
    const int faces[24] = {0, 1, 2, 3, 5, 4, 7, 6, 0, 2, 4, 6, 1, 5, 3, 7, 0, 4, 1, 5, 2, 3, 6, 7};
    int center[6][3], directions[6][2][3];

    void init() {
        for (int i = 0; i < 8; i++) {
            vertices[i * 3] = ((i >> 2) & 1) * 4 - 2;
            vertices[i * 3 + 1] = ((i >> 1) & 1) * 4 - 2;
            vertices[i * 3 + 2] = ((i) & 1) * 4 - 2;
        }
        for (int i = 0; i < 6; i++) {
            center[i][0] = center[i][1] = center[i][2] = 0;
            for (int j = 0; j < 4; j++)
                for (int k = 0; k < 3; k++)
                    center[i][k] += vertices[faces[i * 4 + j] * 3 + k];
            for (int k = 0; k < 3; k++) center[i][k] /= 4;
            for (int k = 0; k < 3; k++) {
                directions[i][0][k] = (vertices[faces[i * 4 + 1] * 3 + k] - vertices[faces[i * 4] * 3 + k]) / 4;
                directions[i][1][k] = (vertices[faces[i * 4 + 2] * 3 + k] - vertices[faces[i * 4] * 3 + k]) / 4;
            }
        }
    }
    bool move(
        int q, int i, int j, int k, int di, int dj, int dk,
        int &nq, int &ni, int &nj, int &nk
    ) {
        if (dk != 0) {
            if (dk == 1 && k >= R - 1 || dk == -1 && k == 0) return 0;
            nq = q; ni = i; nj = j; nk = k + dk;
            return 1;
        }
        nk = k;
        if (di != 0) {
            if (i != 0 && di == -1) {
                nq = q; ni = i - 1; nj = j;
                return 1;
            }
            if (i != L - 1 && di == 1) {
                nq = q; ni = i + 1; nj = j;
                return 1;
            }
            int p[3] = {center[q][0], center[q][1], center[q][2]};
            for (int a = 0; a < 3; a++) p[a] += directions[q][0][a] * di * 2;
            for (int a = 0; a < 3; a++) p[a] += directions[q][1][a];
            for (nq = 0; nq < 6; nq++) {
                if (q == nq) continue;
                for (int dni = -2; dni <= 2; dni++)
                for (int dnj = -2; dnj <= 2; dnj++) {
                    if (dni > -2 && dni < 2 && dnj > -2 && dnj < 2) continue;
                    int np[3] = {center[nq][0], center[nq][1], center[nq][2]};
                    for (int a = 0; a < 3; a++) np[a] += directions[nq][0][a] * dni;
                    for (int a = 0; a < 3; a++) np[a] += directions[nq][1][a] * dnj;
                    if (np[0] == p[0] && p[1] == np[1] && p[2] == np[2]) {
                        if (dni == 2 || dni == -2) {
                            if (dni == -2) ni = 0;
                            else ni = L - 1;
                            if (dnj == 1) nj = j; else nj = L - 1 - j;
                            return 1;
                        }
                        else {
                            if (dnj == -2) nj = 0;
                            else nj = L - 1;
                            if (dni == 1) ni = j; else ni = L - 1 - j;
                            return 1;
                        }
                    }
                }
            }
        }
        if (j != 0 && dj == -1) {
            nq = q; ni = i; nj = j - 1;
            return 1;
        }
        if (j != L - 1 && dj == 1) {
            nq = q; ni = i; nj = j + 1;
            return 1;
        }
        int p[3] = {center[q][0], center[q][1], center[q][2]};
        for (int a = 0; a < 3; a++) p[a] += directions[q][0][a];
        for (int a = 0; a < 3; a++) p[a] += directions[q][1][a] * dj * 2;
        bool flag = 0;
        for (nq = 0; nq < 6; nq++) {
            if (q == nq) continue;
            for (int dni = -2; dni <= 2; dni++) {
                for (int dnj = -2; dnj <= 2; dnj++) {
                    if (dni > -2 && dni < 2 && dnj > -2 && dnj < 2) continue;
                    int np[3] = {center[nq][0], center[nq][1], center[nq][2]};
                    for (int a = 0; a < 3; a++) np[a] += directions[nq][0][a] * dni;
                    for (int a = 0; a < 3; a++) np[a] += directions[nq][1][a] * dnj;
                    if (np[0] == p[0] && p[1] == np[1] && p[2] == np[2]) {
                        if (dni == 2 || dni == -2) {
                            if (dni == -2) ni = 0;
                            else ni = L - 1;
                            if (dnj == 1) nj = i; else nj = L - 1 - i;
                        }
                        else {
                            if (dnj == -2) nj = 0;
                            else nj = L - 1;
                            if (dni == 1) ni = i; else ni = L - 1 - i;
                        }
                        flag = 1;
                        break;
                    }
                }
                if (flag) break;
            }
            if (flag) break;
        }
        return 1;
    }
}

namespace cubespherical_mesh {
    int *faces, *lr_vertices;
    short *qs;
    double *bis_vertices, *vertices;
    int N, M;
    int N_cap, M_cap;
    void init() {
        N_cap = M_cap = 10000;
        N = M = 0;
        HM(faces, M_cap * 3);
        HM(lr_vertices, N_cap * 7);
    }
    
    void finalize() {
        fHM(vertices, N * 3);
        for (int i = 0; i < N * 3; i++) {
            vertices[i] = (bis_vertices[i * 2] + bis_vertices[i * 2 + 1]) / 2;
        }
        safefree(bis_vertices);
    }
    void cleanup() {
        safefree(qs);
        safefree(faces);
        safefree(vertices);
    }
}

namespace pretest {
    int L, R, U, S, cnt, *newfound;
    short *flag;
    int newfound_cap;

    void init(
        int L_, int R_, int U_
    ) {
        L = L_; R = R_; U = U_; S = U + 1;
        newfound_cap = 10000;
        cnt = 0;
        sHM(flag, 6 * L * L * R);
        HM(newfound, newfound_cap * 4);
    }

    void cleanup() {
        safefree(flag);
        safefree(newfound);
    }
}



void normalized_cam_to_world(
    int q,
    double x1, double x2, double r, 
    double *cam,
    double r_min, double r_max,
    double *X
) {
    double x, y, z;
    if (q == 0) {
        x = -1;
        y = tan(-acos(-1) / 4 + acos(-1) / 2 * x2);
        z = tan(-acos(-1) / 4 + acos(-1) / 2 * x1);
    }
    else if (q == 1) {
        // q = 1 is camera q
        x = 1;
        y = tan(-acos(-1) / 4 + acos(-1) / 2 * x2);
        z = tan(acos(-1) / 4 - acos(-1) / 2 * x1);
    }
    else if (q == 2) {
        x = tan(-acos(-1) / 4 + acos(-1) / 2 * x2);
        y = tan(-acos(-1) / 4 + acos(-1) / 2 * x1);
        z = -1;
    }
    else if (q == 3) {
        x = tan(-acos(-1) / 4 + acos(-1) / 2 * x1);
        y = tan(-acos(-1) / 4 + acos(-1) / 2 * x2);
        z = 1;
    }
    else if (q == 4) {
        x = tan(-acos(-1) / 4 + acos(-1) / 2 * x1);
        y = -1;
        z = tan(-acos(-1) / 4 + acos(-1) / 2 * x2);
    }
    else if (q == 5){
        x = tan(-acos(-1) / 4 + acos(-1) / 2 * x2);
        y = 1;
        z = tan(-acos(-1) / 4 + acos(-1) / 2 * x1);
    }
    r = exp(log(r_min) + (log(r_max) - log(r_min)) * r);
    double s = r / sqrt(x * x + y * y + z * z);
    x *= s; y *= s; z *= s;
    double tmp = x; x = -y; y = -z; z = tmp;
    X[0] = cam[0] * x + cam[1] * y + cam[2] * z + cam[3];
    X[1] = cam[4] * x + cam[5] * y + cam[6] * z + cam[7];
    X[2] = cam[8] * x + cam[9] * y + cam[10] * z + cam[11];
}


void multicase_cam_to_world(
    int q,
    double x1, double x2, double r, 
    double *cam,
    double r_min, double r_max,
    double *X
) {
    if (q >= 0 && q <= 5) normalized_cam_to_world(q, 1.0 * x1 / specs::L, 1.0 * x2 / specs::L, 1.0 * r / specs::R, cam, r_min, r_max, X);
    else if (q == 6) {
        normalized_cam_to_world(1,
            (specs::N0 + 1.0 * x1 / finefront::U) / specs::L,
            (specs::N1 + 1.0 * x2 / finefront::U) / specs::L,
            1.0 * r /finefront::U / specs::R, cam, r_min, r_max, X
        );
    }
    else assert(0);
}

extern "C" {

    void init_and_get_emptytest_queries(
        double *cam, double r_min, double r_max, int L_, int R_,
        double *positions,
        int test_downscale, double H_fov, double W_fov, int upscale,
        int N0, int N1
    ) {
        specs::H_fov = H_fov;
        specs::W_fov = W_fov;
        specs::N0 = N0;
        specs::N1 = N1;
        for (int i = 0; i < 12; i++) specs::cam[i] = cam[i];
        specs::r_min = r_min;
        specs::r_max = r_max;
        specs::R = R_;
        specs::L = L_;
        specs::upscale = upscale;
        convention::init();
        cubespherical_mesh::init();
        pretest::init((L_ - 1) / test_downscale + 1, (R_ - 1) / test_downscale + 1, test_downscale);
        {
            using namespace pretest;
            #pragma omp parallel for
            for (int q = 0; q < 6; q++) {
                for (int i = 0; i <= L; i++) {
                    int iH = min(i * U, specs::L);
                    for (int j = 0; j <= L; j++) {
                        int iW = min(j * U, specs::L);
                        for (int k = 0; k <= R; k++) {
                            int iR = min(k * U, specs::R);
                            int t0 = (L + 1) * (L + 1) * (R + 1), t1 = (L + 1) * (R + 1), t2 = R + 1;
                            normalized_cam_to_world(
                                q,
                                iH * 1.0 / specs::L, iW * 1.0 / specs::L, iR * 1.0 / specs::R,
                                specs::cam,
                                specs::r_min, specs::r_max,
                                positions + 3 * (q * t0 + i * t1 + j * t2 + k)
                            );
                        }
                    }
                }
            }
        }

    }
    
    int initial_update(
        double *sdf
    ) {
        using namespace pretest;
        cnt = 0;
        #pragma omp parallel for
        for (int q = 0; q < 6; q++)
        for (int i = 0; i < L; i++)
        for (int j = 0; j < L; j++)
        for (int k = 0; k < R; k++) {
            bool positive = 0, negative = 0, zero = 0;
            for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++) {
                int t2 = R + 1, t1 = (L + 1) * t2, t0 = t1 * (L + 1);
                double sdf_c = sdf[q * t0 + (i + ii) * t1 + (j + jj) * t2 + k + kk];
                positive |= sdf_c > 0;
                negative |= sdf_c < 0;
                zero |= sdf_c == 0;
            }
            int t0 = L * L * R, t1 = L * R, t2 = R;
            if ((flag[q * t0 + i * t1 + j * t2 + k] = has_iso_surface(positive, negative, zero))) {
                #pragma omp critical
                {
                    int mycnt = cnt++;
                    if (mycnt >= newfound_cap) {
                        newfound_cap *= 2;
                        HR(newfound, newfound_cap * 4);
                    }
                    newfound[4 * mycnt] = q;
                    newfound[4 * mycnt + 1] = i;
                    newfound[4 * mycnt + 2] = j;
                    newfound[4 * mycnt + 3] = k;
                }
            }
        }
        if (cnt == 0) {
            pretest::cleanup();
        }
        return cnt;
    }

    void get_coarse_queries(
        double *positions,
        int *position_bounds
    ) {
        using namespace pretest;
        #pragma omp parallel for
        for (int c = 0; c < cnt; c++) {
            int q = newfound[c * 4], i = newfound[c * 4 + 1], j = newfound[c * 4 + 2], k = newfound[c * 4 + 3];
            for (int ii = 0; ii < S && i * U + ii <= specs::L; ii++)
            for (int jj = 0; jj < S && j * U + jj <= specs::L; jj++)
            for (int kk = 0; kk < S && k * U + kk <= specs::R; kk++) {
                int iH = i * U + ii;
                int iW = j * U + jj;
                int iR = k * U + kk;
                normalized_cam_to_world(
                    q,
                    iH * 1.0 / specs::L, iW * 1.0 / specs::L, iR * 1.0 / specs::R,
                    specs::cam,
                    specs::r_min, specs::r_max,
                    positions + 3 * (c * S*S*S + ii*S*S + jj*S + kk)
                );
            }
            position_bounds[c * 3] = min(S - 1, specs::L - i * U);
            position_bounds[c * 3 + 1] = min(S - 1, specs::L - j * U);
            position_bounds[c * 3 + 2] = min(S - 1, specs::R - k * U);
        }
    }

    void update(
        int c,
        double *sdf,
        int *position_bounds,
        double *verts, int N,
        int *faces, int M
    ) {
        using namespace pretest;
        int boundary_x_min[6] = {0, 0, 0, 0, 0, U};
        int boundary_x_max[6] = {U, U, U, U, 0, U};
        int boundary_y_min[6] = {0, 0, 0, U, 0, 0};
        int boundary_y_max[6] = {U, U, 0, U, U, U};
        int boundary_z_min[6] = {0, U, 0, 0, 0, 0};
        int boundary_z_max[6] = {0, U, U, U, U, U};
        int dx[6] = {0, 0, 0, 0, -1, 1};
        int dy[6] = {0, 0, -1, 1, 0, 0};
        int dz[6] = {-1, 1, 0, 0, 0, 0};
        convention::L = L;
        convention::R = R;

        int t0 = L * L * R, t1 = L * R, t2 = R;
        int q = newfound[c * 4], i = newfound[c * 4 + 1], j = newfound[c * 4 + 2], k = newfound[c * 4 + 3];
        flag[q * t0 + i * t1 + j * t2 + k] = 3;
        for (int b = 0; b < 6; b++) {
            int nq, ni, nj, nk;
            bool valid = convention::move(q, i, j, k, dx[b], dy[b], dz[b], nq, ni, nj, nk);
            if (!valid) continue;
            if (flag[nq * t0 + ni * t1 + nj * t2 + nk] != 0) continue;
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
                flag[nq * t0 + ni * t1 + nj * t2 + nk] = 2;
            }
        }

        while (cubespherical_mesh::M_cap <= cubespherical_mesh::M + M) {
            cubespherical_mesh::M_cap *= 2;
            HR(cubespherical_mesh::faces, cubespherical_mesh::M_cap * 3);
        }
        // dont omp
        for (int i = 0; i < 3 * M; i++)
            cubespherical_mesh::faces[3 * cubespherical_mesh::M + i] = faces[i] + cubespherical_mesh::N;
        while (cubespherical_mesh::N_cap <= cubespherical_mesh::N + N) {
            cubespherical_mesh::N_cap *= 2;
            HR(cubespherical_mesh::lr_vertices, cubespherical_mesh::N_cap * 7);
        }
        double *current_sdf = sdf + c * S * S * S;
        t1 = S * S; t2 = S;

        convention::L = specs::L + 1;
        int index_checking[6] = {0, 0, 1, 1, 0, 0};
        int checking_value[6] = {0, 0, 0, specs::L, 0, specs::L};
        // dont omp
        for (int i = 0; i < N; i++) {
            int low = 0, high = 1;
            double sdf_floor = current_sdf[int_floor(verts[i * 3]) * t1 + int_floor(verts[i * 3 + 1]) * t2 + int_floor(verts[i * 3 + 2])];
            double sdf_ceil = current_sdf[int_ceil(verts[i * 3]) * t1 + int_ceil(verts[i * 3 + 1]) * t2 + int_ceil(verts[i * 3 + 2])];
            if (sdf_floor > sdf_ceil) {
                high = 0; low = 1;
            }
            for (int j = 0; j < 3; j++) {
                cubespherical_mesh::lr_vertices[7 * cubespherical_mesh::N + i * 7 + 2 * j + low] = int_floor(verts[i * 3 + j]) + newfound[c * 4 + j + 1] * U;
                cubespherical_mesh::lr_vertices[7 * cubespherical_mesh::N + i * 7 + 2 * j + high] = int_ceil(verts[i * 3 + j]) + newfound[c * 4 + j + 1] * U;
            }
            cubespherical_mesh::lr_vertices[7 * cubespherical_mesh::N + i * 7 + 6] = newfound[c * 4];
            int qq = newfound[c * 4], ii[2], jj[2], kk[2];
            int cq = qq, ci[2], cj[2], ck[2];
            for (int lr = 0; lr < 2; lr++) {
                ci[lr] = ii[lr] = cubespherical_mesh::lr_vertices[7 * cubespherical_mesh::N + i * 7 + lr];
                cj[lr] = jj[lr] = cubespherical_mesh::lr_vertices[7 * cubespherical_mesh::N + i * 7 + 2 + lr];
                ck[lr] = kk[lr] = cubespherical_mesh::lr_vertices[7 * cubespherical_mesh::N + i * 7 + 4 + lr];
            }
            for (int b = 2; b < 6; b++) {
                int nq[2]={qq,qq}, ni[2]={ii[0], ii[1]}, nj[2]={jj[0], jj[1]}, nk[2]={kk[0], kk[1]};
                for (int lr = 0; lr < 2; lr++) {
                    if (cubespherical_mesh::lr_vertices[7 * cubespherical_mesh::N + i * 7 + 2 * index_checking[b] + lr] == checking_value[b]) {
                        convention::move(
                            qq, ii[lr], jj[lr], kk[lr],
                            dx[b], dy[b], 0, nq[lr], ni[lr], nj[lr], nk[lr]
                        );
                    }
                }
                if (nq[0] == nq[1] && nq[0] < cq) {
                    cq = nq[0];
                    for (int lr = 0; lr < 2; lr++) {
                        ci[lr] = ni[lr]; cj[lr] = nj[lr]; ck[lr] = nk[lr];
                    }
                }
            }
            cubespherical_mesh::lr_vertices[7 * cubespherical_mesh::N + i * 7 + 6] = cq;
            for (int lr = 0; lr < 2; lr++) {
                cubespherical_mesh::lr_vertices[7 * cubespherical_mesh::N + i * 7 + lr] = ci[lr];
                cubespherical_mesh::lr_vertices[7 * cubespherical_mesh::N + i * 7 + 2 + lr] = cj[lr];
                cubespherical_mesh::lr_vertices[7 * cubespherical_mesh::N + i * 7 + 4 + lr] = ck[lr];
            }
        }
        cubespherical_mesh::N += N;
        cubespherical_mesh::M += M;
    }

    int get_cnt() {
        using namespace pretest;
        int t0 = L * L * R, t1 = L * R, t2 = R;
        cnt = 0;
        #pragma omp parallel for
        for (int q = 0; q < 6; q++)
        for (int i = 0; i < L; i++)
        for (int j = 0; j < L; j++)
        for (int k = 0; k < R; k++)
            if (flag[q * t0 + i * t1 + j * t2 + k] == 2) {
                flag[q * t0 + i * t1 + j * t2 + k] = 1;
                #pragma omp critical
                {
                    int mycnt = cnt++;
                    if (mycnt >= newfound_cap) {
                        newfound_cap *= 2;
                        HR(newfound, newfound_cap * 4);
                    }
                    newfound[4 * mycnt] = q;
                    newfound[4 * mycnt + 1] = i;
                    newfound[4 * mycnt + 2] = j;
                    newfound[4 * mycnt + 3] = k;
                }
            }
        if (cnt == 0) {
            pretest::cleanup();
        }
        return cnt;
    }


    int finefront_init() {
        using namespace finefront;
        init();
        int t1 = W * R, t2 = R;
        for (int v = 0; v < cubespherical_mesh::N; v++) {
            if (cubespherical_mesh::lr_vertices[v * 7 + 6] == 1) {
                int h = max(0, min(cubespherical_mesh::lr_vertices[v * 7] - specs::N0, H - 1));
                int w = max(0, min(cubespherical_mesh::lr_vertices[v * 7 + 2] - specs::N1, W - 1));
                int r = min(cubespherical_mesh::lr_vertices[v * 7 + 4], R - 1);
                flag[h * W * R + w * R + r] = 1;
            }
        }
        cnt = 0;
        #pragma omp parallel for
        for (int i = 0; i < H; i++)
        for (int j = 0; j < W; j++)
        for (int k = 0; k < R; k++)
            if (flag[i * t1 + j * t2 + k]) {
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

    void finefront_get_queries(double *positions) {
        using namespace finefront;
        #pragma omp parallel for
        for (int c = 0; c < cnt; c++) {
            int i = newfound[c * 3], j = newfound[c * 3 + 1], k = newfound[c * 3 + 2];
            for (int ii = 0; ii < S; ii++)
            for (int jj = 0; jj < S; jj++)
            for (int kk = 0; kk < S; kk++) {
                double iH = (i + specs::N0 + ii * 1.0 / U) / specs::L;
                double iW = (j + specs::N1 + jj * 1.0 / U) / specs::L;
                double iR = (k + kk * 1.0 / U) / specs::R;
                normalized_cam_to_world(
                    1,
                    iH, iW, iR,
                    specs::cam,
                    specs::r_min, specs::r_max,
                    positions + 3 * (c * S*S*S + ii*S*S + jj*S + kk)
                );
            }
        }
    }

    void finefront_update(
        int c,
        double *sdf,
        double *verts, int N,
        int *faces, int M
    ) {
        using namespace finefront;
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
            if (ni < 0 || ni >= H || nj < 0 || nj >= W || nk < 0 || nk >= R) continue;
            if (flag[ni * t1 + nj * t2 + nk] != 0) continue;
            bool positive = 0, negative = 0, zero = 0;
            for (int ii = boundary_x_min[b]; ii <= boundary_x_max[b] && (!(positive && negative)); ii++)
            for (int jj = boundary_y_min[b]; jj <= boundary_y_max[b] && (!(positive && negative)); jj++)
            for (int kk = boundary_z_min[b]; kk <= boundary_z_max[b] && (!(positive && negative)); kk++) {
                double sdf_c = sdf[c*S*S*S + ii*S*S + jj*S + kk];
                positive |= sdf_c > 0;
                negative |= sdf_c < 0;
                zero |= sdf_c == 0;
            }
            if (has_iso_surface(positive, negative, zero)) {
                flag[ni * t1 + nj * t2 + nk] = 2;
            }
        }

        while (cubespherical_mesh::M_cap <= cubespherical_mesh::M + M) {
            cubespherical_mesh::M_cap *= 2;
            HR(cubespherical_mesh::faces, cubespherical_mesh::M_cap * 3);
        }
        // dont omp
        for (int i = 0; i < 3 * M; i++)
            cubespherical_mesh::faces[3 * cubespherical_mesh::M + i] = faces[i] + cubespherical_mesh::N;
        while (cubespherical_mesh::N_cap <= cubespherical_mesh::N + N) {
            cubespherical_mesh::N_cap *= 2;
            HR(cubespherical_mesh::lr_vertices, cubespherical_mesh::N_cap * 7);
        }
        double *current_sdf = sdf + c * S * S * S;
        t1 = S * S; t2 = S;

        // dont omp
        for (int i = 0; i < N; i++) {
            int low = 0, high = 1;
            double sdf_floor = current_sdf[int_floor(verts[i * 3]) * t1 + int_floor(verts[i * 3 + 1]) * t2 + int_floor(verts[i * 3 + 2])];
            double sdf_ceil = current_sdf[int_ceil(verts[i * 3]) * t1 + int_ceil(verts[i * 3 + 1]) * t2 + int_ceil(verts[i * 3 + 2])];
            if (sdf_floor > sdf_ceil) {
                high = 0; low = 1;
            }
            for (int j = 0; j < 3; j++) {
                cubespherical_mesh::lr_vertices[7 * cubespherical_mesh::N + i * 7 + 2 * j + low] = int_floor(verts[i * 3 + j]) + newfound[c * 3 + j] * U;
                cubespherical_mesh::lr_vertices[7 * cubespherical_mesh::N + i * 7 + 2 * j + high] = int_ceil(verts[i * 3 + j]) + newfound[c * 3 + j] * U;
            }
            // 6 means fine front
            cubespherical_mesh::lr_vertices[7 * cubespherical_mesh::N + i * 7 + 6] = 6;
        }
        cubespherical_mesh::N += N;
        cubespherical_mesh::M += M;
    }

    int finefront_get_cnt() {
        using namespace finefront;
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

    void finefront_cleanup() {
        finefront::cleanup();
    }

    void complete_depth_test_get_queries(int relax, int batch, double *positions) {
        int cnt = 0;
        for (int i = 0; i < finefront::H * finefront::U; i += relax)
            for (int k = 0; k < finefront::R * finefront::U; k += relax) {
                multicase_cam_to_world(
                    6, i, batch, k,
                    specs::cam,
                    specs::r_min, specs::r_max,
                    positions + 3 * cnt++
                );
            }
    }

    void complete_depth_test_update(int relax, int batch, double *sdf) {
        int t1 = finefront::W * finefront::R, t2 = finefront::R;
        for (int i = 0; i < finefront::H * finefront::U; i += relax)
            for (int k = 0; k < finefront::R * finefront::U - relax; k += relax) {
                int cnt = i / relax * ((finefront::R * finefront::U - 1) / relax + 1) + k / relax;
                int cnt0 = i / relax * ((finefront::R * finefront::U - 1) / relax + 1) + k / relax + 1;
                if ((sdf[cnt] > 0) != (sdf[cnt0] > 0)) {
                    int ind = i / specs::upscale * t1 + batch / specs::upscale * t2 + k / specs::upscale;
                    if (finefront::flag[ind] == 0) finefront::flag[ind] = 2;
                    ind = i / specs::upscale * t1 + batch / specs::upscale * t2 + (k + relax) / specs::upscale;
                    if (finefront::flag[ind] == 0) finefront::flag[ind] = 2;
                }
            }
    }



    void get_stitching_queries(double *positions, int *identifiers) {
        using namespace finefront;
        int t0 = 0;
        // part 1
        {
            int t[4] = {0, W * U * R * U, W * U * R * U * 2, 2 * W * U * R * U + H * U * R * U};
            int i_start[4] = {-U, H * U, 0, 0}, j_start[4] = {0, 0, -U, W * U};
            int i_end[4] = {-U, H * U, H * U - 1, H * U - 1}, j_end[4] = {W * U - 1, W * U - 1, -U, W * U};
            int i_step[4] = {U, U, 1, 1}, j_step[4] = {1, 1, U, U};
            int contraction[4] = {0, 1, 0, 1};
            for (int q = 0; q < 4; q++) {
                #pragma omp parallel for
                for (int k = 0; k < R * U; k++)
                for (int i = i_start[q]; i <= i_end[q]; i++)
                for (int j = j_start[q]; j <= j_end[q]; j++) {
                    int block_offset = t[q] + (i - i_start[q]) * (j_end[q] - j_start[q] + 1) * R * U + (j - j_start[q]) * R * U + k;
                    for (int ii = 0; ii < 2; ii++)
                    for (int jj = 0; jj < 2; jj++)
                    for (int kk = 0; kk < 2; kk++) {
                        int h = i + ii * i_step[q], w = j + jj * j_step[q], r = k + kk;
                        if (i_step[q] == 1) {
                            if (contraction[q] == jj) {
                                h = (i + U / 2) / U * U;
                                r = (k + U / 2) / U * U;
                            }
                        }
                        else {
                            if (contraction[q] == ii) {
                                w = (j + U / 2) / U * U;
                                r = (k + U / 2) / U * U;
                            }
                        }
                        int o = 3 * (block_offset * 8 + ii * 4 + jj * 2 + kk);
                        if (positions != NULL) {
                            multicase_cam_to_world(
                                6, h, w, r,
                                specs::cam,
                                specs::r_min, specs::r_max,
                                positions + o
                            );
                        }
                        else {
                            identifiers[o] = h;
                            identifiers[o + 1] = w;
                            identifiers[o + 2] = r;
                        }
                    }
                }
            }
        }
        t0 += 2 * W * U * R * U + 2 * H * U * R * U;
        // part 2a
        {
            int t[4] = {t0, t0 + W * R * U, t0 + 2 * W * R * U, t0 + 2 * W * R * U + H * R * U};
            int i_start[4] = {-U, H * U, 0, 0}, j_start[4] = {0, 0, -U, W * U};
            int i_end[4] = {-U, H * U, H * U - U, H * U - U}, j_end[4] = {W * U - U, W * U - U, -U, W * U};
            int contraction[4] = {0, 1, 0, 1};
            for (int q = 0; q < 4; q++) {
                #pragma omp parallel for
                for (int k = 0; k < R * U; k++)
                for (int i = i_start[q]; i <= i_end[q]; i += U)
                for (int j = j_start[q]; j <= j_end[q]; j += U) {
                    int block_offset = t[q] + (i - i_start[q]) / U * ((j_end[q] - j_start[q]) / U + 1) * R * U + (j - j_start[q]) / U * R * U + k;
                    for (int ii = 0; ii < 2; ii++)
                    for (int jj = 0; jj < 2; jj++)
                    for (int kk = 0; kk < 2; kk++) {
                        int h = i + ii * U, w = j + jj * U, r = k + kk;
                        if (i_start[q] == i_end[q]) {
                            if (contraction[q] == ii) {
                                r = (k + U / 2) / U * U;
                            }
                            else {
                                w = j + U / 2;
                            }
                        }
                        else {
                            if (contraction[q] == jj) {
                                r = (k + U / 2) / U * U;
                            }
                            else {
                                h = i + U / 2;
                            }
                        }
                        int o = 3 * (block_offset * 8 + ii * 4 + jj * 2 + kk);
                        if (positions != NULL) {
                            multicase_cam_to_world(
                                6, h, w, r,
                                specs::cam,
                                specs::r_min, specs::r_max,
                                positions + o
                            );
                        }
                        else {
                            identifiers[o] = h;
                            identifiers[o + 1] = w;
                            identifiers[o + 2] = r;
                        }
                    }
                }
            }
        }
        t0 += 2 * W * R * U + 2 * H * R * U;
        // part 2b
        {
            int t[4] = {t0, t0 + W * R * U, t0 + 2 * W * R * U, t0 + 2 * W * R * U + H * R * U};
            int i_start[4] = {-U, H * U, 0, 0}, j_start[4] = {0, 0, -U, W * U};
            int i_end[4] = {-U, H * U, H * U - 1, H * U - 1}, j_end[4] = {W * U - 1, W * U - 1, -U, W * U};
            int i_step[4] = {U, U, 1, 1}, j_step[4] = {1, 1, U, U};
            int contraction[4] = {0, 1, 0, 1};
            for (int q = 0; q < 4; q++) {
                #pragma omp parallel for
                for (int k = 0; k < R * U; k += U)
                for (int i = i_start[q]; i <= i_end[q]; i++)
                for (int j = j_start[q]; j <= j_end[q]; j++) {
                    int block_offset = t[q] + (i - i_start[q]) * (j_end[q] - j_start[q] + 1) * R + (j - j_start[q]) * R + k / U;
                    for (int ii = 0; ii < 2; ii++)
                    for (int jj = 0; jj < 2; jj++)
                    for (int kk = 0; kk < 2; kk++) {
                        int h = i + ii * i_step[q], w = j + jj * j_step[q], r = k + kk * U;
                        if (i_start[q] == i_end[q]) {
                            if (contraction[q] == ii) {
                                w = (j + U / 2) / U * U;
                            }
                            else {
                                r = k + U / 2;
                            }
                        }
                        else {
                            if (contraction[q] == jj) {
                                h = (i + U / 2) / U * U;
                            }
                            else {
                                r = k + U / 2;
                            }
                        }
                        int o = 3 * (block_offset * 8 + ii * 4 + jj * 2 + kk);
                        if (positions != NULL) {
                            multicase_cam_to_world(
                                6, h, w, r,
                                specs::cam,
                                specs::r_min, specs::r_max,
                                positions + o
                            );
                        }
                        else {
                            identifiers[o] = h;
                            identifiers[o + 1] = w;
                            identifiers[o + 2] = r;
                        }
                    }
                }
            }
        }
        t0 += 2 * W * R * U + 2 * H * R * U;
        // part 3
        {
            int t[4] = {t0, t0 + W * R, t0 + W * R * 2, t0 + W * R * 2 + H * R};
            int i_start[4] = {-U, H * U, 0, 0}, j_start[4] = {0, 0, -U, W * U};
            int i_end[4] = {-U, H * U, H * U - U, H * U - U}, j_end[4] = {W * U - U, W * U - U, -U, W * U};
            int contraction[4] = {1, 0, 1, 0};
            for (int q = 0; q < 4; q++) {
                #pragma omp parallel for
                for (int k = 0; k < R * U; k += U)
                for (int i = i_start[q]; i <= i_end[q]; i += U)
                for (int j = j_start[q]; j <= j_end[q]; j += U) {
                    int block_offset = t[q] + (i - i_start[q]) / U * ((j_end[q] - j_start[q]) / U + 1) * R + (j - j_start[q]) / U * R + k / U;
                    for (int ii = 0; ii < 2; ii++)
                    for (int jj = 0; jj < 2; jj++)
                    for (int kk = 0; kk < 2; kk++) {
                        int h = i + ii * U, w = j + jj * U, r = k + kk * U;
                        if (i_start[q] == i_end[q]) {
                            if (contraction[q] == ii) {
                                w = j + U / 2;
                                r = k + U / 2;
                            }
                        }
                        else {
                            if (contraction[q] == jj) {
                                h = i + U / 2;
                                r = k + U / 2;
                            }
                        }
                        int o = 3 * (block_offset * 8 + ii * 4 + jj * 2 + kk);
                        if (positions != NULL) {
                            multicase_cam_to_world(
                                6, h, w, r,
                                specs::cam,
                                specs::r_min, specs::r_max,
                                positions + o
                            );
                        }
                        else {
                            identifiers[o] = h;
                            identifiers[o + 1] = w;
                            identifiers[o + 2] = r;
                        }
                    }
                }
            }
        }
        t0 += W * R * 2 + H * R * 2;
        // corner part
        int is[4] = {-U, -U, H * U, H * U}, js[4] = {-U, W * U, W * U, -U};
        // part 4
        {
            int i_cotraction[4] = {0, 0, 1, 1}, j_contraction[4] = {0, 1, 1, 0};
            for (int q = 0; q < 4; q++) {
                #pragma omp parallel for
                for (int k = 0; k < R * U; k++)
                {
                    int i = is[q], j = js[q];
                    int block_offset = t0 + q * R * U + k;
                    for (int ii = 0; ii < 2; ii++)
                    for (int jj = 0; jj < 2; jj++)
                    for (int kk = 0; kk < 2; kk++) {
                        int h = i + ii * U, w = j + jj * U, r = k + kk;
                        if (ii == i_cotraction[q]) {
                            w = (j + U / 2) / U * U;
                            r = (k + U / 2) / U * U;
                        }
                        else if (jj == j_contraction[q]) {
                            h = (i + U / 2) / U * U;
                            r = (k + U / 2) / U * U;
                        }
                        int o = 3 * (block_offset * 8 + ii * 4 + jj * 2 + kk);
                        if (positions != NULL) {
                            multicase_cam_to_world(
                                6, h, w, r,
                                specs::cam,
                                specs::r_min, specs::r_max,
                                positions + o
                            );
                        }
                        else {
                            identifiers[o] = h;
                            identifiers[o + 1] = w;
                            identifiers[o + 2] = r;
                        }
                    }
                }
            }
        }
        // part 5
        {
            t0 += 4 * R * U;
            int i_cotraction[4] = {1, 1, 0, 0}, j_contraction[4] = {1, 0, 0, 1};
            for (int q = 0; q < 4; q++) {
                #pragma omp parallel for
                for (int k = 0; k < R * U; k += U)
                {
                    int i = is[q], j = js[q];
                    int block_offset = t0 + q * R + k / U;
                    for (int ii = 0; ii < 2; ii++)
                    for (int jj = 0; jj < 2; jj++)
                    for (int kk = 0; kk < 2; kk++) {
                        int h = i + ii * U, w = j + jj * U, r = k + kk * U;
                        if (ii == i_cotraction[q] && jj == j_contraction[q]) {
                            h = i + (1 - ii) * U;
                        }
                        int o = 3 * (block_offset * 8 + ii * 4 + jj * 2 + kk);
                        if (positions != NULL) {
                            multicase_cam_to_world(
                                6, h, w, r,
                                specs::cam,
                                specs::r_min, specs::r_max,
                                positions + o
                            );
                        }
                        else {
                            identifiers[o] = h;
                            identifiers[o + 1] = w;
                            identifiers[o + 2] = r;
                        }
                    }
                }
            }
        }
        // part 6
        {
            t0 += 4 * R;
            int i_cotraction[4] = {1, 1, 0, 0}, j_contraction[4] = {1, 0, 0, 1};
            for (int q = 0; q < 4; q++) {
                #pragma omp parallel for
                for (int k = 0; k < R * U; k += U)
                {
                    int i = is[q], j = js[q];
                    int block_offset = t0 + q * R + k / U;
                    for (int ii = 0; ii < 2; ii++)
                    for (int jj = 0; jj < 2; jj++)
                    for (int kk = 0; kk < 2; kk++) {
                        int h = i + ii * U, w = j + jj * U, r = k + kk * U;
                        if (ii == i_cotraction[q] && jj == j_contraction[q]) {
                            r = k + U / 2;
                        }
                        else if (ii != i_cotraction[q] && jj != j_contraction[q]) {
                            h = i + (1 - ii) * U;
                        }
                        int o = 3 * (block_offset * 8 + ii * 4 + jj * 2 + kk);
                        if (positions != NULL) {
                            multicase_cam_to_world(
                                6, h, w, r,
                                specs::cam,
                                specs::r_min, specs::r_max,
                                positions + o
                            );
                        }
                        else {
                            identifiers[o] = h;
                            identifiers[o + 1] = w;
                            identifiers[o + 2] = r;
                        }
                    }
                }
            }
        }
    }

    void remove_inview_faces() {
        using namespace cubespherical_mesh;
        int N0=specs::N0, N1= specs::N1;
        if (specs::upscale > 0) {
            N1--;
            N0--;
        }
        int new_M = 0;
        for (int i = 0; i < M; i++) {
            bool inview = 1;
            for (int j = 0; j < 3 && inview; j++) {
                int v = faces[i * 3 + j];
                if (lr_vertices[v * 7 + 6] == 1) {
                    for (int k = 0; k < 2; k++) {
                        if (lr_vertices[v * 7 + k] < N0 || lr_vertices[v * 7 + k] > specs::L - N0) {
                            inview = 0;
                            break;
                        }
                        if (lr_vertices[v * 7 + 2 + k] < N1 || lr_vertices[v * 7 + 2 + k] > specs::L - N1) {
                            inview = 0;
                            break;
                        }
                    }
                }
                else {
                    inview = 0;
                    break;
                }
            }
            if (!inview) {
                memcpy(faces + new_M * 3, faces + i * 3, 3 * sizeof(int));
                new_M++;
            }
        }
        M = new_M;
    }

    void stitch_update(
        double *sdf,
        int *verts_int, double *verts_frac, int N, 
        int *faces, int M
    ) {
        using namespace finefront;

        remove_inview_faces();
        int tot = 2 * (W + H) * R * (U * U + 2 * U + 1) + 4 * R * (U + 2);
        HM(all_identifiers, tot * 8 * 3);
        get_stitching_queries(NULL, all_identifiers);

        while (cubespherical_mesh::N_cap <= cubespherical_mesh::N + N) {
            cubespherical_mesh::N_cap *= 2;
            HR(cubespherical_mesh::lr_vertices, cubespherical_mesh::N_cap * 7);
        }
        #pragma omp parallel for
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < 3; j++) assert(verts_frac[i * 3 + j] >= 0 && verts_frac[i * 3 + j] <= 1);
            int int_floors[3] = {
                verts_int[i * 3] + int_floor(verts_frac[i * 3]),
                int_floor(verts_frac[i * 3 + 1]),
                int_floor(verts_frac[i * 3 + 2]),
            };
            int int_ceils[3] = {
                verts_int[i * 3] + int_ceil(verts_frac[i * 3]),
                int_ceil(verts_frac[i * 3 + 1]),
                int_ceil(verts_frac[i * 3 + 2]),
            };
            if (int_floors[0] % 4 > 1) {
                // 7 means useless, byproduct of collective marching cube
                for (int j = 0; j < 7; j++) cubespherical_mesh::lr_vertices[7 * cubespherical_mesh::N + 7 * i + j] = 7;
            }
            else {
                int low = 0, high = 1;
                double sdf_floor = sdf[int_floors[0] * 4 + int_floors[1] * 2 + int_floors[2]];
                double sdf_ceil = sdf[int_ceils[0] * 4 + int_ceils[1] * 2 + int_ceils[2]];
                if (sdf_floor > sdf_ceil) {
                    high = 0; low = 1;
                }
                bool all_divisible = 1;
                int id = int_floors[0] / 4, current_identifiers[2][3];
                int o1 = (int_floors[0] - id * 4), o2 = int_floors[1], o3 = int_floors[2];
                assert(o1 == 0 || o1 == 1);
                assert(o2 == 0 || o2 == 1);
                assert(o3 == 0 || o3 == 1);
                int offset = id * 8 + o1 * 4 + o2 *2 + o3;
                for (int j = 0; j < 3; j++) {
                    all_divisible &= (current_identifiers[0][j] = all_identifiers[offset * 3 + j]) % U == 0;
                }
                all_divisible &= (current_identifiers[0][0] / U < 0 || current_identifiers[0][0] / U > H || current_identifiers[0][1] / U < 0 || current_identifiers[0][1] / U > W);

                o1 = (int_ceils[0] - id * 4); o2 = int_ceils[1]; o3 = int_ceils[2];
                
                assert(o1 == 0 || o1 == 1);
                assert(o2 == 0 || o2 == 1);
                assert(o3 == 0 || o3 == 1);
                offset = id * 8 + o1 * 4 + o2 *2 + o3;
                for (int j = 0; j < 3; j++) {
                    all_divisible &= (current_identifiers[1][j] = all_identifiers[offset * 3 + j]) % U == 0;
                }
                all_divisible &= (current_identifiers[1][0] / U < 0 || current_identifiers[1][0] / U > H || current_identifiers[1][1] / U < 0 || current_identifiers[1][1] / U > W);
                if (!all_divisible) {
                    cubespherical_mesh::lr_vertices[7 * cubespherical_mesh::N + 7 * i + 6] = 6;
                }
                else {
                    cubespherical_mesh::lr_vertices[7 * cubespherical_mesh::N + 7 * i + 6] = 1;
                    for (int j = 0; j < 2; j++) {
                        current_identifiers[j][0] = specs::N0 + current_identifiers[j][0] / U;
                        current_identifiers[j][1] = specs::N1 + current_identifiers[j][1] / U;
                        current_identifiers[j][2] /= U;
                    }
                }
                for (int j = 0; j < 3; j++) {
                    cubespherical_mesh::lr_vertices[7 * cubespherical_mesh::N + i * 7 + 2 * j + low] = current_identifiers[0][j];
                    cubespherical_mesh::lr_vertices[7 * cubespherical_mesh::N + i * 7 + 2 * j + high] = current_identifiers[1][j];
                }
            }
        }
        while (cubespherical_mesh::M_cap <= cubespherical_mesh::M + M) {
            cubespherical_mesh::M_cap *= 2;
            HR(cubespherical_mesh::faces, cubespherical_mesh::M_cap * 3);
        }
        int m = 0;
        for (int i = 0; i < M; i++) {
            int u[3] = {faces[i * 3], faces[i * 3 + 1], faces[i * 3 + 2]};
            int flag = 0, id;
            for (int j = 0; j < 3; j++) {
                if (cubespherical_mesh::lr_vertices[7 * cubespherical_mesh::N + 7 * u[j] + 6] == 7) {
                    flag = 1;
                    break;
                }
            }
            if (flag) continue;
            for (int j = 0; j < 3; j++) cubespherical_mesh::faces[3 * cubespherical_mesh::M + m * 3 + j] = faces[i * 3 + j] + cubespherical_mesh::N;
            m++;
        }
        cubespherical_mesh::N += N;
        cubespherical_mesh::M += m;
    }

    void get_mesh_cnt(int *NM) {
        bool flag = 0;
        if (finefront::all_identifiers != NULL) {
            safefree(finefront::all_identifiers);
            finefront::all_identifiers = NULL;
            flag = 1;
        }
        using namespace cubespherical_mesh;
        if (specs::upscale == -1) remove_inview_faces();
        // add U to avoid negative
        if (flag) for (int i = 0; i < N; i++) for (int j = 0; j < 4; j++) lr_vertices[i * 7 + j] += finefront::U;
        merge_verts(
            lr_vertices, N, 7,
            faces, M
        );
        if (flag) for (int i = 0; i < N; i++) for (int j = 0; j < 4; j++) lr_vertices[i * 7 + j] -= finefront::U;
        clean_faces(N, faces, M);
        clean_verts(lr_vertices, 7, N, faces, M);
        NM[0] = N;
        NM[1] = M;
        fHM(bis_vertices, N * 6);
        sHM(qs, N);
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < 6; j++)
                bis_vertices[i * 6 + j] = (double)lr_vertices[i * 7 + j];
            qs[i] = lr_vertices[i * 7 + 6];
        }
        safefree(lr_vertices);
    }

    void bisection_get_positions(
        double *positions
    ) {
        int N = cubespherical_mesh::N, L=specs::L, R=specs::R;
        double *identifiers = cubespherical_mesh::bis_vertices;
        int cnt = 0;
        for (int i = 0; i < N; i++) {
            double iH = (identifiers[i * 6] + identifiers[i * 6 + 1]) / 2;
            double iW = (identifiers[i * 6 + 2] + identifiers[i * 6 + 3]) / 2;
            double iR = (identifiers[i * 6 + 4] + identifiers[i * 6 + 5]) / 2;
            multicase_cam_to_world(
                cubespherical_mesh::qs[i],
                iH, iW, iR,
                specs::cam,
                specs::r_min, specs::r_max,
                positions + 3 * cnt++
            );
        }
    }

    void bisection_update(
        double *sdfs
    ) {
        int N = cubespherical_mesh::N;
        double *identifiers = cubespherical_mesh::bis_vertices;
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

    void get_final_mesh(double *vertices, int *faces, int *outofview_annotation) {
        cubespherical_mesh::finalize();
        for (int i = 0; i < cubespherical_mesh::N; i++) {
            multicase_cam_to_world(
                cubespherical_mesh::qs[i],
                cubespherical_mesh::vertices[i * 3], cubespherical_mesh::vertices[i * 3 + 1], cubespherical_mesh::vertices[i * 3 + 2],
                specs::cam,
                specs::r_min, specs::r_max,
                vertices + 3 * i
            );
            outofview_annotation[i] = cubespherical_mesh::qs[i] <= 5;
        }
        memcpy(faces, cubespherical_mesh::faces, 3 * cubespherical_mesh::M * sizeof(int));
        cubespherical_mesh::cleanup();
    }
}