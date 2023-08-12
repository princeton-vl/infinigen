// Copyright (c) Princeton University.
// This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

// Authors: Zeyu Ma


#include <stdio.h>
#include <assert.h>
#include <omp.h>
#include <string.h>
#include <math.h>
#include <vector>
#include <algorithm>
using namespace std;

#ifndef __MESHING_UTILS__
#define __MESHING_UTILS__

#define HM(X, N) X = (int*)malloc((N) * sizeof(int));
#define sHM(X, N) X = (short*)malloc((N) * sizeof(short));
#define fHM(X, N) X = (double*)malloc((N) * sizeof(double));
#define HR(X, N) X = (int*)realloc(X, (N) * sizeof(int));
#define sHR(X, N) X = (short*)realloc(X, (N) * sizeof(short));
#define fHR(X, N) X = (double*)realloc(X, (N) * sizeof(double));
#define SET0(X, N) memset(X, 0, (N) * sizeof(int));
#define sSET0(X, N) memset(X, 0, (N) * sizeof(short));
#define SETN(X, N) memset(X, -1, (N) * sizeof(int));

#define safefree(p) free(p); p = NULL;

bool has_iso_surface(bool positive, bool negative, bool zero) {
    return (positive && (negative || zero));
}

unsigned int myhash(int k)
{
    k ^= k >> 16;
    k *= 0x85ebca6b;
    k ^= k >> 13;
    k *= 0xc2b2ae35;
    k ^= k >> 16;
    return (unsigned int)k;
}


inline int int_floor(double x) {
    return int(floor(x));
}


inline int int_ceil(double x) {
    return int(ceil(x));
}


void merge_verts(
    int *ids, int &N, int K,
    int *faces, int M
) {
    int *maxs; HM(maxs, K);
    int *current_order; HM(current_order, N);
    for (int i = 0; i < N; i++) current_order[i] = i;
    SET0(maxs, K);
    for (int i = 0; i < N * K; i++) {
        maxs[i % K] = max(maxs[i % K], ids[i]);
    }
    for (int i = 0; i < K; i++)
        maxs[i] += 1;
    int *head;
    int *index, *nxt;
    HM(index, N); HM(nxt, N);
    for (int iid = 0; iid < K; iid++) {
        int cnt = 0;
        HM(head, maxs[iid]);
        SETN(head, maxs[iid]);
        for (int metai = 0; metai < N; metai++) {
            int i = current_order[metai];
            int idiid = ids[i * K + iid];
            assert(idiid >= 0);
            index[cnt] = i;
            nxt[cnt] = head[idiid];
            head[idiid] = cnt;
            cnt++;
        }
        int tot = 0;
        for (int idiid  = 0; idiid < maxs[iid]; idiid++) {
            for (int i = head[idiid]; i != -1; i = nxt[i]) {
                current_order[tot++] = index[i];
            }
        }
        for (int i = 0; i < N / 2; i++) {
            int tmp = current_order[i];
            current_order[i] = current_order[N - 1 - i];
            current_order[N - 1 - i] = tmp;
        }
        free(head);
    }
    free(index);
    free(nxt);
    
    int *map; HM(map, N);
    for (int i = 0; i < N; i++) {
        int ii = current_order[i];
        bool is_new = 0;
        if (i != 0) {
            for (int j = 0; j < K; j++) {
                if (ids[ii * K + j] != ids[current_order[i - 1] * K + j]) is_new = 1;
            }
        }
        else {
            is_new = 1;
        }
        if (is_new) {
            map[ii] = ii;
        }
        else {
            map[ii] = map[current_order[i - 1]];
        }
    }
    int new_label = -1;
    for (int i = 0; i < N; i++)
        if (map[i] == i) {
            new_label++;
            map[i] = new_label;
            memcpy(ids + K * new_label, ids + K * i, K * sizeof(int));
        }
        else {
            map[i] = map[map[i]];
        }
    N = new_label + 1;
    for (int i = 0; i < 3 * M; i++) {
        faces[i] = map[faces[i]];
    }
    free(map);
    free(maxs);
    free(current_order);
}

void clean_verts_double(double *vertices, int &N, int *faces, int M) {
    const int K = 3;
    int *map;
    HM(map, N);
    SET0(map, N);
    for (int i = 0; i < 3 * M; i++)
        map[faces[i]] = 1;
    int new_label = -1;
    for (int i = 0; i < N; i++)
        if (map[i]) {
            new_label++;
            map[i] = new_label;
            memcpy(vertices + new_label * K, vertices + i * K, K * sizeof(double));
        }
        else {
            map[i] = -1;
        }
    for (int i = 0; i < 3 * M; i++)
        faces[i] = map[faces[i]];
    N = new_label + 1;
    free(map);
}

void clean_verts(int *vertices, int K, int &N, int *faces, int M) {
    int *map;
    HM(map, N);
    SET0(map, N);
    for (int i = 0; i < 3 * M; i++)
        map[faces[i]] = 1;
    int new_label = -1;
    for (int i = 0; i < N; i++)
        if (map[i]) {
            new_label++;
            map[i] = new_label;
            memcpy(vertices + new_label * K, vertices + i * K, K * sizeof(int));
        }
        else {
            map[i] = -1;
        }
    for (int i = 0; i < 3 * M; i++)
        faces[i] = map[faces[i]];
    N = new_label + 1;
    free(map);
}

void clean_faces(int N, int *faces, int &M) {
    int cnt = 0;
    for (int i = 0; i < M; i++) {
        int u[3] = {faces[i * 3], faces[i * 3 + 1], faces[i * 3 + 2]};
        if (u[0] == u[1] || u[0] == u[2] || u[1] == u[2]) continue;
        int min_ind = 2;
        if (u[0] < u[1] && u[0] < u[2]) min_ind = 0;
        else if (u[1] < u[2] && u[1] < u[0]) min_ind = 1;
        faces[cnt * 3] = u[min_ind];
        faces[cnt * 3 + 1] = u[(min_ind + 1) % 3];
        faces[cnt++ * 3 + 2] = u[(min_ind + 2) % 3];
    }
    M = cnt;
    int *head, *index, *nxt, *current_order;
    HM(current_order, M);
    for (int i = 0; i < M; i++) current_order[i] = i;
    HM(head, N);
    HM(index, M);
    HM(nxt, M);
    for (int j = 0; j < 2; j++) {
        SETN(head, N);
        cnt = 0;
        for (int metai = 0; metai < M; metai++) {
            int i = current_order[metai];
            int idiid = faces[i * 3 + j];
            index[cnt] = i;
            nxt[cnt] = head[idiid];
            head[idiid] = cnt;
            cnt++;
        }
        int tot = 0;
        for (int idiid  = 0; idiid < N; idiid++) {
            for (int i = head[idiid]; i != -1; i = nxt[i]) {
                current_order[tot++] = index[i];
            }
        }
    }
    free(head);
    free(nxt);
    SET0(index, M); // reuse as flag
    for (int i = 0; i < M; i++) {
        int ii = current_order[i];
        bool is_new = 0;
        if (i != 0) {
            for (int j = 0; j < 2; j++) {
                if (faces[ii * 3 + j] != faces[current_order[i - 1] * 3 + j]) is_new = 1;
            }
        }
        else {
            is_new = 1;
        }
        index[ii] = is_new;
    }
    free(current_order);
    cnt = 0;
    for (int i = 0; i < M; i++) {
        if (index[i]) {
            memcpy(faces + cnt++ * 3, faces + i * 3, 3 * sizeof(int));
        }
    }
    free(index);
    M = cnt;
}

extern "C" {
    void facewise_mean(double *attrs, int *faces, int M, double *result) {
        #pragma omp parallel for
        for (int i = 0; i < M; i++)
            for (int j = 0; j < 3; j++) {
                result[i] += attrs[faces[i * 3 + j]] / 3;
            }
    }

    void get_adjacency(int M, int E, int *pairs, int *results) {
        for (int i = 0; i < M * 3; i++) results[i] = -1;
        for (int i = 0; i < E; i++) {
            int u = pairs[i * 2], v = pairs[i * 2 + 1];
            int iu = 0, iv = 0;
            while (results[u * 3 + iu] != -1) iu++;
            while (results[v * 3 + iv] != -1) iv++;
            results[u * 3 + iu] = v;
            results[v * 3 + iv] = u;
        }
    }
    void facewise_intmax(int *attrs, int *faces, int M, int *result) {
        #pragma omp parallel for
        for (int i = 0; i < M; i++)
            for (int j = 0; j < 3; j++) {
                result[i] = max(result[i], attrs[faces[i * 3 + j]]);
            }
    }

    void compute_face_normals(double *vertices, int *faces, int M, double *face_normals) {
        #pragma omp parallel for
        for (int i = 0; i < M; i++) {
            int u = faces[i * 3], v = faces[i * 3 + 1], w = faces[i * 3 + 2];
            double x0 = vertices[3 * u], y0 = vertices[3 * u + 1], z0 = vertices[3 * u + 2];
            double x1 = vertices[3 * v], y1 = vertices[3 * v + 1], z1 = vertices[3 * v + 2];
            double x2 = vertices[3 * w], y2 = vertices[3 * w + 1], z2 = vertices[3 * w + 2];
            x1 -= x0; y1 -= y0; z1 -= z0;
            x2 -= x0; y2 -= y0; z2 -= z0;
            double nx = y1 * z2 - y2 * z1, ny = z1 * x2 - z2 * x1, nz = x1 * y2 - x2 * y1, r = sqrt(nx * nx + ny * ny + nz * nz);
            face_normals[i * 3] = nx / r;
            face_normals[i * 3 + 1] = ny / r;
            face_normals[i * 3 + 2] = nz / r;
        }
    }

}

#endif