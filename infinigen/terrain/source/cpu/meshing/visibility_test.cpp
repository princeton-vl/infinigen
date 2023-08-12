// Copyright (c) Princeton University.
// This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

// Authors: Zeyu Ma


#include <stdio.h>
#include <assert.h>
#include <omp.h>
#include <string.h>
#include <math.h>
#include <algorithm>
#include <iostream>
#include "utils.cpp"
using namespace std;


double cross(
    double x1, double y1,
    double x2, double y2
) {
    return x1 * y2 - x2 * y1;
}

bool seg_seg_intersect(
    double x1, double y1,
    double x2, double y2,
    double x3, double y3,
    double x4, double y4
) {
    if ((cross(x2 - x1, y2 - y1, x3 - x1, y3 - y1) >= 0) == (cross(x2 - x1, y2 - y1, x4 - x1, y4 - y1) >= 0)) return 0;
    if ((cross(x4 - x3, y4 - y3, x1 - x3, y1 - y3) >= 0) == (cross(x4 - x3, y4 - y3, x2 - x3, y2 - y3) >= 0)) return 0;
    return 1;
}


namespace depth_table {
    int *head, *nxt, *face_info, cnt, cap, R;

    void init() {
        cnt = 0;
        cap = 10000;
        HM(head, R * 6 + 1);
        SETN(head, R * 6 + 1);
        HM(nxt, cap);
        HM(face_info, 3 * cap);
    };
    void clear() {
        cnt = 0;
        SETN(head, R * 6 + 1);
    }
    void cleanup() {
        safefree(head);
        safefree(nxt);
        safefree(face_info);
    }
    void add(
        int head_id, int id, int h, int w
    ) {
        int mycnt, old_heads;
        #pragma omp critical
        {
            mycnt = cnt++;
            if (mycnt >= cap) {
                cap *= 2;
                HR(nxt, cap);
                HR(face_info, 3 * cap);
            }
            face_info[mycnt * 3] = id;
            face_info[mycnt * 3 + 1] = h;
            face_info[mycnt * 3 + 2] = w;
            old_heads = head[head_id];
            head[head_id] = mycnt;
            nxt[mycnt] = old_heads;
        }
    }
}

namespace mesh {
    int *faces, *visibility;
    double *vertices;
}

namespace face_map {
    int *head, *nxt, *id, *depth, cnt, cap, *new_head;
    int H, W;
    int *backface;

    void init(int H_, int W_) {
        H = H_; W = W_;
        cnt = 0;
        cap = 10000;
        HM(head, H * W);
        SETN(head, H * W);
        HM(nxt, cap);
        HM(id, cap);
        HM(depth, cap);
        HM(new_head, H * W);
    }

    void cleanup() {
        safefree(head);
        safefree(nxt);
        safefree(id);
        safefree(depth);
        safefree(new_head);
    }

    void add(int i, int d, int h, int w) {
        if (cnt >= cap) {
            cap *= 2;
            HR(id, cap);
            HR(depth, cap);
            HR(nxt, cap);
        }
        id[cnt] = i;
        depth[cnt] = d;
        nxt[cnt] = new_head[h * W + w];
        new_head[h * W + w] = cnt;
        cnt++;
    }
}

namespace convex_map {
    int *super_head, *head, *super_nxt, *nxt, cnt, super_cnt;
    double *pix_coords;
    int cap;
    int super_cap;
    int H, W;

    void cleanup() {
        safefree(super_head);
        safefree(head);
        safefree(super_nxt);
        safefree(nxt);
        safefree(pix_coords);
    }
    
    // must be critical in the context
    void add(
        int *head0, double x, double y
    ) {
        int mycnt = cnt++;
        if (mycnt >= cap) {
            cap *= 2;
            HR(nxt, cap);
            fHR(pix_coords, 2 * cap);
        }
        pix_coords[mycnt * 2] = x;
        pix_coords[mycnt * 2 + 1] = y;
        nxt[mycnt] = *head0;
        *head0 = mycnt;
    }
    // must be critical in the context
    void add_to_tail(
        double x, double y, int &tail
    ) {
        int mycnt = cnt++;
        if (mycnt >= cap) {
            cap *= 2;
            HR(nxt, cap);
            fHR(pix_coords, 2 * cap);
        }
        pix_coords[mycnt * 2] = x;
        pix_coords[mycnt * 2 + 1] = y;
        nxt[mycnt] = -1;
        nxt[tail] = mycnt;
        tail = mycnt;
    }
    // must be critical in the context
    void super_add(
        int super_head_id,
        int content
    ) {
        int mycnt;
        // #pragma omp critical
        // {
            mycnt = super_cnt++;
            if (mycnt >= super_cap) {
                super_cap *= 2;
                HR(super_nxt, super_cap);
                HR(head, super_cap);
            }
            head[mycnt] = content;
            super_nxt[mycnt] = super_head[super_head_id];
            super_head[super_head_id] = mycnt;
        // }
    }
    void super_delete(
        int super_head_id,
        int target
    ) {
        int prev = -1;
        for (int i = super_head[super_head_id]; i != -1; i = super_nxt[i]) {
            if (i == target) {
                if (prev != -1) {
                    super_nxt[prev] = super_nxt[i];
                }
                else {
                    super_head[super_head_id] = super_nxt[i];
                }
                return;
            }
            prev = i;
        }
    }

    void init(int H_, int W_) {
        H = H_;
        W = W_;
        cap = super_cap = 10000;
        HM(head, super_cap); SETN(head, super_cap);
        HM(nxt, cap);
        fHM(pix_coords, 2 * cap);
        HM(super_head, H * W);
        SET0(super_head, H * W); // note this is special
        HM(super_nxt, super_cap);
        super_cnt = cnt = 0;
    }

}


void compute_face_order(
    int M1, int M2,
    bool skip_backface
) {
    #pragma omp parallel for
    for (int idx = M1; idx < M2; idx++) {
        const double eps = 1e-3;
        int p[3]{mesh::faces[idx * 3], mesh::faces[idx * 3 + 1], mesh::faces[idx * 3 + 2]};
        double x1 = mesh::vertices[p[1] * 3] - mesh::vertices[p[0] * 3], y1 = mesh::vertices[p[1] * 3 + 1] - mesh::vertices[p[0] * 3 + 1];
        double x2 = mesh::vertices[p[2] * 3] - mesh::vertices[p[0] * 3], y2 = mesh::vertices[p[2] * 3 + 1] - mesh::vertices[p[0] * 3 + 1];
        if (x1 * y2 - x2 * y1 > 0) {
            if (skip_backface) continue;
            else {
                face_map::backface[idx] = 1;
            }
        }
        int total_order = 0;
        double total_theta = 0, total_phi = 0;
        for (int j = 0; j < 3; j++) {
            double r = mesh::vertices[p[j] * 3 + 2];
            int int_r = int_floor(r);
            total_order += int_r * 2;
            if (r > int_r + eps) {
                total_order++;
            }
            double theta = mesh::vertices[p[j] * 3];
            double phi = mesh::vertices[p[j] * 3 + 1];
            total_theta += theta / 3;
            total_phi += phi / 3;
        }
        int int_theta = fmin(int(floor(total_theta)), face_map::H - 1);
        int int_phi = fmin(int(floor(total_phi)), face_map::W - 1);
        depth_table::add(total_order, idx, int_theta, int_phi);
    }
    SETN(face_map::new_head, face_map::H * face_map::W);
    for (int i = depth_table::R * 6; i >= 0; i--) {
        for (int j = depth_table::head[i]; j != -1; j = depth_table::nxt[j]) {
            int id = depth_table::face_info[j * 3];
            int h = depth_table::face_info[j * 3 + 1];
            int w = depth_table::face_info[j * 3 + 2];
            face_map::add(id, i, h, w);
        }
    }
}


bool cut_one_side(
    int head_id,
    double *vertices,
    int *head_result
) {
    using namespace convex_map;
    const double eps = 1e-3;
    double x1 = vertices[0], y1 = vertices[1];
    double x2 = vertices[2], y2 = vertices[3];
    double ex = x2 - x1, ey = y2 - y1;
    double e = sqrt(ex * ex + ey * ey);
    ex /= e; ey /= e;
    double cross_product;
    int enter_cut = -1, exit_cut = -1, prev_k, prev_exit_cut;
    double enter_cut_position[2], exit_cut_position[2];
    bool cut = 0;
    bool uncut = 0;
    for (int k = head[head_id]; k != -1; k = nxt[k]) {
        double cx = pix_coords[k * 2] - x1, cy = pix_coords[k * 2 + 1] - y1;
        double cpk = ex * cy - ey * cx;
        if (cpk >= eps) {
            cut = 1;
            if (k != head[head_id] && cross_product < eps) {
                cross_product = min(cross_product, 0.);
                enter_cut = k;
                enter_cut_position[0] = (pix_coords[k * 2] * cross_product - pix_coords[prev_k * 2] * cpk) / (cross_product - cpk);
                enter_cut_position[1] = (pix_coords[k * 2 + 1] * cross_product - pix_coords[prev_k * 2 + 1] * cpk) / (cross_product - cpk);
            }
        }
        else {
            uncut = 1;
            if (k != head[head_id] && cross_product >= eps) {
                double cpk0 = min(0., cpk);
                exit_cut = k;
                prev_exit_cut = prev_k;
                exit_cut_position[0] = (pix_coords[k * 2] * cross_product - pix_coords[prev_k * 2] * cpk0) / (cross_product - cpk0);
                exit_cut_position[1] = (pix_coords[k * 2 + 1] * cross_product - pix_coords[prev_k * 2 + 1] * cpk0) / (cross_product - cpk0);
            }
        }
        cross_product = cpk;
        prev_k = k;
    }
    if (!cut) {
        return 0;
    }
    if (!uncut) {
        *head_result = head[head_id];
        return 1;
    }
    if (enter_cut == -1) {
        int k = head[head_id];
        enter_cut = k;
        double cx = pix_coords[k * 2] - x1, cy = pix_coords[k * 2 + 1] - y1;
        double cpk = ex * cy - ey * cx;
        enter_cut_position[0] = (pix_coords[k * 2] * cross_product - pix_coords[prev_k * 2] * cpk) / (cross_product - cpk);
        enter_cut_position[1] = (pix_coords[k * 2 + 1] * cross_product - pix_coords[prev_k * 2 + 1] * cpk) / (cross_product - cpk);
    }
    else if (exit_cut == -1) {
        int k = head[head_id];
        exit_cut = k;
        prev_exit_cut = prev_k;
        double cx = pix_coords[k * 2] - x1, cy = pix_coords[k * 2 + 1] - y1;
        double cpk = ex * cy - ey * cx;
        exit_cut_position[0] = (pix_coords[k * 2] * cross_product - pix_coords[prev_k * 2] * cpk) / (cross_product - cpk);
        exit_cut_position[1] = (pix_coords[k * 2 + 1] * cross_product - pix_coords[prev_k * 2 + 1] * cpk) / (cross_product - cpk);
    }
    double x = pix_coords[enter_cut * 2], y = pix_coords[enter_cut * 2 + 1];
    int tail;
    #pragma omp critical
    {
        for (int k = enter_cut;;) {
            x = fmin(fmax(x, 0.), 1.);
            y = fmin(fmax(y, 0.), 1.);
            if (k == enter_cut) {
                add(head_result, x, y);
                tail = head_result[0];
            }
            else {
                add_to_tail(x, y, tail);
            }
            if (k == prev_exit_cut) {
                x = exit_cut_position[0];
                y = exit_cut_position[1];
                k = -2;
            }
            else if (k == -2) {
                x = enter_cut_position[0];
                y = enter_cut_position[1];
                k = -3;
            }
            else if (k == -3) {
                break;
            }
            else {
                k = nxt[k];
                if (k == -1) {
                    k = head[head_id];
                }
                x = pix_coords[k * 2];
                y = pix_coords[k * 2 + 1];
            }
        }
    }
    return 0;
}


bool cut_single_convex(
    int hid,
    int super_head_id,
    double *vertices
) {
    using namespace convex_map;
    int head_i[3] = {-1, -1, -1};
    for (int i = 0; i < 3; i++) {
        double vertices_i[4]{
            vertices[i * 2], vertices[i * 2 + 1],
            vertices[(i + 1) % 3 * 2], vertices[(i + 1) % 3 * 2 + 1]
        };
        bool miss = cut_one_side(
            hid,
            vertices_i,
            head_i + i
        );
        if (miss) {
            return 0;
        }
    }
    super_delete(super_head_id, hid);
    for (int i = 0; i < 3; i++) {
        if (head_i[i] != -1) {
            #pragma omp critical
            super_add(super_head_id, head_i[i]);
        }
    }
    return 1;
}

bool cut_multiple_convex(
    int super_head_id,
    double *vertices
) {
    using namespace convex_map;
    int visible = 0;
    for (int hid = super_head[super_head_id]; hid != -1; hid = super_nxt[hid]) {
        bool visible1 = cut_single_convex(
            hid,
            super_head_id,
            vertices
        );
        visible |= visible1;
    }
    return visible;
}


void convex_hull_cutting_update() {
    using namespace convex_map;
    const double eps = 1e-3;
    // don't omp
    for (int idx = 0; idx < H * W; idx++) {
        if (face_map::new_head[idx] == -1) continue;
        // reinitialize 
        int head_id;
        #pragma omp critical
        {
            super_head[idx] = -1;
            super_add(idx, -1);
            head_id = super_cnt - 1;
        }
        #pragma omp critical
        {
            add(head + head_id, 0., 0.);
            add(head + head_id, 1., 0.);
            add(head + head_id, 1., 1.);
            add(head + head_id, 0., 1.);
        }
        // merge sort
        int *tail = &face_map::head[idx];
        int incoming_i = face_map::new_head[idx], existing_i = face_map::head[idx];
        for (;;) {
            if (incoming_i == -1 && existing_i == -1) break;
            bool flag = 0;
            if (incoming_i == -1 && existing_i != -1) {
                flag = 1;
            }
            else if (!(incoming_i != -1 && existing_i == -1) && face_map::depth[incoming_i] > face_map::depth[existing_i]) {
                flag = 1;
            }
            if (flag) {
                *tail = existing_i;
                tail = &face_map::nxt[existing_i];
                existing_i = face_map::nxt[existing_i];
            }
            else {
                *tail = incoming_i;
                tail = &face_map::nxt[incoming_i];
                incoming_i = face_map::nxt[incoming_i];
            }
        }

        // traverse existing_face_head
        for (int i = face_map::head[idx]; i != -1; i = face_map::nxt[i]) {
            int f = face_map::id[i];
            if (face_map::backface != NULL && face_map::backface[f]) continue;
            int p[3]{mesh::faces[f * 3], mesh::faces[f * 3 + 1], mesh::faces[f * 3 + 2]};
            double v[6] = {
                mesh::vertices[p[0] * 3] - idx / W, mesh::vertices[p[0] * 3 + 1] - idx % W,
                mesh::vertices[p[1] * 3] - idx / W, mesh::vertices[p[1] * 3 + 1] - idx % W,
                mesh::vertices[p[2] * 3] - idx / W, mesh::vertices[p[2] * 3 + 1] - idx % W
            };
            bool aligned[6] = {0, 0, 0, 0, 0, 0};
            for (int iv = 0; iv < 6; iv++) {
                if (abs(v[iv] - round(v[iv])) < eps) {
                    aligned[iv] = 1;
                    v[iv] = round(v[iv]);
                }
            }
            bool aligned_flag = 1;
            for (int iv = 0; iv < 3; iv++)
                aligned_flag &= aligned[iv * 2] || aligned[iv * 2 + 1];
            if (not aligned_flag) {
                mesh::visibility[f] = 1;
                continue;
            }
            bool zero_area = 0;
            for (int j = 0; j < 3; j++)
                if (abs(v[j * 2] - v[(j+1)%3 * 2]) < eps && abs(v[j * 2 + 1] - v[(j+1)%3 * 2 + 1]) < eps)
                {
                    zero_area = 1;
                    break;
                }
            if (zero_area) {
                mesh::visibility[f] = 1;
                continue;
            }
            mesh::visibility[f] = cut_multiple_convex(idx, v);
            if (i == face_map::head[idx]) assert(mesh::visibility[f]);
            if (super_head[idx] == -1) break;
        }
    }
}

namespace extend {
    const int hash_cap = 100663319;
    int *adjacency, *head, *uvf, *nxt, cnt, M;

    void init(int M_) {
        cnt = 0;
        M = M_;
        HM(adjacency, M * 3);
        HM(head, hash_cap);
        HM(uvf, M * 9);
        HM(nxt, M * 3);
        SETN(head, hash_cap);
    }

    void cleanup() {
        safefree(adjacency);
        safefree(head);
        safefree(uvf);
        safefree(nxt);
    }

    void adj_table_fillup(
        int *faces
    ) {
        #pragma omp parallel for
        for (int idx = 0; idx < M; idx++) {
            for (int i = 0; i < 3; i++) {
                int u = faces[idx * 3 + i], v = faces[idx * 3 + (i + 1) % 3];
                int hash_value = myhash(myhash(u) + v) % hash_cap;
                int mycnt;
                #pragma omp critical
                mycnt = cnt++;
                uvf[mycnt * 3] = u;
                uvf[mycnt * 3 + 1] = v;
                uvf[mycnt * 3 + 2] = idx;
                int old_heads;
                #pragma omp critical
                {
                    old_heads = head[hash_value];
                    head[hash_value] = mycnt;
                }
                nxt[mycnt] = old_heads;
            }
        }
    }

    void adj_table_query(
        int *faces
    ) {
        SETN(adjacency, M * 3);
        #pragma omp parallel for
        for (int idx = 0; idx < M; idx++) {
            for (int i = 0; i < 3; i++) {
                int u = faces[idx * 3 + i], v = faces[idx * 3 + (i + 1) % 3];
                int hash_value = myhash(myhash(v) + u) % hash_cap;
                for (int j = head[hash_value]; j != -1; j = nxt[j]) {
                    if (uvf[j * 3] == v && uvf[j * 3 + 1] == u) {
                        adjacency[idx * 3 + i] = uvf[j * 3 + 2];
                    }
                }
            }
        }
    }

    void expand_visibility(
        int *faces, int M_,
        int *visibility,
        int iters
    ) {
        if (iters == 0) return;
        init(M_);
        adj_table_fillup(faces);
        adj_table_query(faces);
        int *visibility_copy;
        HM(visibility_copy, M);
        while (iters--) {
            #pragma omp parallel for
            for (int idx = 0; idx < M; idx++) {
                visibility_copy[idx] = visibility[idx];
                for (int i = 0; i < 3; i++) {
                    if (adjacency[idx * 3 + i] != -1) {
                        visibility_copy[idx] |= visibility[adjacency[idx * 3 + i]];
                    }
                }
            }
            memcpy(visibility, visibility_copy, M * sizeof(int));
        }
        cleanup();
    }

}

void initialize_visibility_engine(
    int H, int W, int R
) {
    face_map::init(H, W);
    convex_map::init(H, W);
    depth_table::R = R;
}

void cleanup_visibility_engine() {
    face_map::cleanup();
    convex_map::cleanup();
}

void visibility_engine_update(
    double *vertices,
    int *visibility,
    int *faces, int M1, int M2,
    int skip_backface=1
) {
    mesh::faces = faces;
    mesh::visibility = visibility;
    mesh::vertices = vertices;
    depth_table::init();
    compute_face_order(M1, M2, skip_backface);
    depth_table::cleanup();
    convex_hull_cutting_update();
}

int visibility_engine_block_query(
    int h_min, int h_max,
    int w_min, int w_max,
    int depth
) {
    int ans = 0;
    for (int h = h_min; h < h_max && !ans; h++)
    for (int w = w_min; w < w_max && !ans; w++) {
        if (convex_map::super_head[h * convex_map::W + w] != -1) ans = 1;
        else {
            bool flag = 0;
            int deepest_visible = 0;
            for (int j = face_map::head[h * convex_map::W + w]; j != -1; j = face_map::nxt[j])
                if (mesh::visibility[face_map::id[j]]) {
                    deepest_visible = face_map::depth[j];
                    flag = 1;
                }
            assert(flag);
            ans = deepest_visible > depth;
        }
    }
    return ans;
}

void deepest_query(int *deepest) {
    for (int h = 0; h < face_map::H; h++)
    for (int w = 0; w < face_map::W; w++) {
        if (convex_map::super_head[h * convex_map::W + w] != -1) deepest[h * face_map::W + w] = depth_table::R;
        else {
            bool flag = 0;
            for (int j = face_map::head[h * convex_map::W + w]; j != -1; j = face_map::nxt[j])
                if (mesh::visibility[face_map::id[j]]) {
                    deepest[h * face_map::W + w] = face_map::depth[j] / 6;
                    flag = 1;
                }
            assert(flag);
        }
    }
}

bool update_coarse_visibility(
    double *vertices,
    int *face,
    int upscale,
    int t0, int t1, int p0, int p1
) {
    double v[6] = {
        vertices[face[0] * 3] * upscale, vertices[face[0] * 3 + 1] * upscale,
        vertices[face[1] * 3] * upscale, vertices[face[1] * 3 + 1] * upscale,
        vertices[face[2] * 3] * upscale, vertices[face[2] * 3 + 1] * upscale
    };
    double d[3] = {
        vertices[face[0] * 3 + 2] * upscale,
        vertices[face[1] * 3 + 2] * upscale,
        vertices[face[2] * 3 + 2] * upscale,
    };

    for (int t = t0; t < t1; t++)
        for (int p = p0; p < p1; p++) {
            int intersect = 0;
            int square[8] = {
                t, p,
                t + 1, p,
                t + 1, p + 1,
                t, p + 1
            };

            for (int v_square = 0; v_square < 4 && !intersect; v_square++)
                for (int v_triangle = 0; v_triangle < 3 && !intersect; v_triangle++) {
                    if (seg_seg_intersect(
                        square[v_square * 2],
                        square[v_square * 2 + 1],
                        square[(v_square + 1) % 4 * 2],
                        square[(v_square + 1) % 4 * 2 + 1],
                        v[v_triangle * 2],
                        v[v_triangle * 2 + 1],
                        v[(v_triangle + 1) % 3 * 2],
                        v[(v_triangle + 1) % 3 * 2 + 1]
                    )) intersect = 1;
                }

            for (int dt = 0; dt < 2 && !intersect; dt++) {
                for (int dp = 0; dp < 2 && !intersect; dp++) {
                    double cross2 = (v[0] - t - dt) * (v[3] - p - dp) - (v[2] - t - dt) * (v[1] - p - dp);
                    double cross0 = (v[2] - t - dt) * (v[5] - p - dp) - (v[4] - t - dt) * (v[3] - p - dp);
                    double cross1 = (v[4] - t - dt) * (v[1] - p - dp) - (v[0] - t - dt) * (v[5] - p - dp);
                    if ((cross0 >= 0 && cross1 >= 0 && cross2 >= 0) || (cross0 <= 0 && cross1 <= 0 && cross2 <= 0)) {
                        intersect = 1;
                        break;
                    }
                }
            }
            if (!intersect) {
                for (int j = 0; j < 3; j++)
                    if (v[2 * j] >= t && v[2 * j] <= t + 1 && v[2 * j + 1] >= p && v[2 * j + 1] <= p + 1) {
                        intersect = 1;
                        break;
                    }
            }
            if (!intersect) continue;
            for (int j = face_map::head[t * face_map::W + p]; j != -1; j = face_map::nxt[j])
            {
                int f = face_map::id[j];
                if (!mesh::visibility[f]) continue;
                if (face_map::depth[j] > d[0] * 6) { // d[1] or d[2] should be the same
                    return 1;
                }
            }
        }
    return 0;
}
