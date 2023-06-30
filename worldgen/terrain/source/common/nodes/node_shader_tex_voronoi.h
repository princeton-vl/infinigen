/* SPDX-License-Identifier: GPL-2.0-or-later
 * Copyright 2005 Blender Foundation. All rights reserved.
 * adapted by Zeyu Ma on date June 4, 2023 to compile geometry nodes */

#ifndef __TEXVORONOI__
#define __TEXVORONOI__

DEVICE_FUNC void node_shader_tex_voronoi(
    // parameters
    int dimensions_,
    int feature_,
    int metric_,
    // input
    float3_nonbuiltin vector,
    float w,
    float scale,
    float smoothness,
    float exponent,
    float randomness,
    // output
    float *r_distance,
    float4_nonbuiltin *r_color,
    float3_nonbuiltin *r_position,
    float *r_w,
    float *r_radius) {
    const bool calc_distance = r_distance != NULL;
    const bool calc_color = r_color != NULL;
    const bool calc_position = r_position != NULL;
    const bool calc_w = r_w != NULL;
    const bool calc_radius = r_radius != NULL;
    const float rand = min(max(randomness, 0.0f), 1.0f);
    const float smth = min(max(smoothness / 2.0f, 0.0f), 0.5f);
    float3_nonbuiltin col;

    bool minowski =
        metric_ == SHD_VORONOI_MINKOWSKI && dimensions_ != 1 &&
        feature_ != SHD_VORONOI_DISTANCE_TO_EDGE && feature_ != SHD_VORONOI_N_SPHERE_RADIUS;
    bool dist_radius = feature_ == SHD_VORONOI_DISTANCE_TO_EDGE || feature_ == SHD_VORONOI_N_SPHERE_RADIUS;
    if (dist_radius) {
        switch (dimensions_) {
            case 1: {
                const float p = w * scale;

                switch (feature_) {
                    case SHD_VORONOI_DISTANCE_TO_EDGE: {
                        if (calc_distance) voronoi_distance_to_edge(p, rand, r_distance);
                        break;
                    }
                    case SHD_VORONOI_N_SPHERE_RADIUS: {
                        if (calc_radius) voronoi_n_sphere_radius(p, rand, r_radius);
                        break;
                    }
                }
                break;
            }
            case 2: {
                const float2_nonbuiltin p = float2_nonbuiltin(vector.x, vector.y) * scale;
                switch (feature_) {
                    case SHD_VORONOI_DISTANCE_TO_EDGE: {
                        if (calc_distance) voronoi_distance_to_edge(p, rand, r_distance);

                        break;
                    }
                    case SHD_VORONOI_N_SPHERE_RADIUS: {
                        if (calc_radius) voronoi_n_sphere_radius(p, rand, r_radius);
                        break;
                    }
                }
                break;
            }
            case 3: {
                switch (feature_) {
                    case SHD_VORONOI_DISTANCE_TO_EDGE: {
                        if (calc_distance) voronoi_distance_to_edge(vector * scale, rand, r_distance);
                        break;
                    }
                    case SHD_VORONOI_N_SPHERE_RADIUS: {
                        if (calc_radius) voronoi_n_sphere_radius(vector * scale, rand, r_radius);
                        break;
                    }
                }
                break;
            }
            case 4: {
                const float4_nonbuiltin p = float4_nonbuiltin(vector.x, vector.y, vector.z, w) * scale;
                switch (feature_) {
                    case SHD_VORONOI_DISTANCE_TO_EDGE: {
                        if (calc_distance) voronoi_distance_to_edge(p, rand, r_distance);

                        break;
                    }
                    case SHD_VORONOI_N_SPHERE_RADIUS: {
                        if (calc_radius) voronoi_n_sphere_radius(p, rand, r_radius);
                        break;
                    }
                }
                break;
            }
        }
    } else if (minowski) {
        switch (dimensions_) {
            case 2: {
                float2_nonbuiltin pos;
                switch (feature_) {
                    case SHD_VORONOI_F1: {
                        voronoi_f1(float2_nonbuiltin(vector.x, vector.y) * scale,
                                   exponent,
                                   rand,
                                   SHD_VORONOI_MINKOWSKI,
                                   r_distance,
                                   calc_color ? &col : nullptr,
                                   calc_position ? &pos : nullptr);

                        break;
                    }
                    case SHD_VORONOI_F2: {
                        voronoi_f2(float2_nonbuiltin(vector.x, vector.y) * scale,
                                   exponent,
                                   rand,
                                   SHD_VORONOI_MINKOWSKI,
                                   r_distance,
                                   calc_color ? &col : nullptr,
                                   calc_position ? &pos : nullptr);

                        break;
                    }
                    case SHD_VORONOI_SMOOTH_F1: {
                        voronoi_smooth_f1(float2_nonbuiltin(vector.x, vector.y) * scale,
                                          smth,
                                          exponent,
                                          rand,
                                          SHD_VORONOI_MINKOWSKI,
                                          r_distance,
                                          calc_color ? &col : nullptr,
                                          calc_position ? &pos : nullptr);
                        break;
                    }
                }
                if (calc_color) {
                    *r_color = float4_nonbuiltin(col.x, col.y, col.z, 1.0f);
                }
                if (calc_position) {
                    pos = safe_divide(pos, scale);
                    *r_position = float3_nonbuiltin(pos.x, pos.y, 0.0f);
                }
                break;
            }
            case 3: {
                switch (feature_) {
                    case SHD_VORONOI_F1: {
                        voronoi_f1(vector * scale,
                                   exponent,
                                   rand,
                                   SHD_VORONOI_MINKOWSKI,
                                   r_distance,
                                   calc_color ? &col : nullptr,
                                   r_position);

                        break;
                    }
                    case SHD_VORONOI_F2: {
                        voronoi_f2(vector * scale,
                                   exponent,
                                   rand,
                                   SHD_VORONOI_MINKOWSKI,
                                   r_distance,
                                   calc_color ? &col : nullptr,
                                   r_position);

                        break;
                    }
                    case SHD_VORONOI_SMOOTH_F1: {
                        voronoi_smooth_f1(vector * scale,
                                          smth,
                                          exponent,
                                          rand,
                                          SHD_VORONOI_MINKOWSKI,
                                          r_distance,
                                          calc_color ? &col : nullptr,
                                          r_position);

                        break;
                    }
                }
                if (calc_color) {
                    *r_color = float4_nonbuiltin(col.x, col.y, col.z, 1.0f);
                }
                if (calc_position) {
                    *r_position = safe_divide(*r_position, scale);
                }
                break;
            }
            case 4: {
                const float4_nonbuiltin p = float4_nonbuiltin(vector.x, vector.y, vector.z, w) * scale;
                float4_nonbuiltin pos;
                switch (feature_) {
                    case SHD_VORONOI_F1: {
                        voronoi_f1(p,
                                   exponent,
                                   rand,
                                   SHD_VORONOI_F1,
                                   r_distance,
                                   calc_color ? &col : nullptr,
                                   calc_position || calc_w ? &pos : nullptr);

                        break;
                    }
                    case SHD_VORONOI_F2: {
                        voronoi_f2(p,
                                   exponent,
                                   rand,
                                   SHD_VORONOI_F1,
                                   r_distance,
                                   calc_color ? &col : nullptr,
                                   calc_position || calc_w ? &pos : nullptr);

                        break;
                    }
                    case SHD_VORONOI_SMOOTH_F1: {
                        voronoi_smooth_f1(p,
                                          smth,
                                          exponent,
                                          rand,
                                          SHD_VORONOI_F1,
                                          r_distance,
                                          calc_color ? &col : nullptr,
                                          calc_position || calc_w ? &pos : nullptr);

                        break;
                    }
                }
                if (calc_color) {
                    *r_color = float4_nonbuiltin(col.x, col.y, col.z, 1.0f);
                }
                if (calc_position || calc_w) {
                    pos = safe_divide(pos, scale);
                    if (calc_position) {
                        *r_position = float3_nonbuiltin(pos.x, pos.y, pos.z);
                    }
                    if (calc_w) {
                        *r_w = pos.w;
                    }
                }
                break;
            }
        }
    } else {
        switch (dimensions_) {
            case 1: {
                const float p = w * scale;
                switch (feature_) {
                    case SHD_VORONOI_F1: {
                        voronoi_f1(p,
                                   rand,
                                   r_distance,
                                   calc_color ? &col : nullptr,
                                   r_w);

                        break;
                    }
                    case SHD_VORONOI_F2: {
                        voronoi_f2(p,
                                   rand,
                                   r_distance,
                                   calc_color ? &col : nullptr,
                                   r_w);

                        break;
                    }
                    case SHD_VORONOI_SMOOTH_F1: {
                        voronoi_smooth_f1(p,
                                          smth,
                                          rand,
                                          r_distance,
                                          calc_color ? &col : nullptr,
                                          r_w);
                        break;
                    }
                }
                if (calc_color) {
                    *r_color = float4_nonbuiltin(col.x, col.y, col.z, 1.0f);
                }
                if (calc_position) {
                    *r_w = safe_divide(*r_w, scale);
                }
                break;
            }
            case 2: {
                float2_nonbuiltin pos;
                switch (feature_) {
                    case SHD_VORONOI_F1: {
                        voronoi_f1(float2_nonbuiltin(vector.x, vector.y) * scale,
                                   0.0f,
                                   rand,
                                   metric_,
                                   r_distance,
                                   calc_color ? &col : nullptr,
                                   calc_position ? &pos : nullptr);

                        break;
                    }
                    case SHD_VORONOI_F2: {
                        voronoi_f2(float2_nonbuiltin(vector.x, vector.y) * scale,
                                   0.0f,
                                   rand,
                                   metric_,
                                   r_distance,
                                   calc_color ? &col : nullptr,
                                   calc_position ? &pos : nullptr);

                        break;
                    }
                    case SHD_VORONOI_SMOOTH_F1: {
                        voronoi_smooth_f1(float2_nonbuiltin(vector.x, vector.y) * scale,
                                          smth,
                                          0.0f,
                                          rand,
                                          metric_,
                                          r_distance,
                                          calc_color ? &col : nullptr,
                                          calc_position ? &pos : nullptr);

                        break;
                    }
                }

                if (calc_color) {
                    *r_color = float4_nonbuiltin(col.x, col.y, col.z, 1.0f);
                }
                if (calc_position) {
                    pos = safe_divide(pos, scale);
                    *r_position = float3_nonbuiltin(pos.x, pos.y, 0.0f);
                }

                break;
            }
            case 3: {
                switch (feature_) {
                    case SHD_VORONOI_F1: {
                        voronoi_f1(vector * scale,
                                   0.0f,
                                   rand,
                                   metric_,
                                   r_distance,
                                   calc_color ? &col : nullptr,
                                   r_position);

                        break;
                    }
                    case SHD_VORONOI_F2: {
                        voronoi_f2(vector * scale,
                                   0.0f,
                                   rand,
                                   metric_,
                                   r_distance,
                                   calc_color ? &col : nullptr,
                                   r_position);
                        break;
                    }
                    case SHD_VORONOI_SMOOTH_F1: {
                        voronoi_smooth_f1(vector * scale,
                                          smth,
                                          0.0f,
                                          rand,
                                          metric_,
                                          r_distance,
                                          calc_color ? &col : nullptr,
                                          r_position);
                        break;
                    }
                }

                if (calc_color) {
                    *r_color = float4_nonbuiltin(col.x, col.y, col.z, 1.0f);
                }
                if (calc_position) {
                    *r_position = safe_divide(*r_position, scale);
                }
                break;
            }
            case 4: {
                const float4_nonbuiltin p = float4_nonbuiltin(vector.x, vector.y, vector.z, w) * scale;
                float4_nonbuiltin pos;
                switch (feature_) {
                    case SHD_VORONOI_F1: {
                        voronoi_f1(p,
                                   0.0f,
                                   rand,
                                   metric_,
                                   r_distance,
                                   calc_color ? &col : nullptr,
                                   calc_position || calc_w ? &pos : nullptr);

                        break;
                    }
                    case SHD_VORONOI_F2: {
                        voronoi_f2(p,
                                   0.0f,
                                   rand,
                                   metric_,
                                   r_distance,
                                   calc_color ? &col : nullptr,
                                   calc_position || calc_w ? &pos : nullptr);
                        break;
                    }
                    case SHD_VORONOI_SMOOTH_F1: {
                        voronoi_smooth_f1(p,
                                          smth,
                                          0.0f,
                                          rand,
                                          metric_,
                                          r_distance,
                                          calc_color ? &col : nullptr,
                                          calc_position || calc_w ? &pos : nullptr);
                        break;
                    }
                }

                if (calc_color) {
                    *r_color = float4_nonbuiltin(col.x, col.y, col.z, 1.0f);
                }
                if (calc_position || calc_w) {
                    pos = safe_divide(pos, scale);
                    if (calc_position) {
                        *r_position = float3_nonbuiltin(pos.x, pos.y, pos.z);
                    }
                    if (calc_w) {
                        *r_w = pos.w;
                    }
                }
                break;
            }
        }
    }
}

#endif