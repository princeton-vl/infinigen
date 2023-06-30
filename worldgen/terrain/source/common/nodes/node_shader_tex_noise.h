/* SPDX-License-Identifier: GPL-2.0-or-later
 * Copyright 2005 Blender Foundation. All rights reserved.
 * adapted by Zeyu Ma on date June 4, 2023 to compile geometry nodes */

#ifndef __TEXNOISEE__
#define __TEXNOISEE__

DEVICE_FUNC void node_shader_tex_noise(
    // parameters
    int dimensions_,
    // input
    float3_nonbuiltin vector,
    float w,
    float scale,
    float detail,
    float roughness,
    float distortion,
    // output,
    float *fac,
    float4_nonbuiltin *color) {
    const bool compute_factor = fac != NULL;
    const bool compute_color = color != NULL;

    switch (dimensions_) {
        case 1: {
            if (compute_color) {
                const float position = w * scale;
                const float3_nonbuiltin c = perlin_float3_fractal_distorted(
                    position, detail, roughness, distortion);
                *color = float4_nonbuiltin(c.x, c.y, c.z, 1.0f);
            }
            if (compute_factor) {
                const float position = w * scale;
                *fac = perlin_fractal_distorted(
                    position, detail, roughness, distortion);
            }
            break;
        }
        case 2: {
            if (compute_color) {
                const float2_nonbuiltin position = float2_nonbuiltin(vector * scale);
                const float3_nonbuiltin c = perlin_float3_fractal_distorted(
                    position, detail, roughness, distortion);
                *color = float4_nonbuiltin(c.x, c.y, c.z, 1.0f);
            }
            if (compute_factor) {
                const float2_nonbuiltin position = float2_nonbuiltin(vector * scale);
                *fac = perlin_fractal_distorted(
                    position, detail, roughness, distortion);
            }
            break;
        }
        case 3: {
            if (compute_color) {
                const float3_nonbuiltin position = vector * scale;
                const float3_nonbuiltin c = perlin_float3_fractal_distorted(
                    position, detail, roughness, distortion);
                *color = float4_nonbuiltin(c.x, c.y, c.z, 1.0f);
            }
            if (compute_factor) {
                const float3_nonbuiltin position = vector * scale;
                *fac = perlin_fractal_distorted(
                    position, detail, roughness, distortion);
            }
            break;
        }
        case 4: {
            if (compute_color) {
                const float3_nonbuiltin position_vector = vector * scale;
                const float position_w = w * scale;
                const float4_nonbuiltin position{
                    position_vector.x, position_vector.y, position_vector.z, position_w};
                const float3_nonbuiltin c = perlin_float3_fractal_distorted(
                    position, detail, roughness, distortion);
                *color = float4_nonbuiltin(c.x, c.y, c.z, 1.0f);
            }
            if (compute_factor) {
                const float3_nonbuiltin position_vector = vector * scale;
                const float position_w = w * scale;
                const float4_nonbuiltin position{position_vector.x, position_vector.y, position_vector.z, position_w};
                *fac = perlin_fractal_distorted(
                    position, detail, roughness, distortion);
                break;
            }
        }
    }
}

#endif