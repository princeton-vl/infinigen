/* SPDX-License-Identifier: GPL-2.0-or-later
 * Copyright 2005 Blender Foundation. All rights reserved.
 * adapted by Zeyu Ma on date June 4, 2023 to compile geometry nodes */

#ifndef __TEXWAVE__
#define __TEXWAVE__

DEVICE_FUNC void node_shader_tex_wave(
    // parameters
    int wave_type_,
    int bands_direction_,
    int rings_direction_,
    int wave_profile_,
    // input
    float3_nonbuiltin vector,
    float scale,
    float distortion,
    float detail,
    float detail_scale,
    float detail_roughness,
    float phase_offset,
    // output,
    float4_nonbuiltin *color,
    float *fac) {
    float3_nonbuiltin p = vector * scale;
    /* Prevent precision issues on unit coordinates. */
    p = (p + 0.000001f) * 0.999999f;

    float n = 0.0f;
    float val = 0.0f;

    switch (wave_type_) {
        case SHD_WAVE_BANDS:
            switch (bands_direction_) {
                case SHD_WAVE_BANDS_DIRECTION_X:
                    n = p.x * 20.0f;
                    break;
                case SHD_WAVE_BANDS_DIRECTION_Y:
                    n = p.y * 20.0f;
                    break;
                case SHD_WAVE_BANDS_DIRECTION_Z:
                    n = p.z * 20.0f;
                    break;
                case SHD_WAVE_BANDS_DIRECTION_DIAGONAL:
                    n = (p.x + p.y + p.z) * 10.0f;
                    break;
            }
            break;
        case SHD_WAVE_RINGS:
            float3_nonbuiltin rp = p;
            switch (rings_direction_) {
                case SHD_WAVE_RINGS_DIRECTION_X:
                    rp.x = 0;
                    break;
                case SHD_WAVE_RINGS_DIRECTION_Y:
                    rp.y = 0;
                    break;
                case SHD_WAVE_RINGS_DIRECTION_Z:
                    rp.z = 0;
                    break;
                case SHD_WAVE_RINGS_DIRECTION_SPHERICAL:
                    /* Ignore. */
                    break;
            }
            n = length(rp) * 20.0f;
            break;
    }

    n += phase_offset;

    if (distortion != 0.0f) {
        n += distortion * (perlin_fractal(p * detail_scale, detail, detail_roughness) * 2.0f - 1.0f);
    }

    switch (wave_profile_) {
        case SHD_WAVE_PROFILE_SIN:
            val = 0.5f + 0.5f * sinf(n - M_PI_2);
            break;
        case SHD_WAVE_PROFILE_SAW:
            n /= M_PI * 2.0f;
            val = n - floorf(n);
            break;
        case SHD_WAVE_PROFILE_TRI:
            n /= M_PI * 2.0f;
            val = fabsf(n - floorf(n + 0.5f)) * 2.0f;
            break;
    }
    if (fac != 0) *fac = val;
    if (color != 0) *color = float4_nonbuiltin(val, val, val, 1);
}

#endif