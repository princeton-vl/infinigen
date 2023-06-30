/* SPDX-License-Identifier: GPL-2.0-or-later
 * Copyright 2005 Blender Foundation. All rights reserved.
 * adapted by Zeyu Ma on date June 4, 2023 to compile geometry nodes */

#ifndef __MIXRGB__
#define __MIXRGB__

DEVICE_FUNC void node_shader_mix_rgb(
    // params
    int type_,
    int clamp_,
    // input
    float fac,
    float4_nonbuiltin color1,
    float4_nonbuiltin color2,
    // output
    float4_nonbuiltin *color) {
    float results[3]{color1.x, color1.y, color1.z};
    float color2_array[3]{color2.x, color2.y, color2.z};

    ramp_blend(type_, results, CLAMPIS(fac, 0.0f, 1.0f), color2_array);
    if (clamp_) {
        results[0] = clamp_range(results[0], 0.0f, 1.0f);
        results[1] = clamp_range(results[1], 0.0f, 1.0f);
        results[2] = clamp_range(results[2], 0.0f, 1.0f);
    }
    if (color != NULL) {
        *color = float4_nonbuiltin(results[0], results[1], results[2], 0);
    }
}

#endif