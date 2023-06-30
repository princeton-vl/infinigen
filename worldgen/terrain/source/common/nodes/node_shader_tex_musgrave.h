/* SPDX-License-Identifier: GPL-2.0-or-later
 * Copyright 2005 Blender Foundation. All rights reserved.
 * adapted by Zeyu Ma on date June 4, 2023 to compile geometry nodes */

#ifndef __TEXMUSGRAVE__
#define __TEXMUSGRAVE__

DEVICE_FUNC void node_shader_tex_musgrave(
    // parameters
    int dimensions_,
    int musgrave_type_,
    // input
    float3_nonbuiltin vector,
    float w,
    float scale,
    float detail,
    float dimension,
    float lacunarity,
    float offset,
    float gain,
    // output,
    float *fac) {
    const bool compute_factor = fac != NULL;
    if (!compute_factor)
        return;
    switch (musgrave_type_) {
        case SHD_MUSGRAVE_MULTIFRACTAL: {
            switch (dimensions_) {
                case 1: {
                    const float position = w * scale;
                    *fac = musgrave_multi_fractal(
                        position, dimension, lacunarity, detail);
                    break;
                }
                case 2: {
                    const float3_nonbuiltin pxyz = vector * scale;
                    const float2_nonbuiltin position = float2_nonbuiltin(pxyz.x, pxyz.y);
                    *fac = musgrave_multi_fractal(
                        position, dimension, lacunarity, detail);
                    break;
                }
                case 3: {
                    const float3_nonbuiltin position = vector * scale;
                    *fac = musgrave_multi_fractal(
                        position, dimension, lacunarity, detail);
                    break;
                }
                case 4: {
                    const float3_nonbuiltin pxyz = vector * scale;
                    const float pw = w * scale;
                    const float4_nonbuiltin position{pxyz.x, pxyz.y, pxyz.z, pw};
                    *fac = musgrave_multi_fractal(
                        position, dimension, lacunarity, detail);
                    break;
                }
            }
            break;
        }
        case SHD_MUSGRAVE_RIDGED_MULTIFRACTAL: {
            switch (dimensions_) {
                case 1: {
                    const float position = w * scale;
                    *fac = musgrave_ridged_multi_fractal(
                        position, dimension, lacunarity, detail, offset, gain);
                    break;
                }
                case 2: {
                    const float3_nonbuiltin pxyz = vector * scale;
                    const float2_nonbuiltin position = float2_nonbuiltin(pxyz.x, pxyz.y);
                    *fac = musgrave_ridged_multi_fractal(
                        position, dimension, lacunarity, detail, offset, gain);
                    break;
                }
                case 3: {
                    const float3_nonbuiltin position = vector * scale;
                    *fac = musgrave_ridged_multi_fractal(
                        position, dimension, lacunarity, detail, offset, gain);
                    break;
                }
                case 4: {
                    const float3_nonbuiltin pxyz = vector * scale;
                    const float pw = w * scale;
                    const float4_nonbuiltin position{pxyz.x, pxyz.y, pxyz.z, pw};
                    *fac = musgrave_ridged_multi_fractal(
                        position, dimension, lacunarity, detail, offset, gain);
                    break;
                }
            }
            break;
        }
        case SHD_MUSGRAVE_HYBRID_MULTIFRACTAL: {
            switch (dimensions_) {
                case 1: {
                    const float position = w * scale;
                    *fac = musgrave_hybrid_multi_fractal(
                        position, dimension, lacunarity, detail, offset, gain);
                    break;
                }
                case 2: {
                    const float3_nonbuiltin pxyz = vector * scale;
                    const float2_nonbuiltin position = float2_nonbuiltin(pxyz.x, pxyz.y);
                    *fac = musgrave_hybrid_multi_fractal(
                        position, dimension, lacunarity, detail, offset, gain);
                    break;
                }
                case 3: {
                    const float3_nonbuiltin position = vector * scale;
                    *fac = musgrave_hybrid_multi_fractal(
                        position, dimension, lacunarity, detail, offset, gain);
                    break;
                }
                case 4: {
                    const float3_nonbuiltin pxyz = vector * scale;
                    const float pw = w * scale;
                    const float4_nonbuiltin position{pxyz.x, pxyz.y, pxyz.z, pw};
                    *fac = musgrave_hybrid_multi_fractal(
                        position, dimension, lacunarity, detail, offset, gain);
                    break;
                }
            }
            break;
        }
        case SHD_MUSGRAVE_FBM: {
            switch (dimensions_) {
                case 1: {
                    const float position = w * scale;
                    *fac = musgrave_fBm(
                        position, dimension, lacunarity, detail);
                    break;
                }
                case 2: {
                    const float3_nonbuiltin pxyz = vector * scale;
                    const float2_nonbuiltin position = float2_nonbuiltin(pxyz.x, pxyz.y);
                    *fac = musgrave_fBm(
                        position, dimension, lacunarity, detail);
                    break;
                }
                case 3: {
                    const float3_nonbuiltin position = vector * scale;
                    *fac = musgrave_fBm(
                        position, dimension, lacunarity, detail);
                    break;
                }
                case 4: {
                    const float3_nonbuiltin pxyz = vector * scale;
                    const float pw = w * scale;
                    const float4_nonbuiltin position{pxyz.x, pxyz.y, pxyz.z, pw};
                    *fac = musgrave_fBm(
                        position, dimension, lacunarity, detail);
                    break;
                }
            }
            break;
        }
        case SHD_MUSGRAVE_HETERO_TERRAIN: {
            switch (dimensions_) {
                case 1: {
                    const float position = w * scale;
                    *fac = musgrave_hetero_terrain(
                        position, dimension, lacunarity, detail, offset);
                    break;
                }
                case 2: {
                    const float3_nonbuiltin pxyz = vector * scale;
                    const float2_nonbuiltin position = float2_nonbuiltin(pxyz.x, pxyz.y);
                    *fac = musgrave_hetero_terrain(
                        position, dimension, lacunarity, detail, offset);
                    break;
                }
                case 3: {
                    const float3_nonbuiltin position = vector * scale;
                    *fac = musgrave_hetero_terrain(
                        position, dimension, lacunarity, detail, offset);
                    break;
                }
                case 4: {
                    const float3_nonbuiltin pxyz = vector * scale;
                    const float pw = w * scale;
                    const float4_nonbuiltin position{pxyz.x, pxyz.y, pxyz.z, pw};
                    *fac = musgrave_hetero_terrain(
                        position, dimension, lacunarity, detail, offset);
                    break;
                }
            }
            break;
        }
    }
}

#endif