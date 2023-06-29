// Copyright (c) Princeton University.
// This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

// Authors: Zeyu Ma


DEVICE_FUNC void warped_rocks(
    float3_nonbuiltin position,
    float *sdf,
    float *auxs,
    int is_caved,
    int *i_params, float *f_params,
    int *caves_i_params, float *caves_f_params
) {

    /* params
    int:
    seed, slope_is_3d
    
    float:
    supressing_param,
    content_min_freq, content_max_freq, content_octaves, content_scale,
    warp_min_freq, warp_max_freq, warp_octaves, warp_scale,
    slope_freq, slope_octaves, slope_scale, slope_shift,
    */
    
    int seed = i_params[0];
    int slope_is_3d = i_params[1];
    float supressing_param = f_params[0];
    float content_min_freq = f_params[1];
    float content_max_freq = f_params[2];
    float content_octaves = f_params[3];
    float content_scale = f_params[4];
    float warp_min_freq = f_params[5];
    float warp_max_freq = f_params[6];
    float warp_octaves = f_params[7];
    float warp_scale = f_params[8];
    float slope_freq = f_params[9];
    float slope_octaves = f_params[10];
    float slope_scale = f_params[11];
    float slope_shift = f_params[12];
    float3_nonbuiltin position_original = position;

    float content_freq = log_uniform(content_min_freq, content_max_freq, myhash(seed, 0));
    float warp_freq = log_uniform(warp_min_freq, warp_max_freq, myhash(seed, 1));
    float3_nonbuiltin warp = float3_nonbuiltin(
        Perlin(position.x, position.y, position.z, myhash(seed, 2), warp_octaves, warp_freq) * warp_scale,
        Perlin(position.x, position.y, position.z, myhash(seed, 3), warp_octaves, warp_freq) * warp_scale,
        Perlin(position.x, position.y, position.z, myhash(seed, 4), warp_octaves, warp_freq) * warp_scale
    );
    float3_nonbuiltin position_warped = position + warp;
    float content = Perlin(position_warped.x, position_warped.y, position_warped.z, myhash(seed, 5), content_octaves, content_freq) * content_scale;
    if (!slope_is_3d) position.z = 0;
    float slope = Perlin(position.x, position.y, position.z, myhash(seed, 6), slope_octaves, slope_freq) * slope_scale + slope_shift;
    *sdf = (position_original.z - slope) * supressing_param + content;
    if (is_caved) {
        float prior_sdf2 = *sdf;
        caves(position_original, sdf, caves_i_params, caves_f_params, prior_sdf2);
        if (auxs != NULL) auxs[0] = prior_sdf2 < *sdf;
    }
    else if (auxs != NULL) auxs[0] = 0;
}