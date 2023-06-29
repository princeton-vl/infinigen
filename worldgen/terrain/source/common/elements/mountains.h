// Copyright (c) Princeton University.
// This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

// Authors: Zeyu Ma


DEVICE_FUNC void mountains(
    float3_nonbuiltin position,
    float *sdf,
    int *i_params,
    float *f_params
) {
    /* params
    int:
    seed, n, is_3d

    float:
    spherical_radius,
    min_freq, max_freq, octaves, scale
    mask_freq, mask_octaves,
    mask_ramp_min, mask_ramp_max
    slope_freq, slope_octaves, slope_scale
    */
    int seed = i_params[0];
    int n = i_params[1];
    int is_3d = i_params[2];
    float spherical_radius = f_params[0];
    float min_freq = f_params[1];
    float max_freq = f_params[2];
    float octaves = f_params[3];
    float scale = f_params[4];
    float mask_freq = f_params[5];
    float mask_octaves = f_params[6];
    float mask_ramp_min = f_params[7];
    float mask_ramp_max = f_params[8];
    float slope_freq = f_params[9];
    float slope_octaves = f_params[10];
    float slope_scale = f_params[11];

    float altitude;
    if (spherical_radius > 0) {
        altitude = sqrt(position.x * position.x + position.y * position.y + position.z * position.z) - spherical_radius;
    }
    else {
        altitude = position.z;
    }
    if (!is_3d) position.z = 0;

    float height = 0;
    for (int i = 0; i < n; i++) {
        float freq = log_uniform(min_freq, max_freq, myhash(seed, i, 0));
        float tmp = Perlin(position.x, position.y, position.z, myhash(seed, i, 1), octaves, freq) * scale;
        float mask = Perlin(position.x, position.y, position.z, myhash(seed, i, 2), mask_octaves, mask_freq);
        tmp *= ramp(mask, mask_ramp_min, mask_ramp_max);
        height = max(tmp, height);
    }
    height += Perlin(position.x, position.y, position.z, myhash(seed, 4), slope_octaves, slope_freq) * slope_scale;

    *sdf = altitude - height;

}