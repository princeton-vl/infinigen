// Copyright (c) Princeton University.
// This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

// Authors: Zeyu Ma


DEVICE_FUNC void ground(
    float3_nonbuiltin position,
    float *sdf,
    float *auxs,
    int is_caved,
    int *i_params, float *f_params,
    int *caves_i_params, float *caves_f_params
) {

    /* params
    int:
    seed, is_3d, with_sand_dunes

    float:
    spherical_radius,
    freq, octaves, scale, offset,
    sand_dunes_warping_freq, sand_dunes_warping_octaves, sand_dunes_warping_scale,
    sand_dunes_freq, sand_dunes_scale
    */

    int seed = i_params[0];
    int is_3d = i_params[1];
    int with_sand_dunes = i_params[2];
    float spherical_radius = f_params[0];
    float freq = f_params[1];
    float octaves = f_params[2];
    float scale = f_params[3];
    float offset = f_params[4];
    float sand_dunes_warping_freq = f_params[5];
    float sand_dunes_warping_octaves = f_params[6];
    float sand_dunes_warping_scale = f_params[7];
    float sand_dunes_freq = f_params[8];
    float sand_dunes_scale = f_params[9];
    float3_nonbuiltin original_position = position;


    float altitude;
    if (spherical_radius > 0) {
        altitude = sqrt(position.x * position.x + position.y * position.y + position.z * position.z) - spherical_radius;
    }
    else {
        altitude = position.z;
    }
    if (!is_3d) position.z = 0;
    float height = Perlin(position.x, position.y, position.z, myhash(seed, 0), octaves, freq) * scale + offset;
    *sdf = altitude - height;

    if (with_sand_dunes) {
        // only 2D now
        float warp = Perlin(position.x, position.y, 0, myhash(seed, 1), sand_dunes_warping_octaves, sand_dunes_warping_freq);
        float x_warped = position.x + warp * sand_dunes_warping_scale;
        warp = Perlin(position.x, position.y, 0, myhash(seed, 2), sand_dunes_warping_octaves, sand_dunes_warping_freq);
        float y_warped = position.y + warp * sand_dunes_warping_scale, dist;
        Voronoi(x_warped, y_warped, 0, myhash(seed, 3), 1, sand_dunes_freq, 1, NULL, &dist, NULL);
        *sdf -= dist * sand_dunes_scale;
    }

    if (is_caved) {
        float prior_sdf = *sdf;
        // 0.02 margin is to ensure when grounds and tiledlandscape coexist, the latter is in the outterlayer
        caves(original_position, sdf, caves_i_params, caves_f_params, prior_sdf, 0.02);
        if (auxs != NULL) auxs[0] = prior_sdf < *sdf;
    }
    else if (auxs != NULL) auxs[0] = 0;
}