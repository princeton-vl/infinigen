// Copyright (c) Princeton University.
// This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

// Authors: Zeyu Ma


#define VROCK_ON_TILED_LANDSCAPE 0
#define VROCK_ON_GROUND 1

DEVICE_FUNC void single_group_voronoi_rocks(
    float3_nonbuiltin position,
    float3_nonbuiltin *cell_center,
    float *distances,
    int *masks,
    float &perturbation, float &freq,
    int *i_params,
    float *f_params
)
{
    /* params
    
    int:
    seed, 

    float:
    min_freq, max_freq,
    gap_min_freq, gap_max_freq, gap_scale, gap_octaves, gap_base,
    warp_min_freq, warp_max_freq, warp_octaves, warp_prob
    warp_modu_sigmoidscale, warp_modu_scale, warp_modu_octaves, warp_modu_freq,
    mask_octaves, mask_freq, mask_shift,
    */

    int seed = i_params[0];
    float min_freq = f_params[0];
    float max_freq = f_params[1];
    float gap_min_freq = f_params[2];
    float gap_max_freq = f_params[3];
    float gap_scale = f_params[4];
    float gap_octaves = f_params[5];
    float gap_base = f_params[6];
    float warp_min_freq = f_params[7];
    float warp_max_freq = f_params[8];
    float warp_octaves = f_params[9];
    float warp_prob = f_params[10];
    float warp_modu_sigmoidscale = f_params[11];
    float warp_modu_scale = f_params[12];
    float warp_modu_octaves = f_params[13];
    float warp_modu_freq = f_params[14];
    float mask_octaves = f_params[15];
    float mask_freq = f_params[16];
    float mask_shift = f_params[17];

    const int n_neighbors = 8;
    float positions[3 * n_neighbors];

    float warp[3];
    float warp_freq = log_uniform(warp_min_freq, warp_max_freq, myhash(seed, 2));

    float3_nonbuiltin warped_position = position;
    if (hash_to_float(seed, 3) < warp_prob) {
        for (int i = 0; i < 3; i++) {
            float warp_modu;
            warp_modu = Perlin(position.x, position.y, position.z, myhash(seed, 0, i), warp_modu_octaves, warp_modu_freq);
            warp_modu = warp_modu_scale / (1 + exp(-warp_modu * warp_modu_sigmoidscale));
            warp[i] = Perlin(position.x, position.y, position.z, myhash(seed, 1, i), warp_octaves, warp_freq);
            warp[i] *= warp_modu / (2 * warp_freq);
        }
        warped_position += float3_nonbuiltin(warp[0], warp[1], warp[2]);
    }

    freq = exp(log(min_freq) + (log(max_freq) - log(min_freq)) * hash_to_float(seed, 0));
    Voronoi(warped_position.x, warped_position.y, warped_position.z, myhash(seed, 4), 1, freq, n_neighbors, positions, distances, 0);
    for (int i = 0; i < n_neighbors; i++) {
        cell_center[i] = float3_nonbuiltin(positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]);
    }
    float gap_freq = exp(log(gap_min_freq) + (log(gap_max_freq) - log(gap_min_freq)) * hash_to_float(seed, 6));
    perturbation = pow(gap_base, Perlin(position.x, position.x, position.x, myhash(seed, 5), gap_octaves, gap_freq)) * gap_scale / freq;
    for (int i = 0; i < n_neighbors; i++) {
        masks[i] = Perlin(positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2], myhash(seed, 7), mask_octaves, mask_freq) + mask_shift < 0;
    }
}


DEVICE_FUNC void voronoi_rocks(
    float3_nonbuiltin position,
    float *sdf,
    float *auxs,
    int attachment,
    int is_caved,
    int *i_params, float *f_params,
    int *attachment_i_params, float *attachment_f_params,
    int *caves_i_params, float *caves_f_params
) {
    const int n_neighbors = 8;
    *sdf = 1e5;
    int seed = i_params[0];
    int n_lattice = i_params[1];
    int attribute_modification = i_params[2];
    float attribute_modification_start_height = f_params[18];
    float attribute_modification_end_height = f_params[19];

    
    for (int i = 0; i < n_lattice; i++) {
        int seed_i = myhash(seed, i);
        float3_nonbuiltin cell_center[n_neighbors];
        float distances[n_neighbors];
        int masks[n_neighbors];
        float perturbation, freq;
        single_group_voronoi_rocks(position, cell_center, distances, masks, perturbation, freq, &seed_i, f_params);
        int solid0;
        float sdf_i;
        int j;
        for (j = 0; j < n_neighbors; j++) {
            int solid;
            if (masks[j]) {
                solid = 0;
            }
            else {
                float sdf_a;
                if (attachment == VROCK_ON_TILED_LANDSCAPE) {
                    landtiles(cell_center[j], &sdf_a, NULL, is_caved, attachment_i_params, attachment_f_params, caves_i_params, caves_f_params);
                }
                else if (attachment == VROCK_ON_GROUND) {
                    ground(cell_center[j], &sdf_a, NULL, is_caved, attachment_i_params, attachment_f_params, caves_i_params, caves_f_params);
                }
                else {
                    assert(0);
                }
                solid = abs(sdf_a) < 0.5 / freq;
            }
            if (j > 0) {
                if (solid != solid0) {
                    sdf_i = (distances[j] - distances[0]) / 2;
                    if (solid0) sdf_i *= -1;
                    sdf_i += perturbation;
                    break;
                }
            }
            else {
                solid0 = solid;
            }
        }
        if (j == n_neighbors) {
            if (solid0) sdf_i = -1e5;
            else sdf_i = 1e5;
        }
        *sdf = min(*sdf, sdf_i);
    }
    if (auxs != NULL) {
        if (attribute_modification) {
            float middle1 = (attribute_modification_start_height * 3 + attribute_modification_end_height) / 4;
            float middle2 = (attribute_modification_start_height + attribute_modification_end_height * 3) / 4;
            auxs[0] = ramp(position.z, attribute_modification_start_height, middle1) - ramp(position.z, middle2, attribute_modification_end_height);
        }
        else auxs[0] = 0;
        if (is_caved) {
            float sdf0;
            caves(position, &sdf0, caves_i_params, caves_f_params, -1e9);
            auxs[1] = sdf0 > 0;
        }
        else auxs[1] = 0;
    }
}
