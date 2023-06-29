// Copyright (c) Princeton University.
// This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

// Authors: Zeyu Ma


DEVICE_FUNC void landtiles(
    float3_nonbuiltin position,
    float *sdf,
    float *auxs,
    int is_caved,
    int *i_params, float *f_params,
    int *caves_i_params, float *caves_f_params
) {
    const int intrinsic_auxiliaries = 1;
    /* params
    int:
    seed,
    n_lattice,
    len_tiles,
    height_modification,
    attribute_modification,
    n_instances, N, use_cblerp

    float:
    randomness, frequency, cover_probability, island_probability, L,
    height_modification_start, height_modification_end,
    attribute_modification_start_height, attribute_modification_end_height,
    attribute_modification_distort_freq, attribute_modification_distort_mag,
    empty_below, y_tilt, y_tilt_clip, sharpen, mask_random_freq
    */

    int seed = i_params[0];
    int n_lattice = i_params[1];
    int len_tiles = i_params[2];
    int height_modification = i_params[3];
    int attribute_modification = i_params[4];
    int n_instances = i_params[5];
    int N = i_params[6];
    int use_cblerp = i_params[7];
    float randomness = f_params[0];
    float frequency = f_params[1];
    float cover_probability = f_params[2];
    float cover_distance_range = f_params[3];
    float island_probability = f_params[4];
    float L = f_params[5];
    float height_modification_start = f_params[6];
    float height_modification_end = f_params[7];
    float attribute_modification_start_height = f_params[8];
    float attribute_modification_end_height = f_params[9];
    float attribute_modification_distort_freq = f_params[10];
    float attribute_modification_distort_mag = f_params[11];
    float empty_below = f_params[12];
    float y_tilt = f_params[13];
    float y_tilt_clip = f_params[14];
    float sharpen = f_params[15];
    float mask_random_freq = f_params[16];
    const int f_offset = 17;

    float *tile_heights = f_params + f_offset;
    float *heightmap = f_params + f_offset + len_tiles;
    float *cover = f_params + f_offset + len_tiles + len_tiles * n_instances * N * N;
    float *coast_direction = f_params + f_offset + len_tiles + \
        len_tiles * n_instances * N * N + intrinsic_auxiliaries * len_tiles * n_instances * N * N;


    const int n_neighbors = 8;
    float heights_i0, height_grad_i0, covers_i0[intrinsic_auxiliaries];
    for (int i = 0; i < n_lattice; i++) {
        float positions[2 * n_neighbors], distances[n_neighbors], weight[n_neighbors];
        int hashes[n_neighbors];
        int is_center_tile[n_neighbors], is_center_band[n_neighbors];
        Voronoi2D(
            position.x, position.y,
            myhash(seed, i),
            randomness,
            frequency,
            n_neighbors,
            positions,
            distances,
            hashes,
            len_tiles==3,
            is_center_tile,
            is_center_band
        );
        float heights_i=0, height_grad_i=0, covers_i[intrinsic_auxiliaries]{0}, weightsum=0;
        for (int j = 0; j < n_neighbors; j++) {
            weight[j] = fmin(1e9f, decaying_distance_weight(distances[j], 0.8 / frequency, 1.0 / frequency, 9.0f));
            weightsum += weight[j];
        }
        for (int j = 0; j < n_neighbors; j++) {
            float heights_j_i, covers_j_i;
            float posj0 = positions[j * 2 + 0];
            float posj1 = positions[j * 2 + 1];
            float offset0 = position.x - posj0;
            float offset1 = position.y - posj1;
            int index = 0;
            int position_hash = myhash(seed, hashes[j]);
            int instance_hash;
            if (i != 0 || !is_center_tile[j]) {
                instance_hash = mod(myhash(position_hash, i, 0), n_instances);
            }
            else instance_hash = 0;
            int with_cover = (hash_to_float(myhash(position_hash, i, 1)) < cover_probability)
                && (positions[0] * positions[0] + positions[1] * positions[1] < cover_distance_range * cover_distance_range);
            float angle = 0;
            if (len_tiles == 1 || (len_tiles == 3 && !is_center_band[j])) {
                angle = hash_to_float(myhash(position_hash, i, 2)) * 2 * acosf(-1.0);
            }
            else {
                angle = coast_direction[instance_hash];
            }
            float tmp = (cos(angle) * offset0 - sin(angle) * offset1) * N / L + N / 2;
            offset1 = (cos(angle) * offset1 + sin(angle) * offset0) * N / L + N / 2;
            offset0 = tmp;
            int instance_offset = instance_hash * N * N;
            int with_island = hash_to_float(myhash(position_hash, i, 2)) < island_probability;

            if (len_tiles == 3) {
                if (posj0 > 0.1 / frequency) {
                    index = 2;
                }
                else if (posj0 < - 0.1 / frequency && !with_island) {
                    index = 0;
                }
                else index = 1;
                instance_offset += n_instances * index * N * N;
            }
            float step = 0.2;
            float hx0 = blerp(heightmap + instance_offset, offset0 - step, offset1, N);
            float hx1 = blerp(heightmap + instance_offset, offset0 + step, offset1, N);
            float hy0 = blerp(heightmap + instance_offset, offset0, offset1 - step, N);
            float hy1 = blerp(heightmap + instance_offset, offset0, offset1 + step, N);
            float scale = sqrt(sqr(L / N) + sqr((hx1 - hx0) / 2 / step) + sqr((hy1 - hy0) / 2 / step)) / L * N;

            if (use_cblerp) {
                heights_j_i = cblerp(heightmap + instance_offset, offset0, offset1, N);
            }
            else heights_j_i = blerp(heightmap + instance_offset, offset0, offset1, N);
            heights_j_i += tile_heights[index];
            heights_i += heights_j_i * (weight[j] / weightsum);
            height_grad_i += scale * (weight[j] / weightsum);
            if (auxs != 0)
                for (int a = 0; a < intrinsic_auxiliaries; a++) {
                    covers_j_i = blerp(cover + instance_offset + a * 3 * n_instances * N * N, offset0, offset1, N) * with_cover;
                    covers_i[a] += covers_j_i * (weight[j] / weightsum);
                }
        }
        if (auxs != 0)
            for (int a = 0; a < intrinsic_auxiliaries; a++) covers_i[a] = fmax(0.0f, fmin(covers_i[a], 1.0f));
        if (i == 0 || heights_i0 < heights_i) {
            heights_i0 = heights_i;
            height_grad_i0 = height_grad_i;
            if (auxs != 0)
                for (int a = 0; a < intrinsic_auxiliaries; a++)
                    covers_i0[a] = covers_i[a];
        }
    }
    float height = heights_i0;
    if (height_modification && height < height_modification_start) {
        height = height_modification_start - vertical_ramp(height_modification_start - height, height_modification_start - height_modification_end);
    }
    height += min(max(position.y * y_tilt, -y_tilt_clip), y_tilt_clip);
    *sdf = position.z - height;
    if (abs(*sdf) < 1) *sdf /= height_grad_i0;
    *sdf = smooth_subtraction(*sdf, position.z - empty_below, 0.5);
    if (auxs != 0) {
        if (abs(*sdf) < 0.2) {
            for (int a = 0; a < intrinsic_auxiliaries; a++) {
                float distort = 0;
                if (mask_random_freq != 0) distort = Perlin(position.x, position.y, position.z, myhash(seed, n_lattice, a), 4, mask_random_freq);
                auxs[a] = ramp(covers_i0[a] + distort, sharpen / 2, 1 - sharpen / 2);
            }
        }
        else {
            for (int a = 0; a < intrinsic_auxiliaries; a++)
                auxs[a] = 0;
        }
        auxs[intrinsic_auxiliaries] = 0;
    }
    if (auxs != 0 && attribute_modification) {
        float middle1 = attribute_modification_start_height * 0.9 + attribute_modification_end_height * 0.1;
        float middle2 = attribute_modification_start_height * 0.1 + attribute_modification_end_height * 0.9;
        float distort = Perlin(position.x, position.y, position.z, myhash(seed, n_lattice), 4, attribute_modification_distort_freq) * attribute_modification_distort_mag;
        auxs[intrinsic_auxiliaries] = fmaxf(auxs[intrinsic_auxiliaries], ramp(position.z + distort, attribute_modification_start_height, middle1) - ramp(position.z + distort, middle2, attribute_modification_end_height));
        for (int a = 0; a < intrinsic_auxiliaries; a++) {
            auxs[a] *= (1 - auxs[intrinsic_auxiliaries]);
        }
    }
    
    if (is_caved) {
        float prior_sdf = *sdf;
        caves(position, sdf, caves_i_params, caves_f_params, prior_sdf);
        if (auxs != NULL) auxs[intrinsic_auxiliaries + 1] = prior_sdf < *sdf;
    }
    else if (auxs != NULL) auxs[intrinsic_auxiliaries + 1] = 0;
}