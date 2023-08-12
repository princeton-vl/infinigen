// Copyright (c) Princeton University.
// This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

// Authors: Zeyu Ma


DEVICE_FUNC void upsidedown_mountains(
    float3_nonbuiltin position,
    float *sdf,
    float *auxs,
    int *i_params,
    float *f_params
) {

    
    /* params
    int:
    seed, n_instances, N
    
    float:
    L, floating_height, randomness, frequency,
    perturb_octaves, perturb_freq, perturb_scale
    */
    int seed = i_params[0];
    int n_instances = i_params[1];
    int N = i_params[2];

    float L = f_params[0];
    float floating_height = f_params[1];
    float randomness = f_params[2];
    float frequency = f_params[3];
    float perturb_octaves = f_params[4];
    float perturb_freq = f_params[5];
    float perturb_scale = f_params[6];

    const int f_offset = 7;
    float *upside = f_params + f_offset;
    float *downside = f_params + f_offset + n_instances * N * N;
    float *peak = f_params + f_offset + 2 * n_instances * N * N;

    float pos[2], dist[1], offset0, offset1;
    int is_center_tile[1];
    int hashes[1];
    Voronoi2D(position.x, position.y, seed, randomness, frequency, 1, pos, dist, hashes, 0, is_center_tile);
    offset0 = position.x - pos[0];
    offset1 = position.y - pos[1];
    int position_hash = myhash(seed, hashes[0]);


    int instance_hash;
    if (!is_center_tile[0]) {
        instance_hash = mod(myhash(position_hash, 0), n_instances) * N * N;
    }
    else instance_hash = 0;
    float angle = hash_to_float(position_hash, 1) * 2 * acos(-1.);
    float tmp = (cos(angle) * offset0 - sin(angle) * offset1) * N / L + N / 2;
    offset1 = (cos(angle) * offset1 + sin(angle) * offset0) * N / L + N / 2;
    offset0 = tmp;
    float upside0 = blerp(upside + instance_hash, offset0, offset1, N);
    float downside0 = blerp(downside + instance_hash, offset0, offset1, N, -2);
    float peak0 = blerp(peak + instance_hash, offset0, offset1, N);

    downside0 += Perlin(position.x, position.y, position.z, myhash(seed, 1), perturb_octaves, perturb_freq) * perturb_scale;
    upside0 += Perlin(position.x, position.y, position.z, myhash(seed, 2), perturb_octaves, perturb_freq) * perturb_scale;

    float v1 = position.z - floating_height - peak0 - (upside0 - peak0) * min(downside0 * 3, 1.f);
    float v2 = peak0 - downside0 - position.z + floating_height;
    float t = max(v1, v2);
    if (downside0 < 0) {
        *sdf = 1e9;
    }
    else *sdf = t;
    if (auxs != 0) {
        auxs[0] = v1 < v2;
    }
}