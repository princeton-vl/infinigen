// Copyright (c) Princeton University.
// This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

// Authors: Zeyu Ma


DEVICE_FUNC void caves(
    float3_nonbuiltin position,
    float *sdf,
    int *i_params,
    float *f_params,
    float prior_sdf,
    float margin=0
)
{
    /* params
    int:
    seed, n_lattice, is_horizontal, n_instances, N
    
    float:
    randomness, frequency, deepest_level, scale_increase,
    noise_octaves, noise_freq, noise_scale, height_offset, smoothness
    */

    int seed = i_params[0];
    int n_lattice = i_params[1];
    int is_horizontal = i_params[2];
    int n_instances = i_params[3];
    int N = i_params[4];

    float randomness = f_params[0];
    float frequency = f_params[1];
    float deepest_level = f_params[2];
    float scale_increase = f_params[3];
    float noise_octaves = f_params[4];
    float noise_freq = f_params[5];
    float noise_scale = f_params[6];
    float height_offset = f_params[7];
    float smoothness = f_params[8];
    
    const int f_offset = 9;
    float *cave_boundingboxes = f_params + f_offset;
    float *cave_occupancies = f_params + f_offset + n_instances * 6;

    const int n_neighbors = 1;
    const int n_neighbors_2 = 2;

    position.z -= height_offset;
    *sdf = 1e5;
    for (int i = 0; i < n_lattice; i++) {
        float positions[3 * n_neighbors], distances[n_neighbors];
        int hashes[n_neighbors];
        int is_center_tile[n_neighbors];
        Voronoi(position.x, position.y, position.z, myhash(seed, i, 0), randomness, frequency, n_neighbors, positions, distances, hashes, is_center_tile);
        int instance_hash;
        if (i != 0 || !is_center_tile[0]) {
            instance_hash = mod(hashes[0], n_instances);
        }
        else {
            instance_hash = 0;
        }
        float3_nonbuiltin offset = position - float3_nonbuiltin(positions[0], positions[1], positions[2]);

        bool toodeep = positions[2] < deepest_level;
        float angle = hash_to_float(hashes[0], i, 1) * 2 * acos(-1.);
        float tmp = cos(angle) * offset.x - sin(angle) * offset.y;
        offset.y = cos(angle) * offset.y + sin(angle) * offset.x;
        offset.x = tmp;
        if (!is_horizontal) {
            angle = hash_to_float(hashes[0], i, 2) * 2 * acos(-1.);
            tmp = cos(angle) * offset.x - sin(angle) * offset.z;
            offset.z = cos(angle) * offset.z + sin(angle) * offset.x;
            offset.x = tmp;
            angle = hash_to_float(hashes[0], i, 3) * 2 * acos(-1.);
            tmp = cos(angle) * offset.z - sin(angle) * offset.y;
            offset.y = cos(angle) * offset.y + sin(angle) * offset.z;
            offset.z = tmp;
        }
        
        float distances_2[n_neighbors_2];
        Voronoi(positions[0], positions[1], positions[2], myhash(seed, i, 0), randomness, frequency, n_neighbors_2, NULL, distances_2, NULL);
        float radius = distances_2[1] / 2;

        float scale = radius / sqrt(sqr(cave_boundingboxes[instance_hash * 6 + 3] - cave_boundingboxes[instance_hash * 6]) \
            + sqr(cave_boundingboxes[instance_hash * 6 + 4] - cave_boundingboxes[instance_hash * 6 + 1]) \
            + sqr(cave_boundingboxes[instance_hash * 6 + 5] - cave_boundingboxes[instance_hash * 6 + 2])) * 2;
        
        scale *= scale_increase;
        offset.x = (offset.x / scale - cave_boundingboxes[instance_hash * 6]) / (cave_boundingboxes[instance_hash * 6 + 3] - cave_boundingboxes[instance_hash * 6]) * (N - 1);
        offset.y = (offset.y / scale - cave_boundingboxes[instance_hash * 6 + 1]) / (cave_boundingboxes[instance_hash * 6 + 4] - cave_boundingboxes[instance_hash * 6 + 1]) * (N - 1);
        offset.z = (offset.z / scale - cave_boundingboxes[instance_hash * 6 + 2]) / (cave_boundingboxes[instance_hash * 6 + 5] - cave_boundingboxes[instance_hash * 6 + 2]) * (N - 1);
        instance_hash *= N * N * N;
        if (toodeep)
            tmp = 1e5;
        else
            tmp = ctlerp(cave_occupancies + instance_hash, offset.x, offset.y, offset.z, N) * scale;
        *sdf = min(*sdf, tmp);
    }
    *sdf += Perlin(position.x, position.y, position.z, myhash(seed, 0, 1), noise_octaves, noise_freq) * noise_scale - margin;
    *sdf = smooth_subtraction(prior_sdf, *sdf, smoothness);

}