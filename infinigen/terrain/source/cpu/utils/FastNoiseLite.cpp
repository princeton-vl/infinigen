// Copyright (c) Princeton University.
// This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

// Authors: Zeyu Ma


#include <assert.h>
using namespace std;
#define DEVICE_FUNC
#define CONSTANT_ARRAY const
#include "../../common/utils/vectors.h"
#include "../../common/utils/FastNoiseLite.h"


extern "C" {

    void perlin_call(
        size_t size,
        float3_nonbuiltin *positions,
        float *values,
        int seed, int octaves, float freq
    ) {
        for (size_t idx = 0; idx < size; idx++) {
            values[idx] = Perlin(positions[idx].x, positions[idx].y, positions[idx].z, seed, octaves, freq);
        }
    }

}