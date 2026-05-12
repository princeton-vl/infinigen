// Copyright (C) 2024, Princeton University.
// This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

// Authors: Zeyu Ma



#include <algorithm>
#include <cmath>
#include <assert.h>
using namespace std;
#define DEVICE_FUNC
#define CONSTANT_ARRAY const
#include "../../common/utils/vectors.h"
#include "../../common/sdf_from_mesh/sdf_from_mesh.h"


extern "C" {
    void call(
        size_t size,
        float3_nonbuiltin *positions,
        float *sdfs,
        float3_nonbuiltin *vertices,
        int *faces,
        int n_faces
    ) {
        #pragma omp parallel for
        for (size_t idx = 0; idx < size; idx++) {
            sdf_from_mesh_common(
                positions[idx], sdfs + idx, vertices, faces, n_faces
            );
        }
    }
}
