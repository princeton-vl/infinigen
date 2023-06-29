// Copyright (c) Princeton University.
// This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

// Authors: Zeyu Ma


#include "header.h"


extern "C" {

    void call(
        size_t size,
        float3_nonbuiltin *positions,
        float *sdfs
    ) {
        using namespace data;
        #pragma omp parallel for
        for (size_t idx = 0; idx < size; idx++) {
            mountains(positions[idx], sdfs + idx, d_i_params, d_f_params);
        }
    }


}