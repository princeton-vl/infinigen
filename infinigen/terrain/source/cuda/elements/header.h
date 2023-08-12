// Copyright (c) Princeton University.
// This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

// Authors: Zeyu Ma


#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <assert.h>
using namespace std;
#define DEVICE_FUNC __device__
#define CONSTANT_ARRAY __device__ __constant__
#include "../../common/utils/vectors.h"
#include "../../../../infinigen_gpl/bnodes/utils/nodes_util.h"
#include "../../../../infinigen_gpl/bnodes/utils/blender_noise.h"
#include "../../common/utils/elements_util.h"
#include "../../common/utils/FastNoiseLite.h"
#include "../../common/utils/smooth_bool_ops.h"
#include "../../common/elements/caves.h"
#include "../../common/elements/landtiles.h"
#include "../../common/elements/ground.h"
#include "../../common/elements/voronoi_rocks.h"
#include "../../common/elements/warped_rocks.h"
#include "../../common/elements/upsidedown_mountains.h"
#include "../../common/elements/mountains.h"
#include "../../common/elements/waterbody.h"
#include "../../common/elements/atmosphere.h"
#include "core.cu"
