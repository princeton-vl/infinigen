// Copyright (c) Princeton University.
// This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

// Authors: Zeyu Ma
// Date Signed: June 5 2023

#include <algorithm>
#include <cmath>
#include <assert.h>
using namespace std;
#define DEVICE_FUNC
#define CONSTANT_ARRAY const
#define POINTER_OR_REFERENCE_ARG
#include "../../common/utils/vectors.h"
#include "../../common/utils/nodes_util.h"
#include "../../common/utils/blender_noise.h"
#include "../../common/nodes/node_shader_tex_noise.h"
#include "../../common/nodes/node_shader_tex_voronoi.h"
#include "../../common/nodes/node_shader_mix_rgb.h"
#include "../../common/nodes/node_float_curve.h"
#include "../../common/nodes/node_shader_map_range.h"
#include "../../common/nodes/node_shader_sepcomb_xyz.h"
#include "../../common/nodes/node_shader_tex_wave.h"
#include "../../common/nodes/node_shader_vector_math.h"
#include "../../common/nodes/node_texture_math.h"
#include "../../common/nodes/node_texture_valToRgb.h"
#include "../../common/nodes/node_shader_tex_musgrave.h"