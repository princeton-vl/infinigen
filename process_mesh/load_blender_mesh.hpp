// Copyright (c) Princeton University.
// This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

// Authors: Lahav Lipson


#include <iostream>
#include <string>
#include <vector>
#include <string>
#include <glm/glm.hpp>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <random>
#include <glm/glm.hpp>
#include "glm/gtx/string_cast.hpp"
#include <glm/gtc/matrix_transform.hpp>
#include "utils.hpp"
#include "cnpy/cnpy.h"
#include "blender_object.hpp"
#include <nlohmann/json.hpp>
#include "io.hpp"

inline std::string truncate(const std::string &str, const size_t width){
    if (str.length() > width)
        return str.substr(0, width-3) + "...";
    return str + std::string(width - str.length(), ' ');
}

std::shared_ptr<BaseBlenderObject> load_blender_mesh(const fs::path json_path);

std::vector<unsigned int> generate_buffer(const std::vector<unsigned int> &indices);


