// Copyright (C) 2023, Princeton University.
// This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

// Authors: Lahav Lipson


#pragma once
#include <iostream>
#include <fstream>
#include <memory>
#include <glad/glad.h>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#include "cnpy/cnpy.h"
#include <array>

void release_assert(const char *file, int line, bool condition, const std::string &msg);
#define RASSERT(c) release_assert(__FILE__, __LINE__, c, "")
#define MRASSERT(c, m) release_assert(__FILE__, __LINE__, c, m)

void glCheckError_(const char *file, int line);
#define glCheckError() glCheckError_(__FILE__, __LINE__)

class loop_obj {
public:
  const int x, y, j;

  loop_obj(int x, int y, int j) : x(x), y(y), j(j) {}

};

const std::vector<loop_obj> image_iterator(const int width, const int height, const std::string desc="");