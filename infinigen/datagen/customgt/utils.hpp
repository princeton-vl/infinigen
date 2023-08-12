// Copyright (c) Princeton University.
// This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

// Authors: Lahav Lipson


#pragma once
#include <iostream>
#include <fstream>
#include <memory>
#include <glad/glad.h>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <tinycolormap.hpp>
typedef tinycolormap::ColormapType clrmap;
#include <indicators/progress_bar.hpp>
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
  const std::shared_ptr<indicators::ProgressBar> progbar;
  
  loop_obj(int xc, int yc, int jc, std::shared_ptr<indicators::ProgressBar> pb) : x(xc), y(yc), j(jc), progbar(pb) {}
  void progressbar() const {if (progbar) progbar->tick();}
};

Eigen::Tensor<unsigned char, 3> compute_flow_viz(const Eigen::Tensor<double, 3> &input_image);

Eigen::Tensor<unsigned char, 3> to_color_map(const Eigen::Tensor<double, 2> &input_image, const double &min_percentile, const double &max_percentile, const double &minval=1e-3, const clrmap &type=clrmap::Jet);

Eigen::Tensor<unsigned char, 3> to_color_map(const Eigen::Tensor<long, 2> &input_image);

const std::vector<loop_obj> image_iterator(const int width, const int height, const std::string desc="");