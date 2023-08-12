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

#if __has_include(<filesystem>)
  #include <filesystem>
  namespace fs = std::filesystem;
#elif __has_include(<experimental/filesystem>)
  #include <experimental/filesystem> 
  namespace fs = std::experimental::filesystem;
#else
  error "Missing the <filesystem> header."
#endif

void imwrite(const fs::path filepath, const Eigen::Tensor<unsigned char, 3> &image);

void imwrite(const fs::path filepath, const Eigen::Array<unsigned char, -1, -1> &image);

template <typename T, int S>
void save_npy(const fs::path filepath, const Eigen::Tensor<T, S> &mat){
  assert (filepath.extension() == ".npy");
  std::vector<size_t> dims(S);
  for (int i=0; i<S; i++)
    dims[i] = mat.dimension(i);

  std::array<int, S> shuffle;
  if constexpr (S == 3)
    shuffle = {2, 1, 0};
  else if constexpr (S == 2)
    shuffle = {1, 0};

  const Eigen::Tensor<T, S, Eigen::RowMajor> swapped = mat.swap_layout().shuffle(shuffle);

  if constexpr(std::is_same<T, double>::value){
    const Eigen::Tensor<float, S, Eigen::RowMajor> tmp = swapped.template cast<float>();
    cnpy::npy_save(filepath.c_str(), tmp.data(), dims, "w");
  } else {
    cnpy::npy_save(filepath.c_str(), swapped.data(), dims, "w");
  }
}

Eigen::MatrixXd read_npy(const fs::path filepath);

void assert_exists(const fs::path &filepath);
