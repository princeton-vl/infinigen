// Copyright (c) Princeton University.
// This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

// Authors: Lahav Lipson


#include <iostream>
#include <fstream>
#include <math.h>
#include <limits>
#include <unsupported/Eigen/CXX11/Tensor>
#include <arpa/inet.h> // or <winsock.h>
#include "io.hpp"
#include "utils.hpp"
#include "cnpy/cnpy.h"
#include "colorcode.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"


void imwrite(const fs::path filepath, const Eigen::Array<unsigned char, -1, -1> &image){
    const int H = image.rows();
    const int W = image.cols();
    std::vector<u_char> color_data_vec(H*W);

    for (int i = 0; i < W*H; i++) {
        const int x = i%W;
        const int y = i/W;
        color_data_vec[i] = image(y,x);
    }

    stbi_write_png(filepath.c_str(), W, H, 1, &color_data_vec[0], W);
}

void imwrite(const fs::path filepath, const Eigen::Tensor<unsigned char, 3> &image){
    const int H = image.dimension(0);
    const int W = image.dimension(1);
    std::vector<u_char> color_data_vec(H*W*3);

    for (int i = 0; i < W*H; i++) {
        const int x = i%W;
        const int y = i/W;
          for (int k=0; k<3; k++)
              color_data_vec[i*3 + k] = image(y,x,k);
    }

    stbi_write_png(filepath.c_str(), W, H, 3, &color_data_vec[0], W*3);
}

Eigen::MatrixXd read_npy(const fs::path filepath){
    assert_exists(filepath);
    const cnpy::NpyArray myarr = cnpy::npy_load(filepath.c_str());
    const double *x = myarr.data<double>();
    const std::vector<double> v(x, x + myarr.num_bytes()/sizeof(double));

    const int W = myarr.shape[1];
    const int H = myarr.shape[0];
    Eigen::ArrayXXd output_data(H,W);
    for (int i=0; i<v.size(); i++){
        assert (fabs(v[i]) < 1e6);
        output_data(i/W, i%W) = v[i];
    }
    return output_data;
}

void assert_exists(const fs::path &filepath){
    MRASSERT(fs::exists(filepath), "Error: " + filepath.string() + " does not exist.");
}
