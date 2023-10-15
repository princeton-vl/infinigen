// Copyright (c) Princeton University.
// This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

// Authors: Lahav Lipson


#include "shader.hpp"
#include "utils.hpp"
#include <glad/glad.h>
#include "io.hpp"
#include "buffer_arrays.hpp"


class CameraView
{

private:

    // float calc_resolution_scale(const npz &camview) const;
    unsigned int create_framebuffer();

    glm::mat4 wc2img, projection, current_frame_view_matrix, next_frame_view_matrix;
    glm::vec3 position;
    double fx, fy, cx, cy;
    float buffer_over_image; // should be >= 1

public:

    const int buffer_width, buffer_height, image_width, image_height;
    const std::string frame_string;
    unsigned int framebuffer, framebuffer_ob, framebuffer_next_faceids;

    CameraView(const std::string fstr, const fs::path input_dir, const int width, const int height);
    void activateShader(Shader &shader) const;
    Eigen::Tensor<double, 3> project(const Eigen::Tensor<double, 3> &cam_coords) const;
    
};
