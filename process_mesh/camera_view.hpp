// Copyright (c) Princeton University.
// This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

// Authors: Lahav Lipson
// Date Signed: May 2 2023

#include "shader.hpp"
#include "utils.hpp"
#include <glad/glad.h>
#include "io.hpp"


class CameraView
{

private:

    float calc_resolution_scale();
    unsigned int create_framebuffer();

    glm::mat4 wc2img, projection, current_frame_view_matrix, next_frame_view_matrix;
    glm::vec3 position;
    double fx, fy, cx, cy;
    const int buffer_width, buffer_height;
    const fs::path frames_directory;

public:

    const std::string frame_string;
    unsigned int framebuffer, framebuffer_ob, framebuffer_next_faceids;

    CameraView(const std::string fstr, const fs::path fdir, const fs::path input_dir, const int width, const int height);
    void activateShader(Shader &shader) const;
    Eigen::Tensor<double, 3> project(const Eigen::Tensor<double, 3> &cam_coords) const;
    
};
