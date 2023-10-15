// Copyright (c) Princeton University.
// This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

// Authors: Lahav Lipson


#include <glm/gtx/string_cast.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "camera_view.hpp"
#include "string_tools.hpp"
#include "utils.hpp"

using Eigen::Matrix4f, Eigen::Matrix3f, Eigen::Tensor;

unsigned int CameraView::create_framebuffer(){
    unsigned int framebuffer;
    glGenFramebuffers(1, &framebuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);  
    constexpr GLenum color_attachments[8] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2, GL_COLOR_ATTACHMENT3, GL_COLOR_ATTACHMENT4, GL_COLOR_ATTACHMENT5, GL_COLOR_ATTACHMENT6, GL_COLOR_ATTACHMENT7};
    constexpr bool is_float[8] = {false, true, true, false, false, false, false, true};
    constexpr size_t num_color_attachments = sizeof(color_attachments)/sizeof(GLenum);
    glDrawBuffers(num_color_attachments, color_attachments);
    unsigned int textureColorbuffers[num_color_attachments];
    for (int ai=0; ai<num_color_attachments; ai++){
        // generate texture
        GLenum col_attach = color_attachments[ai];
        unsigned int textureColorbuffer;
        glGenTextures(1, &textureColorbuffer);
        textureColorbuffers[ai] = textureColorbuffer;
        glActiveTexture(GL_TEXTURE0 + ai);
        glBindTexture(GL_TEXTURE_2D, textureColorbuffer);
        if (is_float[ai])
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, buffer_width, buffer_height, 0, GL_RGBA, GL_FLOAT, NULL);
        else
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32I, buffer_width, buffer_height, 0, GL_RGBA_INTEGER, GL_INT, NULL);
        glCheckError();
        glFramebufferTexture2D(GL_FRAMEBUFFER, col_attach, GL_TEXTURE_2D, textureColorbuffer, 0);  // attach it to currently bound framebuffer object
    }

    unsigned int rbo;
    glGenRenderbuffers(1, &rbo);
    glBindRenderbuffer(GL_RENDERBUFFER, rbo); 
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, buffer_width, buffer_height);
    glBindRenderbuffer(GL_RENDERBUFFER, 0);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, rbo);
    assert(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);
    return framebuffer;
}

const Matrix4f FLIP_Y_Z = Eigen::Vector4f({1,-1,-1,1}).asDiagonal();

template <int h, int w, typename T_orig=double, typename T_final=float>
Eigen::Matrix<T_final, h, w> load_matrix(const npz &camview, const std::string &key){
    const auto blender_camera_pose_data = camview.read_data<T_orig>(key);
    const auto tmp = Eigen::Matrix<T_orig, h, w>(blender_camera_pose_data.data());
    return tmp.transpose().template cast<T_final>();
}

CameraView::CameraView(const std::string fstr, const fs::path input_dir, const int width, const int height) : frame_string(fstr), image_height(height), image_width(width), buffer_height(height*2), buffer_width(width*2)
{
    // Current Frame
    const fs::path current_frame_cam_path = input_dir / ("camview_"+frame_string+".npz");
    const npz current_camview(current_frame_cam_path);
    const Matrix4f blender_camera_pose = load_matrix<4, 4>(current_camview, "T") * FLIP_Y_Z; // TODO REMOVE
    current_frame_view_matrix = glm::make_mat4(Matrix4f(blender_camera_pose.inverse()).data());

    // Next Frame
    const fs::path next_frame_cam_path = increment_int_substr({"frame_([0-9]{4})", "camview_[0-9]+_[0-9]+_([0-9]{4})"}, current_frame_cam_path);
    const npz next_camview(next_frame_cam_path);
    const Matrix4f next_blender_camera_pose = load_matrix<4, 4>(next_camview, "T") * FLIP_Y_Z; // TODO REMOVE
    next_frame_view_matrix = glm::make_mat4(Matrix4f(next_blender_camera_pose.inverse()).data());

    // Set Camera Position
    position = glm::make_vec3(blender_camera_pose.block<3, 1>(0, 3).data());

    // Set WC -> Img Transformation
    const Matrix3f K_mat3x3 = load_matrix<3, 3>(current_camview, "K");
    Matrix4f K_mat = Matrix4f::Identity();
    buffer_over_image = 2;
    K_mat.block<2,3>(0, 0) = buffer_over_image * K_mat3x3.block<2,3>(0, 0);
    wc2img = glm::make_mat4(Matrix4f(K_mat * FLIP_Y_Z * blender_camera_pose.inverse()).data());

    fx = K_mat(0,0);
    fy = K_mat(1,1);
    cx = K_mat(0,2);
    cy = K_mat(1,2);
    // Source https://stackoverflow.com/a/22312303/5057543
    const double near = 0.1;
    const double far = 10000.0;
    projection = glm::mat4(0);
    projection[0][0] = fx / cx;
    projection[1][1] = fy / cy;
    projection[2][2] = -(far+near)/(far-near);
    projection[3][2] = -(2*far*near)/(far-near);
    projection[2][3] = -1.0;

    framebuffer = create_framebuffer();
    framebuffer_ob = create_framebuffer();
    framebuffer_next_faceids = create_framebuffer();
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);

    /* clear the color buffer */
    glClearColor(0.0, 0.0, 0.0, 1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

void CameraView::activateShader(Shader &shader) const {
    shader.use();
    shader.setMat4("projection", projection);
    shader.setMat4("view", current_frame_view_matrix);
    shader.setMat4("viewNext", next_frame_view_matrix);
    shader.setMat4("wc2img", wc2img);
    shader.setVec3("cameraPos", position);
}

Tensor<double, 3> CameraView::project(const Tensor<double, 3> &cam_coords) const {
    const size_t width = cam_coords.dimension(1);
    const size_t height = cam_coords.dimension(0);
    Tensor<double, 3> output(height, width, 3);
    output.setZero();
    for (int x=0; x<width; x++){
        for (int y=0; y<height; y++){
            const double z = cam_coords(y, x, 2);
            const double u = cam_coords(y, x, 0) * (fx / z) + cx;
            const double v = cam_coords(y, x, 1) * (fy / z) + cy;
            output(y, x, 0) = u / buffer_over_image; output(y, x, 1) = v / buffer_over_image; output(y, x, 2) = z;
        }
    }
    return output;
}
