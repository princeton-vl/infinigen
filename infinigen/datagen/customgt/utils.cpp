// Copyright (c) Princeton University.
// This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

// Authors: Lahav Lipson


#include <iostream>
#include <fstream>
#include <math.h>
#include <limits>
#include <unsupported/Eigen/CXX11/Tensor>
#include "utils.hpp"
#include "cnpy/cnpy.h"
#include "colorcode.h"

void release_assert(const char *file, int line, bool condition, const std::string &msg){
    if (!condition)
        throw std::runtime_error(std::string("Assertion failed: ") + file + " (" + std::to_string(line) + ")\n" + msg + "\n");
}

void glCheckError_(const char *file, int line)
{
    GLenum errorCode;
    while ((errorCode = glGetError()) != GL_NO_ERROR)
    {
        std::string error;
        switch (errorCode)
        {
            case GL_INVALID_ENUM:                  error = "INVALID_ENUM"; break;
            case GL_INVALID_VALUE:                 error = "INVALID_VALUE"; break;
            case GL_INVALID_OPERATION:             error = "INVALID_OPERATION"; break;
            case GL_STACK_OVERFLOW:                error = "STACK_OVERFLOW"; break;
            case GL_STACK_UNDERFLOW:               error = "STACK_UNDERFLOW"; break;
            case GL_OUT_OF_MEMORY:                 error = "OUT_OF_MEMORY"; break;
            case GL_INVALID_FRAMEBUFFER_OPERATION: error = "INVALID_FRAMEBUFFER_OPERATION"; break;
        }
        std::cout << error << " | " << file << " (" << line << ")" << std::endl;
        throw std::runtime_error(error + " | " + file + " (" + std::to_string(line) + ")\n");
    }
}

const std::vector<loop_obj> image_iterator(const int width, const int height, const std::string desc){
    using namespace indicators;
    std::shared_ptr<ProgressBar> bar(new ProgressBar{
        option::BarWidth{20},
        option::Start{"["},
        option::End{"]"},
        option::ShowElapsedTime{true},
        option::ShowRemainingTime{true},
        option::ForegroundColor{Color::blue},
        option::FontStyles{std::vector<FontStyle>{FontStyle::bold}},
        option::PrefixText{desc}
    });
    const size_t num_elements = width*height;
    std::vector<loop_obj> output;
    output.reserve(num_elements);
    size_t current_idx = 0;
    int prev_percent = 0;
    for (int y=0; y<height; y++){
        for (int x=0; x<width; x++){
            const int current_percent = ((++current_idx)*100)/num_elements;
            const bool should_tick = (current_percent > prev_percent);
            output.push_back(loop_obj(x, y, (x+(height-y-1)*width) * 4, should_tick ? bar : nullptr));
            prev_percent = current_percent;
        }    
    }
    assert (current_idx == num_elements);
    assert (num_elements == output.size());
    assert (output.back().progbar);
    return output;
}

Eigen::Tensor<unsigned char, 3> compute_flow_viz(const Eigen::Tensor<double, 3> &input_image){
    const int width = input_image.dimension(1);
    const int height = input_image.dimension(0);
    const double fac_x = (160.0/2) / width;
    const double fac_y = (120.0/2) / height;
    Eigen::Tensor<unsigned char, 3> output(height, width, 3);
    output.setZero();

    const int LEGEND_SIZE = 0;

    for (int y=0; y<height; y++){
        for (int x=0; x<width; x++){
            if ((y < LEGEND_SIZE) && (width < x + LEGEND_SIZE)){
                const double u = (x-(width-LEGEND_SIZE/2)) * fac_x;
                const double v = (y - LEGEND_SIZE/2) * fac_y;
                unsigned char px[3];
                computeColor(u, v, px);
                output(y,x,0) = px[0];
                output(y,x,1) = px[1];
                output(y,x,2) = px[2];
            } else {
                const double u = input_image(y, x, 0) * fac_x;
                const double v = input_image(y, x, 1) * fac_y;
                if (!std::isnan(u) && !std::isnan(v)){
                    unsigned char px[3];
                    computeColor(u, v, px);
                    output(y,x,0) = px[0];
                    output(y,x,1) = px[1];
                    output(y,x,2) = px[2];
                }
            }
        }
    }
    return output;
}

Eigen::Tensor<unsigned char, 3> to_color_map(const Eigen::Tensor<double, 2> &input_image, const double &min_percentile, const double &max_percentile, const double &minval, const clrmap &type) {
    std::vector<double> all_nonzero_values;
    const size_t width = input_image.dimension(1);
    const size_t height = input_image.dimension(0);
    all_nonzero_values.reserve(height*width);
    for (int y=0; y<height; y++){
        for (int x=0; x<width; x++){
            if ((input_image(y,x) > minval) && (input_image(y,x) < 1e4))
                all_nonzero_values.push_back(input_image(y,x));
        }
    }
    std::sort(all_nonzero_values.begin(), all_nonzero_values.end());
    const double N = all_nonzero_values.size();
    const double cur_min = all_nonzero_values[int(N * min_percentile)];
    const double cur_max = all_nonzero_values[int(N * max_percentile) - 1];
    assert (cur_max > cur_min);
    Eigen::Tensor<unsigned char, 3> output(height, width, 3);
    output.setZero();
    for (int y=0; y<height; y++){
        for (int x=0; x<width; x++){
            const double value = (input_image(y,x)-cur_min)/(cur_max - cur_min);
            if ((input_image(y,x) > minval) && (input_image(y,x) < 1e4)){
                const tinycolormap::Color color = tinycolormap::GetColor(value, type);
                output(y,x,0) = color.r()*255;
                output(y,x,1) = color.g()*255;
                output(y,x,2) = color.b()*255;
            }
        }
    }
    return output;
}


Eigen::Tensor<unsigned char, 3> to_color_map(const Eigen::Tensor<long, 2> &input_image) {
    std::vector<double> all_nonzero_values;
    const size_t width = input_image.dimension(1);
    const size_t height = input_image.dimension(0);
    Eigen::Tensor<unsigned char, 3> output(height, width, 3);
    output.setZero();
    std::unordered_map<long, double> int2double;
    for (int y=0; y<height; y++){
        for (int x=0; x<width; x++){
            const long key = input_image(y, x);
            double value;
            if (int2double.find(key) != int2double.end()) {
                value = int2double[key];
            } else {
                srand(key);
                value = static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
                int2double[key] = value;
            }
            const tinycolormap::Color color = tinycolormap::GetColor(value, clrmap::Turbo);
            output(y,x,0) = color.r()*255;
            output(y,x,1) = color.g()*255;
            output(y,x,2) = color.b()*255;
        }
    }
    return output;
}
