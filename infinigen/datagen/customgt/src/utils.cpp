// Copyright (C) 2023, Princeton University.
// This source code is licensed under the BSD 3-Clause license found in the
// LICENSE file in the root directory of this source tree.

// Authors: Lahav Lipson

#include "utils.hpp"

#include <math.h>

#include <fstream>
#include <iostream>
#include <limits>
#include <unsupported/Eigen/CXX11/Tensor>

#include "cnpy/cnpy.h"

void release_assert(const char *file, int line, bool condition, const std::string &msg)
{
    if (!condition)
        throw std::runtime_error(std::string("Assertion failed: ") + file + " (" +
                                 std::to_string(line) + ")\n" + msg + "\n");
}

void glCheckError_(const char *file, int line)
{
    GLenum errorCode;
    while ((errorCode = glGetError()) != GL_NO_ERROR)
    {
        std::string error;
        switch (errorCode)
        {
        case GL_INVALID_ENUM:
            error = "INVALID_ENUM";
            break;
        case GL_INVALID_VALUE:
            error = "INVALID_VALUE";
            break;
        case GL_INVALID_OPERATION:
            error = "INVALID_OPERATION";
            break;
        case GL_STACK_OVERFLOW:
            error = "STACK_OVERFLOW";
            break;
        case GL_STACK_UNDERFLOW:
            error = "STACK_UNDERFLOW";
            break;
        case GL_OUT_OF_MEMORY:
            error = "OUT_OF_MEMORY";
            break;
        case GL_INVALID_FRAMEBUFFER_OPERATION:
            error = "INVALID_FRAMEBUFFER_OPERATION";
            break;
        }
        std::cout << error << " | " << file << " (" << line << ")" << std::endl;
        throw std::runtime_error(error + " | " + file + " (" + std::to_string(line) + ")\n");
    }
}

const std::vector<loop_obj> image_iterator(const int width, const int height,
                                           const std::string desc)
{
    const size_t num_elements = width * height;
    std::vector<loop_obj> output;
    output.reserve(num_elements);
    size_t current_idx = 0;
    int prev_percent = 0;
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            const int current_percent = ((++current_idx) * 100) / num_elements;
            const bool should_tick = (current_percent > prev_percent);
            output.push_back(loop_obj(x, y, (x + (height - y - 1) * width) * 4));
            prev_percent = current_percent;
        }
    }
    assert(current_idx == num_elements);
    assert(num_elements == output.size());
    assert(output.back().progbar);
    return output;
}