// Copyright (c) Princeton University.
// This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

// Authors: Lahav Lipson
// Date Signed: May 2 2023

#pragma once
#include <vector>
#include "glm/gtx/string_cast.hpp"
#include <Eigen/Dense>
#include <chrono>
#include "shader.hpp"
#include <nlohmann/json.hpp>
#include "buffer_arrays.hpp"

using json = nlohmann::json;

struct ObjectInfo
{
    int index;
    std::string name, type, mesh_id, npz_filename;

    ObjectInfo(){}

    ObjectInfo(nlohmann::json_abi_v3_11_2::json instance_item) :
    name(instance_item["object_name"].get<std::string>()),
    type(instance_item["object_type"].get<std::string>()),
    index(instance_item["object_idx"].get<int>()),
    mesh_id(instance_item["mesh_id"].get<std::string>()),
    npz_filename(instance_item["filename"].get<std::string>()){}

};

enum ObjectType { Mesh, Hair };

class BaseBlenderObject
{

private:

    void set_matrix_buffer(unsigned int &buffer, const std::vector<Eigen::Matrix4f> &model_matrices_next, int attrib_idx);
    json compute_bbox(const std::vector<unsigned int> &indices, const std::vector<float> &vertex_lookup, const std::vector<int> &instance_ids, const std::vector<Eigen::Matrix4f> &model_matrices, const std::vector<int> &tag_lookup, const int &attrib_stride);

protected:

    static inline std::chrono::duration<double, std::milli> total_elapsed_drawing, total_elapsed_sending;
    static inline size_t num_draw_calls = 0;
    unsigned int VAO, EBO, VBO, VBO_next, VBO_matrices, VBO_matrices_next, VBO_instance_ids, VBO_tag;
    const size_t num_verts, num_instances;
    size_t indices_buf_size, lookup_buf_size;

public:

    json bounding_box;
    const ObjectType type;
    const std::string name;
    const int obj_index;

    BaseBlenderObject(const BufferArrays &current_buf, const BufferArrays &next_buf, const std::vector<int> &instance_ids, const ObjectInfo& object_info, const ObjectType tp, int attrib_stride);
    virtual ~BaseBlenderObject();
    virtual void draw(Shader &shader) const;
    static void print_stats();
};


class MeshBlenderObject : public BaseBlenderObject
{
public:

    MeshBlenderObject(const BufferArrays &current_buf, const BufferArrays &next_buf, const std::vector<int> &instance_ids, const ObjectInfo& object_info);
    ~MeshBlenderObject();
    void draw(Shader &shader) const;
};

class CurvesBlenderObject : public BaseBlenderObject
{
public:

    CurvesBlenderObject(const BufferArrays &current_buf, const BufferArrays &next_buf, const std::vector<int> &instance_ids, const ObjectInfo& object_info);
    ~CurvesBlenderObject();
    void draw(Shader &shader) const;
};
