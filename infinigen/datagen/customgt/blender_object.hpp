// Copyright (c) Princeton University.
// This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

// Authors: Lahav Lipson

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
    int index, num_instances, num_faces, num_verts;
    std::string name, type, mesh_id, npz_filename;
    std::vector<int> children;
    std::vector<std::string> materials, unapplied_modifiers;

    ObjectInfo(){}

    ObjectInfo(const json instance_item) :
    name(instance_item["object_name"]),
    type(instance_item["object_type"]),
    index(instance_item["object_idx"]),
    num_instances(instance_item["num_instances"]),
    num_faces(instance_item["num_faces"]),
    num_verts(instance_item["num_verts"]),
    children(instance_item["children"]),
    materials(instance_item["materials"]),
    unapplied_modifiers(instance_item["unapplied_modifiers"]),
    mesh_id(instance_item["mesh_id"]),
    npz_filename(instance_item["filename"]){}

};

enum ObjectType { Mesh, Hair };

class BaseBlenderObject
{

private:

    void set_matrix_buffer(unsigned int &buffer, const std::vector<Eigen::Matrix4f> &model_matrices_next, int attrib_idx);
    json compute_bbox(const std::vector<unsigned int> &indices, const std::vector<float> &vertex_lookup, const std::vector<InstanceID> &instance_ids, const std::vector<Eigen::Matrix4f> &model_matrices, const std::vector<int> &tag_lookup, const int &attrib_stride);

protected:

    static inline std::chrono::duration<double, std::milli> total_elapsed_drawing, total_elapsed_sending;
    static inline size_t num_draw_calls = 0;
    unsigned int VAO, EBO, VBO, VBO_next, VBO_matrices, VBO_matrices_next, VBO_instance_ids, VBO_tag;
    const size_t num_verts, num_instances;
    size_t indices_buf_size, lookup_buf_size;

public:

    json bounding_box;
    const ObjectType type;
    const ObjectInfo info;

    BaseBlenderObject(const BufferArrays &current_buf, const BufferArrays &next_buf, const std::vector<InstanceID> &instance_ids, const ObjectInfo& object_info, const ObjectType tp, int attrib_stride);
    virtual ~BaseBlenderObject();
    virtual void draw(Shader &shader) const;
    static void print_stats();
};


class MeshBlenderObject : public BaseBlenderObject
{
public:

    MeshBlenderObject(const BufferArrays &current_buf, const BufferArrays &next_buf, const std::vector<InstanceID> &instance_ids, const ObjectInfo& object_info);
    ~MeshBlenderObject();
    void draw(Shader &shader) const;
};

class CurvesBlenderObject : public BaseBlenderObject
{
public:

    CurvesBlenderObject(const BufferArrays &current_buf, const BufferArrays &next_buf, const std::vector<InstanceID> &instance_ids, const ObjectInfo& object_info);
    ~CurvesBlenderObject();
    void draw(Shader &shader) const;
};
