// Copyright (c) Princeton University.
// This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

// Authors: Lahav Lipson

#include "blender_object.hpp"
#include <glad/glad.h>
#include <random>
#include <iostream>
#include <regex>
#include <set>
#include <limits>
#include "utils.hpp"
#include "buffer_arrays.hpp"
using std::cout;
using std::endl;

template <typename T>
void set_regular_buffer(unsigned int &buffer, const std::vector<T> &data_vec, int attrib_idx, int attrib_size, int attrib_stride){
    glGenBuffers(1, &buffer);
    glBindBuffer(GL_ARRAY_BUFFER, buffer);
    glBufferData(GL_ARRAY_BUFFER, data_vec.size() * sizeof(T), data_vec.data(), GL_STATIC_DRAW);

    glEnableVertexAttribArray(attrib_idx);
    static_assert(std::is_same<T,float>::value || std::is_same<T,int>::value);
    if constexpr (std::is_same<T,float>::value)
        glVertexAttribPointer(attrib_idx, attrib_size, GL_FLOAT, GL_FALSE, attrib_stride * sizeof(T), 0);
    else if constexpr (std::is_same<T,int>::value)
        glVertexAttribIPointer(attrib_idx, attrib_size, GL_INT, attrib_stride * sizeof(T), 0);
}

void BaseBlenderObject::set_matrix_buffer(unsigned int &buffer, const std::vector<Eigen::Matrix4f> &model_matrices_next, int attrib_idx){
    glGenBuffers(1, &buffer);
    glBindBuffer(GL_ARRAY_BUFFER, buffer);
    glBufferData(GL_ARRAY_BUFFER, num_instances * sizeof(Eigen::Matrix4f), model_matrices_next.data(), GL_STATIC_DRAW);

    // vertex attributes
    std::size_t vec4Size = sizeof(Eigen::RowVector4f);
    for (int ii=0; ii<4; ii++){
        glEnableVertexAttribArray(ii+attrib_idx);
        glVertexAttribPointer(ii+attrib_idx, 4, GL_FLOAT, GL_FALSE, 4 * vec4Size, (void*)(ii * vec4Size));
    }

    for (int ii=0; ii<4; ii++)
        glVertexAttribDivisor(ii+attrib_idx, 1);
}

json BaseBlenderObject::compute_bbox(const std::vector<unsigned int> &indices, const std::vector<float> &vertex_lookup, const std::vector<InstanceID> &instance_ids, const std::vector<Eigen::Matrix4f> &model_matrices, const std::vector<int> &tag_lookup, const int &attrib_stride){
    constexpr float inf = std::numeric_limits<float>::infinity();
    Eigen::Vector3f max({-inf, -inf, -inf}), min({inf, inf, inf});
    for (const auto &idx : indices){
        for (int i=0; i<3; i++){
            max(i) = std::max(max(i), vertex_lookup[idx * attrib_stride + i]);
            min(i) = std::min(min(i), vertex_lookup[idx * attrib_stride + i]);
        }
    }

    std::vector<std::vector<long>> json_serializable_instance_ids(num_instances);
    std::vector<json> json_serializable_model_matrices(num_instances);
    for (int idx=0; idx<num_instances; idx++){
        const auto instance_id = instance_ids[idx];
        json_serializable_instance_ids[idx] = {instance_id.n1, instance_id.n2, instance_id.n3};
        const auto m = model_matrices[idx];
        json_serializable_model_matrices[idx] = {
            {m(0,0), m(0,1), m(0,2), m(0,3)},
            {m(1,0), m(1,1), m(1,2), m(1,3)},
            {m(2,0), m(2,1), m(2,2), m(2,3)},
            {m(3,0), m(3,1), m(3,2), m(3,3)},
        };
    }

    const std::set<int> unique_tags(tag_lookup.begin(), tag_lookup.end());

    json output = {
        {"instance_ids", json_serializable_instance_ids},
        {"model_matrices", json_serializable_model_matrices},
        {"tags", std::vector<int>(unique_tags.begin(), unique_tags.end())},
        {"name", info.name},
        {"num_verts", info.num_verts},
        {"num_faces", info.num_faces},
        {"children", info.children},
        {"materials", info.materials},
        {"unapplied_modifiers", info.unapplied_modifiers},
        {"object_index", info.index}
    };

    if ((num_verts > 0) && ((max - min).norm() > 1e-4)){
        output["min"] = {min(0), min(1), min(2)};
        output["max"] = {max(0), max(1), max(2)};
    } else {
        output["min"] = nullptr;
        output["max"] = nullptr;
    }

    return output;
}

BaseBlenderObject::BaseBlenderObject(const BufferArrays &current_buf, const BufferArrays &next_buf, const std::vector<InstanceID> &instance_ids, const ObjectInfo& object_info, const ObjectType tp, int attrib_stride)
 : num_verts(current_buf.indices.size()), type(tp), info(object_info), num_instances(instance_ids.size()) {

        const std::vector<Eigen::Matrix4f> &model_matrices = current_buf.get_instances(instance_ids);
        const std::vector<Eigen::Matrix4f> &model_matrices_next = next_buf.get_instances(instance_ids);

        MRASSERT(model_matrices.size() == num_instances, "Incorrect number of instances");
        MRASSERT(model_matrices_next.size() == num_instances, "Incorrect number of instances");
        const auto t1 = std::chrono::high_resolution_clock::now();

        const auto &indices = current_buf.indices;
        const auto &vertex_lookup = current_buf.lookup;
        const auto &vertex_lookup_next = next_buf.lookup;
        const auto &tag_lookup = current_buf.tag_lookup;

        glGenVertexArrays(1, &VAO);
        glBindVertexArray(VAO);

        // Vertices
        glGenBuffers(1, &EBO);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size()*sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);
        indices_buf_size = indices.size()*sizeof(unsigned int);
        lookup_buf_size = vertex_lookup.size() * sizeof(float);

        set_matrix_buffer(VBO_matrices, model_matrices, 0);
        set_matrix_buffer(VBO_matrices_next, model_matrices_next, 4);

        set_regular_buffer(VBO, vertex_lookup, 8, 3, attrib_stride);
        set_regular_buffer(VBO_next, vertex_lookup_next, 9, 3, attrib_stride);

        static_assert(sizeof(int)*3 == sizeof(InstanceID));
        glGenBuffers(1, &VBO_instance_ids);
        glBindBuffer(GL_ARRAY_BUFFER, VBO_instance_ids);
        glBufferData(GL_ARRAY_BUFFER, num_instances * sizeof(InstanceID), instance_ids.data(), GL_STATIC_DRAW);
        glEnableVertexAttribArray(10);
        glVertexAttribIPointer(10, 3, GL_INT, sizeof(InstanceID), 0);
        glVertexAttribDivisor(10, 1);

        set_regular_buffer(VBO_tag, tag_lookup, 11, 1, 1);

        total_elapsed_sending += (std::chrono::high_resolution_clock::now() - t1);
        num_draw_calls++;

        bounding_box = compute_bbox(indices, vertex_lookup, instance_ids, model_matrices, tag_lookup, attrib_stride);
}

void BaseBlenderObject::draw(Shader &shader) const {
    throw std::runtime_error("Base class draw() called!");
}

void BaseBlenderObject::print_stats() {
    std::cout << "Spent " << total_elapsed_drawing.count() << "milliseconds in draw calls." << std::endl;
    std::cout << "Spent " << total_elapsed_sending.count() << "milliseconds across " << num_draw_calls << " buffer calls." << std::endl;
}

BaseBlenderObject::~BaseBlenderObject(){
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, lookup_buf_size, nullptr, GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, VBO_next);
    glBufferData(GL_ARRAY_BUFFER, lookup_buf_size, nullptr, GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, VBO_matrices);
    glBufferData(GL_ARRAY_BUFFER, num_instances * sizeof(Eigen::Matrix4f), nullptr, GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, VBO_matrices_next);
    glBufferData(GL_ARRAY_BUFFER, num_instances * sizeof(Eigen::Matrix4f), nullptr, GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, VBO_instance_ids);
    glBufferData(GL_ARRAY_BUFFER, num_instances * sizeof(int), nullptr, GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices_buf_size, nullptr, GL_STATIC_DRAW);
    glDeleteVertexArrays(1, &VAO);
    unsigned int to_delete[7] = {EBO, VBO, VBO_next, VBO_matrices, VBO_matrices_next, VBO_instance_ids, VBO_tag};
    glDeleteBuffers(7, to_delete);
}

MeshBlenderObject::MeshBlenderObject(const BufferArrays &current_buf, const BufferArrays &next_buf, const std::vector<InstanceID> &instance_ids, const ObjectInfo& object_info)
 : BaseBlenderObject(current_buf, next_buf, instance_ids, object_info, Mesh, 3) {
    glBindVertexArray(0);
}
MeshBlenderObject::~MeshBlenderObject(){}

void MeshBlenderObject::draw(Shader &shader) const {
    const auto t1 = std::chrono::high_resolution_clock::now();
    shader.setInt("object_index", info.index);
    glBindVertexArray(VAO);
    glDrawElementsInstanced(GL_LINES_ADJACENCY, num_verts, GL_UNSIGNED_INT, 0, num_instances);
    glCheckError();
    total_elapsed_drawing += (std::chrono::high_resolution_clock::now() - t1);
    glBindVertexArray(0);
}


CurvesBlenderObject::CurvesBlenderObject(const BufferArrays &current_buf, const BufferArrays &next_buf, const std::vector<InstanceID> &instance_ids, const ObjectInfo& object_info)
 : BaseBlenderObject(current_buf, next_buf, instance_ids, object_info, Hair, 4){
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glEnableVertexAttribArray(12); // Radius
    glVertexAttribPointer(12, 1, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(sizeof(float)*3));
    glBindVertexArray(0);
}
CurvesBlenderObject::~CurvesBlenderObject(){}

void CurvesBlenderObject::draw(Shader &shader) const {
    const auto t1 = std::chrono::high_resolution_clock::now();
    glBindVertexArray(VAO);
    glDrawElementsInstanced(GL_LINES_ADJACENCY, num_verts, GL_UNSIGNED_INT, 0, num_instances);
    glCheckError();
    total_elapsed_drawing += (std::chrono::high_resolution_clock::now() - t1);
    glBindVertexArray(0);
}
