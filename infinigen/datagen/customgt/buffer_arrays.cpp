// Copyright (c) Princeton University.
// This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

// Authors: Lahav Lipson

#include "buffer_arrays.hpp"
#include <string>
#include <unordered_map>
#include <map>
#include <set>
#include "utils.hpp"

template <typename T>
std::vector<T> triangulate_face(std::vector<T> &polygon){
    std::vector<T> output;
    const int num_tri_verts = ((int)(polygon.size()) - 2) * 3;
    output.reserve(num_tri_verts);

    while (polygon.size() > 2){
        const T a = polygon.back();
        polygon.pop_back();
        const T b = polygon.back();
        polygon.pop_back();
        const T c = polygon.back();

        output.push_back(a);
        output.push_back(b);
        output.push_back(c);

        polygon.push_back(a);
    }
    return output;
}

void curve_quartet(const float *verts, const float *radii, std::vector<float> &point_lookup, std::vector<unsigned int> &indices, int offs){
    using Eigen::Vector3f, Eigen::Map;
    std::array<Vector3f, 7> P;
    for (int pt=0; pt<5; pt++)
        P[pt+1] = Map<const Vector3f>(verts+pt*3); // fill 1-5, still need 0 & 6

    P[0] = P[1] + (P[1] - P[2]);
    P[6] = P[5] + (P[5] - P[4]);

    for (int p=0; p<7; p++){
        const auto v = P[p];
        point_lookup.insert(point_lookup.end(), v.begin(), v.end());
        const int r = std::clamp(p-1, 0, 4);
        point_lookup.push_back(radii[r]);
    }

    for (int seg=0; seg<4; seg++){
        for (int p=0; p<4; p++)
            indices.push_back(offs + seg + p);
    }
}

std::vector<unsigned int> generate_buffer(const std::vector<unsigned int> &indices){

    using std::unordered_map, std::map, std::set, std::vector;

    unordered_map<uint, map<uint, set<uint>>> edge_neighbors;
    for (int i=0; i < indices.size(); i+=3){
        std::array<uint, 3> tri = {indices[i], indices[i+1], indices[i+2]};
        std::sort(tri.begin(), tri.end());
        edge_neighbors[tri[0]][tri[1]].insert(tri[2]);
        edge_neighbors[tri[0]][tri[2]].insert(tri[1]);
        edge_neighbors[tri[1]][tri[2]].insert(tri[0]);
    }

    vector<unsigned int> vertices;
    for (const auto &keyval : edge_neighbors){
        const uint &v1 = keyval.first;
        const map<uint, set<uint>> &mapping = keyval.second;
        for (const auto &v2_v3s : mapping){
            const auto &v2 = v2_v3s.first;
            vector<uint> v3s;
            v3s.insert(v3s.end(), v2_v3s.second.begin(), v2_v3s.second.end());
            RASSERT(v3s.size() == v2_v3s.second.size());
            if (v3s.size() == 1)
                v3s.push_back(v3s[0]); // i.e. treat dangling edges as two triangles folded together
            RASSERT(v3s.size() >= 2);
            for (int i=0; i<v3s.size(); i++){
                for (int j=0; j<i; j++){
                    for (const uint idx : {v1, v2, v3s[i], v3s[j]}){
                        vertices.push_back(idx);
                    }
                }
            }
        }
    }
    return vertices;
  }

BufferArrays::BufferArrays(const npz &my_npz, std::string mesh_id, std::string obj_type, bool skip_indices){

    {
        const std::vector<int> instance_ids = my_npz.read_data<int>(mesh_id + "_instance_ids");
        const auto mwd = my_npz.read_data<float>(mesh_id + "_transformations");
        RASSERT((mwd.size()/16) == instance_ids.size()/3);
        for (int i=0; i<(mwd.size()/16); i++)
            model_mats[{instance_ids[i*3], instance_ids[i*3+1], instance_ids[i*3+2]}] = Eigen::Matrix<float, 4, 4, Eigen::RowMajor>(&mwd[i*16]);
    }

    if (obj_type == "MESH") {

        lookup = my_npz.read_data<float>(mesh_id + "_vertices");
        // const auto face_tag_lookup = my_npz.read_data<int>(mesh_id + "_masktag");
        // tag_lookup.resize(lookup.size());
        tag_lookup = my_npz.read_data<int>(mesh_id + "_masktag");

        if (skip_indices)
            return;

        const auto loop_totals = my_npz.read_data<int>(mesh_id + "_loop_totals");
        // MRASSERT(loop_totals.size() == face_tag_lookup.size(), "loop_totals.size() ["+std::to_string(loop_totals.size())+"] != tag_lookup.size() ["+std::to_string(face_tag_lookup.size())+"]");

        const auto npy_data = my_npz.read_data<int>(mesh_id + "_indices");
        auto it = npy_data.begin();
        int face_index = 0;
        for (const int &polygon_size : loop_totals){
            // const int face_id = face_tag_lookup[face_index++];
            std::vector<int> polygon(it, it+polygon_size);
            const std::vector<int> triangles = triangulate_face(polygon);
            // for (const int &t : triangles)
            //     tag_lookup[t] = std::max(face_id, tag_lookup[t]); // tag is per-vertex, ideally should be per-face
            indices.insert(indices.end(),triangles.begin(),triangles.end());
            it += polygon_size;
        }

        indices = generate_buffer(indices);

    } else if (obj_type == "CURVES") {
        const auto vert_data = my_npz.read_data<float>(mesh_id + "_vertices");
        const auto radii_data = my_npz.read_data<float>(mesh_id + "_radii");
        const size_t num_hairs = vert_data.size() / (5*3);
        // std::cout << "We've got hair!: " << num_hairs << std::endl;
        for (int hair_idx=0; hair_idx < num_hairs; hair_idx++)
            curve_quartet(vert_data.data()+(hair_idx*5*3), radii_data.data()+(hair_idx*5), lookup, indices, hair_idx*7);
    } else {
        throw std::runtime_error("");
    }
}
