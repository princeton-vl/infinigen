#pragma once
#include <vector>
#include <unordered_map>
#include <Eigen/Dense>
#include <iostream>
#include "cnpy/cnpy.h"
#include "utils.hpp"
#include "io.hpp"

class npz {
private:
  cnpy::npz_t lookup;

public:
  fs::path path;

  npz() {}

  npz(const fs::path p) : path(p) {
    assert_exists(path);
    RASSERT(path.extension() == ".npz");
    try {
        lookup = cnpy::npz_load(path);
    }
    catch(const std::exception& e) {
        throw std::runtime_error("Couldn't open " + path.string() + " - probably a BadZipFile error.");
    }
  }

  bool check_key(const std::string &key) const {
    return (lookup.count(key) != 0);
  }

    template<typename T>
    std::vector<T> read_data(const std::string &key) const {
        if (!check_key(key)){
            std::cerr << "Key " << key << " not found in " << path << std::endl;
            exit(1);
        }
        cnpy::NpyArray arr = lookup.at(key);
        T* arr_ptr = arr.data<T>();
        return std::vector<T>(arr_ptr, arr_ptr+arr.num_vals);
    }

};

class BufferArrays
{
public:
    std::vector<int> tag_lookup;
    std::vector<float> lookup;
    std::vector<unsigned int> indices;
    std::unordered_map<int, Eigen::Matrix4f> model_mats;

    BufferArrays(){}

    BufferArrays(const npz &my_npz, std::string mesh_id, std::string obj_type);

    size_t sizeof_instance() const {
        return indices.size();
    }

    void clear(){
        model_mats.clear();
        RASSERT(is_empty());
    }

    bool is_empty() const {
        return model_mats.empty();
    }

    std::vector<int> get_some_instance_ids(const size_t max_size=std::numeric_limits<size_t>::max()) const {
        std::vector<int> keys;
        for (const auto &kv : model_mats){
            if ((keys.size() * sizeof_instance()) < max_size)
                keys.push_back(kv.first);
        }
        return keys;
    }

    void remove_instances(const std::vector<int> &keys) {
        for (const auto &key : keys){
            const auto keyval = model_mats.find(key);
            if (keyval != model_mats.end())
                model_mats.erase(keyval);
        }
    }

    std::vector<Eigen::Matrix4f> get_instances(const std::vector<int> &keys) const {
        std::vector<Eigen::Matrix4f> matrix_worlds_subset;
        for (const auto &key : keys){
            const auto keyval = model_mats.find(key);
            if (keyval != model_mats.end())
                matrix_worlds_subset.push_back(keyval->second);
            else
                matrix_worlds_subset.push_back(Eigen::Matrix4f::Zero(4,4));
        }
        return matrix_worlds_subset;
    }

};
