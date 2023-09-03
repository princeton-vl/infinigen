// Copyright (c) Princeton University.
// This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

// Authors: Lahav Lipson


#include <random>
#include <iostream>
#include <functional> //without .h
#include <indicators/progress_bar.hpp>
#include <regex>
#include <unordered_map>
#include <unordered_set>
#include "string_tools.hpp"
#include "load_blender_mesh.hpp"
#include "buffer_arrays.hpp"

using json = nlohmann::json;
using namespace indicators;
using namespace indicators::option;

auto parse_json(const fs::path json_path){
    const json data = json::parse(std::ifstream(json_path));
    std::unordered_map<std::string, ObjectInfo> output;
    for (const auto &instance_item : data){
        if (instance_item.contains("filename") && (instance_item["object_type"] == "MESH")){ // Ignore CURVES objects, for now
            const ObjectInfo ii(instance_item);
            MRASSERT(output.count(ii.mesh_id) == 0, ii.mesh_id);
            output[ii.mesh_id] = ii;
        }
    }
    return output;
}

std::shared_ptr<BaseBlenderObject> load_blender_mesh(const fs::path json_path){
    assert_exists(json_path);

    // Progress bar
    static int object_idx = 0;
    static ProgressBar bar{
        BarWidth{20}, Start{"["}, End{"]"},
        ShowElapsedTime{true}, ShowRemainingTime{true},
        ForegroundColor{Color::blue},
        FontStyles{std::vector<FontStyle>{FontStyle::bold}}
    };
    if (object_idx == 0){
        bar.set_option(PrefixText{truncate(std::string("Loading..."), 20)});
        bar.set_progress(0);
    }

    static std::unordered_map<std::string, npz> npz_lookup;

    // Current frame
    static BufferArrays current_buf;
    static ObjectInfo current_obj;
    static const auto info_lookup = parse_json(json_path);

    // Next frame
    static BufferArrays next_buf;
    static ObjectInfo next_obj;
    static const auto next_info_lookup = parse_json(increment_int_substr({"frame_([0-9]{4})"}, json_path));

    static auto it = info_lookup.begin();

    if (current_buf.is_empty()){
        // while ((it != info_lookup.end()) && match_regex(".*errain.*", it->second.name).empty()) it++;
        if (it == info_lookup.end())
            return nullptr;
        const std::string current_mesh_id = it->first;

        // Current frame
        current_obj = it->second;
        const fs::path current_npz_path = json_path.parent_path() / current_obj.npz_filename;
        if (npz_lookup.count(current_npz_path.string()) == 0)
            npz_lookup[current_npz_path.string()] = current_npz_path;
        const npz &current_npz = npz_lookup.at(current_npz_path.string());
        current_buf = BufferArrays(current_npz, current_mesh_id, current_obj.type);

        bar.set_option(PrefixText{"Loading " + truncate(current_obj.name, 20) + " " + std::to_string(object_idx++) + "/" + std::to_string(info_lookup.size())});

        if (next_info_lookup.count(current_mesh_id) > 0){
            next_obj = next_info_lookup.at(current_mesh_id);
            const fs::path next_npz_path = fs::path(increment_int_substr({"frame_([0-9]{4})"}, json_path.parent_path())) / next_obj.npz_filename;
            if (npz_lookup.count(next_npz_path.string()) == 0)
                npz_lookup[next_npz_path.string()] = next_npz_path;
            const npz &next_npz = npz_lookup.at(next_npz_path.string());
            next_buf = BufferArrays(next_npz, current_mesh_id, next_obj.type, true);
        } else {
            next_buf = current_buf;
        }

        it++;
    }

    std::shared_ptr<BaseBlenderObject> new_obj;
    if (current_obj.type == "MESH"){
        const auto instance_ids = current_buf.get_some_instance_ids(1e8);
        new_obj = std::shared_ptr<BaseBlenderObject>(new MeshBlenderObject(current_buf, next_buf, instance_ids, current_obj));
        current_buf.remove_instances(instance_ids);
    } else if (current_obj.type == "CURVES"){
        const auto all_instance_ids = current_buf.get_some_instance_ids();
        new_obj = std::shared_ptr<BaseBlenderObject>(new CurvesBlenderObject(current_buf, next_buf, all_instance_ids, current_obj));
        current_buf.remove_instances(all_instance_ids);
    }
    bar.set_progress((object_idx * 100)/info_lookup.size());
    RASSERT((object_idx == info_lookup.size()) == (it == info_lookup.end()));
    return new_obj;
}
