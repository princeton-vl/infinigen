// Copyright (c) Princeton University.
// This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

// Authors: Lahav Lipson


#include <string>
#include <iostream>
#include "string_tools.hpp"

std::vector<std::string> match_regex(const std::string &pattern, const std::string &input){
        const std::regex regex{pattern.c_str()};
        std::smatch m;
        std::vector<std::string> output;
        if (std::regex_match(input, m, regex)){
            for (auto it : m)
                output.push_back(it.str());
        }
        return output;
}

std::string increment_int_substr(const std::vector<std::string> &patterns, const std::string &input_orig){
    std::string input(input_orig);
    std::smatch res;

    for (const auto pattern : patterns){
        std::string::const_iterator searchStart( input.cbegin() );
        std::regex exp(pattern.c_str());
        while ( std::regex_search( searchStart, input.cend(), res, exp ) )
        {
            std::string number = res[1];
            std::string new_number = zfill(number.size(), std::atoi(number.c_str())+1);
            input.replace(res[1].first, res[1].second, new_number);

            searchStart = res.suffix().first;
        }
    }

    return input;

}