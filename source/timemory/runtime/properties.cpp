// MIT License
//
// Copyright (c) 2020, The Regents of the University of California,
// through Lawrence Berkeley National Laboratory (subject to receipt of any
// required approvals from the U.S. Dept. of Energy).  All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include "timemory/components/properties.hpp"
#include "timemory/components/placeholder.hpp"
#include "timemory/components/types.hpp"
#include "timemory/enum.h"

#include <set>
#include <string>
#include <type_traits>
#include <unordered_map>

#include "timemory/components/types.hpp"
#include "timemory/runtime/properties.hpp"

namespace tim
{
//
//--------------------------------------------------------------------------------------//
//
namespace runtime
{
//
//--------------------------------------------------------------------------------------//
//
int
enumerate(std::string key)
{
    using data_t = std::tuple<component_hash_map_t, std::function<void(const char*)>>;
    static auto _data = []() {
        component_hash_map_t _map;
        component_key_set_t  _set;
        enumerator_enumerate(_map, _set, make_int_sequence<TIMEMORY_COMPONENTS_END>{});
        std::stringstream ss;
        ss << "Valid choices are: [";
        for(auto itr = _set.begin(); itr != _set.end(); ++itr)
        {
            ss << "'" << (*itr) << "'";
            size_t _dist = std::distance(_set.begin(), itr);
            if(_dist + 1 < _set.size())
                ss << ", ";
        }
        ss << ']';
        auto _choices = ss.str();
        auto _msg     = [_choices](const char* itr) {
            fprintf(stderr, "Unknown component: '%s'. %s\n", itr, _choices.c_str());
        };
        return data_t(_map, _msg);
    }();

    auto itr = std::get<0>(_data).find(settings::tolower(key));
    if(itr != std::get<0>(_data).end())
        return itr->second;

    std::get<1>(_data)(key.c_str());
    return TIMEMORY_COMPONENTS_END;
}
//
//--------------------------------------------------------------------------------------//
//
int
enumerate(const char* key)
{
    return enumerate(std::string(key));
}
//
//--------------------------------------------------------------------------------------//
//
}  // namespace runtime
}  // namespace tim
