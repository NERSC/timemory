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

/**
 * \file timemory/settings/types.hpp
 * \brief Declare the settings types
 */

#pragma once

#include "timemory/compat/macros.h"
#include "timemory/settings/macros.hpp"

#include <functional>
#include <map>
#include <string>
#include <vector>

namespace tim
{
//
//--------------------------------------------------------------------------------------//
//
//                              settings
//
//--------------------------------------------------------------------------------------//
//
using setting_callback_t        = std::function<void()>;
using setting_callback_map_t    = std::map<std::string, setting_callback_t>;
using setting_description_map_t = std::map<std::string, std::string>;
//
//--------------------------------------------------------------------------------------//
//
std::string
get_local_datetime(const char* dt_format);
//
//--------------------------------------------------------------------------------------//
//
struct settings;
//
//--------------------------------------------------------------------------------------//
//
template <typename... T>
inline setting_callback_map_t&
get_parse_callback_map()
{
    static setting_callback_map_t _instance;
    return _instance;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename... T>
inline setting_description_map_t&
get_descriptions()
{
    static setting_description_map_t _instance;
    return _instance;
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_SETTINGS_DLL setting_callback_map_t&
                      get_parse_callbacks() TIMEMORY_VISIBILITY("default");
//
TIMEMORY_SETTINGS_DLL setting_description_map_t&
                      get_setting_descriptions() TIMEMORY_VISIBILITY("default");
//
//--------------------------------------------------------------------------------------//
//
}  // namespace tim
