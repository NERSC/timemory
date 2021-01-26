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
#include "timemory/utility/argparse.hpp"
#include "timemory/utility/serializer.hpp"

#include <functional>
#include <map>
#include <set>
#include <string>
#include <typeindex>
#include <typeinfo>
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
std::string
get_local_datetime(const char* dt_format, std::time_t* dt_curr = nullptr);
//
//--------------------------------------------------------------------------------------//
//
struct TIMEMORY_VISIBILITY("default") vsettings;
//
template <typename Tp, typename Vp = Tp>
struct TIMEMORY_VISIBILITY("default") tsettings;
//
struct TIMEMORY_VISIBILITY("default") settings;
//
//--------------------------------------------------------------------------------------//
//
}  // namespace tim

TIMEMORY_SET_CLASS_VERSION(2, ::tim::settings)
TIMEMORY_SET_CLASS_VERSION(0, ::tim::vsettings)
