// MIT License
//
// Copyright (c) 2019, The Regents of the University of California,
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

#include "timemory/utility/macros.hpp"

#include <cstdint>
#include <cstring>
#include <limits>
#include <string>

#if !defined(TIMEMORY_DEFAULT_ENABLED)
#    define TIMEMORY_DEFAULT_ENABLED true
#endif

namespace tim
{
namespace settings
{
//--------------------------------------------------------------------------------------//

using string_t = std::string;

#define DEFINE_STATIC_ACCESSOR_FUNCTION(TYPE, FUNC, INIT)                                \
    inline TYPE& FUNC()                                                                  \
    {                                                                                    \
        static TYPE instance = INIT;                                                     \
        return instance;                                                                 \
    }

//--------------------------------------------------------------------------------------//
// logic
DEFINE_STATIC_ACCESSOR_FUNCTION(bool, enabled, true)
DEFINE_STATIC_ACCESSOR_FUNCTION(bool, suppress_parsing, false)
DEFINE_STATIC_ACCESSOR_FUNCTION(bool, auto_output, true)
DEFINE_STATIC_ACCESSOR_FUNCTION(bool, file_output, true)
DEFINE_STATIC_ACCESSOR_FUNCTION(bool, text_output, true)
DEFINE_STATIC_ACCESSOR_FUNCTION(bool, json_output, false)
DEFINE_STATIC_ACCESSOR_FUNCTION(bool, cout_output, true)

// general settings
DEFINE_STATIC_ACCESSOR_FUNCTION(int, verbose, 0)
DEFINE_STATIC_ACCESSOR_FUNCTION(bool, debug, false)
DEFINE_STATIC_ACCESSOR_FUNCTION(bool, banner, true)
DEFINE_STATIC_ACCESSOR_FUNCTION(uint16_t, max_depth, std::numeric_limits<uint16_t>::max())

// general formatting
DEFINE_STATIC_ACCESSOR_FUNCTION(int16_t, precision, -1)
DEFINE_STATIC_ACCESSOR_FUNCTION(int16_t, width, -1)
DEFINE_STATIC_ACCESSOR_FUNCTION(bool, scientific, false)

// timing formatting
DEFINE_STATIC_ACCESSOR_FUNCTION(int16_t, timing_precision, -1)
DEFINE_STATIC_ACCESSOR_FUNCTION(int16_t, timing_width, -1)
DEFINE_STATIC_ACCESSOR_FUNCTION(string_t, timing_units, "")
DEFINE_STATIC_ACCESSOR_FUNCTION(bool, timing_scientific, false)

// memory formatting
DEFINE_STATIC_ACCESSOR_FUNCTION(int16_t, memory_precision, -1)
DEFINE_STATIC_ACCESSOR_FUNCTION(int16_t, memory_width, -1)
DEFINE_STATIC_ACCESSOR_FUNCTION(string_t, memory_units, "")
DEFINE_STATIC_ACCESSOR_FUNCTION(bool, memory_scientific, false)

// output control
DEFINE_STATIC_ACCESSOR_FUNCTION(string_t, output_path, "timemory_output/")  // folder
DEFINE_STATIC_ACCESSOR_FUNCTION(string_t, output_prefix, "")                // file prefix

//--------------------------------------------------------------------------------------//
}  // namespace settings
}  // namespace tim
