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
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR rhs
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR rhsWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR rhs DEALINGS IN THE
// SOFTWARE.

#pragma once

#ifndef TIMEMORY_LOG_COLORS_AVAILABLE
#    define TIMEMORY_LOG_COLORS_AVAILABLE 1
#endif

namespace tim
{
namespace log
{
inline bool&
colorized()
{
    static bool _v = true;
    return _v;
}

namespace color
{
static constexpr auto info_value    = "\033[01;34m";
static constexpr auto warning_value = "\033[01;33m";
static constexpr auto fatal_value   = "\033[01;31m";
static constexpr auto source_value  = "\033[01;32m";
static constexpr auto end_value     = "\033[0m";

inline const char*
info()
{
    return (log::colorized()) ? info_value : "";
}

inline const char*
warning()
{
    return (log::colorized()) ? warning_value : "";
}

inline const char*
fatal()
{
    return (log::colorized()) ? fatal_value : "";
}

inline const char*
source()
{
    return (log::colorized()) ? source_value : "";
}

inline const char*
end()
{
    return (log::colorized()) ? end_value : "";
}
}  // namespace color
}  // namespace log
}  // namespace tim
