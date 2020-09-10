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

#include "timemory/backends/types/papi.hpp"

#include <functional>
#include <string>
#include <vector>

namespace tim
{
namespace hardware_counters
{
using string_t = std::string;
//
struct api
{
    enum type
    {
        papi = 0,
        cupti,
        unknown
    };
};
//
struct info
: public std::tuple<bool, int, int32_t, int32_t, string_t, string_t, string_t, string_t>
{
    using base_type =
        std::tuple<bool, int, int32_t, int32_t, string_t, string_t, string_t, string_t>;

    info()            = default;
    ~info()           = default;
    info(info&&)      = default;
    info(const info&) = default;

    info& operator=(info&&) = default;
    info& operator=(const info&) = default;

    info(const base_type& rhs)
    : base_type(rhs)
    {}

    info(base_type&& rhs)
    : base_type(std::forward<base_type>(rhs))
    {}

    info(bool _avail, int _cat, int32_t _idx, int32_t _off, const string_t& _sym,
         const string_t& _pysym, const string_t& _short, const string_t& _long)
    : base_type(_avail, _cat, _idx, _off, _sym, _pysym, _short, _long)
    {}

#define TIMEMORY_HWCOUNTER_INFO_ACCESSOR(NAME, INDEX)                                    \
    auto&       NAME() { return std::get<INDEX>(*this); }                                \
    const auto& NAME() const { return std::get<INDEX>(*this); }
    //
    TIMEMORY_HWCOUNTER_INFO_ACCESSOR(available, 0)
    TIMEMORY_HWCOUNTER_INFO_ACCESSOR(iface, 1)
    TIMEMORY_HWCOUNTER_INFO_ACCESSOR(index, 2)
    TIMEMORY_HWCOUNTER_INFO_ACCESSOR(offset, 3)
    TIMEMORY_HWCOUNTER_INFO_ACCESSOR(symbol, 4)
    TIMEMORY_HWCOUNTER_INFO_ACCESSOR(python_symbol, 5)
    TIMEMORY_HWCOUNTER_INFO_ACCESSOR(short_description, 6)
    TIMEMORY_HWCOUNTER_INFO_ACCESSOR(long_description, 7)
    //
#undef TIMEMORY_HWCOUNTER_INFO_ACCESSOR

    auto id() const { return (this->index() | this->offset()); }
};
//
//--------------------------------------------------------------------------------------//
//
using info_vec_t = std::vector<info>;
//
inline info_vec_t&
get_info()
{
    static info_vec_t _instance;
    return _instance;
}
//
}  // namespace hardware_counters
}  // namespace tim
