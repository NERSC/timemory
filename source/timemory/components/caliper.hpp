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

#include "timemory/components/base.hpp"
#include "timemory/components/types.hpp"
#include "timemory/units.hpp"
#include "timemory/utility/storage.hpp"

#if defined(TIMEMORY_USE_CALIPER)
#    include <caliper/cali.h>
#endif

namespace tim
{
namespace component
{
struct caliper : public base<caliper>
{
    using value_type = int64_t;
    using base_type  = base<caliper, value_type>;

    static const short                   precision = 3;
    static const short                   width     = 6;
    static const std::ios_base::fmtflags format_flags =
        std::ios_base::fixed | std::ios_base::dec | std::ios_base::showpoint;

    static int64_t     unit() { return 1; }
    static std::string label() { return ""; }
    static std::string descript() { return "caliper"; }
    static std::string display_unit() { return ""; }

    static value_type record() { return 0; }

    float compute_display() const { return 0.0f; }

    float get() const { return compute_display(); }

    caliper() { prefix = std::to_string(m_count); }
    void start()
    {
#if defined(TIMEMORY_USE_CALIPER)
        cali_begin_string(id, prefix.c_str());
#endif
    }

    void stop()
    {
#if defined(TIMEMORY_USE_CALIPER)
        cali_end(id);
#endif
    }

#if defined(TIMEMORY_USE_CALIPER)
    cali_id_t id = cali_create_attribute("timemory", CALI_TYPE_STRING, CALI_ATTR_NESTED);
#endif

    std::string prefix;
};

}  // namespace component
}  // namespace tim
