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
//

#pragma once

#if defined(USE_TIMEMORY)

#    include <timemory/timemory.hpp>

using tim::component::cpu_clock;
using tim::component::cpu_roofline;
using tim::component::cpu_util;
using tim::component::current_rss;
using tim::component::peak_rss;
using tim::component::real_clock;
using tim::component::thread_cpu_clock;
using tim::component::thread_cpu_util;

// some using statements
using roofline_t = cpu_roofline<double, PAPI_DP_OPS>;
using auto_tuple_t =
    tim::auto_tuple<real_clock, cpu_clock, cpu_util, peak_rss, current_rss, roofline_t>;
using auto_tuple_thr = tim::auto_tuple<real_clock, thread_cpu_clock, thread_cpu_util,
                                       peak_rss, current_rss, roofline_t>;

#else

#    include <string>
#    define TIMEMORY_AUTO_TUPLE(...)
#    define TIMEMORY_BASIC_AUTO_TUPLE(...)
#    define TIMEMORY_BLANK_AUTO_TUPLE(...)
#    define TIMEMORY_AUTO_TUPLE_CALIPER(...)
#    define TIMEMORY_BASIC_AUTO_TUPLE_CALIPER(...)
#    define TIMEMORY_BLANK_AUTO_TUPLE_CALIPER(...)
#    define TIMEMORY_CALIPER_APPLY(...)

namespace tim
{
void print_env() {}
void timemory_init(int, char**, const std::string& = "", const std::string& = "") {}
void timemory_init(const std::string&, const std::string& = "", const std::string& = "")
{}
}

#endif
