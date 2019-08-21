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

#    include <ostream>
#    include <string>

namespace tim
{
void                              print_env() {}
template <typename... _Args> void timemory_init(_Args...) {}
void                              timemory_finalize() {}

/// this provides "functionality" for *_INSTANCE macros
/// and can be omitted if these macros are not utilized
struct dummy
{
    template <typename... _Args> dummy(_Args&&...) {}
    ~dummy()            = default;
    dummy(const dummy&) = default;
    dummy(dummy&&)      = default;
    dummy& operator=(const dummy&) = default;
    dummy& operator=(dummy&&) = default;

    void                              start() {}
    void                              stop() {}
    void                              conditional_start() {}
    void                              conditional_stop() {}
    void                              report_at_exit(bool) {}
    template <typename... _Args> void mark_begin(_Args&&...) {}
    template <typename... _Args> void mark_end(_Args&&...) {}
    friend std::ostream& operator<<(std::ostream& os, const dummy&) { return os; }
};
}  // namespace tim

// creates a label
#    define TIMEMORY_BASIC_LABEL(...) std::string("")
#    define TIMEMORY_LABEL(...) std::string("")

// define an object
#    define TIMEMORY_BLANK_OBJECT(...)
#    define TIMEMORY_BASIC_OBJECT(...)
#    define TIMEMORY_OBJECT(...)

// define an object with a caliper reference
#    define TIMEMORY_BLANK_CALIPER(...)
#    define TIMEMORY_BASIC_CALIPER(...)
#    define TIMEMORY_CALIPER(...)
#    define TIMEMORY_CALIPER_APPLY(...)
#    define TIMEMORY_CALIPER_TYPE_APPLY(...)

// define an object
#    define TIMEMORY_BLANK_INSTANCE(...) tim::dummy()
#    define TIMEMORY_BASIC_INSTANCE(...) tim::dummy()
#    define TIMEMORY_INSTANCE(...) tim::dummy()

// debug only
#    define TIMEMORY_DEBUG_BASIC_OBJECT(...)
#    define TIMEMORY_DEBUG_OBJECT(...)
#    define TIMEMORY_DEBUG_BASIC_OBJECT(...)
#    define TIMEMORY_DEBUG_OBJECT(...)

// for CUDA
#    define TIMEMORY_CALIPER_MARK_STREAM_BEGIN(...)
#    define TIMEMORY_CALIPER_MARK_STREAM_END(...)

#endif
