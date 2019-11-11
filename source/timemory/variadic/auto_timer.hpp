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

/** \file auto_tuple.hpp
 * \headerfile auto_tuple.hpp "timemory/variadic/auto_tuple.hpp"
 * Automatic timers. Exist for backwards compatibility. In C++, use auto_tuple.
 * Usage with macros (recommended):
 *    \param TIMEMORY_AUTO_TIMER("")
 *    \param TIMEMORY_BASIC_AUTO_TIMER("")
 *    \param auto t = TIMEMORY_AUTO_TIMER_HANDLE("")
 *    \param auto t = TIMEMORY_BASIC_AUTO_TIMER_HANDLE("")
 */

#pragma once

#include "timemory/components/types.hpp"
#include "timemory/variadic/macros.hpp"
#include "timemory/variadic/types.hpp"

namespace tim
{
//--------------------------------------------------------------------------------------//
namespace auto_timer_types
{
using namespace component;

using tuple_t = component_tuple<real_clock, system_clock, user_clock, cpu_util, peak_rss>;

//--------------------------------------------------------------------------------------//

#if defined(TIMEMORY_MINIMAL_AUTO_TIMER_LIST)

using list_t = component_list<page_rss, virtual_memory, cpu_clock, caliper, papi_array_t,
                              cuda_event, nvtx_marker, cupti_activity, cupti_counters>;

#else

using list_t =
    component_list<page_rss, virtual_memory, cpu_clock, thread_cpu_clock, thread_cpu_util,
                   process_cpu_clock, process_cpu_util, priority_context_switch,
                   voluntary_context_switch, num_major_page_faults, num_minor_page_faults,
                   read_bytes, written_bytes, caliper, papi_array_t,
                   cpu_roofline_sp_flops, cpu_roofline_dp_flops, gperf_cpu_profiler,
                   gperf_heap_profiler, cuda_event, nvtx_marker, cupti_activity,
                   cupti_counters, gpu_roofline_flops, gpu_roofline_hp_flops,
                   gpu_roofline_sp_flops, gpu_roofline_dp_flops>;

#endif
}  // namespace auto_timer_types

//--------------------------------------------------------------------------------------//

using auto_timer_tuple_t = auto_timer_types::tuple_t;
using auto_timer_list_t  = auto_timer_types::list_t;

#if defined(TIMEMORY_MINIMAL_AUTO_TIMER)
/// if compilation overhead is too high for C++, define TIMEMORY_MINIMAL_AUTO_TIMER before
/// including any timemory headers
using auto_timer = auto_timer_types::tuple_t;
#else
using auto_timer = auto_hybrid<auto_timer_tuple_t, auto_timer_list_t>;
#endif

//--------------------------------------------------------------------------------------//

}  // namespace tim

//======================================================================================//

#define TIMEMORY_BLANK_AUTO_TIMER(...)                                                   \
    TIMEMORY_BLANK_POINTER(::tim::auto_timer, __VA_ARGS__)

#define TIMEMORY_BASIC_AUTO_TIMER(...)                                                   \
    TIMEMORY_BASIC_POINTER(::tim::auto_timer, __VA_ARGS__)

#define TIMEMORY_AUTO_TIMER(...) TIMEMORY_POINTER(::tim::auto_timer, __VA_ARGS__)

//--------------------------------------------------------------------------------------//
// instance versions

#define TIMEMORY_BLANK_AUTO_TIMER_HANDLE(...)                                            \
    TIMEMORY_BLANK_HANDLE(::tim::auto_timer, __VA_ARGS__)

#define TIMEMORY_BASIC_AUTO_TIMER_HANDLE(...)                                            \
    TIMEMORY_BASIC_HANDLE(::tim::auto_timer, __VA_ARGS__)

#define TIMEMORY_AUTO_TIMER_HANDLE(...) TIMEMORY_HANDLE(::tim::auto_timer, __VA_ARGS__)

//--------------------------------------------------------------------------------------//
// debug versions

#define TIMEMORY_DEBUG_BASIC_AUTO_TIMER(...)                                             \
    TIMEMORY_DEBUG_BASIC_MARKER(::tim::auto_timer, __VA_ARGS__)

#define TIMEMORY_DEBUG_AUTO_TIMER(...)                                                   \
    TIMEMORY_DEBUG_MARKER(::tim::auto_timer, __VA_ARGS__)

//======================================================================================//
