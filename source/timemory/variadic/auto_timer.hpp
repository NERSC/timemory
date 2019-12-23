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

/** \file timemory/variadic/auto_timer.hpp
 * \headerfile timemory/variadic/auto_timer.hpp "timemory/variadic/auto_timer.hpp"
 * Automatic timers. Exist for backwards compatibility. In C++, use auto_tuple.
 * Usage with macros (recommended):
 *    \param TIMEMORY_AUTO_TIMER("")
 *    \param TIMEMORY_BASIC_AUTO_TIMER("")
 *    auto t = \param TIMEMORY_AUTO_TIMER_HANDLE("")
 *    auto t = \param TIMEMORY_BASIC_AUTO_TIMER_HANDLE("")
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

using minimal_tuple_t =
    component_tuple<wall_clock, cpu_clock, cpu_util, peak_rss, user_tuple_bundle>;

using full_tuple_t = component_tuple<wall_clock, system_clock, user_clock, cpu_util,
                                     peak_rss, user_tuple_bundle>;

//--------------------------------------------------------------------------------------//

using minimal_list_t =
    component_list<user_list_bundle, caliper, tau_marker, papi_array_t, cuda_event,
                   nvtx_marker, cupti_activity, cupti_counters>;

using full_list_t =
    component_list<user_list_bundle, gperf_cpu_profiler, gperf_heap_profiler, caliper,
                   tau_marker, papi_array_t, cpu_roofline_sp_flops, cpu_roofline_dp_flops,
                   cuda_event, nvtx_marker, cupti_activity, cupti_counters,
                   gpu_roofline_flops, gpu_roofline_hp_flops, gpu_roofline_sp_flops,
                   gpu_roofline_dp_flops>;

}  // namespace auto_timer_types

//--------------------------------------------------------------------------------------//

using minimal_auto_timer =
    auto_hybrid<auto_timer_types::minimal_tuple_t, auto_timer_types::minimal_list_t>;
using full_auto_timer =
    auto_hybrid<auto_timer_types::full_tuple_t, auto_timer_types::full_list_t>;

#if defined(TIMEMORY_MINIMAL_AUTO_TIMER)

using auto_timer_tuple_t = auto_timer_types::minimal_tuple_t;
using auto_timer_list_t  = auto_timer_types::minimal_list_t;
using auto_timer         = minimal_auto_timer;

#else

using auto_timer_tuple_t = auto_timer_types::full_tuple_t;
using auto_timer_list_t  = auto_timer_types::full_list_t;
using auto_timer         = full_auto_timer;

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
