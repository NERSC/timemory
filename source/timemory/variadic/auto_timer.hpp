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

#include "timemory/variadic/auto_hybrid.hpp"
#include "timemory/variadic/auto_list.hpp"
#include "timemory/variadic/auto_tuple.hpp"
#include "timemory/variadic/component_hybrid.hpp"
#include "timemory/variadic/component_list.hpp"
#include "timemory/variadic/component_tuple.hpp"
#include "timemory/variadic/macros.hpp"

namespace tim
{
//--------------------------------------------------------------------------------------//

using auto_timer_tuple_t =
    component_tuple<component::real_clock, component::system_clock, component::user_clock,
                    component::cpu_util, component::page_rss, component::peak_rss>;

using auto_timer_list_t =
    component_list<component::caliper, component::papi_array_t, component::cuda_event,
                   component::nvtx_marker, component::cupti_activity,
                   component::cupti_counters, component::cpu_roofline_flops,
                   component::cpu_roofline_sp_flops, component::cpu_roofline_dp_flops,
                   component::gpu_roofline_flops, component::gpu_roofline_hp_flops,
                   component::gpu_roofline_sp_flops, component::gpu_roofline_dp_flops,
                   component::gperf_cpu_profiler, component::gperf_heap_profiler>;

using auto_timer = auto_hybrid<auto_timer_tuple_t, auto_timer_list_t>;

//--------------------------------------------------------------------------------------//

}  // namespace tim

//======================================================================================//

#define TIMEMORY_BLANK_AUTO_TIMER(...) TIMEMORY_BLANK_MARKER(tim::auto_timer, __VA_ARGS__)

#define TIMEMORY_BASIC_AUTO_TIMER(...) TIMEMORY_BASIC_MARKER(tim::auto_timer, __VA_ARGS__)

#define TIMEMORY_AUTO_TIMER(...) TIMEMORY_MARKER(tim::auto_timer, __VA_ARGS__)

//--------------------------------------------------------------------------------------//
// instance versions

#define TIMEMORY_BLANK_AUTO_TIMER_HANDLE(...)                                            \
    TIMEMORY_BLANK_HANDLE(tim::auto_timer, __VA_ARGS__)

#define TIMEMORY_BASIC_AUTO_TIMER_HANDLE(...)                                            \
    TIMEMORY_BASIC_HANDLE(tim::auto_timer, __VA_ARGS__)

#define TIMEMORY_AUTO_TIMER_HANDLE(...) TIMEMORY_HANDLE(tim::auto_timer, __VA_ARGS__)

//--------------------------------------------------------------------------------------//
// debug versions

#define TIMEMORY_DEBUG_BASIC_AUTO_TIMER(...)                                             \
    TIMEMORY_DEBUG_BASIC_MARKER(tim::auto_timer, __VA_ARGS__)

#define TIMEMORY_DEBUG_AUTO_TIMER(...) TIMEMORY_DEBUG_MARKER(tim::auto_timer, __VA_ARGS__)

//======================================================================================//
