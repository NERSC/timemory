//  MIT License
//
//  Copyright (c) 2019, The Regents of the University of California,
// through Lawrence Berkeley National Laboratory (subject to receipt of any
// required approvals from the U.S. Dept. of Energy).  All rights reserved.
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to
//  deal in the Software without restriction, including without limitation the
//  rights to use, copy, modify, merge, publish, distribute, sublicense, and
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in
//  all copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
//  IN THE SOFTWARE.

/** \file ctimemory.cpp
 * This is the C++ proxy for the C interface. Compilation of this file is not
 * required for C++ codes but is compiled into "libtimemory.*" (timemory-cxx-library)
 * so that the "libctimemory.*" can be linked during the TiMemory build and
 * "libctimemory.*" can be stand-alone linked to C code.
 *
 */

#define TIMEMORY_EXTERN_TEMPLATE_BUILD

#include "timemory/variadic/auto_timer.hpp"  // for auto_timer_list_t, auto_...
#include "timemory/components.hpp"           // for papi_array_t
#include "timemory/components/types.hpp"     // for cpu_roofline_dp_flops
#include "timemory/utility/macros.hpp"       // for TIMEMORY_INSTANTIATE_EXT...
#include "timemory/variadic/auto_hybrid.hpp"
#include "timemory/variadic/auto_list.hpp"
#include "timemory/variadic/auto_tuple.hpp"

//======================================================================================//
//
//                      C++ extern template instantiation
//
//======================================================================================//

namespace component = ::tim::component;

//--------------------------------------------------------------------------------------//
// auto-timer
//
TIMEMORY_INSTANTIATE_EXTERN_TUPLE(auto_timer_t, component::real_clock,
                                  component::system_clock, component::user_clock,
                                  component::cpu_util, component::page_rss,
                                  component::peak_rss)

TIMEMORY_INSTANTIATE_EXTERN_LIST(
    auto_timer_t, component::caliper, component::papi_array_t, component::cuda_event,
    component::nvtx_marker, component::cupti_activity, component::cupti_counters,
    component::cpu_roofline_flops, component::cpu_roofline_sp_flops,
    component::cpu_roofline_dp_flops, component::gpu_roofline_flops,
    component::gpu_roofline_hp_flops, component::gpu_roofline_sp_flops,
    component::gpu_roofline_dp_flops, component::gperf_cpu_profiler,
    component::gperf_heap_profiler)

using _tuple_t = tim::auto_timer_tuple_t;
using _list_t  = tim::auto_timer_list_t;

TIMEMORY_INSTANTIATE_EXTERN_HYBRID(auto_timer_t)
