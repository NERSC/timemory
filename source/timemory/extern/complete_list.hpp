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

/** \file extern/complete_list.hpp
 * \headerfile extern/complete_list.hpp "timemory/extern/complete_list.hpp"
 * Extern template declarations
 *
 */

#pragma once

//--------------------------------------------------------------------------------------//
// complete_list
//
#if defined(TIMEMORY_EXTERN_TEMPLATES) && !defined(TIMEMORY_BUILD_EXTERN_TEMPLATE)

#    include "timemory/components.hpp"
#    include "timemory/utility/macros.hpp"
#    include "timemory/variadic/auto_hybrid.hpp"
#    include "timemory/variadic/auto_list.hpp"
#    include "timemory/variadic/auto_tuple.hpp"

TIMEMORY_DECLARE_EXTERN_LIST(
    complete_list_t, ::tim::component::caliper, ::tim::component::cpu_clock,
    ::tim::component::cpu_roofline_dp_flops, ::tim::component::cpu_roofline_flops,
    ::tim::component::cpu_roofline_sp_flops, ::tim::component::cpu_util,
    ::tim::component::cuda_event, ::tim::component::cuda_profiler,
    ::tim::component::cupti_activity, ::tim::component::cupti_counters,
    ::tim::component::data_rss, ::tim::component::gperf_cpu_profiler,
    ::tim::component::gperf_heap_profiler, ::tim::component::gpu_roofline_dp_flops,
    ::tim::component::gpu_roofline_flops, ::tim::component::gpu_roofline_hp_flops,
    ::tim::component::gpu_roofline_sp_flops, ::tim::component::likwid_nvmon,
    ::tim::component::likwid_perfmon, ::tim::component::monotonic_clock,
    ::tim::component::monotonic_raw_clock, ::tim::component::num_io_in,
    ::tim::component::num_io_out, ::tim::component::num_major_page_faults,
    ::tim::component::num_minor_page_faults, ::tim::component::num_msg_recv,
    ::tim::component::num_msg_sent, ::tim::component::num_signals,
    ::tim::component::num_swap, ::tim::component::nvtx_marker, ::tim::component::page_rss,
    ::tim::component::papi_array_t, ::tim::component::peak_rss,
    ::tim::component::priority_context_switch, ::tim::component::process_cpu_clock,
    ::tim::component::process_cpu_util, ::tim::component::read_bytes,
    ::tim::component::wall_clock, ::tim::component::stack_rss,
    ::tim::component::system_clock, ::tim::component::tau_marker,
    ::tim::component::thread_cpu_clock, ::tim::component::thread_cpu_util,
    ::tim::component::trip_count, ::tim::component::user_tuple_bundle,
    ::tim::component::user_list_bundle, ::tim::component::user_clock,
    ::tim::component::virtual_memory, ::tim::component::voluntary_context_switch,
    ::tim::component::written_bytes)

#endif

//======================================================================================//
