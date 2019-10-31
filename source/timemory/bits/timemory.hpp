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

/** \file bits/timemory.hpp
 * \headerfile bits/timemory.hpp "timemory/bits/timemory.hpp"
 * Provides collection types
 *
 */

#pragma once

#include "timemory/components/types.hpp"
#include "timemory/variadic/types.hpp"

#include <tuple>

namespace tim
{
//--------------------------------------------------------------------------------------//
//
//
using complete_tuple_t = std::tuple<
    component::caliper, component::cpu_clock, component::cpu_roofline_dp_flops,
    component::cpu_roofline_flops, component::cpu_roofline_sp_flops, component::cpu_util,
    component::cuda_event, component::cupti_activity, component::cupti_counters,
    component::data_rss, component::gperf_cpu_profiler, component::gperf_heap_profiler,
    component::gpu_roofline_dp_flops, component::gpu_roofline_flops,
    component::gpu_roofline_hp_flops, component::gpu_roofline_sp_flops,
    component::monotonic_clock, component::monotonic_raw_clock, component::num_io_in,
    component::num_io_out, component::num_major_page_faults,
    component::num_minor_page_faults, component::num_msg_recv, component::num_msg_sent,
    component::num_signals, component::num_swap, component::nvtx_marker,
    component::page_rss, component::papi_array_t, component::peak_rss,
    component::priority_context_switch, component::process_cpu_clock,
    component::process_cpu_util, component::read_bytes, component::real_clock,
    component::stack_rss, component::system_clock, component::thread_cpu_clock,
    component::thread_cpu_util, component::trip_count, component::user_clock,
    component::virtual_memory, component::voluntary_context_switch,
    component::written_bytes>;

using complete_auto_list_t = auto_list<
    component::caliper, component::cpu_clock, component::cpu_roofline_dp_flops,
    component::cpu_roofline_flops, component::cpu_roofline_sp_flops, component::cpu_util,
    component::cuda_event, component::cupti_activity, component::cupti_counters,
    component::data_rss, component::gperf_cpu_profiler, component::gperf_heap_profiler,
    component::gpu_roofline_dp_flops, component::gpu_roofline_flops,
    component::gpu_roofline_hp_flops, component::gpu_roofline_sp_flops,
    component::monotonic_clock, component::monotonic_raw_clock, component::num_io_in,
    component::num_io_out, component::num_major_page_faults,
    component::num_minor_page_faults, component::num_msg_recv, component::num_msg_sent,
    component::num_signals, component::num_swap, component::nvtx_marker,
    component::page_rss, component::papi_array_t, component::peak_rss,
    component::priority_context_switch, component::process_cpu_clock,
    component::process_cpu_util, component::read_bytes, component::real_clock,
    component::stack_rss, component::system_clock, component::thread_cpu_clock,
    component::thread_cpu_util, component::trip_count, component::user_clock,
    component::virtual_memory, component::voluntary_context_switch,
    component::written_bytes>;

using complete_list_t = component_list<
    component::caliper, component::cpu_clock, component::cpu_roofline_dp_flops,
    component::cpu_roofline_flops, component::cpu_roofline_sp_flops, component::cpu_util,
    component::cuda_event, component::cupti_activity, component::cupti_counters,
    component::data_rss, component::gperf_cpu_profiler, component::gperf_heap_profiler,
    component::gpu_roofline_dp_flops, component::gpu_roofline_flops,
    component::gpu_roofline_hp_flops, component::gpu_roofline_sp_flops,
    component::monotonic_clock, component::monotonic_raw_clock, component::num_io_in,
    component::num_io_out, component::num_major_page_faults,
    component::num_minor_page_faults, component::num_msg_recv, component::num_msg_sent,
    component::num_signals, component::num_swap, component::nvtx_marker,
    component::page_rss, component::papi_array_t, component::peak_rss,
    component::priority_context_switch, component::process_cpu_clock,
    component::process_cpu_util, component::read_bytes, component::real_clock,
    component::stack_rss, component::system_clock, component::thread_cpu_clock,
    component::thread_cpu_util, component::trip_count, component::user_clock,
    component::virtual_memory, component::voluntary_context_switch,
    component::written_bytes>;

using recommended_auto_tuple_t =
    auto_tuple<component::real_clock, component::system_clock, component::user_clock,
               component::cpu_util, component::page_rss, component::peak_rss,
               component::read_bytes, component::written_bytes,
               component::num_minor_page_faults, component::num_major_page_faults,
               component::voluntary_context_switch, component::priority_context_switch>;

using recommended_tuple_t =
    component_tuple<component::real_clock, component::system_clock, component::user_clock,
                    component::cpu_util, component::page_rss, component::peak_rss,
                    component::read_bytes, component::written_bytes,
                    component::num_minor_page_faults, component::num_major_page_faults,
                    component::voluntary_context_switch,
                    component::priority_context_switch>;

using recommended_auto_list_t =
    auto_list<component::caliper, component::papi_array_t, component::cuda_event,
              component::nvtx_marker, component::cupti_counters,
              component::cupti_activity, component::cpu_roofline_flops,
              component::gpu_roofline_flops, component::gperf_cpu_profiler,
              component::gperf_heap_profiler>;

using recommended_list_t =
    component_list<component::caliper, component::papi_array_t, component::cuda_event,
                   component::nvtx_marker, component::cupti_counters,
                   component::cupti_activity, component::cpu_roofline_flops,
                   component::gpu_roofline_flops, component::gperf_cpu_profiler,
                   component::gperf_heap_profiler>;

using recommended_auto_hybrid_t = auto_hybrid<recommended_tuple_t, recommended_list_t>;

using recommended_hybrid_t = component_hybrid<recommended_tuple_t, recommended_list_t>;

//--------------------------------------------------------------------------------------//
//  category configurations
//
using rusage_components_t = component_tuple<
    component::page_rss, component::peak_rss, component::stack_rss, component::data_rss,
    component::num_swap, component::num_io_in, component::num_io_out,
    component::num_minor_page_faults, component::num_major_page_faults,
    component::num_msg_sent, component::num_msg_recv, component::num_signals,
    component::voluntary_context_switch, component::priority_context_switch>;

using timing_components_t =
    component_tuple<component::real_clock, component::system_clock, component::user_clock,
                    component::cpu_clock, component::monotonic_clock,
                    component::monotonic_raw_clock, component::thread_cpu_clock,
                    component::process_cpu_clock, component::cpu_util,
                    component::thread_cpu_util, component::process_cpu_util>;

//--------------------------------------------------------------------------------------//
//  standard configurations
//
using standard_rusage_t =
    component_tuple<component::page_rss, component::peak_rss, component::num_io_in,
                    component::num_io_out, component::num_minor_page_faults,
                    component::num_major_page_faults, component::priority_context_switch,
                    component::voluntary_context_switch>;

using standard_timing_t =
    component_tuple<component::real_clock, component::user_clock, component::system_clock,
                    component::cpu_clock, component::cpu_util>;

}  // namespace tim
