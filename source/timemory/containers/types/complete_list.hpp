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

/**
 * \file timemory/containers/types/complete_list.hpp
 * \brief Include the extern declarations for complete_list in containers
 */

#pragma once

//======================================================================================//
//
#include "timemory/containers/macros.hpp"
//
#include "timemory/containers/types.hpp"
//
#include "timemory/containers/declaration.hpp"
//
#include "timemory/components/types.hpp"
//
#include "timemory/components/extern.hpp"
#include "timemory/manager/extern.hpp"
#include "timemory/storage/extern.hpp"
//
#include "timemory/variadic/component_hybrid.hpp"
#include "timemory/variadic/component_list.hpp"
#include "timemory/variadic/component_tuple.hpp"
//
#include "timemory/variadic/auto_hybrid.hpp"
#include "timemory/variadic/auto_list.hpp"
#include "timemory/variadic/auto_tuple.hpp"
//
#include "timemory/runtime/configure.hpp"
#include "timemory/runtime/enumerate.hpp"
#include "timemory/runtime/initialize.hpp"
#include "timemory/runtime/properties.hpp"
//
//======================================================================================//
//
// clang-format off
//
TIMEMORY_EXTERN_LIST(complete_list_t,
                     component::caliper,
                     component::cpu_clock,
                     component::cpu_roofline_dp_flops,
                     component::cpu_roofline_flops,
                     component::cpu_roofline_sp_flops,
                     component::cpu_util,
                     component::cuda_event,
                     component::cuda_profiler,
                     component::cupti_activity,
                     component::cupti_counters,
                     component::current_peak_rss,
                     component::data_rss,
                     component::gperf_cpu_profiler,
                     component::gperf_heap_profiler,
                     component::gpu_roofline_dp_flops,
                     component::gpu_roofline_flops,
                     component::gpu_roofline_hp_flops,
                     component::gpu_roofline_sp_flops,
                     component::kernel_mode_time,
                     component::likwid_marker,
                     component::likwid_nvmarker,
                     component::malloc_gotcha,
                     component::monotonic_clock,
                     component::monotonic_raw_clock,
                     component::num_io_in,
                     component::num_io_out,
                     component::num_major_page_faults,
                     component::num_minor_page_faults,
                     component::num_msg_recv,
                     component::num_msg_sent,
                     component::num_signals,
                     component::num_swap,
                     component::nvtx_marker,
                     component::page_rss,
                     component::papi_array_t,
                     component::papi_vector,
                     component::peak_rss,
                     component::priority_context_switch,
                     component::process_cpu_clock,
                     component::process_cpu_util,
                     component::read_bytes,
                     component::stack_rss,
                     component::system_clock,
                     component::tau_marker,
                     component::thread_cpu_clock,
                     component::thread_cpu_util,
                     component::trip_count,
                     component::user_clock,
                     component::user_global_bundle,
                     component::user_mode_time,
                     component::virtual_memory,
                     component::voluntary_context_switch,
                     component::vtune_event,
                     component::vtune_frame,
                     component::vtune_profiler,
                     component::wall_clock,
                     component::written_bytes)
//
//--------------------------------------------------------------------------------------//
//
// clang-format off
//
