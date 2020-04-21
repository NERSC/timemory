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

#pragma once

// clang-format off
#include "timemory/components/opaque.hpp"
#include "timemory/components/opaque/definition.hpp"
#include "timemory/components/extern.hpp"
#include "timemory/components.hpp"
#include "timemory/dll.hpp"
// clang-format on
#include "timemory/storage/declaration.hpp"
#include "timemory/storage/definition.hpp"
#include "timemory/storage/extern.hpp"

#if defined(TIMEMORY_FACTORY_SOURCE)
#    define TIMEMORY_FACTORY_DLL tim_dll_export
#elif defined(TIMEMORY_USE_EXTERN) || defined(TIMEMORY_USE_FACTORY_EXTERN)
#    define TIMEMORY_FACTORY_DLL tim_dll_import
#else
#    define TIMEMORY_FACTORY_DLL
#endif

#if !defined(TIMEMORY_EXTERN_FACTORY_TEMPLATE)
#    if defined(TIMEMORY_FACTORY_SOURCE)
#        define TIMEMORY_EXTERN_FACTORY_TEMPLATE(TYPE)                                   \
            namespace tim                                                                \
            {                                                                            \
            namespace component                                                          \
            {                                                                            \
            namespace factory                                                            \
            {                                                                            \
            template TIMEMORY_FACTORY_DLL opaque get_opaque<TYPE>();                     \
            template TIMEMORY_FACTORY_DLL opaque get_opaque<TYPE>(bool);                 \
            template TIMEMORY_FACTORY_DLL opaque get_opaque<TYPE>(scope::config);        \
            template TIMEMORY_FACTORY_DLL std::set<size_t> get_typeids<TYPE>();          \
            }                                                                            \
            }                                                                            \
            }
#    elif defined(TIMEMORY_USE_EXTERN) || defined(TIMEMORY_USE_FACTORY_EXTERN)
#        define TIMEMORY_EXTERN_FACTORY_TEMPLATE(TYPE)                                   \
            namespace tim                                                                \
            {                                                                            \
            namespace component                                                          \
            {                                                                            \
            namespace factory                                                            \
            {                                                                            \
            extern template TIMEMORY_FACTORY_DLL opaque get_opaque<TYPE>();              \
            extern template TIMEMORY_FACTORY_DLL opaque get_opaque<TYPE>(bool);          \
            extern template TIMEMORY_FACTORY_DLL opaque get_opaque<TYPE>(scope::config); \
            extern template TIMEMORY_FACTORY_DLL std::set<size_t> get_typeids<TYPE>();   \
            }                                                                            \
            }                                                                            \
            }
#    else
#        define TIMEMORY_EXTERN_FACTORY_TEMPLATE(...)
#    endif
#endif

TIMEMORY_EXTERN_FACTORY_TEMPLATE(allinea_map)
TIMEMORY_EXTERN_FACTORY_TEMPLATE(caliper)
TIMEMORY_EXTERN_FACTORY_TEMPLATE(craypat_record)
TIMEMORY_EXTERN_FACTORY_TEMPLATE(craypat_region)
TIMEMORY_EXTERN_FACTORY_TEMPLATE(craypat_counters)
TIMEMORY_EXTERN_FACTORY_TEMPLATE(craypat_heap_stats)
TIMEMORY_EXTERN_FACTORY_TEMPLATE(craypat_flush_buffer)
TIMEMORY_EXTERN_FACTORY_TEMPLATE(cpu_clock)
TIMEMORY_EXTERN_FACTORY_TEMPLATE(cpu_roofline_dp_flops)
TIMEMORY_EXTERN_FACTORY_TEMPLATE(cpu_roofline_flops)
TIMEMORY_EXTERN_FACTORY_TEMPLATE(cpu_roofline_sp_flops)
TIMEMORY_EXTERN_FACTORY_TEMPLATE(cpu_util)
TIMEMORY_EXTERN_FACTORY_TEMPLATE(cuda_event)
TIMEMORY_EXTERN_FACTORY_TEMPLATE(cuda_profiler)
TIMEMORY_EXTERN_FACTORY_TEMPLATE(cupti_activity)
TIMEMORY_EXTERN_FACTORY_TEMPLATE(cupti_counters)
TIMEMORY_EXTERN_FACTORY_TEMPLATE(current_peak_rss)
TIMEMORY_EXTERN_FACTORY_TEMPLATE(data_rss)
TIMEMORY_EXTERN_FACTORY_TEMPLATE(gperftools_cpu_profiler)
TIMEMORY_EXTERN_FACTORY_TEMPLATE(gperftools_heap_profiler)
TIMEMORY_EXTERN_FACTORY_TEMPLATE(gpu_roofline_dp_flops)
TIMEMORY_EXTERN_FACTORY_TEMPLATE(gpu_roofline_flops)
TIMEMORY_EXTERN_FACTORY_TEMPLATE(gpu_roofline_hp_flops)
TIMEMORY_EXTERN_FACTORY_TEMPLATE(gpu_roofline_sp_flops)
TIMEMORY_EXTERN_FACTORY_TEMPLATE(kernel_mode_time)
TIMEMORY_EXTERN_FACTORY_TEMPLATE(likwid_marker)
TIMEMORY_EXTERN_FACTORY_TEMPLATE(likwid_nvmarker)
TIMEMORY_EXTERN_FACTORY_TEMPLATE(malloc_gotcha)
TIMEMORY_EXTERN_FACTORY_TEMPLATE(monotonic_clock)
TIMEMORY_EXTERN_FACTORY_TEMPLATE(monotonic_raw_clock)
TIMEMORY_EXTERN_FACTORY_TEMPLATE(num_io_in)
TIMEMORY_EXTERN_FACTORY_TEMPLATE(num_io_out)
TIMEMORY_EXTERN_FACTORY_TEMPLATE(num_major_page_faults)
TIMEMORY_EXTERN_FACTORY_TEMPLATE(num_minor_page_faults)
TIMEMORY_EXTERN_FACTORY_TEMPLATE(num_msg_recv)
TIMEMORY_EXTERN_FACTORY_TEMPLATE(num_msg_sent)
TIMEMORY_EXTERN_FACTORY_TEMPLATE(num_signals)
TIMEMORY_EXTERN_FACTORY_TEMPLATE(num_swap)
TIMEMORY_EXTERN_FACTORY_TEMPLATE(nvtx_marker)
TIMEMORY_EXTERN_FACTORY_TEMPLATE(ompt_native_handle)
TIMEMORY_EXTERN_FACTORY_TEMPLATE(page_rss)
TIMEMORY_EXTERN_FACTORY_TEMPLATE(papi_array_t)
TIMEMORY_EXTERN_FACTORY_TEMPLATE(papi_vector)
TIMEMORY_EXTERN_FACTORY_TEMPLATE(peak_rss)
TIMEMORY_EXTERN_FACTORY_TEMPLATE(priority_context_switch)
TIMEMORY_EXTERN_FACTORY_TEMPLATE(process_cpu_clock)
TIMEMORY_EXTERN_FACTORY_TEMPLATE(process_cpu_util)
TIMEMORY_EXTERN_FACTORY_TEMPLATE(read_bytes)
TIMEMORY_EXTERN_FACTORY_TEMPLATE(stack_rss)
TIMEMORY_EXTERN_FACTORY_TEMPLATE(system_clock)
TIMEMORY_EXTERN_FACTORY_TEMPLATE(tau_marker)
TIMEMORY_EXTERN_FACTORY_TEMPLATE(thread_cpu_clock)
TIMEMORY_EXTERN_FACTORY_TEMPLATE(thread_cpu_util)
TIMEMORY_EXTERN_FACTORY_TEMPLATE(trip_count)
TIMEMORY_EXTERN_FACTORY_TEMPLATE(user_clock)
TIMEMORY_EXTERN_FACTORY_TEMPLATE(user_mode_time)
TIMEMORY_EXTERN_FACTORY_TEMPLATE(virtual_memory)
TIMEMORY_EXTERN_FACTORY_TEMPLATE(voluntary_context_switch)
TIMEMORY_EXTERN_FACTORY_TEMPLATE(vtune_event)
TIMEMORY_EXTERN_FACTORY_TEMPLATE(vtune_frame)
TIMEMORY_EXTERN_FACTORY_TEMPLATE(vtune_profiler)
TIMEMORY_EXTERN_FACTORY_TEMPLATE(wall_clock)
TIMEMORY_EXTERN_FACTORY_TEMPLATE(written_bytes)
