//  MIT License
//
//  Copyright (c) 2020, The Regents of the University of California,
//  through Lawrence Berkeley National Laboratory (subject to receipt of any
//  required approvals from the U.S. Dept. of Energy).  All rights reserved.
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to deal
//  in the Software without restriction, including without limitation the rights
//  to use, copy, modify, merge, publish, distribute, sublicense, and
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in all
//  copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//  SOFTWARE.

/** \file components/types.hpp
 * \headerfile components/types.hpp "timemory/components/types.hpp"
 *
 * This is a pre-declaration of all the component structs.
 * Care should be taken to make sure that this includes a minimal
 * number of additional headers.
 *
 */

#pragma once

#include <cstdint>
#include <iostream>
#include <string>
#include <type_traits>

#include "timemory/backends/cuda.hpp"
#include "timemory/components/properties.hpp"

#if !defined(TIMEMORY_PAPI_ARRAY_SIZE)
#    define TIMEMORY_PAPI_ARRAY_SIZE 8
#endif

//======================================================================================//
//
namespace tim
{
//======================================================================================//
//  components that provide implementations (i.e. HOW to record a component)
//
namespace component
{
// this is a type for tagging native types
struct native_tag
{};

// define this short-hand from C++14 for C++11
template <bool B, typename T = int>
using enable_if_t = typename std::enable_if<B, T>::type;

// generic static polymorphic base class
template <typename _Tp, typename value_type = int64_t>
struct base;

// holder that provides nothing
template <typename... _Types>
struct placeholder;

// general
struct trip_count;
struct gperf_heap_profiler;
struct gperf_cpu_profiler;

// timing
struct wall_clock;
struct system_clock;
struct user_clock;
struct cpu_clock;
struct monotonic_clock;
struct monotonic_raw_clock;
struct thread_cpu_clock;
struct process_cpu_clock;
struct cpu_util;
struct process_cpu_util;
struct thread_cpu_util;

// resource usage
struct peak_rss;
struct page_rss;
struct stack_rss;
struct data_rss;
struct num_swap;
struct num_io_in;
struct num_io_out;
struct num_minor_page_faults;
struct num_major_page_faults;
struct num_msg_sent;
struct num_msg_recv;
struct num_signals;
struct voluntary_context_switch;
struct priority_context_switch;
struct virtual_memory;

// filesystem
struct read_bytes;
struct written_bytes;

// marker-forwarding
struct caliper;
struct likwid_perfmon;
struct likwid_nvmon;
struct tau_marker;

// vtune
struct vtune_frame;
struct vtune_event;

// cuda
struct cuda_event;
struct cuda_profiler;
struct nvtx_marker;
using cuda_nvtx = nvtx_marker;

template <size_t _N, typename _Components, typename _Differentiator = void>
struct gotcha;

// aliases
// papi
template <int... EventTypes>
struct papi_tuple;

template <size_t MaxNumEvents>
struct papi_array;

template <typename... _Types>
struct cpu_roofline;

// always defined
using papi_array8_t  = papi_array<8>;
using papi_array16_t = papi_array<16>;
using papi_array32_t = papi_array<32>;
using papi_array_t   = papi_array<TIMEMORY_PAPI_ARRAY_SIZE>;

// cupti
struct cupti_counters;
struct cupti_activity;

template <typename... _Types>
struct gpu_roofline;

//
// roofline aliases
//
using cpu_roofline_sp_flops = cpu_roofline<float>;
using cpu_roofline_dp_flops = cpu_roofline<double>;
using cpu_roofline_flops    = cpu_roofline<float, double>;

using gpu_roofline_sp_flops = gpu_roofline<float>;
using gpu_roofline_dp_flops = gpu_roofline<double>;
using gpu_roofline_hp_flops = gpu_roofline<cuda::fp16_t>;
using gpu_roofline_flops    = gpu_roofline<cuda::fp16_t, float, double>;

template <size_t _Idx, typename _Tag = native_tag>
struct user_bundle;

// reserved
using user_tuple_bundle = user_bundle<10101, native_tag>;
using user_list_bundle  = user_bundle<11011, native_tag>;

}  // namespace component
}  // namespace tim

//======================================================================================//

#if !defined(TIMEMORY_PROPERTY_SPECIALIZATION)
#    define TIMEMORY_PROPERTY_SPECIALIZATION(TYPE, ENUM, ID, ...)                        \
        template <>                                                                      \
        struct properties<TYPE>                                                          \
        {                                                                                \
            using type                                = TYPE;                            \
            using value_type                          = TIMEMORY_COMPONENT;              \
            static constexpr TIMEMORY_COMPONENT value = ENUM;                            \
            static constexpr const char*        enum_string() { return #ENUM; }          \
            static constexpr const char*        id() { return ID; }                      \
            static const idset_t&               ids()                                    \
            {                                                                            \
                static idset_t _instance{ ID, __VA_ARGS__ };                             \
                return _instance;                                                        \
            }                                                                            \
        };                                                                               \
        template <>                                                                      \
        struct enumerator<ENUM> : properties<TYPE>                                       \
        {                                                                                \
            using type = TYPE;                                                           \
        };
#endif

//======================================================================================//

#include <string>
#include <unordered_set>

//======================================================================================//

namespace tim
{
namespace component
{
//--------------------------------------------------------------------------------------//
//
TIMEMORY_PROPERTY_SPECIALIZATION(caliper, CALIPER, "caliper", "cali")

//--------------------------------------------------------------------------------------//

TIMEMORY_PROPERTY_SPECIALIZATION(cpu_clock, CPU_CLOCK, "cpu_clock")

//--------------------------------------------------------------------------------------//

TIMEMORY_PROPERTY_SPECIALIZATION(cpu_roofline_dp_flops, CPU_ROOFLINE_DP_FLOPS,
                                 "cpu_roofline_dp_flops", "cpu_roofline_dp",
                                 "cpu_roofline_double")

//--------------------------------------------------------------------------------------//

TIMEMORY_PROPERTY_SPECIALIZATION(cpu_roofline_flops, CPU_ROOFLINE_FLOPS,
                                 "cpu_roofline_flops", "cpu_roofline")

//--------------------------------------------------------------------------------------//

TIMEMORY_PROPERTY_SPECIALIZATION(cpu_roofline_sp_flops, CPU_ROOFLINE_SP_FLOPS,
                                 "cpu_roofline_sp_flops", "cpu_roofline_sp",
                                 "cpu_roofline_single")

//--------------------------------------------------------------------------------------//

TIMEMORY_PROPERTY_SPECIALIZATION(cpu_util, CPU_UTIL, "cpu_util")

//--------------------------------------------------------------------------------------//

TIMEMORY_PROPERTY_SPECIALIZATION(cuda_event, CUDA_EVENT, "cuda_event")

//--------------------------------------------------------------------------------------//

TIMEMORY_PROPERTY_SPECIALIZATION(cuda_profiler, CUDA_PROFILER, "cuda_profiler")

//--------------------------------------------------------------------------------------//

TIMEMORY_PROPERTY_SPECIALIZATION(cupti_activity, CUPTI_ACTIVITY, "cupti_activity")

//--------------------------------------------------------------------------------------//

TIMEMORY_PROPERTY_SPECIALIZATION(cupti_counters, CUPTI_COUNTERS, "cupti_counters")

//--------------------------------------------------------------------------------------//

TIMEMORY_PROPERTY_SPECIALIZATION(data_rss, DATA_RSS, "data_rss")

//--------------------------------------------------------------------------------------//

TIMEMORY_PROPERTY_SPECIALIZATION(gperf_cpu_profiler, GPERF_CPU_PROFILER,
                                 "gperf_cpu_profiler", "gperf_cpu", "gperftools-cpu")

//--------------------------------------------------------------------------------------//

TIMEMORY_PROPERTY_SPECIALIZATION(gperf_heap_profiler, GPERF_HEAP_PROFILER,
                                 "gperf_heap_profiler", "gperf_heap", "gperftools-heap")

//--------------------------------------------------------------------------------------//

TIMEMORY_PROPERTY_SPECIALIZATION(gpu_roofline_dp_flops, GPU_ROOFLINE_DP_FLOPS,
                                 "gpu_roofline_dp_flops", "gpu_roofline_dp",
                                 "gpu_roofline_double")

//--------------------------------------------------------------------------------------//

TIMEMORY_PROPERTY_SPECIALIZATION(gpu_roofline_flops, GPU_ROOFLINE_FLOPS,
                                 "gpu_roofline_flops", "gpu_roofline")

//--------------------------------------------------------------------------------------//

TIMEMORY_PROPERTY_SPECIALIZATION(gpu_roofline_hp_flops, GPU_ROOFLINE_HP_FLOPS,
                                 "gpu_roofline_hp_flops", "gpu_roofline_hp",
                                 "gpu_roofline_half")

//--------------------------------------------------------------------------------------//

TIMEMORY_PROPERTY_SPECIALIZATION(gpu_roofline_sp_flops, GPU_ROOFLINE_SP_FLOPS,
                                 "gpu_roofline_sp_flops", "gpu_roofline_sp",
                                 "gpu_roofline_single")

//--------------------------------------------------------------------------------------//

TIMEMORY_PROPERTY_SPECIALIZATION(likwid_nvmon, LIKWID_NVMON, "likwid_nvmon", "likwid_gpu")

//--------------------------------------------------------------------------------------//

TIMEMORY_PROPERTY_SPECIALIZATION(likwid_perfmon, LIKWID_PERFMON, "likwid_perfmon",
                                 "likwid_cpu")

//--------------------------------------------------------------------------------------//

TIMEMORY_PROPERTY_SPECIALIZATION(monotonic_clock, MONOTONIC_CLOCK, "monotonic_clock")

//--------------------------------------------------------------------------------------//

TIMEMORY_PROPERTY_SPECIALIZATION(monotonic_raw_clock, MONOTONIC_RAW_CLOCK,
                                 "monotonic_raw_clock")

//--------------------------------------------------------------------------------------//

TIMEMORY_PROPERTY_SPECIALIZATION(num_io_in, NUM_IO_IN, "num_io_in")

//--------------------------------------------------------------------------------------//

TIMEMORY_PROPERTY_SPECIALIZATION(num_io_out, NUM_IO_OUT, "num_io_out")

//--------------------------------------------------------------------------------------//

TIMEMORY_PROPERTY_SPECIALIZATION(num_major_page_faults, NUM_MAJOR_PAGE_FAULTS,
                                 "num_major_page_faults")

//--------------------------------------------------------------------------------------//

TIMEMORY_PROPERTY_SPECIALIZATION(num_minor_page_faults, NUM_MINOR_PAGE_FAULTS,
                                 "num_minor_page_faults")

//--------------------------------------------------------------------------------------//

TIMEMORY_PROPERTY_SPECIALIZATION(num_msg_recv, NUM_MSG_RECV, "num_msg_recv")

//--------------------------------------------------------------------------------------//

TIMEMORY_PROPERTY_SPECIALIZATION(num_msg_sent, NUM_MSG_SENT, "num_msg_sent")

//--------------------------------------------------------------------------------------//

TIMEMORY_PROPERTY_SPECIALIZATION(num_signals, NUM_SIGNALS, "num_signals")

//--------------------------------------------------------------------------------------//

TIMEMORY_PROPERTY_SPECIALIZATION(num_swap, NUM_SWAP, "num_swap")

//--------------------------------------------------------------------------------------//

TIMEMORY_PROPERTY_SPECIALIZATION(nvtx_marker, NVTX_MARKER, "nvtx_marker", "nvtx")

//--------------------------------------------------------------------------------------//

TIMEMORY_PROPERTY_SPECIALIZATION(page_rss, PAGE_RSS, "page_rss")

//--------------------------------------------------------------------------------------//

TIMEMORY_PROPERTY_SPECIALIZATION(papi_array_t, PAPI_ARRAY, "papi_array_t", "papi_array",
                                 "papi")

//--------------------------------------------------------------------------------------//

TIMEMORY_PROPERTY_SPECIALIZATION(peak_rss, PEAK_RSS, "peak_rss")

//--------------------------------------------------------------------------------------//

TIMEMORY_PROPERTY_SPECIALIZATION(priority_context_switch, PRIORITY_CONTEXT_SWITCH,
                                 "priority_context_switch")

//--------------------------------------------------------------------------------------//

TIMEMORY_PROPERTY_SPECIALIZATION(process_cpu_clock, PROCESS_CPU_CLOCK,
                                 "process_cpu_clock")

//--------------------------------------------------------------------------------------//

TIMEMORY_PROPERTY_SPECIALIZATION(process_cpu_util, PROCESS_CPU_UTIL, "process_cpu_util")

//--------------------------------------------------------------------------------------//

TIMEMORY_PROPERTY_SPECIALIZATION(read_bytes, READ_BYTES, "read_bytes")

//--------------------------------------------------------------------------------------//

TIMEMORY_PROPERTY_SPECIALIZATION(stack_rss, STACK_RSS, "stack_rss")

//--------------------------------------------------------------------------------------//

TIMEMORY_PROPERTY_SPECIALIZATION(system_clock, SYS_CLOCK, "system_clock", "sys_clock")

//--------------------------------------------------------------------------------------//

TIMEMORY_PROPERTY_SPECIALIZATION(tau_marker, TAU_MARKER, "tau_marker", "tau")

//--------------------------------------------------------------------------------------//

TIMEMORY_PROPERTY_SPECIALIZATION(thread_cpu_clock, THREAD_CPU_CLOCK, "thread_cpu_clock")

//--------------------------------------------------------------------------------------//

TIMEMORY_PROPERTY_SPECIALIZATION(thread_cpu_util, THREAD_CPU_UTIL, "thread_cpu_util")

//--------------------------------------------------------------------------------------//

TIMEMORY_PROPERTY_SPECIALIZATION(trip_count, TRIP_COUNT, "trip_count")

//--------------------------------------------------------------------------------------//

TIMEMORY_PROPERTY_SPECIALIZATION(user_clock, USER_CLOCK, "user_clock")

//--------------------------------------------------------------------------------------//

TIMEMORY_PROPERTY_SPECIALIZATION(user_list_bundle, USER_LIST_BUNDLE, "user_list_bundle")

//--------------------------------------------------------------------------------------//

TIMEMORY_PROPERTY_SPECIALIZATION(user_tuple_bundle, USER_TUPLE_BUNDLE,
                                 "user_tuple_bundle")

//--------------------------------------------------------------------------------------//

TIMEMORY_PROPERTY_SPECIALIZATION(virtual_memory, VIRTUAL_MEMORY, "virtual_memory")

//--------------------------------------------------------------------------------------//

TIMEMORY_PROPERTY_SPECIALIZATION(voluntary_context_switch, VOLUNTARY_CONTEXT_SWITCH,
                                 "voluntary_context_switch")

//--------------------------------------------------------------------------------------//

TIMEMORY_PROPERTY_SPECIALIZATION(vtune_event, VTUNE_EVENT, "vtune_event")

//--------------------------------------------------------------------------------------//

TIMEMORY_PROPERTY_SPECIALIZATION(vtune_frame, VTUNE_FRAME, "vtune_frame")

//--------------------------------------------------------------------------------------//

TIMEMORY_PROPERTY_SPECIALIZATION(wall_clock, WALL_CLOCK, "wall_clock", "real_clock",
                                 "virtual_clock")

//--------------------------------------------------------------------------------------//

TIMEMORY_PROPERTY_SPECIALIZATION(written_bytes, WRITTEN_BYTES, "written_bytes",
                                 "write_bytes")

//--------------------------------------------------------------------------------------//
//
//--------------------------------------------------------------------------------------//
}  // namespace component
}  // namespace tim

//======================================================================================//
