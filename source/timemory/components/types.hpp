//  MIT License
//
//  Copyright (c) 2019, The Regents of the University of California,
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
#include "timemory/bits/types.hpp"

#if !defined(TIMEMORY_PAPI_ARRAY_SIZE)
#    define TIMEMORY_PAPI_ARRAY_SIZE 32
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
// define this short-hand from C++14 for C++11
template <bool B, typename T = int>
using enable_if_t = typename std::enable_if<B, T>::type;

// generic static polymorphic base class
template <typename _Tp, typename value_type = int64_t, typename... _Policies>
struct base;

// holder that provides nothing
template <typename... _Types>
struct placeholder;

// general
struct trip_count;
struct gperf_heap_profiler;
struct gperf_cpu_profiler;

// timing
struct real_clock;
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

// caliper
struct caliper;

// cuda
struct cuda_event;

// nvtx
struct nvtx_marker;

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

}  // namespace component
}  // namespace tim

//======================================================================================//
