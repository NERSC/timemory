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

#pragma once

#include "timemory/components/types.hpp"
#include "timemory/ctimemory.h"

#include <unordered_map>

namespace tim
{
namespace component
{
//--------------------------------------------------------------------------------------//

template <typename _Tp>
struct properties
{
    static constexpr TIMEMORY_COMPONENT value = TIMEMORY_COMPONENTS_END;
    static bool&                        has_storage()
    {
        static thread_local bool _instance = false;
        return _instance;
    }
};

//--------------------------------------------------------------------------------------//

template <>
struct properties<caliper>
{
    static constexpr TIMEMORY_COMPONENT value = CALIPER;
    static bool&                        has_storage()
    {
        static thread_local bool _instance = false;
        return _instance;
    }
};

//--------------------------------------------------------------------------------------//

template <>
struct properties<cpu_clock>
{
    static constexpr TIMEMORY_COMPONENT value = CPU_CLOCK;
    static bool&                        has_storage()
    {
        static thread_local bool _instance = false;
        return _instance;
    }
};

//--------------------------------------------------------------------------------------//

template <>
struct properties<cpu_roofline_dp_flops>
{
    static constexpr TIMEMORY_COMPONENT value = CPU_ROOFLINE_DP_FLOPS;
    static bool&                        has_storage()
    {
        static thread_local bool _instance = false;
        return _instance;
    }
};

//--------------------------------------------------------------------------------------//

template <>
struct properties<cpu_roofline_flops>
{
    static constexpr TIMEMORY_COMPONENT value = CPU_ROOFLINE_FLOPS;
    static bool&                        has_storage()
    {
        static thread_local bool _instance = false;
        return _instance;
    }
};

//--------------------------------------------------------------------------------------//

template <>
struct properties<cpu_roofline_sp_flops>
{
    static constexpr TIMEMORY_COMPONENT value = CPU_ROOFLINE_SP_FLOPS;
    static bool&                        has_storage()
    {
        static thread_local bool _instance = false;
        return _instance;
    }
};

//--------------------------------------------------------------------------------------//

template <>
struct properties<cpu_util>
{
    static constexpr TIMEMORY_COMPONENT value = CPU_UTIL;
    static bool&                        has_storage()
    {
        static thread_local bool _instance = false;
        return _instance;
    }
};

//--------------------------------------------------------------------------------------//

template <>
struct properties<cuda_event>
{
    static constexpr TIMEMORY_COMPONENT value = CUDA_EVENT;
    static bool&                        has_storage()
    {
        static thread_local bool _instance = false;
        return _instance;
    }
};

//--------------------------------------------------------------------------------------//

template <>
struct properties<cupti_activity>
{
    static constexpr TIMEMORY_COMPONENT value = CUPTI_ACTIVITY;
    static bool&                        has_storage()
    {
        static thread_local bool _instance = false;
        return _instance;
    }
};

//--------------------------------------------------------------------------------------//

template <>
struct properties<cupti_counters>
{
    static constexpr TIMEMORY_COMPONENT value = CUPTI_COUNTERS;
    static bool&                        has_storage()
    {
        static thread_local bool _instance = false;
        return _instance;
    }
};

//--------------------------------------------------------------------------------------//

template <>
struct properties<data_rss>
{
    static constexpr TIMEMORY_COMPONENT value = DATA_RSS;
    static bool&                        has_storage()
    {
        static thread_local bool _instance = false;
        return _instance;
    }
};

//--------------------------------------------------------------------------------------//

template <>
struct properties<gperf_cpu_profiler>
{
    static constexpr TIMEMORY_COMPONENT value = GPERF_CPU_PROFILER;
    static bool&                        has_storage()
    {
        static thread_local bool _instance = false;
        return _instance;
    }
};

//--------------------------------------------------------------------------------------//

template <>
struct properties<gperf_heap_profiler>
{
    static constexpr TIMEMORY_COMPONENT value = GPERF_HEAP_PROFILER;
    static bool&                        has_storage()
    {
        static thread_local bool _instance = false;
        return _instance;
    }
};

//--------------------------------------------------------------------------------------//

template <>
struct properties<gpu_roofline_dp_flops>
{
    static constexpr TIMEMORY_COMPONENT value = GPU_ROOFLINE_DP_FLOPS;
    static bool&                        has_storage()
    {
        static thread_local bool _instance = false;
        return _instance;
    }
};

//--------------------------------------------------------------------------------------//

template <>
struct properties<gpu_roofline_flops>
{
    static constexpr TIMEMORY_COMPONENT value = GPU_ROOFLINE_FLOPS;
    static bool&                        has_storage()
    {
        static thread_local bool _instance = false;
        return _instance;
    }
};

//--------------------------------------------------------------------------------------//

template <>
struct properties<gpu_roofline_hp_flops>
{
    static constexpr TIMEMORY_COMPONENT value = GPU_ROOFLINE_HP_FLOPS;
    static bool&                        has_storage()
    {
        static thread_local bool _instance = false;
        return _instance;
    }
};

//--------------------------------------------------------------------------------------//

template <>
struct properties<gpu_roofline_sp_flops>
{
    static constexpr TIMEMORY_COMPONENT value = GPU_ROOFLINE_SP_FLOPS;
    static bool&                        has_storage()
    {
        static thread_local bool _instance = false;
        return _instance;
    }
};

//--------------------------------------------------------------------------------------//

template <>
struct properties<monotonic_clock>
{
    static constexpr TIMEMORY_COMPONENT value = MONOTONIC_CLOCK;
    static bool&                        has_storage()
    {
        static thread_local bool _instance = false;
        return _instance;
    }
};

//--------------------------------------------------------------------------------------//

template <>
struct properties<monotonic_raw_clock>
{
    static constexpr TIMEMORY_COMPONENT value = MONOTONIC_RAW_CLOCK;
    static bool&                        has_storage()
    {
        static thread_local bool _instance = false;
        return _instance;
    }
};

//--------------------------------------------------------------------------------------//

template <>
struct properties<num_io_in>
{
    static constexpr TIMEMORY_COMPONENT value = NUM_IO_IN;
    static bool&                        has_storage()
    {
        static thread_local bool _instance = false;
        return _instance;
    }
};

//--------------------------------------------------------------------------------------//

template <>
struct properties<num_io_out>
{
    static constexpr TIMEMORY_COMPONENT value = NUM_IO_OUT;
    static bool&                        has_storage()
    {
        static thread_local bool _instance = false;
        return _instance;
    }
};

//--------------------------------------------------------------------------------------//

template <>
struct properties<num_major_page_faults>
{
    static constexpr TIMEMORY_COMPONENT value = NUM_MAJOR_PAGE_FAULTS;
    static bool&                        has_storage()
    {
        static thread_local bool _instance = false;
        return _instance;
    }
};

//--------------------------------------------------------------------------------------//

template <>
struct properties<num_minor_page_faults>
{
    static constexpr TIMEMORY_COMPONENT value = NUM_MINOR_PAGE_FAULTS;
    static bool&                        has_storage()
    {
        static thread_local bool _instance = false;
        return _instance;
    }
};

//--------------------------------------------------------------------------------------//

template <>
struct properties<num_msg_recv>
{
    static constexpr TIMEMORY_COMPONENT value = NUM_MSG_RECV;
    static bool&                        has_storage()
    {
        static thread_local bool _instance = false;
        return _instance;
    }
};

//--------------------------------------------------------------------------------------//

template <>
struct properties<num_msg_sent>
{
    static constexpr TIMEMORY_COMPONENT value = NUM_MSG_SENT;
    static bool&                        has_storage()
    {
        static thread_local bool _instance = false;
        return _instance;
    }
};

//--------------------------------------------------------------------------------------//

template <>
struct properties<num_signals>
{
    static constexpr TIMEMORY_COMPONENT value = NUM_SIGNALS;
    static bool&                        has_storage()
    {
        static thread_local bool _instance = false;
        return _instance;
    }
};

//--------------------------------------------------------------------------------------//

template <>
struct properties<num_swap>
{
    static constexpr TIMEMORY_COMPONENT value = NUM_SWAP;
    static bool&                        has_storage()
    {
        static thread_local bool _instance = false;
        return _instance;
    }
};

//--------------------------------------------------------------------------------------//

template <>
struct properties<nvtx_marker>
{
    static constexpr TIMEMORY_COMPONENT value = NVTX_MARKER;
    static bool&                        has_storage()
    {
        static thread_local bool _instance = false;
        return _instance;
    }
};

//--------------------------------------------------------------------------------------//

template <>
struct properties<page_rss>
{
    static constexpr TIMEMORY_COMPONENT value = PAGE_RSS;
    static bool&                        has_storage()
    {
        static thread_local bool _instance = false;
        return _instance;
    }
};

//--------------------------------------------------------------------------------------//

template <>
struct properties<papi_array_t>
{
    static constexpr TIMEMORY_COMPONENT value = PAPI_ARRAY;
    static bool&                        has_storage()
    {
        static thread_local bool _instance = false;
        return _instance;
    }
};

//--------------------------------------------------------------------------------------//

template <>
struct properties<peak_rss>
{
    static constexpr TIMEMORY_COMPONENT value = PEAK_RSS;
    static bool&                        has_storage()
    {
        static thread_local bool _instance = false;
        return _instance;
    }
};

//--------------------------------------------------------------------------------------//

template <>
struct properties<priority_context_switch>
{
    static constexpr TIMEMORY_COMPONENT value = PRIORITY_CONTEXT_SWITCH;
    static bool&                        has_storage()
    {
        static thread_local bool _instance = false;
        return _instance;
    }
};

//--------------------------------------------------------------------------------------//

template <>
struct properties<process_cpu_clock>
{
    static constexpr TIMEMORY_COMPONENT value = PROCESS_CPU_CLOCK;
    static bool&                        has_storage()
    {
        static thread_local bool _instance = false;
        return _instance;
    }
};

//--------------------------------------------------------------------------------------//

template <>
struct properties<process_cpu_util>
{
    static constexpr TIMEMORY_COMPONENT value = PROCESS_CPU_UTIL;
    static bool&                        has_storage()
    {
        static thread_local bool _instance = false;
        return _instance;
    }
};

//--------------------------------------------------------------------------------------//

template <>
struct properties<read_bytes>
{
    static constexpr TIMEMORY_COMPONENT value = READ_BYTES;
    static bool&                        has_storage()
    {
        static thread_local bool _instance = false;
        return _instance;
    }
};

//--------------------------------------------------------------------------------------//

template <>
struct properties<real_clock>
{
    static constexpr TIMEMORY_COMPONENT value = WALL_CLOCK;
    static bool&                        has_storage()
    {
        static thread_local bool _instance = false;
        return _instance;
    }
};

//--------------------------------------------------------------------------------------//

template <>
struct properties<stack_rss>
{
    static constexpr TIMEMORY_COMPONENT value = STACK_RSS;
    static bool&                        has_storage()
    {
        static thread_local bool _instance = false;
        return _instance;
    }
};

//--------------------------------------------------------------------------------------//

template <>
struct properties<system_clock>
{
    static constexpr TIMEMORY_COMPONENT value = SYS_CLOCK;
    static bool&                        has_storage()
    {
        static thread_local bool _instance = false;
        return _instance;
    }
};

//--------------------------------------------------------------------------------------//

template <>
struct properties<thread_cpu_clock>
{
    static constexpr TIMEMORY_COMPONENT value = THREAD_CPU_CLOCK;
    static bool&                        has_storage()
    {
        static thread_local bool _instance = false;
        return _instance;
    }
};

//--------------------------------------------------------------------------------------//

template <>
struct properties<thread_cpu_util>
{
    static constexpr TIMEMORY_COMPONENT value = THREAD_CPU_UTIL;
    static bool&                        has_storage()
    {
        static thread_local bool _instance = false;
        return _instance;
    }
};

//--------------------------------------------------------------------------------------//

template <>
struct properties<trip_count>
{
    static constexpr TIMEMORY_COMPONENT value = TRIP_COUNT;
    static bool&                        has_storage()
    {
        static thread_local bool _instance = false;
        return _instance;
    }
};

//--------------------------------------------------------------------------------------//

template <>
struct properties<user_clock>
{
    static constexpr TIMEMORY_COMPONENT value = USER_CLOCK;
    static bool&                        has_storage()
    {
        static thread_local bool _instance = false;
        return _instance;
    }
};

//--------------------------------------------------------------------------------------//

template <>
struct properties<voluntary_context_switch>
{
    static constexpr TIMEMORY_COMPONENT value = VOLUNTARY_CONTEXT_SWITCH;
    static bool&                        has_storage()
    {
        static thread_local bool _instance = false;
        return _instance;
    }
};

//--------------------------------------------------------------------------------------//

template <>
struct properties<written_bytes>
{
    static constexpr TIMEMORY_COMPONENT value = WRITTEN_BYTES;
    static bool&                        has_storage()
    {
        static thread_local bool _instance = false;
        return _instance;
    }
};

//--------------------------------------------------------------------------------------//

}  // namespace component
}  // namespace tim
