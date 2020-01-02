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

/** \file runtime/initialize.hpp
 * \headerfile runtime/initialize.hpp "timemory/runtime/initialize.hpp"
 * Provides implementation for initialize, enumerate_components which regularly
 * change as more features are added
 *
 */

#pragma once

#include "timemory/components/types.hpp"
#include "timemory/enum.h"
#include "timemory/runtime/enumerate.hpp"
#include "timemory/runtime/types.hpp"

#include <unordered_map>

namespace tim
{
//--------------------------------------------------------------------------------------//
//
//                  specializations for std::initializer_list
//
//--------------------------------------------------------------------------------------//

template <template <typename...> class _CompList, typename... _CompTypes,
          typename _EnumT = int>
inline void
initialize(_CompList<_CompTypes...>& obj, std::initializer_list<_EnumT> components)
{
    initialize(obj, std::vector<_EnumT>(components));
}

//--------------------------------------------------------------------------------------//

template <template <typename...> class _CompList, typename... _CompTypes>
inline void
initialize(_CompList<_CompTypes...>& obj, std::initializer_list<std::string> components)
{
    initialize(obj, enumerate_components(components));
}

//--------------------------------------------------------------------------------------//

template <template <typename...> class _CompList, typename... _CompTypes,
          typename... _ExtraArgs, template <typename, typename...> class _Container>
inline void
initialize(_CompList<_CompTypes...>&                     obj,
           const _Container<std::string, _ExtraArgs...>& components)
{
    initialize(obj, enumerate_components(components));
}

//--------------------------------------------------------------------------------------//
//
/// this is for initializing with a string
//
template <template <typename...> class _CompList, typename... _CompTypes>
inline void
initialize(_CompList<_CompTypes...>& obj, const std::string& components)
{
    initialize(obj, enumerate_components(tim::delimit(components)));
}

//--------------------------------------------------------------------------------------//

template <template <typename...> class _CompList, typename... _CompTypes>
inline void
initialize(const TIMEMORY_COMPONENT& comp, _CompList<_CompTypes...>& obj)
{
    using namespace component;
    switch(comp)
    {
        case CALIPER: obj.template init<caliper>(); break;
        case CPU_CLOCK: obj.template init<cpu_clock>(); break;
        case CPU_ROOFLINE_DP_FLOPS: obj.template init<cpu_roofline_dp_flops>(); break;
        case CPU_ROOFLINE_FLOPS: obj.template init<cpu_roofline_flops>(); break;
        case CPU_ROOFLINE_SP_FLOPS: obj.template init<cpu_roofline_sp_flops>(); break;
        case CPU_UTIL: obj.template init<cpu_util>(); break;
        case CUDA_EVENT: obj.template init<cuda_event>(); break;
        case CUDA_PROFILER: obj.template init<cuda_profiler>(); break;
        case CUPTI_ACTIVITY: obj.template init<cupti_activity>(); break;
        case CUPTI_COUNTERS: obj.template init<cupti_counters>(); break;
        case DATA_RSS: obj.template init<data_rss>(); break;
        case GPERF_CPU_PROFILER: obj.template init<gperf_cpu_profiler>(); break;
        case GPERF_HEAP_PROFILER: obj.template init<gperf_heap_profiler>(); break;
        case GPU_ROOFLINE_DP_FLOPS: obj.template init<gpu_roofline_dp_flops>(); break;
        case GPU_ROOFLINE_FLOPS: obj.template init<gpu_roofline_flops>(); break;
        case GPU_ROOFLINE_HP_FLOPS: obj.template init<gpu_roofline_hp_flops>(); break;
        case GPU_ROOFLINE_SP_FLOPS: obj.template init<gpu_roofline_sp_flops>(); break;
        case LIKWID_NVMON: obj.template init<likwid_nvmon>(); break;
        case LIKWID_PERFMON: obj.template init<likwid_perfmon>(); break;
        case MONOTONIC_CLOCK: obj.template init<monotonic_clock>(); break;
        case MONOTONIC_RAW_CLOCK: obj.template init<monotonic_raw_clock>(); break;
        case NUM_IO_IN: obj.template init<num_io_in>(); break;
        case NUM_IO_OUT: obj.template init<num_io_out>(); break;
        case NUM_MAJOR_PAGE_FAULTS: obj.template init<num_major_page_faults>(); break;
        case NUM_MINOR_PAGE_FAULTS: obj.template init<num_minor_page_faults>(); break;
        case NUM_MSG_RECV: obj.template init<num_msg_recv>(); break;
        case NUM_MSG_SENT: obj.template init<num_msg_sent>(); break;
        case NUM_SIGNALS: obj.template init<num_signals>(); break;
        case NUM_SWAP: obj.template init<num_swap>(); break;
        case NVTX_MARKER: obj.template init<nvtx_marker>(); break;
        case PAGE_RSS: obj.template init<page_rss>(); break;
        case PAPI_ARRAY: obj.template init<papi_array_t>(); break;
        case PEAK_RSS: obj.template init<peak_rss>(); break;
        case PRIORITY_CONTEXT_SWITCH: obj.template init<priority_context_switch>(); break;
        case PROCESS_CPU_CLOCK: obj.template init<process_cpu_clock>(); break;
        case PROCESS_CPU_UTIL: obj.template init<process_cpu_util>(); break;
        case READ_BYTES: obj.template init<read_bytes>(); break;
        case STACK_RSS: obj.template init<stack_rss>(); break;
        case SYS_CLOCK: obj.template init<system_clock>(); break;
        case TAU_MARKER: obj.template init<tau_marker>(); break;
        case THREAD_CPU_CLOCK: obj.template init<thread_cpu_clock>(); break;
        case THREAD_CPU_UTIL: obj.template init<thread_cpu_util>(); break;
        case TRIP_COUNT: obj.template init<trip_count>(); break;
        case USER_CLOCK: obj.template init<user_clock>(); break;
        case USER_LIST_BUNDLE: obj.template init<user_list_bundle>(); break;
        case USER_TUPLE_BUNDLE: obj.template init<user_tuple_bundle>(); break;
        case VIRTUAL_MEMORY: obj.template init<virtual_memory>(); break;
        case VOLUNTARY_CONTEXT_SWITCH:
            obj.template init<voluntary_context_switch>();
            break;
        case VTUNE_EVENT: obj.template init<vtune_event>(); break;
        case VTUNE_FRAME: obj.template init<vtune_frame>(); break;
        case WALL_CLOCK: obj.template init<wall_clock>(); break;
        case WRITTEN_BYTES: obj.template init<written_bytes>(); break;
        case TIMEMORY_COMPONENTS_END:
        default: break;
    }
}

//--------------------------------------------------------------------------------------//

template <template <typename...> class _CompList, typename... _CompTypes,
          template <typename, typename...> class _Container, typename _Intp,
          typename... _ExtraArgs,
          typename std::enable_if<(std::is_integral<_Intp>::value ||
                                   std::is_same<_Intp, TIMEMORY_COMPONENT>::value),
                                  int>::type>
void
initialize(_CompList<_CompTypes...>&               obj,
           const _Container<_Intp, _ExtraArgs...>& components)
{
    for(auto itr : components)
        initialize(static_cast<TIMEMORY_COMPONENT>(itr), obj);
}

//--------------------------------------------------------------------------------------//

template <template <typename...> class _CompList, typename... _CompTypes,
          template <typename, typename...> class _Container, typename... _ExtraArgs>
void
initialize(_CompList<_CompTypes...>&                     obj,
           const _Container<const char*, _ExtraArgs...>& components)
{
    std::vector<std::string> _components;
    _components.reserve(components.size());
    for(auto itr : components)
        _components.emplace_back(std::string(itr));
    initialize(obj, _components);
}

//--------------------------------------------------------------------------------------//

template <template <typename...> class _CompList, typename... _CompTypes>
void
initialize(_CompList<_CompTypes...>& obj, const int ncomponents, const int* components)
{
    for(int i = 0; i < ncomponents; ++i)
        initialize(static_cast<TIMEMORY_COMPONENT>(components[i]), obj);
}

//======================================================================================//

}  // namespace tim

//======================================================================================//
