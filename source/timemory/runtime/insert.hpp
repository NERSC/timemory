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

/** \file runtime/insert.hpp
 * \headerfile runtime/insert.hpp "timemory/runtime/insert.hpp"
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
//======================================================================================//

template <size_t _Idx, typename _Type, template <size_t, typename> class _Bundle,
          typename _EnumT = int>
inline void
insert(_Bundle<_Idx, _Type>& obj, std::initializer_list<_EnumT> components)
{
    insert(obj, std::vector<_EnumT>(components));
}

//--------------------------------------------------------------------------------------//

template <size_t _Idx, typename _Type, template <size_t, typename> class _Bundle>
inline void
insert(_Bundle<_Idx, _Type>& obj, const std::initializer_list<std::string>& components)
{
    insert(obj, enumerate_components(components));
}

//--------------------------------------------------------------------------------------//
//
/// this is for initializing with a container of string
//
template <size_t _Idx, typename _Type, template <size_t, typename> class _Bundle,
          typename... _ExtraArgs, template <typename, typename...> class _Container>
inline void
insert(_Bundle<_Idx, _Type>&                         obj,
       const _Container<std::string, _ExtraArgs...>& components)
{
    insert(obj, enumerate_components(components));
}

//--------------------------------------------------------------------------------------//
//
/// this is for initializing with a string
//
template <size_t _Idx, typename _Type, template <size_t, typename> class _Bundle>
inline void
insert(_Bundle<_Idx, _Type>& obj, const std::string& components)
{
    insert(obj, enumerate_components(tim::delimit(components)));
}

//======================================================================================//

template <size_t _Idx, typename _Type, template <size_t, typename> class _Bundle>
inline void
insert(const TIMEMORY_COMPONENT& comp, _Bundle<_Idx, _Type>& obj)
{
    using namespace component;
    switch(comp)
    {
        case CALIPER: obj.template insert<caliper>(); break;
        case CPU_CLOCK: obj.template insert<cpu_clock>(); break;
        case CPU_ROOFLINE_DP_FLOPS: obj.template insert<cpu_roofline_dp_flops>(); break;
        case CPU_ROOFLINE_FLOPS: obj.template insert<cpu_roofline_flops>(); break;
        case CPU_ROOFLINE_SP_FLOPS: obj.template insert<cpu_roofline_sp_flops>(); break;
        case CPU_UTIL: obj.template insert<cpu_util>(); break;
        case CUDA_EVENT: obj.template insert<cuda_event>(); break;
        case CUDA_PROFILER: obj.template insert<cuda_profiler>(); break;
        case CUPTI_ACTIVITY: obj.template insert<cupti_activity>(); break;
        case CUPTI_COUNTERS: obj.template insert<cupti_counters>(); break;
        case DATA_RSS: obj.template insert<data_rss>(); break;
        case GPERF_CPU_PROFILER: obj.template insert<gperf_cpu_profiler>(); break;
        case GPERF_HEAP_PROFILER: obj.template insert<gperf_heap_profiler>(); break;
        case GPU_ROOFLINE_DP_FLOPS: obj.template insert<gpu_roofline_dp_flops>(); break;
        case GPU_ROOFLINE_FLOPS: obj.template insert<gpu_roofline_flops>(); break;
        case GPU_ROOFLINE_HP_FLOPS: obj.template insert<gpu_roofline_hp_flops>(); break;
        case GPU_ROOFLINE_SP_FLOPS: obj.template insert<gpu_roofline_sp_flops>(); break;
        case LIKWID_NVMON: obj.template insert<likwid_nvmon>(); break;
        case LIKWID_PERFMON: obj.template insert<likwid_perfmon>(); break;
        case MONOTONIC_CLOCK: obj.template insert<monotonic_clock>(); break;
        case MONOTONIC_RAW_CLOCK: obj.template insert<monotonic_raw_clock>(); break;
        case NUM_IO_IN: obj.template insert<num_io_in>(); break;
        case NUM_IO_OUT: obj.template insert<num_io_out>(); break;
        case NUM_MAJOR_PAGE_FAULTS: obj.template insert<num_major_page_faults>(); break;
        case NUM_MINOR_PAGE_FAULTS: obj.template insert<num_minor_page_faults>(); break;
        case NUM_MSG_RECV: obj.template insert<num_msg_recv>(); break;
        case NUM_MSG_SENT: obj.template insert<num_msg_sent>(); break;
        case NUM_SIGNALS: obj.template insert<num_signals>(); break;
        case NUM_SWAP: obj.template insert<num_swap>(); break;
        case NVTX_MARKER: obj.template insert<nvtx_marker>(); break;
        case PAGE_RSS: obj.template insert<page_rss>(); break;
        case PAPI_ARRAY: obj.template insert<papi_array_t>(); break;
        case PEAK_RSS: obj.template insert<peak_rss>(); break;
        case PRIORITY_CONTEXT_SWITCH:
            obj.template insert<priority_context_switch>();
            break;
        case PROCESS_CPU_CLOCK: obj.template insert<process_cpu_clock>(); break;
        case PROCESS_CPU_UTIL: obj.template insert<process_cpu_util>(); break;
        case READ_BYTES: obj.template insert<read_bytes>(); break;
        case STACK_RSS: obj.template insert<stack_rss>(); break;
        case SYS_CLOCK: obj.template insert<system_clock>(); break;
        case TAU_MARKER: obj.template insert<tau_marker>(); break;
        case THREAD_CPU_CLOCK: obj.template insert<thread_cpu_clock>(); break;
        case THREAD_CPU_UTIL: obj.template insert<thread_cpu_util>(); break;
        case TRIP_COUNT: obj.template insert<trip_count>(); break;
        case USER_CLOCK: obj.template insert<user_clock>(); break;
        case USER_LIST_BUNDLE: obj.template insert<user_list_bundle>(); break;
        case USER_TUPLE_BUNDLE: obj.template insert<user_tuple_bundle>(); break;
        case VIRTUAL_MEMORY: obj.template insert<virtual_memory>(); break;
        case VOLUNTARY_CONTEXT_SWITCH:
            obj.template insert<voluntary_context_switch>();
            break;
        case VTUNE_EVENT: obj.template insert<vtune_event>(); break;
        case VTUNE_FRAME: obj.template insert<vtune_frame>(); break;
        case WALL_CLOCK: obj.template insert<wall_clock>(); break;
        case WRITTEN_BYTES: obj.template insert<written_bytes>(); break;
        case TIMEMORY_COMPONENTS_END:
        default: break;
    }
}

//--------------------------------------------------------------------------------------//

template <size_t _Idx, typename _Type, template <size_t, typename> class _Bundle,
          template <typename, typename...> class _Container, typename _Intp,
          typename... _ExtraArgs,
          typename std::enable_if<(std::is_integral<_Intp>::value ||
                                   std::is_same<_Intp, TIMEMORY_COMPONENT>::value),
                                  int>::type>
void
insert(_Bundle<_Idx, _Type>& obj, const _Container<_Intp, _ExtraArgs...>& components)
{
    for(auto itr : components)
        insert(static_cast<TIMEMORY_COMPONENT>(itr), obj);
}

//--------------------------------------------------------------------------------------//

template <size_t _Idx, typename _Type, template <size_t, typename> class _Bundle,
          template <typename, typename...> class _Container, typename... _ExtraArgs>
void
insert(_Bundle<_Idx, _Type>&                         obj,
       const _Container<const char*, _ExtraArgs...>& components)
{
    std::vector<std::string> _components;
    _components.reserve(components.size());
    for(auto itr : components)
        _components.emplace_back(std::string(itr));
    insert(obj, _components);
}

//--------------------------------------------------------------------------------------//

template <size_t _Idx, typename _Type, template <size_t, typename> class _Bundle>
void
insert(_Bundle<_Idx, _Type>& obj, const int ncomponents, const int* components)
{
    for(int i = 0; i < ncomponents; ++i)
        insert(static_cast<TIMEMORY_COMPONENT>(components[i]), obj);
}

//======================================================================================//

}  // namespace tim

//======================================================================================//
