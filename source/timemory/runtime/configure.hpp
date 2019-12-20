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

/** \file runtime/configure.hpp
 * \headerfile runtime/configure.hpp "timemory/runtime/configure.hpp"
 * Provides implementation for initialize, enumerate_components which regularly
 * change as more features are added
 *
 */

#pragma once

#include "timemory/components/types.hpp"
#include "timemory/enum.h"
#include "timemory/runtime/enumerate.hpp"
#include "timemory/runtime/types.hpp"
#include "timemory/settings.hpp"

#include <unordered_map>

namespace tim
{
//======================================================================================//

template <typename _Bundle, typename _EnumT = int>
inline void
configure(std::initializer_list<_EnumT> components)
{
    if(settings::debug())
        PRINT_HERE("%s", demangle<_Bundle>().c_str());
    configure<_Bundle>(std::vector<_EnumT>(components));
}

//--------------------------------------------------------------------------------------//

template <typename _Bundle>
inline void
configure(const std::initializer_list<std::string>& components)
{
    if(settings::debug())
        PRINT_HERE("%s", demangle<_Bundle>().c_str());
    configure<_Bundle>(enumerate_components(components));
}

//--------------------------------------------------------------------------------------//
//
/// this is for initializing with a container of string
//
template <typename _Bundle, typename... _ExtraArgs,
          template <typename, typename...> class _Container>
inline void
configure(const _Container<std::string, _ExtraArgs...>& components)
{
    if(settings::debug())
        PRINT_HERE("%s", demangle<_Bundle>().c_str());
    configure<_Bundle>(enumerate_components(components));
}

//--------------------------------------------------------------------------------------//
//
/// this is for initializing with a string
//
template <typename _Bundle>
inline void
configure(const std::string& components)
{
    if(settings::debug())
        PRINT_HERE("%s", demangle<_Bundle>().c_str());
    configure<_Bundle>(enumerate_components(tim::delimit(components)));
}

//======================================================================================//

template <typename _Bundle>
inline void
configure(const TIMEMORY_COMPONENT& comp)
{
    if(settings::debug())
        PRINT_HERE("%s", demangle<_Bundle>().c_str());
    using namespace component;
    switch(comp)
    {
        case CALIPER: _Bundle::template configure<caliper>(); break;
        case CPU_CLOCK: _Bundle::template configure<cpu_clock>(); break;
        case CPU_ROOFLINE_DP_FLOPS:
            _Bundle::template configure<cpu_roofline_dp_flops>();
            break;
        case CPU_ROOFLINE_FLOPS: _Bundle::template configure<cpu_roofline_flops>(); break;
        case CPU_ROOFLINE_SP_FLOPS:
            _Bundle::template configure<cpu_roofline_sp_flops>();
            break;
        case CPU_UTIL: _Bundle::template configure<cpu_util>(); break;
        case CUDA_EVENT: _Bundle::template configure<cuda_event>(); break;
        case CUDA_PROFILER: _Bundle::template configure<cuda_profiler>(); break;
        case CUPTI_ACTIVITY: _Bundle::template configure<cupti_activity>(); break;
        case CUPTI_COUNTERS: _Bundle::template configure<cupti_counters>(); break;
        case DATA_RSS: _Bundle::template configure<data_rss>(); break;
        case GPERF_CPU_PROFILER: _Bundle::template configure<gperf_cpu_profiler>(); break;
        case GPERF_HEAP_PROFILER:
            _Bundle::template configure<gperf_heap_profiler>();
            break;
        case GPU_ROOFLINE_DP_FLOPS:
            _Bundle::template configure<gpu_roofline_dp_flops>();
            break;
        case GPU_ROOFLINE_FLOPS: _Bundle::template configure<gpu_roofline_flops>(); break;
        case GPU_ROOFLINE_HP_FLOPS:
            _Bundle::template configure<gpu_roofline_hp_flops>();
            break;
        case GPU_ROOFLINE_SP_FLOPS:
            _Bundle::template configure<gpu_roofline_sp_flops>();
            break;
        case LIKWID_NVMON: _Bundle::template configure<likwid_nvmon>(); break;
        case LIKWID_PERFMON: _Bundle::template configure<likwid_perfmon>(); break;
        case MONOTONIC_CLOCK: _Bundle::template configure<monotonic_clock>(); break;
        case MONOTONIC_RAW_CLOCK:
            _Bundle::template configure<monotonic_raw_clock>();
            break;
        case NUM_IO_IN: _Bundle::template configure<num_io_in>(); break;
        case NUM_IO_OUT: _Bundle::template configure<num_io_out>(); break;
        case NUM_MAJOR_PAGE_FAULTS:
            _Bundle::template configure<num_major_page_faults>();
            break;
        case NUM_MINOR_PAGE_FAULTS:
            _Bundle::template configure<num_minor_page_faults>();
            break;
        case NUM_MSG_RECV: _Bundle::template configure<num_msg_recv>(); break;
        case NUM_MSG_SENT: _Bundle::template configure<num_msg_sent>(); break;
        case NUM_SIGNALS: _Bundle::template configure<num_signals>(); break;
        case NUM_SWAP: _Bundle::template configure<num_swap>(); break;
        case NVTX_MARKER: _Bundle::template configure<nvtx_marker>(); break;
        case PAGE_RSS: _Bundle::template configure<page_rss>(); break;
        case PAPI_ARRAY: _Bundle::template configure<papi_array_t>(); break;
        case PEAK_RSS: _Bundle::template configure<peak_rss>(); break;
        case PRIORITY_CONTEXT_SWITCH:
            _Bundle::template configure<priority_context_switch>();
            break;
        case PROCESS_CPU_CLOCK: _Bundle::template configure<process_cpu_clock>(); break;
        case PROCESS_CPU_UTIL: _Bundle::template configure<process_cpu_util>(); break;
        case READ_BYTES: _Bundle::template configure<read_bytes>(); break;
        case STACK_RSS: _Bundle::template configure<stack_rss>(); break;
        case SYS_CLOCK: _Bundle::template configure<system_clock>(); break;
        case TAU_MARKER: _Bundle::template configure<tau_marker>(); break;
        case THREAD_CPU_CLOCK: _Bundle::template configure<thread_cpu_clock>(); break;
        case THREAD_CPU_UTIL: _Bundle::template configure<thread_cpu_util>(); break;
        case TRIP_COUNT: _Bundle::template configure<trip_count>(); break;
        case USER_CLOCK: _Bundle::template configure<user_clock>(); break;
        case USER_LIST_BUNDLE: _Bundle::template configure<user_list_bundle>(); break;
        case USER_TUPLE_BUNDLE: _Bundle::template configure<user_tuple_bundle>(); break;
        case VIRTUAL_MEMORY: _Bundle::template configure<virtual_memory>(); break;
        case VOLUNTARY_CONTEXT_SWITCH:
            _Bundle::template configure<voluntary_context_switch>();
            break;
        case VTUNE_EVENT: _Bundle::template configure<vtune_event>(); break;
        case VTUNE_FRAME: _Bundle::template configure<vtune_frame>(); break;
        case WALL_CLOCK: _Bundle::template configure<wall_clock>(); break;
        case WRITTEN_BYTES: _Bundle::template configure<written_bytes>(); break;
        case TIMEMORY_COMPONENTS_END:
        default: break;
    }
}

//--------------------------------------------------------------------------------------//

template <typename _Bundle, template <typename, typename...> class _Container,
          typename _Intp, typename... _ExtraArgs>
void
configure(const _Container<_Intp, _ExtraArgs...>& components)
{
    if(settings::debug())
        PRINT_HERE("%s", demangle<_Bundle>().c_str());
    for(auto itr : components)
        configure<_Bundle>(static_cast<TIMEMORY_COMPONENT>(itr));
}

//--------------------------------------------------------------------------------------//

template <typename _Bundle, template <typename, typename...> class _Container,
          typename... _ExtraArgs>
void
configure(const _Container<const char*, _ExtraArgs...>& components)
{
    if(settings::debug())
        PRINT_HERE("%s", demangle<_Bundle>().c_str());
    std::vector<std::string> _components;
    _components.reserve(components.size());
    for(auto itr : components)
        _components.emplace_back(std::string(itr));
    configure<_Bundle>(_components);
}

//--------------------------------------------------------------------------------------//

template <typename _Bundle>
void
configure(const int ncomponents, const int* components)
{
    if(settings::debug())
        PRINT_HERE("%s", demangle<_Bundle>().c_str());
    for(int i = 0; i < ncomponents; ++i)
        configure<_Bundle>(static_cast<TIMEMORY_COMPONENT>(components[i]));
}

//======================================================================================//

}  // namespace tim

//======================================================================================//
