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

template <typename Bundle, typename EnumT = int>
inline void
configure(std::initializer_list<EnumT> components, bool flat = false)
{
    if(settings::debug())
        PRINT_HERE("%s", demangle<Bundle>().c_str());
    configure<Bundle>(std::vector<EnumT>(components), flat);
}

//--------------------------------------------------------------------------------------//

template <typename Bundle>
inline void
configure(const std::initializer_list<std::string>& components, bool flat = false)
{
    if(settings::debug())
        PRINT_HERE("%s", demangle<Bundle>().c_str());
    configure<Bundle>(enumerate_components(components), flat);
}

//--------------------------------------------------------------------------------------//
//
/// this is for initializing with a container of string
//
template <typename Bundle, typename... ExtraArgs,
          template <typename, typename...> class Container>
inline void
configure(const Container<std::string, ExtraArgs...>& components, bool flat = false)
{
    if(settings::debug())
        PRINT_HERE("%s", demangle<Bundle>().c_str());
    configure<Bundle>(enumerate_components(components), flat);
}

//--------------------------------------------------------------------------------------//
//
/// this is for initializing with a string
//
template <typename Bundle>
inline void
configure(const std::string& components, bool flat = false)
{
    if(settings::debug())
        PRINT_HERE("%s", demangle<Bundle>().c_str());
    configure<Bundle>(enumerate_components(tim::delimit(components)), flat);
}

//======================================================================================//

template <typename Bundle>
inline void
configure(const TIMEMORY_COMPONENT& comp, bool flat = false)
{
    if(settings::debug())
        PRINT_HERE("%s", demangle<Bundle>().c_str());
    using namespace component;
    switch(comp)
    {
        case CALIPER: Bundle::template configure<caliper>(flat); break;
        case CPU_CLOCK: Bundle::template configure<cpu_clock>(flat); break;
        case CPU_ROOFLINE_DP_FLOPS:
            Bundle::template configure<cpu_roofline_dp_flops>(flat);
            break;
        case CPU_ROOFLINE_FLOPS:
            Bundle::template configure<cpu_roofline_flops>(flat);
            break;
        case CPU_ROOFLINE_SP_FLOPS:
            Bundle::template configure<cpu_roofline_sp_flops>(flat);
            break;
        case CPU_UTIL: Bundle::template configure<cpu_util>(flat); break;
        case CUDA_EVENT: Bundle::template configure<cuda_event>(flat); break;
        case CUDA_PROFILER: Bundle::template configure<cuda_profiler>(flat); break;
        case CUPTI_ACTIVITY: Bundle::template configure<cupti_activity>(flat); break;
        case CUPTI_COUNTERS: Bundle::template configure<cupti_counters>(flat); break;
        case DATA_RSS: Bundle::template configure<data_rss>(flat); break;
        case GPERF_CPU_PROFILER:
            Bundle::template configure<gperf_cpu_profiler>(flat);
            break;
        case GPERF_HEAP_PROFILER:
            Bundle::template configure<gperf_heap_profiler>(flat);
            break;
        case GPU_ROOFLINE_DP_FLOPS:
            Bundle::template configure<gpu_roofline_dp_flops>(flat);
            break;
        case GPU_ROOFLINE_FLOPS:
            Bundle::template configure<gpu_roofline_flops>(flat);
            break;
        case GPU_ROOFLINE_HP_FLOPS:
            Bundle::template configure<gpu_roofline_hp_flops>(flat);
            break;
        case GPU_ROOFLINE_SP_FLOPS:
            Bundle::template configure<gpu_roofline_sp_flops>(flat);
            break;
        case LIKWID_NVMARKER: Bundle::template configure<likwid_nvmarker>(flat); break;
        case LIKWID_MARKER: Bundle::template configure<likwid_marker>(flat); break;
        case MONOTONIC_CLOCK: Bundle::template configure<monotonic_clock>(flat); break;
        case MONOTONIC_RAW_CLOCK:
            Bundle::template configure<monotonic_raw_clock>(flat);
            break;
        case NUM_IO_IN: Bundle::template configure<num_io_in>(flat); break;
        case NUM_IO_OUT: Bundle::template configure<num_io_out>(flat); break;
        case NUM_MAJOR_PAGE_FAULTS:
            Bundle::template configure<num_major_page_faults>(flat);
            break;
        case NUM_MINOR_PAGE_FAULTS:
            Bundle::template configure<num_minor_page_faults>(flat);
            break;
        case NUM_MSG_RECV: Bundle::template configure<num_msg_recv>(flat); break;
        case NUM_MSG_SENT: Bundle::template configure<num_msg_sent>(flat); break;
        case NUM_SIGNALS: Bundle::template configure<num_signals>(flat); break;
        case NUM_SWAP: Bundle::template configure<num_swap>(flat); break;
        case NVTX_MARKER: Bundle::template configure<nvtx_marker>(flat); break;
        case PAGE_RSS: Bundle::template configure<page_rss>(flat); break;
        case PAPI_ARRAY: Bundle::template configure<papi_array_t>(flat); break;
        case PEAK_RSS: Bundle::template configure<peak_rss>(flat); break;
        case PRIORITY_CONTEXT_SWITCH:
            Bundle::template configure<priority_context_switch>(flat);
            break;
        case PROCESS_CPU_CLOCK:
            Bundle::template configure<process_cpu_clock>(flat);
            break;
        case PROCESS_CPU_UTIL: Bundle::template configure<process_cpu_util>(flat); break;
        case READ_BYTES: Bundle::template configure<read_bytes>(flat); break;
        case STACK_RSS: Bundle::template configure<stack_rss>(flat); break;
        case SYS_CLOCK: Bundle::template configure<system_clock>(flat); break;
        case TAU_MARKER: Bundle::template configure<tau_marker>(flat); break;
        case THREAD_CPU_CLOCK: Bundle::template configure<thread_cpu_clock>(flat); break;
        case THREAD_CPU_UTIL: Bundle::template configure<thread_cpu_util>(flat); break;
        case TRIP_COUNT: Bundle::template configure<trip_count>(flat); break;
        case USER_CLOCK: Bundle::template configure<user_clock>(flat); break;
        case USER_LIST_BUNDLE: Bundle::template configure<user_list_bundle>(flat); break;
        case USER_TUPLE_BUNDLE:
            Bundle::template configure<user_tuple_bundle>(flat);
            break;
        case VIRTUAL_MEMORY: Bundle::template configure<virtual_memory>(flat); break;
        case VOLUNTARY_CONTEXT_SWITCH:
            Bundle::template configure<voluntary_context_switch>(flat);
            break;
        case VTUNE_EVENT: Bundle::template configure<vtune_event>(flat); break;
        case VTUNE_FRAME: Bundle::template configure<vtune_frame>(flat); break;
        case WALL_CLOCK: Bundle::template configure<wall_clock>(flat); break;
        case WRITTEN_BYTES: Bundle::template configure<written_bytes>(flat); break;
        case TIMEMORY_COMPONENTS_END:
        default: break;
    }
}

//--------------------------------------------------------------------------------------//

template <typename Bundle, template <typename, typename...> class Container,
          typename _Intp, typename... ExtraArgs,
          typename std::enable_if<(std::is_integral<_Intp>::value ||
                                   std::is_same<_Intp, TIMEMORY_COMPONENT>::value),
                                  int>::type>
void
configure(const Container<_Intp, ExtraArgs...>& components, bool flat)
{
    if(settings::debug())
        PRINT_HERE("%s", demangle<Bundle>().c_str());

    for(auto itr : components)
        configure<Bundle>(static_cast<TIMEMORY_COMPONENT>(itr), flat);
}

//--------------------------------------------------------------------------------------//

template <typename Bundle, template <typename, typename...> class Container,
          typename... ExtraArgs>
void
configure(const Container<const char*, ExtraArgs...>& components, bool flat = false)
{
    if(settings::debug())
        PRINT_HERE("%s", demangle<Bundle>().c_str());
    std::vector<std::string> _components;
    _components.reserve(components.size());
    for(auto itr : components)
        _components.emplace_back(std::string(itr), flat);
    configure<Bundle>(_components);
}

//--------------------------------------------------------------------------------------//

template <typename Bundle>
void
configure(const int ncomponents, const int* components, bool flat = false)
{
    if(settings::debug())
        PRINT_HERE("%s", demangle<Bundle>().c_str());
    for(int i = 0; i < ncomponents; ++i)
        configure<Bundle>(static_cast<TIMEMORY_COMPONENT>(components[i]), flat);
}

//======================================================================================//

}  // namespace tim

//======================================================================================//
