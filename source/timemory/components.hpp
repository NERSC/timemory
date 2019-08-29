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

/** \file components.hpp
 * \headerfile components.hpp "timemory/components.hpp"
 * These are core tools provided by TiMemory. These tools can be used individually
 * or bundled together in a component_tuple (C++) or component_list (C, Python)
 *
 */

#pragma once

// general components
#include "timemory/components/caliper.hpp"
#include "timemory/components/cuda_event.hpp"
#include "timemory/components/cupti_activity.hpp"
#include "timemory/components/general.hpp"
#include "timemory/components/rusage.hpp"
#include "timemory/components/timing.hpp"

// hardware counter components
#include "timemory/components/cupti_counters.hpp"
#include "timemory/components/papi_array.hpp"
#include "timemory/components/papi_tuple.hpp"

// advanced components
#include "timemory/components/cpu_roofline.hpp"
#include "timemory/components/gpu_roofline.hpp"

// device backend
#include "timemory/backends/device.hpp"

//======================================================================================//
//
//      helpers for generating components
//
//======================================================================================//

/// this is the default maximum number of CPU counters that can be used in papi_array
#if !defined(TIMEMORY_CPU_COUNTERS)
#    define TIMEMORY_CPU_COUNTERS 32
#endif

/// this is the default maximum number of CPU counters that can be used in cupti_array
#if !defined(TIMEMORY_GPU_COUNTERS)
#    define TIMEMORY_GPU_COUNTERS 32
#endif

#include "timemory/mpl/apply.hpp"

extern "C"
{
#include "timemory/ctimemory.h"
}

#include <algorithm>
#include <cctype>
#include <string>
#include <vector>

namespace tim
{
//--------------------------------------------------------------------------------------//
///  provide a default size for papi_array
//
namespace component
{
using papi_array_t = papi_array<TIMEMORY_CPU_COUNTERS>;
}  // namespace component

//--------------------------------------------------------------------------------------//
//
///  description:
///      use this function to initialize a auto_list or component_list from a list
///      of enumerations
//
///  usage:
///      using namespace tim::component;
///      using optional_t = tim::auto_list<real_clock, cpu_clock, cpu_util, cuda_event>;
//
///      auto obj = new optional_t(__FUNCTION__, __LINE__);
///      tim::initialize(*obj, { CPU_CLOCK, CPU_UTIL });
//
///  typename... _ExtraArgs
///      required because of extra "hidden" template parameters in STL containers
//
template <typename... _CompTypes, template <typename...> class _CompList, typename _Intp,
          typename... _ExtraArgs, template <typename, typename...> class _Container>
void
initialize(_CompList<_CompTypes...>&               obj,
           const _Container<_Intp, _ExtraArgs...>& components)
{
    using namespace component;
    for(auto itr : components)
    {
        TIMEMORY_COMPONENT comp = static_cast<TIMEMORY_COMPONENT>(itr);
        switch(comp)
        {
            case CALIPER: obj.template init<caliper>(); break;
            case CPU_CLOCK: obj.template init<cpu_clock>(); break;
            case CPU_ROOFLINE_DP_FLOPS: obj.template init<cpu_roofline_dp_flops>(); break;
            case CPU_ROOFLINE_SP_FLOPS: obj.template init<cpu_roofline_sp_flops>(); break;
            case CPU_UTIL: obj.template init<cpu_util>(); break;
            case CUDA_EVENT: obj.template init<cuda_event>(); break;
            case CUPTI_ACTIVITY: obj.template init<cupti_activity>(); break;
            case CUPTI_COUNTERS: obj.template init<cupti_counters>(); break;
            case CURRENT_RSS: obj.template init<current_rss>(); break;
            case DATA_RSS: obj.template init<data_rss>(); break;
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
            case PAPI_ARRAY: obj.template init<papi_array_t>(); break;
            case PEAK_RSS: obj.template init<peak_rss>(); break;
            case PRIORITY_CONTEXT_SWITCH:
                obj.template init<priority_context_switch>();
                break;
            case PROCESS_CPU_CLOCK: obj.template init<process_cpu_clock>(); break;
            case PROCESS_CPU_UTIL: obj.template init<process_cpu_util>(); break;
            case READ_BYTES: obj.template init<read_bytes>(); break;
            case WALL_CLOCK: obj.template init<real_clock>(); break;
            case STACK_RSS: obj.template init<stack_rss>(); break;
            case SYS_CLOCK: obj.template init<system_clock>(); break;
            case THREAD_CPU_CLOCK: obj.template init<thread_cpu_clock>(); break;
            case THREAD_CPU_UTIL: obj.template init<thread_cpu_util>(); break;
            case TRIP_COUNT: obj.template init<trip_count>(); break;
            case USER_CLOCK: obj.template init<user_clock>(); break;
            case VOLUNTARY_CONTEXT_SWITCH:
                obj.template init<voluntary_context_switch>();
                break;
            case WRITTEN_BYTES: obj.template init<written_bytes>(); break;
            case TIMEMORY_COMPONENTS_END:
            default: break;
        }
    }
}

//--------------------------------------------------------------------------------------//
//
///  description:
///      use this function to generate an array of enumerations from a list of string
///      that can be subsequently used to initialize an auto_list or a component_list
///
///  usage:
///      using namespace tim::component;
///      using optional_t = tim::auto_list<real_clock, cpu_clock, cpu_util, cuda_event>;
///
///      auto obj = new optional_t(__FUNCTION__, __LINE__);
///      tim::initialize(*obj, tim::enumerate_components({ "cpu_clock", "cpu_util"}));
///
template <typename _StringT, typename... _ExtraArgs,
          template <typename, typename...> class _Container>
_Container<TIMEMORY_COMPONENT>
enumerate_components(const _Container<_StringT, _ExtraArgs...>& component_names)
{
    _Container<TIMEMORY_COMPONENT> vec;
    for(auto itr : component_names)
    {
        std::transform(itr.begin(), itr.end(), itr.begin(),
                       [](unsigned char c) -> unsigned char { return std::tolower(c); });

        if(itr == "cali" || itr == "caliper")
        {
            vec.push_back(CALIPER);
        }
        else if(itr == "cpu_clock")
        {
            vec.push_back(CPU_CLOCK);
        }
        else if(itr == "cpu_roofline_double" || itr == "cpu_roofline_dp" ||
                itr == "cpu_roofline_dp_flops")
        {
            vec.push_back(CPU_ROOFLINE_DP_FLOPS);
        }
        else if(itr == "cpu_roofline_single" || itr == "cpu_roofline_sp" ||
                itr == "cpu_roofline_sp_flops")
        {
            vec.push_back(CPU_ROOFLINE_SP_FLOPS);
        }
        else if(itr == "cpu_util")
        {
            vec.push_back(CPU_UTIL);
        }
        else if(itr == "cuda_event")
        {
            vec.push_back(CUDA_EVENT);
        }
        else if(itr == "cupti_activity")
        {
            vec.push_back(CUPTI_ACTIVITY);
        }
        else if(itr == "cupti_counters")
        {
            vec.push_back(CUPTI_COUNTERS);
        }
        else if(itr == "current_rss")
        {
            vec.push_back(CURRENT_RSS);
        }
        else if(itr == "data_rss")
        {
            vec.push_back(DATA_RSS);
        }
        else if(itr == "monotonic_clock")
        {
            vec.push_back(MONOTONIC_CLOCK);
        }
        else if(itr == "monotonic_raw_clock")
        {
            vec.push_back(MONOTONIC_RAW_CLOCK);
        }
        else if(itr == "num_io_in")
        {
            vec.push_back(NUM_IO_IN);
        }
        else if(itr == "num_io_out")
        {
            vec.push_back(NUM_IO_OUT);
        }
        else if(itr == "num_major_page_faults")
        {
            vec.push_back(NUM_MAJOR_PAGE_FAULTS);
        }
        else if(itr == "num_minor_page_faults")
        {
            vec.push_back(NUM_MINOR_PAGE_FAULTS);
        }
        else if(itr == "num_msg_recv")
        {
            vec.push_back(NUM_MSG_RECV);
        }
        else if(itr == "num_msg_sent")
        {
            vec.push_back(NUM_MSG_SENT);
        }
        else if(itr == "num_signals")
        {
            vec.push_back(NUM_SIGNALS);
        }
        else if(itr == "num_swap")
        {
            vec.push_back(NUM_SWAP);
        }
        else if(itr == "nvtx" || itr == "nvtx_marker")
        {
            vec.push_back(NVTX_MARKER);
        }
        else if(itr == "papi" || itr == "papi_array" || itr == "papi_array_t")
        {
            vec.push_back(PAPI_ARRAY);
        }
        else if(itr == "peak_rss")
        {
            vec.push_back(PEAK_RSS);
        }
        else if(itr == "priority_context_switch")
        {
            vec.push_back(PRIORITY_CONTEXT_SWITCH);
        }
        else if(itr == "process_cpu_clock")
        {
            vec.push_back(PROCESS_CPU_CLOCK);
        }
        else if(itr == "process_cpu_util")
        {
            vec.push_back(PROCESS_CPU_UTIL);
        }
        else if(itr == "read_bytes")
        {
            vec.push_back(READ_BYTES);
        }
        else if(itr == "real_clock")
        {
            vec.push_back(WALL_CLOCK);
        }
        else if(itr == "stack_rss")
        {
            vec.push_back(STACK_RSS);
        }
        else if(itr == "sys_clock" || itr == "system_clock")
        {
            vec.push_back(SYS_CLOCK);
        }
        else if(itr == "thread_cpu_clock")
        {
            vec.push_back(THREAD_CPU_CLOCK);
        }
        else if(itr == "thread_cpu_util")
        {
            vec.push_back(THREAD_CPU_UTIL);
        }
        else if(itr == "trip_count")
        {
            vec.push_back(TRIP_COUNT);
        }
        else if(itr == "user_clock")
        {
            vec.push_back(USER_CLOCK);
        }
        else if(itr == "voluntary_context_switch")
        {
            vec.push_back(VOLUNTARY_CONTEXT_SWITCH);
        }
        else if(itr == "write_bytes" || itr == "written_bytes")
        {
            vec.push_back(WRITTEN_BYTES);
        }
        else
        {
            fprintf(
                stderr,
                "Unknown component label: %s. Valid choices are: ['cali', 'caliper', "
                "'cpu_clock', 'cpu_roofline_double', 'cpu_roofline_dp', "
                "'cpu_roofline_dp_flops', 'cpu_roofline_single', 'cpu_roofline_sp', "
                "'cpu_roofline_sp_flops', 'cpu_util', 'cuda_event', 'cupti_activity', "
                "'cupti_counters', 'current_rss', 'data_rss', 'monotonic_clock', "
                "'monotonic_raw_clock', 'num_io_in', 'num_io_out', "
                "'num_major_page_faults', 'num_minor_page_faults', 'num_msg_recv', "
                "'num_msg_sent', 'num_signals', 'num_swap', 'nvtx', 'nvtx_marker', "
                "'papi', 'papi_array', 'papi_array_t', 'peak_rss', "
                "'priority_context_switch', 'process_cpu_clock', 'process_cpu_util', "
                "'read_bytes', 'real_clock', 'stack_rss', 'sys_clock', 'system_clock', "
                "'thread_cpu_clock', 'thread_cpu_util', 'trip_count', 'user_clock', "
                "'voluntary_context_switch', 'write_bytes', 'written_bytes']\n",
                itr.c_str());
        }
    }
    return vec;
}

//--------------------------------------------------------------------------------------//
//
//                  specializations for std::initializer_list
//
//--------------------------------------------------------------------------------------//

template <typename... _CompTypes, template <typename...> class _CompList,
          typename _EnumT = int>
inline void
initialize(_CompList<_CompTypes...>& obj, const std::initializer_list<_EnumT>& components)
{
    initialize(obj, std::vector<_EnumT>(components));
}

//--------------------------------------------------------------------------------------//

inline std::vector<TIMEMORY_COMPONENT>
enumerate_components(const std::initializer_list<std::string>& component_names)
{
    return enumerate_components(std::vector<std::string>(component_names));
}

//--------------------------------------------------------------------------------------//

inline std::vector<TIMEMORY_COMPONENT>
enumerate_components(const std::string& names, const std::string& env_id = "")
{
    if(env_id.length() > 0)
        return enumerate_components(tim::delimit(get_env<std::string>(env_id, names)));
    else
        return enumerate_components(tim::delimit(names));
}

//--------------------------------------------------------------------------------------//
//
//                  extra specializations for std::string
//
//--------------------------------------------------------------------------------------//
//
/// this is for initializing with a container of string
//
template <typename... _CompTypes, template <typename...> class _CompList,
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
template <typename... _CompTypes, template <typename...> class _CompList>
inline void
initialize(_CompList<_CompTypes...>& obj, const std::string& components)
{
    initialize(obj, enumerate_components(tim::delimit(components)));
}

//--------------------------------------------------------------------------------------//
//
/// this is for initializing reading an environment variable, getting a string, breaking
/// into list of components, and initializing
//
namespace env
{
template <typename... _CompTypes, template <typename...> class _CompList>
inline void
initialize(_CompList<_CompTypes...>& obj, const std::string& env_var,
           const std::string& default_env)
{
    auto env_result = tim::get_env(env_var, default_env);
    initialize(obj, enumerate_components(tim::delimit(env_result)));
}
}  // namespace env

//--------------------------------------------------------------------------------------//

}  // namespace tim
