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

/** \file bits/components.hpp
 * \headerfile bits/components.hpp "timemory/bits/components.hpp"
 * Provides implementation for initialize, enumerate_components which regularly
 * change as more features are added
 *
 */

#pragma once

#include "timemory/components/types.hpp"
#include "timemory/enum.h"

#include <unordered_map>

namespace tim
{
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
        case USER_TUPLE_BUNDLE: obj.template init<user_tuple_bundle>(); break;
        case USER_LIST_BUNDLE: obj.template init<user_list_bundle>(); break;
        case USER_CLOCK: obj.template init<user_clock>(); break;
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
          typename... _ExtraArgs>
void
initialize(_CompList<_CompTypes...>&               obj,
           const _Container<_Intp, _ExtraArgs...>& components)
{
    for(auto itr : components)
        initialize(static_cast<TIMEMORY_COMPONENT>(itr), obj);
}

//--------------------------------------------------------------------------------------//

template <template <typename...> class _CompList, typename... _CompTypes>
void
initialize(_CompList<_CompTypes...>& obj, const int ncomponents, const int* components)
{
    for(int i = 0; i < ncomponents; ++i)
        initialize(static_cast<TIMEMORY_COMPONENT>(components[i]), obj);
}

//--------------------------------------------------------------------------------------//

template <typename _StringT, typename... _ExtraArgs,
          template <typename, typename...> class _Container>
_Container<TIMEMORY_COMPONENT>
enumerate_components(const _Container<_StringT, _ExtraArgs...>& component_names)
{
    _Container<TIMEMORY_COMPONENT> vec;

    using hash_type = std::hash<std::string>;
    using component_hash_map_t =
        std::unordered_map<std::string, TIMEMORY_COMPONENT, hash_type>;

    static auto _generate = []() {
        component_hash_map_t _instance;
        _instance["cali"]                     = CALIPER;
        _instance["caliper"]                  = CALIPER;
        _instance["cpu_clock"]                = CPU_CLOCK;
        _instance["cpu_roofline_double"]      = CPU_ROOFLINE_DP_FLOPS;
        _instance["cpu_roofline_dp"]          = CPU_ROOFLINE_DP_FLOPS;
        _instance["cpu_roofline_dp_flops"]    = CPU_ROOFLINE_DP_FLOPS;
        _instance["cpu_roofline"]             = CPU_ROOFLINE_FLOPS;
        _instance["cpu_roofline_flops"]       = CPU_ROOFLINE_FLOPS;
        _instance["cpu_roofline_single"]      = CPU_ROOFLINE_SP_FLOPS;
        _instance["cpu_roofline_sp"]          = CPU_ROOFLINE_SP_FLOPS;
        _instance["cpu_roofline_sp_flops"]    = CPU_ROOFLINE_SP_FLOPS;
        _instance["cpu_util"]                 = CPU_UTIL;
        _instance["cuda_event"]               = CUDA_EVENT;
        _instance["cupti_activity"]           = CUPTI_ACTIVITY;
        _instance["cupti_counters"]           = CUPTI_COUNTERS;
        _instance["data_rss"]                 = DATA_RSS;
        _instance["gperf_cpu"]                = GPERF_CPU_PROFILER;
        _instance["gperf_cpu_profiler"]       = GPERF_CPU_PROFILER;
        _instance["gperftools-cpu"]           = GPERF_CPU_PROFILER;
        _instance["gperf_heap"]               = GPERF_HEAP_PROFILER;
        _instance["gperf_heap_profiler"]      = GPERF_HEAP_PROFILER;
        _instance["gperftools-heap"]          = GPERF_HEAP_PROFILER;
        _instance["gpu_roofline_double"]      = GPU_ROOFLINE_DP_FLOPS;
        _instance["gpu_roofline_dp"]          = GPU_ROOFLINE_DP_FLOPS;
        _instance["gpu_roofline_dp_flops"]    = GPU_ROOFLINE_DP_FLOPS;
        _instance["gpu_roofline"]             = GPU_ROOFLINE_FLOPS;
        _instance["gpu_roofline_flops"]       = GPU_ROOFLINE_FLOPS;
        _instance["gpu_roofline_half"]        = GPU_ROOFLINE_HP_FLOPS;
        _instance["gpu_roofline_hp"]          = GPU_ROOFLINE_HP_FLOPS;
        _instance["gpu_roofline_hp_flops"]    = GPU_ROOFLINE_HP_FLOPS;
        _instance["gpu_roofline_single"]      = GPU_ROOFLINE_SP_FLOPS;
        _instance["gpu_roofline_sp"]          = GPU_ROOFLINE_SP_FLOPS;
        _instance["gpu_roofline_sp_flops"]    = GPU_ROOFLINE_SP_FLOPS;
        _instance["likwid_gpu"]               = LIKWID_NVMON;
        _instance["likwid_nvmon"]             = LIKWID_NVMON;
        _instance["likwid_cpu"]               = LIKWID_PERFMON;
        _instance["likwid_perfmon"]           = LIKWID_PERFMON;
        _instance["monotonic_clock"]          = MONOTONIC_CLOCK;
        _instance["monotonic_raw_clock"]      = MONOTONIC_RAW_CLOCK;
        _instance["num_io_in"]                = NUM_IO_IN;
        _instance["num_io_out"]               = NUM_IO_OUT;
        _instance["num_major_page_faults"]    = NUM_MAJOR_PAGE_FAULTS;
        _instance["num_minor_page_faults"]    = NUM_MINOR_PAGE_FAULTS;
        _instance["num_msg_recv"]             = NUM_MSG_RECV;
        _instance["num_msg_sent"]             = NUM_MSG_SENT;
        _instance["num_signals"]              = NUM_SIGNALS;
        _instance["num_swap"]                 = NUM_SWAP;
        _instance["nvtx"]                     = NVTX_MARKER;
        _instance["nvtx_marker"]              = NVTX_MARKER;
        _instance["page_rss"]                 = PAGE_RSS;
        _instance["papi"]                     = PAPI_ARRAY;
        _instance["papi_array"]               = PAPI_ARRAY;
        _instance["papi_array_t"]             = PAPI_ARRAY;
        _instance["peak_rss"]                 = PEAK_RSS;
        _instance["priority_context_switch"]  = PRIORITY_CONTEXT_SWITCH;
        _instance["process_cpu_clock"]        = PROCESS_CPU_CLOCK;
        _instance["process_cpu_util"]         = PROCESS_CPU_UTIL;
        _instance["read_bytes"]               = READ_BYTES;
        _instance["stack_rss"]                = STACK_RSS;
        _instance["sys_clock"]                = SYS_CLOCK;
        _instance["system_clock"]             = SYS_CLOCK;
        _instance["tau"]                      = TAU_MARKER;
        _instance["tau_marker"]               = TAU_MARKER;
        _instance["thread_cpu_clock"]         = THREAD_CPU_CLOCK;
        _instance["thread_cpu_util"]          = THREAD_CPU_UTIL;
        _instance["trip_count"]               = TRIP_COUNT;
        _instance["user_clock"]               = USER_CLOCK;
        _instance["user_list_bundle"]         = USER_LIST_BUNDLE;
        _instance["user_tuple_bundle"]        = USER_TUPLE_BUNDLE;
        _instance["virtual_memory"]           = VIRTUAL_MEMORY;
        _instance["voluntary_context_switch"] = VOLUNTARY_CONTEXT_SWITCH;
        _instance["vtune_event"]              = VTUNE_EVENT;
        _instance["vtune_frame"]              = VTUNE_FRAME;
        _instance["real_clock"]               = WALL_CLOCK;
        _instance["virtual_clock"]            = WALL_CLOCK;
        _instance["wall_clock"]               = WALL_CLOCK;
        _instance["write_bytes"]              = WRITTEN_BYTES;
        _instance["written_bytes"]            = WRITTEN_BYTES;
        return _instance;
    };

    static auto errmsg = [](const std::string& itr) {
        fprintf(
            stderr,
            "Unknown component label: %s. Valid choices are: ['cali', 'caliper', "
            "'cpu_clock', 'cpu_roofline', 'cpu_roofline_double', 'cpu_roofline_dp', "
            "'cpu_roofline_dp_flops', 'cpu_roofline_flops', 'cpu_roofline_single', "
            "'cpu_roofline_sp', 'cpu_roofline_sp_flops', 'cpu_util', 'cuda_event', "
            "'cupti_activity', 'cupti_counters', 'data_rss', 'gperf-cpu', 'gperf-heap', "
            "'gperf_cpu_profiler', 'gperf_heap_profiler', 'gperftools-cpu', "
            "'gperftools-heap', 'gpu_roofline', 'gpu_roofline_double', "
            "'gpu_roofline_dp', 'gpu_roofline_dp_flops', 'gpu_roofline_flops', "
            "'gpu_roofline_half', 'gpu_roofline_hp', 'gpu_roofline_hp_flops', "
            "'gpu_roofline_single', 'gpu_roofline_sp', 'gpu_roofline_sp_flops', "
            "'likwid_cpu', 'likwid_gpu', 'likwid_nvmon', 'likwid_perfmon', "
            "'monotonic_clock', 'monotonic_raw_clock', 'num_io_in', 'num_io_out', "
            "'num_major_page_faults', 'num_minor_page_faults', 'num_msg_recv', "
            "'num_msg_sent', 'num_signals', 'num_swap', 'nvtx', 'nvtx_marker', "
            "'page_rss', 'papi', 'papi_array', 'papi_array_t', 'peak_rss', "
            "'priority_context_switch', 'process_cpu_clock', 'process_cpu_util', "
            "'read_bytes', 'real_clock', 'stack_rss', 'sys_clock', 'system_clock', "
            "'tau', 'tau_marker', 'thread_cpu_clock', 'thread_cpu_util', 'trip_count', "
            "'user_clock', 'user_list_bundle', 'user_tuple_bundle', 'virtual_clock', "
            "'virtual_memory', 'voluntary_context_switch', 'vtune_event', 'vtune_frame', "
            "'wall_clock', 'write_bytes', 'written_bytes']\n",
            itr.c_str());
    };

    static auto _hashmap = _generate();
    for(auto itr : component_names)
    {
        std::transform(itr.begin(), itr.end(), itr.begin(),
                       [](unsigned char c) -> unsigned char { return std::tolower(c); });

        auto _eitr = _hashmap.find(itr);
        if(_eitr != _hashmap.end())
            vec.push_back(_eitr->second);
        else
            errmsg(itr);
    }

    return vec;
}

//======================================================================================//

}  // namespace tim

//======================================================================================//
