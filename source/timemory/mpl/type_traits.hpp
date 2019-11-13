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

#pragma once

#include "timemory/components/types.hpp"
#include "timemory/mpl/types.hpp"
#include <type_traits>

//======================================================================================//
//
//                                 Type Traits
//
//======================================================================================//

namespace tim
{
namespace trait
{
//--------------------------------------------------------------------------------------//
/// trait that signifies that an implementation (e.g. PAPI) is available
///
template <typename _Tp>
struct is_available : std::true_type
{};

//--------------------------------------------------------------------------------------//
/// trait that signifies that updating w.r.t. another instance should
/// be a max of the two instances
//
template <typename _Tp>
struct record_max : std::false_type
{};

//--------------------------------------------------------------------------------------//
/// trait that signifies that data is an array type
///
template <typename _Tp>
struct array_serialization : std::false_type
{};

//--------------------------------------------------------------------------------------//
/// trait that signifies that a component uses the timemory output handling
///
template <typename _Tp>
struct external_output_handling : std::false_type
{};

//--------------------------------------------------------------------------------------//
/// trait that signifies that a component requires the prefix to be set right after
/// construction. Types with this trait must contain a member string variable named
/// prefix
///
template <typename _Tp>
struct requires_prefix : std::false_type
{};

//--------------------------------------------------------------------------------------//
/// trait that signifies that a component handles it's label when printing
///
template <typename _Tp>
struct custom_label_printing : std::false_type
{};

//--------------------------------------------------------------------------------------//
/// trait that signifies that a component includes it's units when printing
///
template <typename _Tp>
struct custom_unit_printing : std::false_type
{};

//--------------------------------------------------------------------------------------//
/// trait that signifies that a component includes it's laps when printing
///
template <typename _Tp>
struct custom_laps_printing : std::false_type
{};

//--------------------------------------------------------------------------------------//
/// trait that designates whether there is a priority when starting the type w.r.t.
/// other types.
///
template <typename _Tp>
struct start_priority : std::false_type
{};

//--------------------------------------------------------------------------------------//
/// trait that designates whether there is a priority when stopping the type w.r.t.
/// other types.
///
template <typename _Tp>
struct stop_priority : std::false_type
{};

//--------------------------------------------------------------------------------------//
/// trait that designates the width and precision should follow env specified
/// timing settings
///
template <typename _Tp>
struct is_timing_category : std::false_type
{};

//--------------------------------------------------------------------------------------//
/// trait that designates the width and precision should follow env specified
/// memory settings
///
template <typename _Tp>
struct is_memory_category : std::false_type
{};

//--------------------------------------------------------------------------------------//
/// trait that designates the units should follow env specified timing settings
///
template <typename _Tp>
struct uses_timing_units : std::false_type
{};

//--------------------------------------------------------------------------------------//
/// trait that designates the units should follow env specified memory settings
///
template <typename _Tp>
struct uses_memory_units : std::false_type
{};

//--------------------------------------------------------------------------------------//
/// trait that designates the units are a percentage
///
template <typename _Tp>
struct uses_percent_units : std::false_type
{};

//--------------------------------------------------------------------------------------//
/// trait that designates a type should always print a JSON output
///
template <typename _Tp>
struct requires_json : std::false_type
{};

//--------------------------------------------------------------------------------------//
/// trait that designates the type is a gotcha... ONLY gotcha should set to TRUE!
///
template <typename _Tp>
struct is_gotcha : std::false_type
{};

//--------------------------------------------------------------------------------------//
/// trait that designates the type supports calling a function with a certain
/// set of argument types (passed via a tuple)
///
template <typename _Tp, typename _Tuple>
struct supports_args : std::false_type
{};

//--------------------------------------------------------------------------------------//
/// trait that designates the type supports changing the record() static function
/// per-instance
///
template <typename _Tp>
struct supports_custom_record : std::false_type
{};

//--------------------------------------------------------------------------------------//
/// trait that signifies that get() returns an iterable type
///
template <typename _Tp>
struct iterable_measurement : std::false_type
{};

//--------------------------------------------------------------------------------------//
/// trait that signifies that secondary data resembling the original data
/// exists but should be another node entry in the graph. These types
/// must provide a get_secondary() member function and that member function
/// must return a pair-wise iterable container, e.g. std::map, of types:
///     - std::string
///     - value_type
///
template <typename _Tp>
struct secondary_data : std::false_type
{};

//--------------------------------------------------------------------------------------//

template <typename _Trait>
inline std::string
as_string()
{
    constexpr bool _val = _Trait::value;
    return (_val) ? "true" : "false";
}

//--------------------------------------------------------------------------------------//
}  // namespace trait
}  // namespace tim

//======================================================================================//
//
//                              Specifications
//
//======================================================================================//

namespace tim
{
namespace trait
{
//--------------------------------------------------------------------------------------//
//
//                              RECORD MAX
//
//--------------------------------------------------------------------------------------//

template <>
struct record_max<component::peak_rss> : std::true_type
{};

template <>
struct record_max<component::page_rss> : std::true_type
{};

template <>
struct record_max<component::stack_rss> : std::true_type
{};

template <>
struct record_max<component::data_rss> : std::true_type
{};

template <>
struct record_max<component::virtual_memory> : std::true_type
{};

//--------------------------------------------------------------------------------------//
//
//                              ARRAY SERIALIZATION
//
//--------------------------------------------------------------------------------------//
#if defined(TIMEMORY_USE_PAPI)
template <int... EventTypes>
struct array_serialization<component::papi_tuple<EventTypes...>> : std::true_type
{};

template <std::size_t MaxNumEvents>
struct array_serialization<component::papi_array<MaxNumEvents>> : std::true_type
{};

template <>
struct array_serialization<component::cupti_counters> : std::true_type
{};
#endif
//--------------------------------------------------------------------------------------//
//
//                              START PRIORITY
//
//--------------------------------------------------------------------------------------//
#if defined(TIMEMORY_USE_CUPTI)
/// component::cuda_event should be stopped before other types
template <>
struct start_priority<component::cupti_activity> : std::true_type
{};
#endif
//--------------------------------------------------------------------------------------//
//
//                              STOP PRIORITY
//
//--------------------------------------------------------------------------------------//
#if defined(TIMEMORY_USE_CUDA)
/// component::cuda_event should be stopped before other types
template <>
struct stop_priority<component::cuda_event> : std::true_type
{};
#endif

#if defined(TIMEMORY_USE_CUPTI)
/// component::cuda_event should be stopped before other types
template <>
struct stop_priority<component::cupti_activity> : std::true_type
{};
#endif
//--------------------------------------------------------------------------------------//
//
//                              CUSTOM UNIT PRINTING
//
//--------------------------------------------------------------------------------------//

template <>
struct custom_unit_printing<component::read_bytes> : std::true_type
{};

template <>
struct custom_unit_printing<component::written_bytes> : std::true_type
{};

#if defined(TIMEMORY_USE_CUPTI)
template <>
struct custom_unit_printing<component::cupti_counters> : std::true_type
{};

template <typename... _Types>
struct custom_unit_printing<component::gpu_roofline<_Types...>> : std::true_type
{};
#endif

/*
template <typename... _Types>
struct custom_unit_printing<component::cpu_roofline<_Types...>> : std::true_type
{
};
*/

//--------------------------------------------------------------------------------------//
//
//                              CUSTOM LABEL PRINTING
//
//--------------------------------------------------------------------------------------//

template <>
struct custom_label_printing<component::read_bytes> : std::true_type
{};

template <>
struct custom_label_printing<component::written_bytes> : std::true_type
{};

#if defined(TIMEMORY_USE_CUPTI)
template <>
struct custom_laps_printing<component::cupti_counters> : std::true_type
{};

template <typename... _Types>
struct custom_label_printing<component::gpu_roofline<_Types...>> : std::true_type
{};
#endif
/*
template <typename... _Types>
struct custom_label_printing<component::cpu_roofline<_Types...>> : std::true_type
{
};
*/

//--------------------------------------------------------------------------------------//
//
//                              CUSTOM LAPS PRINTING
//
//--------------------------------------------------------------------------------------//

template <>
struct custom_laps_printing<component::trip_count> : std::true_type
{};

//--------------------------------------------------------------------------------------//
//
//                              NOT UNIX (i.e. Windows)
//
//--------------------------------------------------------------------------------------//
// if not UNIX (i.e. Windows)
//
#if !defined(_UNIX)

template <>
struct is_available<component::stack_rss> : std::false_type
{};

template <>
struct is_available<component::data_rss> : std::false_type
{};

template <>
struct is_available<component::num_io_in> : std::false_type
{};

template <>
struct is_available<component::num_io_out> : std::false_type
{};

template <>
struct is_available<component::num_major_page_faults> : std::false_type
{};

template <>
struct is_available<component::num_minor_page_faults> : std::false_type
{};

template <>
struct is_available<component::num_msg_recv> : std::false_type
{};

template <>
struct is_available<component::num_msg_sent> : std::false_type
{};

template <>
struct is_available<component::num_signals> : std::false_type
{};

template <>
struct is_available<component::num_swap> : std::false_type
{};

template <>
struct is_available<component::read_bytes> : std::false_type
{};

template <>
struct is_available<component::written_bytes> : std::false_type
{};

template <>
struct is_available<component::virtual_memory> : std::false_type
{};

#endif

//--------------------------------------------------------------------------------------//
//
//                              PAPI / CPU_ROOFLINE
//
//--------------------------------------------------------------------------------------//
//  disable if not enabled via preprocessor TIMEMORY_USE_PAPI
//
#if !defined(TIMEMORY_USE_PAPI)

template <int... EventTypes>
struct is_available<component::papi_tuple<EventTypes...>> : std::false_type
{};

template <std::size_t MaxNumEvents>
struct is_available<component::papi_array<MaxNumEvents>> : std::false_type
{};

template <typename... _Types>
struct is_available<component::cpu_roofline<_Types...>> : std::false_type
{};

template <>
struct is_available<component::cpu_roofline_sp_flops> : std::false_type
{};

template <>
struct is_available<component::cpu_roofline_dp_flops> : std::false_type
{};

template <>
struct is_available<component::cpu_roofline_flops> : std::false_type
{};

#else

template <typename... _Types>
struct requires_json<component::cpu_roofline<_Types...>> : std::true_type
{};

template <>
struct requires_json<component::cpu_roofline_sp_flops> : std::true_type
{};

template <>
struct requires_json<component::cpu_roofline_dp_flops> : std::true_type
{};

template <>
struct requires_json<component::cpu_roofline_flops> : std::true_type
{};

template <typename... _Types>
struct supports_custom_record<component::cpu_roofline<_Types...>> : std::true_type
{};

#endif  // TIMEMORY_USE_PAPI

//--------------------------------------------------------------------------------------//
//
//                              CUDA
//
//--------------------------------------------------------------------------------------//
//  disable if not enabled via preprocessor TIMEMORY_USE_CUDA
//
#if !defined(TIMEMORY_USE_CUDA)

template <>
struct is_available<component::cuda_event> : std::false_type
{};

#endif  // TIMEMORY_USE_CUDA

//--------------------------------------------------------------------------------------//
//
//                              CUPTI / GPU ROOFLINE
//
//--------------------------------------------------------------------------------------//
//  disable if not enabled via preprocessor TIMEMORY_USE_CUPTI
//
#if !defined(TIMEMORY_USE_CUPTI)

template <>
struct is_available<component::cupti_counters> : std::false_type
{};

template <>
struct is_available<component::cupti_activity> : std::false_type
{};

template <typename... _Types>
struct is_available<component::gpu_roofline<_Types...>> : std::false_type
{};

template <>
struct is_available<component::gpu_roofline_hp_flops> : std::false_type
{};

template <>
struct is_available<component::gpu_roofline_sp_flops> : std::false_type
{};

template <>
struct is_available<component::gpu_roofline_dp_flops> : std::false_type
{};

template <>
struct is_available<component::gpu_roofline_flops> : std::false_type
{};

#else

template <typename... _Types>
struct requires_json<component::gpu_roofline<_Types...>> : std::true_type
{};

template <>
struct requires_json<component::gpu_roofline_hp_flops> : std::true_type
{};

template <>
struct requires_json<component::gpu_roofline_sp_flops> : std::true_type
{};

template <>
struct requires_json<component::gpu_roofline_dp_flops> : std::true_type
{};

template <>
struct requires_json<component::gpu_roofline_flops> : std::true_type
{};

template <typename... _Types>
struct iterable_measurement<component::gpu_roofline<_Types...>> : std::true_type
{};

template <>
struct iterable_measurement<component::gpu_roofline_hp_flops> : std::true_type
{};

template <>
struct iterable_measurement<component::gpu_roofline_sp_flops> : std::true_type
{};

template <>
struct iterable_measurement<component::gpu_roofline_dp_flops> : std::true_type
{};

template <>
struct iterable_measurement<component::gpu_roofline_flops> : std::true_type
{};

//
//  secondary data
//

template <>
struct secondary_data<component::cupti_activity> : std::true_type
{};

template <>
struct secondary_data<component::cupti_counters> : std::true_type
{};

#endif  // TIMEMORY_USE_CUPTI

//--------------------------------------------------------------------------------------//
//  disable if not enabled via preprocessor TIMEMORY_USE_NVTX
//
#if !defined(TIMEMORY_USE_NVTX)

template <>
struct is_available<component::nvtx_marker> : std::false_type
{};

#else

template <>
struct requires_prefix<component::nvtx_marker> : std::true_type
{};

template <>
struct external_output_handling<component::nvtx_marker> : std::true_type
{};

#endif  // TIMEMORY_USE_NVTX

//--------------------------------------------------------------------------------------//
//
//                              CALIPER
//
//--------------------------------------------------------------------------------------//
//  disable if not enabled via preprocessor TIMEMORY_USE_CALIPER
//
#if !defined(TIMEMORY_USE_CALIPER)

template <>
struct is_available<component::caliper> : std::false_type
{};

#else

template <>
struct requires_prefix<component::caliper> : std::true_type
{};

#endif  // TIMEMORY_USE_CALIPER

template <>
struct external_output_handling<component::caliper> : std::true_type
{};

//--------------------------------------------------------------------------------------//
//
//                              GOTCHA
//
//--------------------------------------------------------------------------------------//
//  disable if not enabled via preprocessor TIMEMORY_USE_GOTCHA
//
#if !defined(TIMEMORY_USE_GOTCHA)

template <size_t _N, typename _Comp, typename _Diff>
struct is_available<component::gotcha<_N, _Comp, _Diff>> : std::false_type
{};

#else  // TIMEMORY_USE_GOTCHA

template <size_t _N, typename _Comp, typename _Diff>
struct external_output_handling<component::gotcha<_N, _Comp, _Diff>> : std::true_type
{};

template <size_t _N, typename _Comp, typename _Diff>
struct is_gotcha<component::gotcha<_N, _Comp, _Diff>> : std::true_type
{};

#endif  // TIMEMORY_USE_GOTCHA

//--------------------------------------------------------------------------------------//
//
//                              GPERFTOOLS
//
//--------------------------------------------------------------------------------------//
//  disable if not enabled via preprocessor TIMEMORY_USE_GPERF_HEAP_PROFILER or
//  TIMEMORY_USE_GPERF
//
#if defined(TIMEMORY_USE_GPERF) || defined(TIMEMORY_USE_GPERF_HEAP_PROFILER)

//--------------------------------------------------------------------------------------//
//
template <>
struct requires_prefix<component::gperf_heap_profiler> : std::true_type
{};

//--------------------------------------------------------------------------------------//
//
template <>
struct external_output_handling<component::gperf_heap_profiler> : std::true_type
{};

#else

//--------------------------------------------------------------------------------------//
//
template <>
struct is_available<component::gperf_heap_profiler> : std::false_type
{};

#endif

//--------------------------------------------------------------------------------------//
//  disable if not enabled via preprocessor TIMEMORY_USE_GPERF_CPU_PROFILER or
//  TIMEMORY_USE_GPERF
//

#if defined(TIMEMORY_USE_GPERF) || defined(TIMEMORY_USE_GPERF_CPU_PROFILER)

//--------------------------------------------------------------------------------------//
//
template <>
struct external_output_handling<component::gperf_cpu_profiler> : std::true_type
{};

#else

//--------------------------------------------------------------------------------------//
//
template <>
struct is_available<component::gperf_cpu_profiler> : std::false_type
{};

#endif

//--------------------------------------------------------------------------------------//
}  // namespace trait
}  // namespace tim

#include "timemory/mpl/bits/type_traits.hpp"
