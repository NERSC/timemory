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
/// trait that signifies that updating w.r.t. another instance should
/// be a max of the two instances
//
template <typename _Tp>
struct record_max : std::false_type
{
};

//--------------------------------------------------------------------------------------//
/// trait that signifies that data is an array type
///
template <typename _Tp>
struct array_serialization : std::false_type
{
};

//--------------------------------------------------------------------------------------//
/// trait that signifies that an implementation (e.g. PAPI) is available
///
template <typename _Tp>
struct is_available : std::true_type
{
};

//--------------------------------------------------------------------------------------//
/// trait that signifies that a component uses the timemory output handling
///
template <typename _Tp>
struct external_output_handling : std::false_type
{
};

//--------------------------------------------------------------------------------------//
/// trait that signifies that a component requires the prefix to be set right after
/// construction. Types with this trait must contain a member string variable named
/// prefix
///
template <typename _Tp>
struct requires_prefix : std::false_type
{
};

//--------------------------------------------------------------------------------------//
/// trait that signifies that a component handles it's label when printing
///
template <typename _Tp>
struct custom_label_printing : std::false_type
{
};

//--------------------------------------------------------------------------------------//
/// trait that signifies that a component includes it's units when printing
///
template <typename _Tp>
struct custom_unit_printing : std::false_type
{
};

//--------------------------------------------------------------------------------------//
/// trait that signifies that a component includes it's laps when printing
///
template <typename _Tp>
struct custom_laps_printing : std::false_type
{
};

//--------------------------------------------------------------------------------------//
/// trait that designates whether there is a priority when starting the type w.r.t.
/// other types.
///
template <typename _Tp>
struct start_priority : std::false_type
{
};

//--------------------------------------------------------------------------------------//
/// trait that designates whether there is a priority when stopping the type w.r.t.
/// other types.
///
template <typename _Tp>
struct stop_priority : std::false_type
{
};

//--------------------------------------------------------------------------------------//
/// trait that designates the width and precision should follow env specified
/// timing settings
///
template <typename _Tp>
struct is_timing_category : std::false_type
{
};

//--------------------------------------------------------------------------------------//
/// trait that designates the width and precision should follow env specified
/// memory settings
///
template <typename _Tp>
struct is_memory_category : std::false_type
{
};

//--------------------------------------------------------------------------------------//
/// trait that designates the units should follow env specified timing settings
///
template <typename _Tp>
struct uses_timing_units : std::false_type
{
};

//--------------------------------------------------------------------------------------//
/// trait that designates the units should follow env specified memory settings
///
template <typename _Tp>
struct uses_memory_units : std::false_type
{
};

//--------------------------------------------------------------------------------------//
/// trait that designates a type should always print a JSON output
///
template <typename _Tp>
struct requires_json : std::false_type
{
};

//--------------------------------------------------------------------------------------//
}  // trait
}  // tim

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
//      record_max
//--------------------------------------------------------------------------------------//

template <>
struct record_max<component::peak_rss> : std::true_type
{
};

template <>
struct record_max<component::current_rss> : std::true_type
{
};

template <>
struct record_max<component::stack_rss> : std::true_type
{
};

template <>
struct record_max<component::data_rss> : std::true_type
{
};

//--------------------------------------------------------------------------------------//
//      array_serialization
//--------------------------------------------------------------------------------------//

template <int... EventTypes>
struct array_serialization<component::papi_tuple<EventTypes...>> : std::true_type
{
};

template <std::size_t MaxNumEvents>
struct array_serialization<component::papi_array<MaxNumEvents>> : std::true_type
{
};

template <>
struct array_serialization<component::cupti_counters> : std::true_type
{
};

//--------------------------------------------------------------------------------------//
//      start_priority
//--------------------------------------------------------------------------------------//

/// component::cuda_event should be stopped before other types
template <>
struct start_priority<component::cupti_activity> : std::true_type
{
};

//--------------------------------------------------------------------------------------//
//      stop_priority
//--------------------------------------------------------------------------------------//

/// component::cuda_event should be stopped before other types
template <>
struct stop_priority<component::cuda_event> : std::true_type
{
};

/// component::cuda_event should be stopped before other types
template <>
struct stop_priority<component::cupti_activity> : std::true_type
{
};

//--------------------------------------------------------------------------------------//
//      custom_unit_printing
//--------------------------------------------------------------------------------------//

template <>
struct custom_unit_printing<component::read_bytes> : std::true_type
{
};

template <>
struct custom_unit_printing<component::written_bytes> : std::true_type
{
};

//--------------------------------------------------------------------------------------//
//      custom_label_printing
//--------------------------------------------------------------------------------------//

template <>
struct custom_label_printing<component::read_bytes> : std::true_type
{
};

template <>
struct custom_label_printing<component::written_bytes> : std::true_type
{
};

template <>
struct custom_laps_printing<component::trip_count> : std::true_type
{
};

//--------------------------------------------------------------------------------------//
//		is_timing_category
//--------------------------------------------------------------------------------------//

template <>
struct is_timing_category<component::real_clock> : std::true_type
{
};

template <>
struct is_timing_category<component::system_clock> : std::true_type
{
};

template <>
struct is_timing_category<component::user_clock> : std::true_type
{
};

template <>
struct is_timing_category<component::cpu_clock> : std::true_type
{
};

template <>
struct is_timing_category<component::monotonic_clock> : std::true_type
{
};

template <>
struct is_timing_category<component::monotonic_raw_clock> : std::true_type
{
};

template <>
struct is_timing_category<component::thread_cpu_clock> : std::true_type
{
};

template <>
struct is_timing_category<component::process_cpu_clock> : std::true_type
{
};

template <>
struct is_timing_category<component::cuda_event> : std::true_type
{
};

template <>
struct is_timing_category<component::cupti_activity> : std::true_type
{
};

//--------------------------------------------------------------------------------------//
//		is_memory_category
//--------------------------------------------------------------------------------------//

template <>
struct is_memory_category<component::peak_rss> : std::true_type
{
};

template <>
struct is_memory_category<component::current_rss> : std::true_type
{
};

template <>
struct is_memory_category<component::stack_rss> : std::true_type
{
};

template <>
struct is_memory_category<component::data_rss> : std::true_type
{
};

template <>
struct is_memory_category<component::num_swap> : std::true_type
{
};

template <>
struct is_memory_category<component::num_io_in> : std::true_type
{
};

template <>
struct is_memory_category<component::num_io_out> : std::true_type
{
};

template <>
struct is_memory_category<component::num_minor_page_faults> : std::true_type
{
};

template <>
struct is_memory_category<component::num_major_page_faults> : std::true_type
{
};

template <>
struct is_memory_category<component::num_msg_sent> : std::true_type
{
};

template <>
struct is_memory_category<component::num_msg_recv> : std::true_type
{
};

template <>
struct is_memory_category<component::num_signals> : std::true_type
{
};

template <>
struct is_memory_category<component::voluntary_context_switch> : std::true_type
{
};

template <>
struct is_memory_category<component::priority_context_switch> : std::true_type
{
};

template <>
struct is_memory_category<component::read_bytes> : std::true_type
{
};

template <>
struct is_memory_category<component::written_bytes> : std::true_type
{
};

//--------------------------------------------------------------------------------------//
//		uses_timing_units
//--------------------------------------------------------------------------------------//

template <>
struct uses_timing_units<component::real_clock> : std::true_type
{
};

template <>
struct uses_timing_units<component::system_clock> : std::true_type
{
};

template <>
struct uses_timing_units<component::user_clock> : std::true_type
{
};

template <>
struct uses_timing_units<component::cpu_clock> : std::true_type
{
};

template <>
struct uses_timing_units<component::monotonic_clock> : std::true_type
{
};

template <>
struct uses_timing_units<component::monotonic_raw_clock> : std::true_type
{
};

template <>
struct uses_timing_units<component::thread_cpu_clock> : std::true_type
{
};

template <>
struct uses_timing_units<component::process_cpu_clock> : std::true_type
{
};

template <>
struct uses_timing_units<component::cuda_event> : std::true_type
{
};

template <>
struct uses_timing_units<component::cupti_activity> : std::true_type
{
};

//--------------------------------------------------------------------------------------//
//		uses_memory_units
//--------------------------------------------------------------------------------------//

template <>
struct uses_memory_units<component::peak_rss> : std::true_type
{
};

template <>
struct uses_memory_units<component::current_rss> : std::true_type
{
};

template <>
struct uses_memory_units<component::stack_rss> : std::true_type
{
};

template <>
struct uses_memory_units<component::data_rss> : std::true_type
{
};

template <>
struct uses_memory_units<component::read_bytes> : std::true_type
{
};

template <>
struct uses_memory_units<component::written_bytes> : std::true_type
{
};

//--------------------------------------------------------------------------------------//
// if not UNIX (i.e. Windows)
//
#if !defined(_UNIX)

template <>
struct is_available<component::stack_rss> : std::false_type
{
};

template <>
struct is_available<component::data_rss> : std::false_type
{
};

template <>
struct is_available<component::num_io_in> : std::false_type
{
};

template <>
struct is_available<component::num_io_out> : std::false_type
{
};

template <>
struct is_available<component::num_major_page_faults> : std::false_type
{
};

template <>
struct is_available<component::num_minor_page_faults> : std::false_type
{
};

template <>
struct is_available<component::num_msg_recv> : std::false_type
{
};

template <>
struct is_available<component::num_msg_sent> : std::false_type
{
};

template <>
struct is_available<component::num_signals> : std::false_type
{
};

template <>
struct is_available<component::num_swap> : std::false_type
{
};

template <>
struct is_available<component::read_bytes> : std::false_type
{
};

template <>
struct is_available<component::written_bytes> : std::false_type
{
};

#endif

//--------------------------------------------------------------------------------------//
//  disable if not enabled via preprocessor TIMEMORY_USE_PAPI
//
#if !defined(TIMEMORY_USE_PAPI)

template <int... EventTypes>
struct is_available<component::papi_tuple<EventTypes...>> : std::false_type
{
};

template <std::size_t MaxNumEvents>
struct is_available<component::papi_array<MaxNumEvents>> : std::false_type
{
};

template <typename _Tp, int... EventTypes>
struct is_available<component::cpu_roofline<_Tp, EventTypes...>> : std::false_type
{
};

template <typename _Tp, int... EventTypes>
struct requires_json<component::cpu_roofline<_Tp, EventTypes...>> : std::true_type
{
};

#endif  // TIMEMORY_USE_PAPI

//--------------------------------------------------------------------------------------//
//  disable if not enabled via preprocessor TIMEMORY_USE_CUDA
//
#if !defined(TIMEMORY_USE_CUDA)

template <>
struct is_available<component::cuda_event> : std::false_type
{
};

#endif  // TIMEMORY_USE_CUDA

//--------------------------------------------------------------------------------------//
//  disable if not enabled via preprocessor TIMEMORY_USE_CUPTI
//
#if !defined(TIMEMORY_USE_CUPTI)

template <>
struct is_available<component::cupti_counters> : std::false_type
{
};

template <>
struct is_available<component::cupti_activity> : std::false_type
{
};

template <typename _Tp>
struct is_available<component::gpu_roofline<_Tp>> : std::false_type
{
};

#endif  // TIMEMORY_USE_CUPTI

//--------------------------------------------------------------------------------------//
//  disable if not enabled via preprocessor TIMEMORY_USE_NVTX
//
#if !defined(TIMEMORY_USE_NVTX)

template <>
struct is_available<component::nvtx_marker> : std::false_type
{
};

#else

template <>
struct requires_prefix<component::nvtx_marker> : std::true_type
{
};

#endif  // TIMEMORY_USE_NVTX

template <>
struct external_output_handling<component::nvtx_marker> : std::true_type
{
};

//--------------------------------------------------------------------------------------//
//  disable if not enabled via preprocessor TIMEMORY_USE_CALIPER
//
#if !defined(TIMEMORY_USE_CALIPER)

template <>
struct is_available<component::caliper> : std::false_type
{
};

#else

template <>
struct requires_prefix<component::caliper> : std::true_type
{
};

#endif  // TIMEMORY_USE_CALIPER

template <>
struct external_output_handling<component::caliper> : std::true_type
{
};

//--------------------------------------------------------------------------------------//
}  // component
}  // tim
