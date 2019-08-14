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
struct internal_output_handling : std::true_type
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
//
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

template <int... EventTypes>
struct array_serialization<component::papi_tuple<EventTypes...>> : std::true_type
{
};

template <std::size_t MaxNumEvents>
struct array_serialization<component::papi_array<MaxNumEvents>> : std::true_type
{
};

template <>
struct array_serialization<component::cupti_event> : std::true_type
{
};

/// component::cuda_event should be stopped before other types
template <>
struct stop_priority<component::cuda_event> : std::true_type
{
};

template <>
struct custom_unit_printing<component::read_bytes> : std::true_type
{
};

template <>
struct custom_unit_printing<component::written_bytes> : std::true_type
{
};

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
struct is_available<component::cupti_event> : std::false_type
{
};

#endif  // TIMEMORY_USE_CUPTI

//--------------------------------------------------------------------------------------//
//  disable if not enabled via preprocessor TIMEMORY_USE_CALIPER
//
#if !defined(TIMEMORY_USE_CALIPER)

template <>
struct is_available<component::caliper> : std::false_type
{
};

#endif  // TIMEMORY_USE_CALIPER

template <>
struct internal_output_handling<component::caliper> : std::false_type
{
};

//--------------------------------------------------------------------------------------//
}  // component
}  // tim
