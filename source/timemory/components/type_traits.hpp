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
namespace component
{
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
struct array_serialization
{
    using type = std::false_type;
};

//--------------------------------------------------------------------------------------//
/// trait that signifies that an implementation (e.g. PAPI) is available
///
template <typename _Tp>
struct impl_available : std::true_type
{};

//--------------------------------------------------------------------------------------//
/// trait that designates whether there is a priority ordering for variadic list
/// e.g. -1 for type A would mean it should come before type B with 0 and
/// type C with 1 should come after A and B
///
template <typename _Tp>
struct ordering_priority : std::integral_constant<int16_t, 0>
{};

//--------------------------------------------------------------------------------------//
}  // component
}  // tim

//======================================================================================//
//
//                              Specifications
//
//======================================================================================//

namespace tim
{
namespace component
{
//--------------------------------------------------------------------------------------//
//
template <>
struct record_max<peak_rss> : std::true_type
{};

template <>
struct record_max<current_rss> : std::true_type
{};

template <>
struct record_max<stack_rss> : std::true_type
{};

template <>
struct record_max<data_rss> : std::true_type
{};

template <int... EventTypes>
struct array_serialization<papi_tuple<EventTypes...>>
{
    using type = std::true_type;
};

template <std::size_t MaxNumEvents>
struct array_serialization<papi_array<MaxNumEvents>>
{
    using type = std::true_type;
};

template <>
struct array_serialization<cupti_event>
{
    using type = std::true_type;
};

//--------------------------------------------------------------------------------------//
//  disable if not enabled via preprocessor TIMEMORY_USE_PAPI
//
#if !defined(TIMEMORY_USE_PAPI)

template <int... EventTypes>
struct impl_available<papi_tuple<EventTypes...>> : std::false_type
{};

template <std::size_t MaxNumEvents>
struct impl_available<papi_array<MaxNumEvents>> : std::false_type
{};

template <typename _Tp, int... EventTypes>
struct impl_available<cpu_roofline<_Tp, EventTypes...>> : std::false_type
{};

#endif  // TIMEMORY_USE_PAPI

//--------------------------------------------------------------------------------------//
//  disable if not enabled via preprocessor TIMEMORY_USE_CUDA
//
#if !defined(TIMEMORY_USE_CUDA)

template <>
struct impl_available<cuda_event> : std::false_type
{};

#endif  // TIMEMORY_USE_CUDA

//--------------------------------------------------------------------------------------//
//  disable if not enabled via preprocessor TIMEMORY_USE_CUPTI
//
#if !defined(TIMEMORY_USE_CUPTI)

template <>
struct impl_available<cupti_event> : std::false_type
{};

#endif  // TIMEMORY_USE_CUPTI

//--------------------------------------------------------------------------------------//
}  // component
}  // tim
