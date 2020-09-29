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

/**
 * \file timemory/components/papi/types.hpp
 * \brief Declare the papi component types
 */

#pragma once

#include "timemory/backends/types/papi.hpp"
#include "timemory/components/macros.hpp"
#include "timemory/enum.h"
#include "timemory/mpl/type_traits.hpp"
#include "timemory/mpl/types.hpp"

#if !defined(TIMEMORY_PAPI_ARRAY_SIZE)
#    define TIMEMORY_PAPI_ARRAY_SIZE 8
#endif

//======================================================================================//
//
TIMEMORY_DECLARE_COMPONENT(papi_vector)
TIMEMORY_DECLARE_TEMPLATE_COMPONENT(papi_tuple, int... EventTypes)
TIMEMORY_DECLARE_TEMPLATE_COMPONENT(papi_array, size_t MaxNumEvents)
TIMEMORY_DECLARE_TEMPLATE_COMPONENT(papi_rate_tuple, int... EventTypes)
//
TIMEMORY_COMPONENT_ALIAS(papi_array8_t, papi_array<8>)
TIMEMORY_COMPONENT_ALIAS(papi_array16_t, papi_array<16>)
TIMEMORY_COMPONENT_ALIAS(papi_array32_t, papi_array<32>)
TIMEMORY_COMPONENT_ALIAS(papi_array_t, papi_array<TIMEMORY_PAPI_ARRAY_SIZE>)
//
//======================================================================================//
//
//                              STATISTICS
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_STATISTICS_TYPE(component::papi_vector, std::vector<double>)
TIMEMORY_TEMPLATE_STATISTICS_TYPE(component::papi_array, std::vector<double>, size_t)
// TIMEMORY_VARIADIC_STATISTICS_TYPE(component::papi_tuple, std::vector<double>, int)
namespace tim
{
namespace trait
{
template <int... Idx>
struct statistics<component::papi_tuple<Idx...>>
{
    using type = std::array<double, sizeof...(Idx)>;
};
//
template <int... Idx>
struct statistics<component::papi_rate_tuple<Idx...>>
{
    using type = std::array<double, sizeof...(Idx)>;
};
}  // namespace trait
}  // namespace tim
//
//--------------------------------------------------------------------------------------//
//
//                              IS AVAILABLE
//
//--------------------------------------------------------------------------------------//
//
#if !defined(TIMEMORY_USE_PAPI)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::papi_vector, false_type)
TIMEMORY_DEFINE_TEMPLATE_TRAIT(is_available, component::papi_array, false_type, size_t)
TIMEMORY_DEFINE_VARIADIC_TRAIT(is_available, component::papi_tuple, false_type, int)
TIMEMORY_DEFINE_VARIADIC_TRAIT(is_available, component::papi_rate_tuple, false_type, int)
#endif
//
//--------------------------------------------------------------------------------------//
//
//                              ARRAY SERIALIZATION
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_DEFINE_CONCRETE_TRAIT(array_serialization, component::papi_vector, true_type)
TIMEMORY_DEFINE_TEMPLATE_TRAIT(array_serialization, component::papi_array, true_type,
                               size_t)
TIMEMORY_DEFINE_VARIADIC_TRAIT(array_serialization, component::papi_tuple, true_type, int)
TIMEMORY_DEFINE_VARIADIC_TRAIT(array_serialization, component::papi_rate_tuple, true_type,
                               int)
//
//--------------------------------------------------------------------------------------//
//
//                              CUSTOM SERIALIZATION
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_DEFINE_CONCRETE_TRAIT(custom_serialization, component::papi_vector, true_type)
TIMEMORY_DEFINE_TEMPLATE_TRAIT(custom_serialization, component::papi_array, true_type,
                               size_t)
TIMEMORY_DEFINE_VARIADIC_TRAIT(custom_serialization, component::papi_tuple, true_type,
                               int)
//
//--------------------------------------------------------------------------------------//
//
//                              SAMPLER
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_DEFINE_CONCRETE_TRAIT(sampler, component::papi_vector, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(sampler, component::papi_array_t, true_type)
TIMEMORY_DEFINE_VARIADIC_TRAIT(sampler, component::papi_tuple, true_type, int)
//
//--------------------------------------------------------------------------------------//
//
//                              BASE HAS ACCUM
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_DEFINE_VARIADIC_TRAIT(base_has_accum, component::papi_rate_tuple, false_type,
                               int)
//
//--------------------------------------------------------------------------------------//
//
//                              PROPERTIES
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_PROPERTY_SPECIALIZATION(papi_vector, PAPI_VECTOR, "papi_vector", "papi")
TIMEMORY_PROPERTY_SPECIALIZATION(papi_array_t, PAPI_ARRAY, "papi_array_t", "papi_array")
