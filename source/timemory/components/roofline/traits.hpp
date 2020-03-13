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
 * \file timemory/components/roofline/traits.hpp
 * \brief Configure the type-traits for the roofline components
 */

#pragma once

#include "timemory/components/macros.hpp"
#include "timemory/components/roofline/types.hpp"
#include "timemory/mpl/type_traits.hpp"
#include "timemory/mpl/types.hpp"

//--------------------------------------------------------------------------------------//
//
//                              STATISTICS
//
//--------------------------------------------------------------------------------------//

TIMEMORY_VARIADIC_STATISTICS_TYPE(component::cpu_roofline, std::vector<double>, typename)
TIMEMORY_VARIADIC_STATISTICS_TYPE(component::gpu_roofline, std::vector<double>, typename)

//--------------------------------------------------------------------------------------//
//
//                              IS AVAILABLE
//
//--------------------------------------------------------------------------------------//
//
//      PAPI
//
#if !defined(TIMEMORY_USE_PAPI)
TIMEMORY_DEFINE_VARIADIC_TRAIT(is_available, component::cpu_roofline, false_type,
                               typename)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::cpu_roofline_flops, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::cpu_roofline_sp_flops, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::cpu_roofline_dp_flops, false_type)
#endif

//
//      CUDA and CUPTI
//
#if !defined(TIMEMORY_USE_CUPTI) || !defined(TIMEMORY_USE_CUDA)
TIMEMORY_DEFINE_VARIADIC_TRAIT(is_available, component::gpu_roofline, false_type,
                               typename)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::gpu_roofline_flops, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::gpu_roofline_hp_flops, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::gpu_roofline_sp_flops, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::gpu_roofline_dp_flops, false_type)
#endif

//--------------------------------------------------------------------------------------//
//
//                              CUSTOM UNIT PRINTING
//
//--------------------------------------------------------------------------------------//

TIMEMORY_DEFINE_VARIADIC_TRAIT(custom_unit_printing, component::cpu_roofline, true_type,
                               typename)
TIMEMORY_DEFINE_VARIADIC_TRAIT(custom_unit_printing, component::gpu_roofline, true_type,
                               typename)

//--------------------------------------------------------------------------------------//
//
//                              CUSTOM LABEL PRINTING
//
//--------------------------------------------------------------------------------------//

TIMEMORY_DEFINE_VARIADIC_TRAIT(custom_label_printing, component::cpu_roofline, true_type,
                               typename)
TIMEMORY_DEFINE_VARIADIC_TRAIT(custom_label_printing, component::gpu_roofline, true_type,
                               typename)

//--------------------------------------------------------------------------------------//
//
//                              ARRAY SERIALIZATION
//
//--------------------------------------------------------------------------------------//

TIMEMORY_DEFINE_VARIADIC_TRAIT(array_serialization, component::cpu_roofline, true_type,
                               typename)
// TIMEMORY_DEFINE_VARIADIC_TRAIT(array_serialization, component::gpu_roofline, true_type,
//                                typename)

//--------------------------------------------------------------------------------------//
//
//                              REQUIRES JSON
//
//--------------------------------------------------------------------------------------//

TIMEMORY_DEFINE_VARIADIC_TRAIT(requires_json, component::cpu_roofline, true_type,
                               typename)

TIMEMORY_DEFINE_VARIADIC_TRAIT(requires_json, component::gpu_roofline, true_type,
                               typename)

//--------------------------------------------------------------------------------------//
//
//                              SUPPORTS CUSTOM RECORD
//
//--------------------------------------------------------------------------------------//

TIMEMORY_DEFINE_VARIADIC_TRAIT(supports_custom_record, component::cpu_roofline, true_type,
                               typename)

TIMEMORY_DEFINE_VARIADIC_TRAIT(supports_custom_record, component::gpu_roofline, true_type,
                               typename)

//--------------------------------------------------------------------------------------//
//
//                              ITERABLE MEASUREMENT
//
//--------------------------------------------------------------------------------------//

TIMEMORY_DEFINE_VARIADIC_TRAIT(iterable_measurement, component::cpu_roofline, true_type,
                               typename)

TIMEMORY_DEFINE_VARIADIC_TRAIT(iterable_measurement, component::gpu_roofline, true_type,
                               typename)

//--------------------------------------------------------------------------------------//
//
//                              CUSTOM SERIALIZATION
//
//--------------------------------------------------------------------------------------//

TIMEMORY_DEFINE_VARIADIC_TRAIT(custom_serialization, component::cpu_roofline, true_type,
                               typename)

TIMEMORY_DEFINE_VARIADIC_TRAIT(custom_serialization, component::gpu_roofline, true_type,
                               typename)

//--------------------------------------------------------------------------------------//
//
//                              SECONDARY DATA
//
//--------------------------------------------------------------------------------------//

TIMEMORY_DEFINE_VARIADIC_TRAIT(secondary_data, component::gpu_roofline, true_type,
                               typename)

//--------------------------------------------------------------------------------------//
//
//                              UNITS SPECIALIZATIONS
//
//--------------------------------------------------------------------------------------//

namespace tim
{
namespace trait
{
//
template <typename... _Types>
struct units<component::cpu_roofline<_Types...>>
{
    using type         = double;
    using display_type = std::vector<std::string>;
};
//
}  // namespace trait
}  // namespace tim
