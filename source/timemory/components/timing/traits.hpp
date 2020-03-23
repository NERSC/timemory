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
 * \file timemory/components/timing/traits.hpp
 * \brief Configure the type-traits for the components
 */

#pragma once

#include "timemory/components/macros.hpp"
#include "timemory/components/timing/types.hpp"
#include "timemory/mpl/type_traits.hpp"
#include "timemory/mpl/types.hpp"

//--------------------------------------------------------------------------------------//
//
//                              STATISTICS
//
//--------------------------------------------------------------------------------------//

TIMEMORY_STATISTICS_TYPE(component::wall_clock, double)
TIMEMORY_STATISTICS_TYPE(component::system_clock, double)
TIMEMORY_STATISTICS_TYPE(component::user_clock, double)
TIMEMORY_STATISTICS_TYPE(component::cpu_clock, double)
TIMEMORY_STATISTICS_TYPE(component::monotonic_clock, double)
TIMEMORY_STATISTICS_TYPE(component::monotonic_raw_clock, double)
TIMEMORY_STATISTICS_TYPE(component::thread_cpu_clock, double)
TIMEMORY_STATISTICS_TYPE(component::process_cpu_clock, double)
TIMEMORY_STATISTICS_TYPE(component::cpu_util, double)
TIMEMORY_STATISTICS_TYPE(component::process_cpu_util, double)
TIMEMORY_STATISTICS_TYPE(component::thread_cpu_util, double)

//--------------------------------------------------------------------------------------//
//
//                              THREAD SCOPE ONLY
//
//--------------------------------------------------------------------------------------//

TIMEMORY_DEFINE_CONCRETE_TRAIT(thread_scope_only, component::thread_cpu_clock, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(thread_scope_only, component::thread_cpu_util, true_type)

//--------------------------------------------------------------------------------------//
//
//                              IS TIMING CATEGORY
//
//--------------------------------------------------------------------------------------//

TIMEMORY_DEFINE_CONCRETE_TRAIT(is_timing_category, component::wall_clock, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_timing_category, component::system_clock, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_timing_category, component::user_clock, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_timing_category, component::cpu_clock, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_timing_category, component::monotonic_clock, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_timing_category, component::monotonic_raw_clock,
                               true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_timing_category, component::thread_cpu_clock, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_timing_category, component::process_cpu_clock,
                               true_type)

//--------------------------------------------------------------------------------------//
//
//                              USES TIMING UNITS
//
//--------------------------------------------------------------------------------------//

TIMEMORY_DEFINE_CONCRETE_TRAIT(uses_timing_units, component::wall_clock, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(uses_timing_units, component::system_clock, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(uses_timing_units, component::user_clock, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(uses_timing_units, component::cpu_clock, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(uses_timing_units, component::monotonic_clock, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(uses_timing_units, component::monotonic_raw_clock,
                               true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(uses_timing_units, component::thread_cpu_clock, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(uses_timing_units, component::process_cpu_clock, true_type)

//--------------------------------------------------------------------------------------//
//
//                              USES PERCENT UNITS
//
//--------------------------------------------------------------------------------------//

TIMEMORY_DEFINE_CONCRETE_TRAIT(uses_percent_units, component::cpu_util, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(uses_percent_units, component::process_cpu_util, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(uses_percent_units, component::thread_cpu_util, true_type)

//--------------------------------------------------------------------------------------//
//
//                                  DERIVATION
//
//--------------------------------------------------------------------------------------//

namespace tim
{
namespace trait
{
//
//--------------------------------------------------------------------------------------//
//
template <>
struct derivation_types<component::cpu_util>
{
    using type = std::tuple<
        type_list<component::wall_clock, component::cpu_clock>,
        type_list<component::wall_clock, component::user_clock, component::system_clock>>;
};
//
//--------------------------------------------------------------------------------------//
//
template <>
struct derivation_types<component::process_cpu_util>
{
    using type =
        std::tuple<type_list<component::wall_clock, component::process_cpu_clock>>;
};
//
//--------------------------------------------------------------------------------------//
//
template <>
struct derivation_types<component::thread_cpu_util>
{
    using type =
        std::tuple<type_list<component::wall_clock, component::thread_cpu_clock>>;
};
//
//--------------------------------------------------------------------------------------//
//
}  // namespace trait
}  // namespace tim
