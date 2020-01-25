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

/** \file mpl/bits/type_traits.hpp
 * \headerfile mpl/bits/type_traits.hpp "timemory/mpl/bits/type_traits.hpp"
 * This provides the generated type-traits for various types
 *
 */

#pragma once

#include "timemory/components/types.hpp"
#include "timemory/mpl/type_traits.hpp"

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
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_timing_category, component::user_mode_time, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_timing_category, component::kernel_mode_time, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_timing_category, component::cuda_event, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_timing_category, component::cupti_activity, true_type)

//--------------------------------------------------------------------------------------//
//
//                              IS MEMORY CATEGORY
//
//--------------------------------------------------------------------------------------//

TIMEMORY_DEFINE_CONCRETE_TRAIT(is_memory_category, component::peak_rss, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_memory_category, component::page_rss, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_memory_category, component::stack_rss, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_memory_category, component::data_rss, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_memory_category, component::num_swap, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_memory_category, component::num_io_in, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_memory_category, component::num_io_out, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_memory_category, component::num_minor_page_faults,
                               true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_memory_category, component::num_major_page_faults,
                               true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_memory_category, component::num_msg_sent, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_memory_category, component::num_msg_recv, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_memory_category, component::num_signals, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_memory_category, component::voluntary_context_switch,
                               true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_memory_category, component::priority_context_switch,
                               true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_memory_category, component::read_bytes, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_memory_category, component::written_bytes, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_memory_category, component::virtual_memory, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_memory_category, component::malloc_gotcha, true_type)

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
TIMEMORY_DEFINE_CONCRETE_TRAIT(uses_timing_units, component::user_mode_time, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(uses_timing_units, component::kernel_mode_time, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(uses_timing_units, component::cuda_event, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(uses_timing_units, component::cupti_activity, true_type)

//--------------------------------------------------------------------------------------//
//
//                              USES MEMORY UNITS
//
//--------------------------------------------------------------------------------------//

TIMEMORY_DEFINE_CONCRETE_TRAIT(uses_memory_units, component::peak_rss, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(uses_memory_units, component::page_rss, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(uses_memory_units, component::stack_rss, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(uses_memory_units, component::data_rss, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(uses_memory_units, component::read_bytes, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(uses_memory_units, component::written_bytes, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(uses_memory_units, component::virtual_memory, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(uses_memory_units, component::malloc_gotcha, true_type)

//--------------------------------------------------------------------------------------//
//
//                              USES PERCENT UNITS
//
//--------------------------------------------------------------------------------------//

TIMEMORY_DEFINE_CONCRETE_TRAIT(uses_percent_units, component::cpu_util, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(uses_percent_units, component::process_cpu_util, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(uses_percent_units, component::thread_cpu_util, true_type)

namespace tim
{
namespace trait
{
//--------------------------------------------------------------------------------------//
//
//                              UNITS SPECIALIZATIONS
//
//--------------------------------------------------------------------------------------//

template <>
struct units<component::current_peak_rss>
{
    using type         = std::pair<double, double>;
    using display_type = std::pair<std::string, std::string>;
};

//--------------------------------------------------------------------------------------//

template <>
struct units<component::read_bytes>
{
    using type         = std::tuple<double, double>;
    using display_type = std::tuple<std::string, std::string>;
};

//--------------------------------------------------------------------------------------//

template <>
struct units<component::written_bytes>
{
    using type         = std::tuple<double, double>;
    using display_type = std::tuple<std::string, std::string>;
};

//--------------------------------------------------------------------------------------//

template <typename... _Types>
struct units<component::cpu_roofline<_Types...>>
{
    using type         = double;
    using display_type = std::vector<std::string>;
};

//--------------------------------------------------------------------------------------//
/*
template <>
struct units<component::cupti_counters>
{
    using type         = std::vector<double>;
    using display_type = std::string;
};
*/
//--------------------------------------------------------------------------------------//

}  // namespace trait
}  // namespace tim
