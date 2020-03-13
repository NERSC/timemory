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
 * \file timemory/components/rusage/traits.hpp
 * \brief Configure the type-traits for the rusage components
 */

#pragma once

#include "timemory/components/macros.hpp"
#include "timemory/components/rusage/types.hpp"
#include "timemory/mpl/type_traits.hpp"
#include "timemory/mpl/types.hpp"

namespace tim
{
namespace resource_usage
{
namespace alias
{
using tuple_dd_t = std::tuple<double, double>;
using pair_dd_t  = std::pair<double, double>;
template <size_t N>
using farray_t = std::array<double, N>;
}  // namespace alias
}  // namespace resource_usage
}  // namespace tim

//--------------------------------------------------------------------------------------//
//
//                              STATISTICS
//
//--------------------------------------------------------------------------------------//

TIMEMORY_STATISTICS_TYPE(component::peak_rss, double)
TIMEMORY_STATISTICS_TYPE(component::page_rss, double)
TIMEMORY_STATISTICS_TYPE(component::stack_rss, double)
TIMEMORY_STATISTICS_TYPE(component::data_rss, double)
TIMEMORY_STATISTICS_TYPE(component::num_swap, int64_t)
TIMEMORY_STATISTICS_TYPE(component::num_io_in, int64_t)
TIMEMORY_STATISTICS_TYPE(component::num_io_out, int64_t)
TIMEMORY_STATISTICS_TYPE(component::num_minor_page_faults, int64_t)
TIMEMORY_STATISTICS_TYPE(component::num_major_page_faults, int64_t)
TIMEMORY_STATISTICS_TYPE(component::num_msg_sent, int64_t)
TIMEMORY_STATISTICS_TYPE(component::num_msg_recv, int64_t)
TIMEMORY_STATISTICS_TYPE(component::num_signals, int64_t)
TIMEMORY_STATISTICS_TYPE(component::voluntary_context_switch, int64_t)
TIMEMORY_STATISTICS_TYPE(component::priority_context_switch, int64_t)
TIMEMORY_STATISTICS_TYPE(component::read_bytes, resource_usage::alias::tuple_dd_t)
TIMEMORY_STATISTICS_TYPE(component::written_bytes, resource_usage::alias::farray_t<2>)
TIMEMORY_STATISTICS_TYPE(component::virtual_memory, double)
TIMEMORY_STATISTICS_TYPE(component::user_mode_time, double)
TIMEMORY_STATISTICS_TYPE(component::kernel_mode_time, double)
TIMEMORY_STATISTICS_TYPE(component::current_peak_rss, resource_usage::alias::pair_dd_t)

//--------------------------------------------------------------------------------------//
//
//                              RECORD MAX
//
//--------------------------------------------------------------------------------------//

TIMEMORY_DEFINE_CONCRETE_TRAIT(record_max, component::peak_rss, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(record_max, component::page_rss, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(record_max, component::stack_rss, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(record_max, component::data_rss, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(record_max, component::virtual_memory, true_type)

//--------------------------------------------------------------------------------------//
//
//                              REPORT SUM
//
//--------------------------------------------------------------------------------------//

TIMEMORY_DEFINE_CONCRETE_TRAIT(report_sum, component::current_peak_rss, false_type)

//--------------------------------------------------------------------------------------//
//
//                              ECHO ENABLED TRAIT
//
//--------------------------------------------------------------------------------------//

TIMEMORY_DEFINE_CONCRETE_TRAIT(echo_enabled, component::written_bytes, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(echo_enabled, component::read_bytes, false_type)

//--------------------------------------------------------------------------------------//
//
//                              FILE SAMPLER
//
//--------------------------------------------------------------------------------------//

#if defined(_LINUX) || (defined(_UNIX) && !defined(_MACOS))

TIMEMORY_DEFINE_CONCRETE_TRAIT(file_sampler, component::page_rss, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(file_sampler, component::data_rss, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(file_sampler, component::written_bytes, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(file_sampler, component::read_bytes, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(file_sampler, component::virtual_memory, true_type)

#endif

//--------------------------------------------------------------------------------------//
//
//                              CUSTOM UNIT PRINTING
//
//--------------------------------------------------------------------------------------//

TIMEMORY_DEFINE_CONCRETE_TRAIT(custom_unit_printing, component::read_bytes, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(custom_unit_printing, component::written_bytes, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(custom_unit_printing, component::current_peak_rss,
                               true_type)

//--------------------------------------------------------------------------------------//
//
//                              CUSTOM LABEL PRINTING
//
//--------------------------------------------------------------------------------------//

TIMEMORY_DEFINE_CONCRETE_TRAIT(custom_label_printing, component::read_bytes, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(custom_label_printing, component::written_bytes, true_type)

//--------------------------------------------------------------------------------------//
//
//                              ARRAY SERIALIZATION
//
//--------------------------------------------------------------------------------------//

TIMEMORY_DEFINE_CONCRETE_TRAIT(array_serialization, component::read_bytes, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(array_serialization, component::written_bytes, true_type)

//--------------------------------------------------------------------------------------//
//
//                              IS AVAILABLE
//
//--------------------------------------------------------------------------------------//

//
//      WINDOWS (non-UNIX)
//
#if !defined(_UNIX)

TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::stack_rss, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::data_rss, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::num_io_in, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::num_io_out, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::num_major_page_faults, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::num_minor_page_faults, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::num_msg_recv, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::num_msg_sent, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::num_signals, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::num_swap, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::read_bytes, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::written_bytes, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::virtual_memory, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::user_mode_time, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::kernel_mode_time, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::current_peak_rss, false_type)

#endif

//
//      UNIX
//
#if defined(UNIX)

/// \param TIMEMORY_USE_UNMAINTAINED_RUSAGE
/// \brief This macro enables the globally disable rusage structures that are
/// unmaintained by the Linux kernel and are zero on macOS
///
#    if !defined(TIMEMORY_USE_UNMAINTAINED_RUSAGE)

TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::stack_rss, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::data_rss, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::num_swap, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::num_msg_recv, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::num_msg_sent, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::num_signals, false_type)

#        if defined(_MACOS)

TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::num_io_in, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::num_io_out, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::read_bytes, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::written_bytes, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::virtual_memory, false_type)

#        endif
#    endif  // !defined(TIMEMORY_USE_UNMAINTAINED_RUSAGE)

#endif

//--------------------------------------------------------------------------------------//
//
//                              IS TIMING CATEGORY
//
//--------------------------------------------------------------------------------------//

TIMEMORY_DEFINE_CONCRETE_TRAIT(is_timing_category, component::user_mode_time, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_timing_category, component::kernel_mode_time, true_type)

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

//--------------------------------------------------------------------------------------//
//
//                              USES TIMING UNITS
//
//--------------------------------------------------------------------------------------//

TIMEMORY_DEFINE_CONCRETE_TRAIT(uses_timing_units, component::user_mode_time, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(uses_timing_units, component::kernel_mode_time, true_type)

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

//--------------------------------------------------------------------------------------//
//
//                              UNITS SPECIALIZATIONS
//
//--------------------------------------------------------------------------------------//

namespace tim
{
namespace trait
{
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
    using type         = std::array<double, 2>;
    using display_type = std::array<std::string, 2>;
};

//--------------------------------------------------------------------------------------//

}  // namespace trait
}  // namespace tim
