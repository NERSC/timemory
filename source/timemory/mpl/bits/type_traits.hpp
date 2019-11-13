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

/** \file mpl/bits/type_traits.hpp
 * \headerfile mpl/bits/type_traits.hpp "timemory/mpl/bits/type_traits.hpp"
 * This provides the generated type-traits for various types
 *
 */

#pragma once

#include "timemory/components/types.hpp"
#include "timemory/mpl/type_traits.hpp"

namespace tim
{
namespace trait
{
//--------------------------------------------------------------------------------------//
//
//                              IS TIMING CATEGORY
//
//--------------------------------------------------------------------------------------//

template <>
struct is_timing_category<component::real_clock> : std::true_type
{};

template <>
struct is_timing_category<component::system_clock> : std::true_type
{};

template <>
struct is_timing_category<component::user_clock> : std::true_type
{};

template <>
struct is_timing_category<component::cpu_clock> : std::true_type
{};

template <>
struct is_timing_category<component::monotonic_clock> : std::true_type
{};

template <>
struct is_timing_category<component::monotonic_raw_clock> : std::true_type
{};

template <>
struct is_timing_category<component::thread_cpu_clock> : std::true_type
{};

template <>
struct is_timing_category<component::process_cpu_clock> : std::true_type
{};

template <>
struct is_timing_category<component::cuda_event> : std::true_type
{};

template <>
struct is_timing_category<component::cupti_activity> : std::true_type
{};

//--------------------------------------------------------------------------------------//
//
//                              IS MEMORY CATEGORY
//
//--------------------------------------------------------------------------------------//

template <>
struct is_memory_category<component::peak_rss> : std::true_type
{};

template <>
struct is_memory_category<component::page_rss> : std::true_type
{};

template <>
struct is_memory_category<component::stack_rss> : std::true_type
{};

template <>
struct is_memory_category<component::data_rss> : std::true_type
{};

template <>
struct is_memory_category<component::num_swap> : std::true_type
{};

template <>
struct is_memory_category<component::num_io_in> : std::true_type
{};

template <>
struct is_memory_category<component::num_io_out> : std::true_type
{};

template <>
struct is_memory_category<component::num_minor_page_faults> : std::true_type
{};

template <>
struct is_memory_category<component::num_major_page_faults> : std::true_type
{};

template <>
struct is_memory_category<component::num_msg_sent> : std::true_type
{};

template <>
struct is_memory_category<component::num_msg_recv> : std::true_type
{};

template <>
struct is_memory_category<component::num_signals> : std::true_type
{};

template <>
struct is_memory_category<component::voluntary_context_switch> : std::true_type
{};

template <>
struct is_memory_category<component::priority_context_switch> : std::true_type
{};

template <>
struct is_memory_category<component::read_bytes> : std::true_type
{};

template <>
struct is_memory_category<component::written_bytes> : std::true_type
{};

template <>
struct is_memory_category<component::virtual_memory> : std::true_type
{};

//--------------------------------------------------------------------------------------//
//
//                              USES TIMING UNITS
//
//--------------------------------------------------------------------------------------//

template <>
struct uses_timing_units<component::real_clock> : std::true_type
{};

template <>
struct uses_timing_units<component::system_clock> : std::true_type
{};

template <>
struct uses_timing_units<component::user_clock> : std::true_type
{};

template <>
struct uses_timing_units<component::cpu_clock> : std::true_type
{};

template <>
struct uses_timing_units<component::monotonic_clock> : std::true_type
{};

template <>
struct uses_timing_units<component::monotonic_raw_clock> : std::true_type
{};

template <>
struct uses_timing_units<component::thread_cpu_clock> : std::true_type
{};

template <>
struct uses_timing_units<component::process_cpu_clock> : std::true_type
{};

template <>
struct uses_timing_units<component::cuda_event> : std::true_type
{};

template <>
struct uses_timing_units<component::cupti_activity> : std::true_type
{};

//--------------------------------------------------------------------------------------//
//
//                              USES MEMORY UNITS
//
//--------------------------------------------------------------------------------------//

template <>
struct uses_memory_units<component::peak_rss> : std::true_type
{};

template <>
struct uses_memory_units<component::page_rss> : std::true_type
{};

template <>
struct uses_memory_units<component::stack_rss> : std::true_type
{};

template <>
struct uses_memory_units<component::data_rss> : std::true_type
{};

template <>
struct uses_memory_units<component::read_bytes> : std::true_type
{};

template <>
struct uses_memory_units<component::written_bytes> : std::true_type
{};

template <>
struct uses_memory_units<component::virtual_memory> : std::true_type
{};

//--------------------------------------------------------------------------------------//
//
//                              USES PERCENT UNITS
//
//--------------------------------------------------------------------------------------//

template <>
struct uses_percent_units<component::cpu_util> : std::true_type
{};

template <>
struct uses_percent_units<component::process_cpu_util> : std::true_type
{};

template <>
struct uses_percent_units<component::thread_cpu_util> : std::true_type
{};

}  // namespace trait
}  // namespace tim
