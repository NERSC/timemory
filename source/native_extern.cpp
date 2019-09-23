//  MIT License
//
//  Copyright (c) 2019, The Regents of the University of California,
// through Lawrence Berkeley National Laboratory (subject to receipt of any
// required approvals from the U.S. Dept. of Energy).  All rights reserved.
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to
//  deal in the Software without restriction, including without limitation the
//  rights to use, copy, modify, merge, publish, distribute, sublicense, and
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in
//  all copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
//  IN THE SOFTWARE.

/** \file native_extern.cpp
 * Generates extern templates for native components (no external linking required)
 *
 */

#define EXTERN_TEMPLATE_BUILD

#include "timemory/components/types.hpp"
#include "timemory/utility/macros.hpp"
#include "timemory/variadic/auto_hybrid.hpp"
#include "timemory/variadic/auto_list.hpp"
#include "timemory/variadic/auto_tuple.hpp"

//======================================================================================//
//
//                      C++ extern template instantiation
//
//======================================================================================//

namespace component = ::tim::component;

/*
//--------------------------------------------------------------------------------------//
// individual
//
TIMEMORY_INSTANTIATE_EXTERN_TUPLE(cpu_clock_t, ::tim::component::cpu_clock)
TIMEMORY_INSTANTIATE_EXTERN_TUPLE(cpu_util_t, ::tim::component::cpu_util)
TIMEMORY_INSTANTIATE_EXTERN_TUPLE(data_rss_t, ::tim::component::data_rss)
TIMEMORY_INSTANTIATE_EXTERN_TUPLE(monotonic_clock_t, ::tim::component::monotonic_clock)
TIMEMORY_INSTANTIATE_EXTERN_TUPLE(monotonic_raw_clock_t,
                                  ::tim::component::monotonic_raw_clock)
TIMEMORY_INSTANTIATE_EXTERN_TUPLE(num_io_in_t, ::tim::component::num_io_in)
TIMEMORY_INSTANTIATE_EXTERN_TUPLE(num_io_out_t, ::tim::component::num_io_out)
TIMEMORY_INSTANTIATE_EXTERN_TUPLE(num_major_page_faults_t,
                                  ::tim::component::num_major_page_faults)
TIMEMORY_INSTANTIATE_EXTERN_TUPLE(num_minor_page_faults_t,
                                  ::tim::component::num_minor_page_faults)
TIMEMORY_INSTANTIATE_EXTERN_TUPLE(num_msg_recv_t, ::tim::component::num_msg_recv)
TIMEMORY_INSTANTIATE_EXTERN_TUPLE(num_msg_sent_t, ::tim::component::num_msg_sent)
TIMEMORY_INSTANTIATE_EXTERN_TUPLE(num_signals_t, ::tim::component::num_signals)
TIMEMORY_INSTANTIATE_EXTERN_TUPLE(num_swap_t, ::tim::component::num_swap)
TIMEMORY_INSTANTIATE_EXTERN_TUPLE(page_rss_t, ::tim::component::page_rss)
TIMEMORY_INSTANTIATE_EXTERN_TUPLE(peak_rss_t, ::tim::component::peak_rss)
TIMEMORY_INSTANTIATE_EXTERN_TUPLE(priority_context_switch_t,
                                  ::tim::component::priority_context_switch)
TIMEMORY_INSTANTIATE_EXTERN_TUPLE(process_cpu_clock_t,
                                  ::tim::component::process_cpu_clock)
TIMEMORY_INSTANTIATE_EXTERN_TUPLE(process_cpu_util_t, ::tim::component::process_cpu_util)
TIMEMORY_INSTANTIATE_EXTERN_TUPLE(read_bytes_t, ::tim::component::read_bytes)
TIMEMORY_INSTANTIATE_EXTERN_TUPLE(real_clock_t, ::tim::component::real_clock)
TIMEMORY_INSTANTIATE_EXTERN_TUPLE(stack_rss_t, ::tim::component::stack_rss)
TIMEMORY_INSTANTIATE_EXTERN_TUPLE(system_clock_t, ::tim::component::system_clock)
TIMEMORY_INSTANTIATE_EXTERN_TUPLE(thread_cpu_clock_t, ::tim::component::thread_cpu_clock)
TIMEMORY_INSTANTIATE_EXTERN_TUPLE(thread_cpu_util_t, ::tim::component::thread_cpu_util)
TIMEMORY_INSTANTIATE_EXTERN_TUPLE(trip_count_t, ::tim::component::trip_count)
TIMEMORY_INSTANTIATE_EXTERN_TUPLE(user_clock_t, ::tim::component::user_clock)
TIMEMORY_INSTANTIATE_EXTERN_TUPLE(voluntary_context_switch_t,
                                  ::tim::component::voluntary_context_switch)
TIMEMORY_INSTANTIATE_EXTERN_TUPLE(written_bytes_t, ::tim::component::written_bytes)

TIMEMORY_INSTANTIATE_EXTERN_LIST(cpu_clock_t, ::tim::component::cpu_clock)
TIMEMORY_INSTANTIATE_EXTERN_LIST(cpu_util_t, ::tim::component::cpu_util)
TIMEMORY_INSTANTIATE_EXTERN_LIST(data_rss_t, ::tim::component::data_rss)
TIMEMORY_INSTANTIATE_EXTERN_LIST(monotonic_clock_t, ::tim::component::monotonic_clock)
TIMEMORY_INSTANTIATE_EXTERN_LIST(monotonic_raw_clock_t,
                                 ::tim::component::monotonic_raw_clock)
TIMEMORY_INSTANTIATE_EXTERN_LIST(num_io_in_t, ::tim::component::num_io_in)
TIMEMORY_INSTANTIATE_EXTERN_LIST(num_io_out_t, ::tim::component::num_io_out)
TIMEMORY_INSTANTIATE_EXTERN_LIST(num_major_page_faults_t,
                                 ::tim::component::num_major_page_faults)
TIMEMORY_INSTANTIATE_EXTERN_LIST(num_minor_page_faults_t,
                                 ::tim::component::num_minor_page_faults)
TIMEMORY_INSTANTIATE_EXTERN_LIST(num_msg_recv_t, ::tim::component::num_msg_recv)
TIMEMORY_INSTANTIATE_EXTERN_LIST(num_msg_sent_t, ::tim::component::num_msg_sent)
TIMEMORY_INSTANTIATE_EXTERN_LIST(num_signals_t, ::tim::component::num_signals)
TIMEMORY_INSTANTIATE_EXTERN_LIST(num_swap_t, ::tim::component::num_swap)
TIMEMORY_INSTANTIATE_EXTERN_LIST(page_rss_t, ::tim::component::page_rss)
TIMEMORY_INSTANTIATE_EXTERN_LIST(peak_rss_t, ::tim::component::peak_rss)
TIMEMORY_INSTANTIATE_EXTERN_LIST(priority_context_switch_t,
                                 ::tim::component::priority_context_switch)
TIMEMORY_INSTANTIATE_EXTERN_LIST(process_cpu_clock_t, ::tim::component::process_cpu_clock)
TIMEMORY_INSTANTIATE_EXTERN_LIST(process_cpu_util_t, ::tim::component::process_cpu_util)
TIMEMORY_INSTANTIATE_EXTERN_LIST(read_bytes_t, ::tim::component::read_bytes)
TIMEMORY_INSTANTIATE_EXTERN_LIST(real_clock_t, ::tim::component::real_clock)
TIMEMORY_INSTANTIATE_EXTERN_LIST(stack_rss_t, ::tim::component::stack_rss)
TIMEMORY_INSTANTIATE_EXTERN_LIST(system_clock_t, ::tim::component::system_clock)
TIMEMORY_INSTANTIATE_EXTERN_LIST(thread_cpu_clock_t, ::tim::component::thread_cpu_clock)
TIMEMORY_INSTANTIATE_EXTERN_LIST(thread_cpu_util_t, ::tim::component::thread_cpu_util)
TIMEMORY_INSTANTIATE_EXTERN_LIST(trip_count_t, ::tim::component::trip_count)
TIMEMORY_INSTANTIATE_EXTERN_LIST(user_clock_t, ::tim::component::user_clock)
TIMEMORY_INSTANTIATE_EXTERN_LIST(voluntary_context_switch_t,
                                 ::tim::component::voluntary_context_switch)
TIMEMORY_INSTANTIATE_EXTERN_LIST(written_bytes_t, ::tim::component::written_bytes)

TIMEMORY_INSTANTIATE_EXTERN_HYBRID(cpu_clock_t)
TIMEMORY_INSTANTIATE_EXTERN_HYBRID(cpu_util_t)
TIMEMORY_INSTANTIATE_EXTERN_HYBRID(data_rss_t)
TIMEMORY_INSTANTIATE_EXTERN_HYBRID(monotonic_clock_t)
TIMEMORY_INSTANTIATE_EXTERN_HYBRID(monotonic_raw_clock_t)
TIMEMORY_INSTANTIATE_EXTERN_HYBRID(num_io_in_t)
TIMEMORY_INSTANTIATE_EXTERN_HYBRID(num_io_out_t)
TIMEMORY_INSTANTIATE_EXTERN_HYBRID(num_major_page_faults_t)
TIMEMORY_INSTANTIATE_EXTERN_HYBRID(num_minor_page_faults_t)
TIMEMORY_INSTANTIATE_EXTERN_HYBRID(num_msg_recv_t)
TIMEMORY_INSTANTIATE_EXTERN_HYBRID(num_msg_sent_t)
TIMEMORY_INSTANTIATE_EXTERN_HYBRID(num_signals_t)
TIMEMORY_INSTANTIATE_EXTERN_HYBRID(num_swap_t)
TIMEMORY_INSTANTIATE_EXTERN_HYBRID(page_rss_t)
TIMEMORY_INSTANTIATE_EXTERN_HYBRID(peak_rss_t)
TIMEMORY_INSTANTIATE_EXTERN_HYBRID(priority_context_switch_t)
TIMEMORY_INSTANTIATE_EXTERN_HYBRID(process_cpu_clock_t)
TIMEMORY_INSTANTIATE_EXTERN_HYBRID(process_cpu_util_t)
TIMEMORY_INSTANTIATE_EXTERN_HYBRID(read_bytes_t)
TIMEMORY_INSTANTIATE_EXTERN_HYBRID(real_clock_t)
TIMEMORY_INSTANTIATE_EXTERN_HYBRID(stack_rss_t)
TIMEMORY_INSTANTIATE_EXTERN_HYBRID(system_clock_t)
TIMEMORY_INSTANTIATE_EXTERN_HYBRID(thread_cpu_clock_t)
TIMEMORY_INSTANTIATE_EXTERN_HYBRID(thread_cpu_util_t)
TIMEMORY_INSTANTIATE_EXTERN_HYBRID(trip_count_t)
TIMEMORY_INSTANTIATE_EXTERN_HYBRID(user_clock_t)
TIMEMORY_INSTANTIATE_EXTERN_HYBRID(voluntary_context_switch_t)
TIMEMORY_INSTANTIATE_EXTERN_HYBRID(written_bytes_t)
*/

// clang-format off
namespace
{
// designed to provide notify and not warn about unused-{parameter,function}
// declare func
static bool
native_extern_symbol();
// variable that defaults to func
static bool _native_extern_symbol = native_extern_symbol();
// impl of func consumes variable
static bool
native_extern_symbol() { tim::consume_parameters(_native_extern_symbol); return false; }
}  // namespace
