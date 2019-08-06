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

/** \file cuda_extern.hpp
 * \headerfile cuda_extern.hpp "timemory/templates/cuda_extern.hpp"
 * Extern template declarations that include CUDA
 *
 */

#pragma once

#include "timemory/components.hpp"
#include "timemory/utility/macros.hpp"

//--------------------------------------------------------------------------------------//
// individual
//
TIMEMORY_DECLARE_EXTERN_TUPLE(tim::component::cuda_event)

//--------------------------------------------------------------------------------------//
//  standard configurations
//

// standard_timing_t
TIMEMORY_DECLARE_EXTERN_TUPLE(tim::component::cuda_event, tim::component::real_clock,
                              tim::component::user_clock, tim::component::system_clock,
                              tim::component::cpu_clock, tim::component::cpu_util)

//--------------------------------------------------------------------------------------//
// auto_timer_t
//
TIMEMORY_DECLARE_EXTERN_TUPLE(tim::component::cuda_event, tim::component::real_clock,
                              tim::component::system_clock, tim::component::user_clock,
                              tim::component::cpu_clock, tim::component::cpu_util)

TIMEMORY_DECLARE_EXTERN_TUPLE(tim::component::cuda_event, tim::component::real_clock,
                              tim::component::system_clock, tim::component::user_clock,
                              tim::component::cpu_clock, tim::component::cpu_util,
                              tim::component::current_rss, tim::component::peak_rss)

TIMEMORY_DECLARE_EXTERN_TUPLE(tim::component::cuda_event, tim::component::real_clock,
                              tim::component::system_clock, tim::component::cpu_clock,
                              tim::component::cpu_util, tim::component::current_rss,
                              tim::component::peak_rss)

//--------------------------------------------------------------------------------------//
// all_tuple_t
//

TIMEMORY_DECLARE_EXTERN_TUPLE(
    tim::component::cuda_event, tim::component::real_clock, tim::component::system_clock,
    tim::component::user_clock, tim::component::cpu_clock,
    tim::component::monotonic_clock, tim::component::monotonic_raw_clock,
    tim::component::thread_cpu_clock, tim::component::process_cpu_clock,
    tim::component::cpu_util, tim::component::thread_cpu_util,
    tim::component::process_cpu_util, tim::component::current_rss,
    tim::component::peak_rss, tim::component::stack_rss, tim::component::data_rss,
    tim::component::num_swap, tim::component::num_io_in, tim::component::num_io_out,
    tim::component::num_minor_page_faults, tim::component::num_major_page_faults,
    tim::component::num_msg_sent, tim::component::num_msg_recv,
    tim::component::num_signals, tim::component::voluntary_context_switch,
    tim::component::priority_context_switch)

//--------------------------------------------------------------------------------------//
// miscellaneous
//

TIMEMORY_DECLARE_EXTERN_TUPLE(tim::component::cuda_event, tim::component::real_clock,
                              tim::component::cpu_clock)

TIMEMORY_DECLARE_EXTERN_TUPLE(tim::component::cuda_event, tim::component::real_clock,
                              tim::component::cpu_clock, tim::component::cpu_util)

TIMEMORY_DECLARE_EXTERN_TUPLE(tim::component::cuda_event, tim::component::real_clock,
                              tim::component::system_clock, tim::component::user_clock)

TIMEMORY_DECLARE_EXTERN_TUPLE(tim::component::cuda_event, tim::component::real_clock,
                              tim::component::system_clock, tim::component::cpu_clock,
                              tim::component::cpu_util, tim::component::peak_rss,
                              tim::component::current_rss)

TIMEMORY_DECLARE_EXTERN_TUPLE(tim::component::cuda_event, tim::component::real_clock,
                              tim::component::system_clock,
                              tim::component::thread_cpu_clock,
                              tim::component::thread_cpu_util,
                              tim::component::process_cpu_clock,
                              tim::component::process_cpu_util, tim::component::peak_rss,
                              tim::component::current_rss)

//======================================================================================//
