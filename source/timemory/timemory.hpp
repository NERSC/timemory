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

/** \file timemory.hpp
 * \headerfile timemory.hpp "timemory/timemory.hpp"
 * All-inclusive timemory header
 *
 */

#pragma once

#include "timemory/components.hpp"
#include "timemory/manager.hpp"
#include "timemory/settings.hpp"
#include "timemory/units.hpp"
#include "timemory/utility/macros.hpp"
#include "timemory/utility/utility.hpp"
#include "timemory/variadic/auto_hybrid.hpp"
#include "timemory/variadic/auto_list.hpp"
#include "timemory/variadic/auto_timer.hpp"
#include "timemory/variadic/macros.hpp"

#include "timemory/ctimemory.h"

//======================================================================================//

namespace tim
{
using complete_auto_list_t =
    auto_list<component::caliper, component::cpu_clock, component::cpu_roofline_dp_flops,
              component::cpu_roofline_sp_flops, component::cpu_util,
              component::cuda_event, component::cupti_event, component::current_rss,
              component::data_rss, component::monotonic_clock,
              component::monotonic_raw_clock, component::num_io_in, component::num_io_out,
              component::num_major_page_faults, component::num_minor_page_faults,
              component::num_msg_recv, component::num_msg_sent, component::num_signals,
              component::num_swap, component::papi_array_t, component::peak_rss,
              component::priority_context_switch, component::process_cpu_clock,
              component::process_cpu_util, component::read_bytes, component::real_clock,
              component::stack_rss, component::system_clock, component::thread_cpu_clock,
              component::thread_cpu_util, component::trip_count, component::user_clock,
              component::voluntary_context_switch, component::written_bytes>;

using complete_list_t = complete_auto_list_t::component_type;

using recommended_auto_tuple_t =
    auto_tuple<component::real_clock, component::system_clock, component::user_clock,
               component::cpu_util, component::current_rss, component::peak_rss,
               component::read_bytes, component::written_bytes,
               component::num_minor_page_faults, component::num_major_page_faults,
               component::voluntary_context_switch, component::priority_context_switch>;

using recommended_tuple_t = recommended_auto_tuple_t::component_type;

using recommended_auto_list_t = auto_list<component::caliper, component::papi_array_t,
                                          component::cuda_event, component::cupti_event>;

using recommended_list_t = recommended_auto_list_t::component_type;

using recommended_auto_hybrid_t = auto_hybrid<recommended_tuple_t, recommended_list_t>;

using recommended_hybrid_t = component_hybrid<recommended_tuple_t, recommended_list_t>;
}

//======================================================================================//

#if defined(TIMEMORY_EXTERN_TEMPLATES)
#    include "timemory/templates/native_extern.hpp"
#endif

//======================================================================================//

#if defined(TIMEMORY_EXTERN_TEMPLATES) && defined(TIMEMORY_USE_CUDA)
// not yet implemented
// #    include "timemory/templates/cuda_extern.hpp"
#endif

//======================================================================================//

#if defined(TIMEMORY_EXTERN_INIT)
#    include "timemory/utility/storage.hpp"

TIMEMORY_DECLARE_EXTERN_STORAGE(real_clock)
TIMEMORY_DECLARE_EXTERN_STORAGE(system_clock)
TIMEMORY_DECLARE_EXTERN_STORAGE(user_clock)
TIMEMORY_DECLARE_EXTERN_STORAGE(cpu_clock)
TIMEMORY_DECLARE_EXTERN_STORAGE(monotonic_clock)
TIMEMORY_DECLARE_EXTERN_STORAGE(monotonic_raw_clock)
TIMEMORY_DECLARE_EXTERN_STORAGE(thread_cpu_clock)
TIMEMORY_DECLARE_EXTERN_STORAGE(process_cpu_clock)
TIMEMORY_DECLARE_EXTERN_STORAGE(cpu_util)
TIMEMORY_DECLARE_EXTERN_STORAGE(thread_cpu_util)
TIMEMORY_DECLARE_EXTERN_STORAGE(process_cpu_util)
TIMEMORY_DECLARE_EXTERN_STORAGE(current_rss)
TIMEMORY_DECLARE_EXTERN_STORAGE(peak_rss)
TIMEMORY_DECLARE_EXTERN_STORAGE(stack_rss)
TIMEMORY_DECLARE_EXTERN_STORAGE(data_rss)
TIMEMORY_DECLARE_EXTERN_STORAGE(num_swap)
TIMEMORY_DECLARE_EXTERN_STORAGE(num_io_in)
TIMEMORY_DECLARE_EXTERN_STORAGE(num_io_out)
TIMEMORY_DECLARE_EXTERN_STORAGE(num_minor_page_faults)
TIMEMORY_DECLARE_EXTERN_STORAGE(num_major_page_faults)
TIMEMORY_DECLARE_EXTERN_STORAGE(num_msg_sent)
TIMEMORY_DECLARE_EXTERN_STORAGE(num_msg_recv)
TIMEMORY_DECLARE_EXTERN_STORAGE(num_signals)
TIMEMORY_DECLARE_EXTERN_STORAGE(voluntary_context_switch)
TIMEMORY_DECLARE_EXTERN_STORAGE(priority_context_switch)

#endif

//======================================================================================//
