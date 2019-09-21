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

/** \file init.cpp
 * This file defined the extern init
 *
 */

#include "timemory/components.hpp"
#include "timemory/manager.hpp"
#include "timemory/utility/macros.hpp"
#include "timemory/utility/serializer.hpp"
#include "timemory/utility/singleton.hpp"
#include "timemory/utility/utility.hpp"

using namespace tim::component;

//======================================================================================//
#if defined(TIMEMORY_EXTERN_INIT)
namespace tim
{
std::atomic<int32_t>&
manager::f_manager_instance_count()
{
    static std::atomic<int32_t> instance;
    return instance;
}

//======================================================================================//
// get either master or thread-local instance
//
manager::pointer
manager::instance()
{
    return details::manager_singleton().instance();
}

//======================================================================================//
// get master instance
//
manager::pointer
manager::master_instance()
{
    return details::manager_singleton().master_instance();
}

//======================================================================================//
// static function
manager::pointer
manager::noninit_instance()
{
    return details::manager_singleton().instance_ptr();
}

//======================================================================================//
// static function
manager::pointer
manager::noninit_master_instance()
{
    return details::manager_singleton().master_instance_ptr();
}

}  // namespace tim

TIMEMORY_INSTANTIATE_EXTERN_STORAGE(caliper)
TIMEMORY_INSTANTIATE_EXTERN_STORAGE(cpu_clock)
TIMEMORY_INSTANTIATE_EXTERN_STORAGE(cpu_roofline_dp_flops)
TIMEMORY_INSTANTIATE_EXTERN_STORAGE(cpu_roofline_flops)
TIMEMORY_INSTANTIATE_EXTERN_STORAGE(cpu_roofline_sp_flops)
TIMEMORY_INSTANTIATE_EXTERN_STORAGE(cpu_util)
TIMEMORY_INSTANTIATE_EXTERN_STORAGE(cuda_event)
TIMEMORY_INSTANTIATE_EXTERN_STORAGE(cupti_activity)
TIMEMORY_INSTANTIATE_EXTERN_STORAGE(cupti_counters)
TIMEMORY_INSTANTIATE_EXTERN_STORAGE(page_rss)
TIMEMORY_INSTANTIATE_EXTERN_STORAGE(data_rss)
TIMEMORY_INSTANTIATE_EXTERN_STORAGE(gpu_roofline_dp_flops)
TIMEMORY_INSTANTIATE_EXTERN_STORAGE(gpu_roofline_flops)
TIMEMORY_INSTANTIATE_EXTERN_STORAGE(gpu_roofline_hp_flops)
TIMEMORY_INSTANTIATE_EXTERN_STORAGE(gpu_roofline_sp_flops)
TIMEMORY_INSTANTIATE_EXTERN_STORAGE(monotonic_clock)
TIMEMORY_INSTANTIATE_EXTERN_STORAGE(monotonic_raw_clock)
TIMEMORY_INSTANTIATE_EXTERN_STORAGE(num_io_in)
TIMEMORY_INSTANTIATE_EXTERN_STORAGE(num_io_out)
TIMEMORY_INSTANTIATE_EXTERN_STORAGE(num_major_page_faults)
TIMEMORY_INSTANTIATE_EXTERN_STORAGE(num_minor_page_faults)
TIMEMORY_INSTANTIATE_EXTERN_STORAGE(num_msg_recv)
TIMEMORY_INSTANTIATE_EXTERN_STORAGE(num_msg_sent)
TIMEMORY_INSTANTIATE_EXTERN_STORAGE(num_signals)
TIMEMORY_INSTANTIATE_EXTERN_STORAGE(num_swap)
TIMEMORY_INSTANTIATE_EXTERN_STORAGE(nvtx_marker)
TIMEMORY_INSTANTIATE_EXTERN_STORAGE(papi_array_t)
TIMEMORY_INSTANTIATE_EXTERN_STORAGE(peak_rss)
TIMEMORY_INSTANTIATE_EXTERN_STORAGE(priority_context_switch)
TIMEMORY_INSTANTIATE_EXTERN_STORAGE(process_cpu_clock)
TIMEMORY_INSTANTIATE_EXTERN_STORAGE(process_cpu_util)
TIMEMORY_INSTANTIATE_EXTERN_STORAGE(read_bytes)
TIMEMORY_INSTANTIATE_EXTERN_STORAGE(real_clock)
TIMEMORY_INSTANTIATE_EXTERN_STORAGE(stack_rss)
TIMEMORY_INSTANTIATE_EXTERN_STORAGE(system_clock)
TIMEMORY_INSTANTIATE_EXTERN_STORAGE(thread_cpu_clock)
TIMEMORY_INSTANTIATE_EXTERN_STORAGE(thread_cpu_util)
TIMEMORY_INSTANTIATE_EXTERN_STORAGE(trip_count)
TIMEMORY_INSTANTIATE_EXTERN_STORAGE(user_clock)
TIMEMORY_INSTANTIATE_EXTERN_STORAGE(voluntary_context_switch)
TIMEMORY_INSTANTIATE_EXTERN_STORAGE(written_bytes)

#endif  // defined(TIMEMORY_EXTERN_INIT
