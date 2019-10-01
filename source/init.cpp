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

#define TIMEMORY_BUILD_EXTERN_INIT

#include "timemory/components.hpp"
#include "timemory/manager.hpp"
#include "timemory/utility/macros.hpp"
#include "timemory/utility/serializer.hpp"
#include "timemory/utility/singleton.hpp"
#include "timemory/utility/utility.hpp"

using namespace tim::component;

#if defined(TIMEMORY_EXTERN_INIT)

//======================================================================================//

__library_ctor__
void
timemory_manager_ctor_init()
{
#if defined(DEBUG)
    auto _debug = tim::settings::debug();
    auto _verbose = tim::settings::verbose();
#endif

#if defined(DEBUG)
    if(_debug || _verbose > 3)
        printf("[%s]> initializing manager...\n", __FUNCTION__);
#endif

    // fully initialize manager
    auto _instance = tim::manager::instance();
    auto _master = tim::manager::master_instance();

    if(_instance != _master)
        printf("[%s]> master_instance() != instance() : %p vs. %p\n", __FUNCTION__,
               (void*) _instance, (void*) _master);

#if defined(DEBUG)
    if(_debug || _verbose > 3)
        printf("[%s]> initializing storage...\n", __FUNCTION__);
#endif

    // initialize storage
    using tuple_type = tim::available_tuple<tim::complete_tuple_t>;
    tim::manager::get_storage<tuple_type>::initialize(_master);
}

//======================================================================================//

namespace tim
{

//======================================================================================//

env_settings* env_settings::instance()
{
    static env_settings* _instance = new env_settings();
    return _instance;
}

//======================================================================================//

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

//======================================================================================//
// implements:
//      template <> get_storage_singleton<TYPE>();
//      template <> get_noninit_storage_singleton<TYPE>();
//
TIMEMORY_INSTANTIATE_EXTERN_INIT(caliper)
TIMEMORY_INSTANTIATE_EXTERN_INIT(cpu_clock)
TIMEMORY_INSTANTIATE_EXTERN_INIT(cpu_roofline_dp_flops)
TIMEMORY_INSTANTIATE_EXTERN_INIT(cpu_roofline_flops)
TIMEMORY_INSTANTIATE_EXTERN_INIT(cpu_roofline_sp_flops)
TIMEMORY_INSTANTIATE_EXTERN_INIT(cpu_util)
TIMEMORY_INSTANTIATE_EXTERN_INIT(cuda_event)
TIMEMORY_INSTANTIATE_EXTERN_INIT(cupti_activity)
TIMEMORY_INSTANTIATE_EXTERN_INIT(cupti_counters)
TIMEMORY_INSTANTIATE_EXTERN_INIT(data_rss)
TIMEMORY_INSTANTIATE_EXTERN_INIT(gperf_cpu_profiler)
TIMEMORY_INSTANTIATE_EXTERN_INIT(gperf_heap_profiler)
TIMEMORY_INSTANTIATE_EXTERN_INIT(gpu_roofline_dp_flops)
TIMEMORY_INSTANTIATE_EXTERN_INIT(gpu_roofline_flops)
TIMEMORY_INSTANTIATE_EXTERN_INIT(gpu_roofline_hp_flops)
TIMEMORY_INSTANTIATE_EXTERN_INIT(gpu_roofline_sp_flops)
TIMEMORY_INSTANTIATE_EXTERN_INIT(monotonic_clock)
TIMEMORY_INSTANTIATE_EXTERN_INIT(monotonic_raw_clock)
TIMEMORY_INSTANTIATE_EXTERN_INIT(num_io_in)
TIMEMORY_INSTANTIATE_EXTERN_INIT(num_io_out)
TIMEMORY_INSTANTIATE_EXTERN_INIT(num_major_page_faults)
TIMEMORY_INSTANTIATE_EXTERN_INIT(num_minor_page_faults)
TIMEMORY_INSTANTIATE_EXTERN_INIT(num_msg_recv)
TIMEMORY_INSTANTIATE_EXTERN_INIT(num_msg_sent)
TIMEMORY_INSTANTIATE_EXTERN_INIT(num_signals)
TIMEMORY_INSTANTIATE_EXTERN_INIT(num_swap)
TIMEMORY_INSTANTIATE_EXTERN_INIT(nvtx_marker)
TIMEMORY_INSTANTIATE_EXTERN_INIT(page_rss)
TIMEMORY_INSTANTIATE_EXTERN_INIT(papi_array_t)
TIMEMORY_INSTANTIATE_EXTERN_INIT(peak_rss)
TIMEMORY_INSTANTIATE_EXTERN_INIT(priority_context_switch)
TIMEMORY_INSTANTIATE_EXTERN_INIT(process_cpu_clock)
TIMEMORY_INSTANTIATE_EXTERN_INIT(process_cpu_util)
TIMEMORY_INSTANTIATE_EXTERN_INIT(read_bytes)
TIMEMORY_INSTANTIATE_EXTERN_INIT(real_clock)
TIMEMORY_INSTANTIATE_EXTERN_INIT(stack_rss)
TIMEMORY_INSTANTIATE_EXTERN_INIT(system_clock)
TIMEMORY_INSTANTIATE_EXTERN_INIT(thread_cpu_clock)
TIMEMORY_INSTANTIATE_EXTERN_INIT(thread_cpu_util)
TIMEMORY_INSTANTIATE_EXTERN_INIT(trip_count)
TIMEMORY_INSTANTIATE_EXTERN_INIT(user_clock)
TIMEMORY_INSTANTIATE_EXTERN_INIT(virtual_memory)
TIMEMORY_INSTANTIATE_EXTERN_INIT(voluntary_context_switch)
TIMEMORY_INSTANTIATE_EXTERN_INIT(written_bytes)

}  // namespace tim

#endif  // defined(TIMEMORY_EXTERN_INIT)
