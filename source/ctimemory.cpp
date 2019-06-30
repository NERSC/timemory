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

/** \file ctimemory.cpp
 * This is the C++ proxy for the C interface. Compilation of this file is not
 * required for C++ codes but is compiled into "libtimemory.*" (timemory-cxx-library)
 * so that the "libctimemory.*" can be linked during the TiMemory build and
 * "libctimemory.*" can be stand-alone linked to C code.
 *
 */

#include "timemory/auto_list.hpp"
#include "timemory/auto_tuple.hpp"
#include "timemory/component_list.hpp"
#include "timemory/component_tuple.hpp"
#include "timemory/macros.hpp"
#include "timemory/manager.hpp"
#include "timemory/serializer.hpp"
#include "timemory/signal_detection.hpp"
#include "timemory/singleton.hpp"
#include "timemory/timemory.hpp"
#include "timemory/utility.hpp"

EXTERN_C_BEGIN
#include "timemory/ctimemory.h"
EXTERN_C_END

using namespace tim::component;

using auto_timer_t =
    tim::auto_tuple<real_clock, system_clock, cpu_clock, cpu_util, current_rss, peak_rss>;

using all_tuple_t =
    tim::auto_tuple<real_clock, system_clock, user_clock, cpu_clock, monotonic_clock,
                    monotonic_raw_clock, thread_cpu_clock, process_cpu_clock, cpu_util,
                    thread_cpu_util, process_cpu_util, current_rss, peak_rss, stack_rss,
                    data_rss, num_swap, num_io_in, num_io_out, num_minor_page_faults,
                    num_major_page_faults, num_msg_sent, num_msg_recv, num_signals,
                    voluntary_context_switch, priority_context_switch>;

using auto_list_t =
    tim::auto_list<real_clock, system_clock, user_clock, cpu_clock, monotonic_clock,
                   monotonic_raw_clock, thread_cpu_clock, process_cpu_clock, cpu_util,
                   thread_cpu_util, process_cpu_util, current_rss, peak_rss, stack_rss,
                   data_rss, num_swap, num_io_in, num_io_out, num_minor_page_faults,
                   num_major_page_faults, num_msg_sent, num_msg_recv, num_signals,
                   voluntary_context_switch, priority_context_switch, cuda_event>;

//======================================================================================//
//
//                      C++ extern template instantiation
//
//======================================================================================//

#if defined(TIMEMORY_BUILD_EXTERN_TEMPLATES)

//--------------------------------------------------------------------------------------//
//  category configurations
//

// rusage_components_t
TIMEMORY_INSTANTIATE_EXTERN_TUPLE(
    tim::component::current_rss, tim::component::peak_rss, tim::component::stack_rss,
    tim::component::data_rss, tim::component::num_swap, tim::component::num_io_in,
    tim::component::num_io_out, tim::component::num_minor_page_faults,
    tim::component::num_major_page_faults, tim::component::num_msg_sent,
    tim::component::num_msg_recv, tim::component::num_signals,
    tim::component::voluntary_context_switch, tim::component::priority_context_switch)

// timing_components_t
TIMEMORY_INSTANTIATE_EXTERN_TUPLE(
    tim::component::real_clock, tim::component::system_clock, tim::component::user_clock,
    tim::component::cpu_clock, tim::component::monotonic_clock,
    tim::component::monotonic_raw_clock, tim::component::thread_cpu_clock,
    tim::component::process_cpu_clock, tim::component::cpu_util,
    tim::component::thread_cpu_util, tim::component::process_cpu_util)

//--------------------------------------------------------------------------------------//
//  standard configurations
//

// standard_rusage_t
TIMEMORY_INSTANTIATE_EXTERN_TUPLE(tim::component::current_rss, tim::component::peak_rss,
                                  tim::component::num_io_in, tim::component::num_io_out,
                                  tim::component::num_minor_page_faults,
                                  tim::component::num_major_page_faults,
                                  tim::component::priority_context_switch,
                                  tim::component::voluntary_context_switch)

// standard_timing_t
TIMEMORY_INSTANTIATE_EXTERN_TUPLE(tim::component::real_clock, tim::component::user_clock,
                                  tim::component::system_clock, tim::component::cpu_clock,
                                  tim::component::cpu_util)

//--------------------------------------------------------------------------------------//
// auto_timer_t
//
TIMEMORY_INSTANTIATE_EXTERN_TUPLE(tim::component::real_clock,
                                  tim::component::system_clock,
                                  tim::component::user_clock, tim::component::cpu_clock,
                                  tim::component::cpu_util)

TIMEMORY_INSTANTIATE_EXTERN_TUPLE(tim::component::real_clock,
                                  tim::component::system_clock,
                                  tim::component::user_clock, tim::component::cpu_clock,
                                  tim::component::cpu_util, tim::component::current_rss,
                                  tim::component::peak_rss)

TIMEMORY_INSTANTIATE_EXTERN_TUPLE(tim::component::real_clock,
                                  tim::component::system_clock, tim::component::cpu_clock,
                                  tim::component::cpu_util, tim::component::current_rss,
                                  tim::component::peak_rss)

//--------------------------------------------------------------------------------------//
// all_tuple_t
//

TIMEMORY_INSTANTIATE_EXTERN_TUPLE(
    tim::component::real_clock, tim::component::system_clock, tim::component::user_clock,
    tim::component::cpu_clock, tim::component::monotonic_clock,
    tim::component::monotonic_raw_clock, tim::component::thread_cpu_clock,
    tim::component::process_cpu_clock, tim::component::cpu_util,
    tim::component::thread_cpu_util, tim::component::process_cpu_util,
    tim::component::current_rss, tim::component::peak_rss, tim::component::stack_rss,
    tim::component::data_rss, tim::component::num_swap, tim::component::num_io_in,
    tim::component::num_io_out, tim::component::num_minor_page_faults,
    tim::component::num_major_page_faults, tim::component::num_msg_sent,
    tim::component::num_msg_recv, tim::component::num_signals,
    tim::component::voluntary_context_switch, tim::component::priority_context_switch)

//--------------------------------------------------------------------------------------//
// miscellaneous
//

TIMEMORY_INSTANTIATE_EXTERN_TUPLE(tim::component::real_clock)
TIMEMORY_INSTANTIATE_EXTERN_TUPLE(tim::component::system_clock)
TIMEMORY_INSTANTIATE_EXTERN_TUPLE(tim::component::user_clock)
TIMEMORY_INSTANTIATE_EXTERN_TUPLE(tim::component::cpu_clock)
TIMEMORY_INSTANTIATE_EXTERN_TUPLE(tim::component::monotonic_clock)
TIMEMORY_INSTANTIATE_EXTERN_TUPLE(tim::component::monotonic_raw_clock)
TIMEMORY_INSTANTIATE_EXTERN_TUPLE(tim::component::thread_cpu_clock)
TIMEMORY_INSTANTIATE_EXTERN_TUPLE(tim::component::process_cpu_clock)
TIMEMORY_INSTANTIATE_EXTERN_TUPLE(tim::component::cpu_util)
TIMEMORY_INSTANTIATE_EXTERN_TUPLE(tim::component::process_cpu_util)
TIMEMORY_INSTANTIATE_EXTERN_TUPLE(tim::component::thread_cpu_util)
TIMEMORY_INSTANTIATE_EXTERN_TUPLE(tim::component::peak_rss)
TIMEMORY_INSTANTIATE_EXTERN_TUPLE(tim::component::current_rss)
TIMEMORY_INSTANTIATE_EXTERN_TUPLE(tim::component::stack_rss)
TIMEMORY_INSTANTIATE_EXTERN_TUPLE(tim::component::data_rss)
TIMEMORY_INSTANTIATE_EXTERN_TUPLE(tim::component::num_swap)
TIMEMORY_INSTANTIATE_EXTERN_TUPLE(tim::component::num_io_in)
TIMEMORY_INSTANTIATE_EXTERN_TUPLE(tim::component::num_io_out)
TIMEMORY_INSTANTIATE_EXTERN_TUPLE(tim::component::num_minor_page_faults)
TIMEMORY_INSTANTIATE_EXTERN_TUPLE(tim::component::num_major_page_faults)
TIMEMORY_INSTANTIATE_EXTERN_TUPLE(tim::component::num_msg_sent)
TIMEMORY_INSTANTIATE_EXTERN_TUPLE(tim::component::num_msg_recv)
TIMEMORY_INSTANTIATE_EXTERN_TUPLE(tim::component::num_signals)
TIMEMORY_INSTANTIATE_EXTERN_TUPLE(tim::component::voluntary_context_switch)
TIMEMORY_INSTANTIATE_EXTERN_TUPLE(tim::component::priority_context_switch)

TIMEMORY_INSTANTIATE_EXTERN_TUPLE(tim::component::real_clock, tim::component::cpu_clock)

TIMEMORY_INSTANTIATE_EXTERN_TUPLE(tim::component::real_clock, tim::component::cpu_clock,
                                  tim::component::cpu_util)

TIMEMORY_INSTANTIATE_EXTERN_TUPLE(tim::component::real_clock,
                                  tim::component::system_clock,
                                  tim::component::user_clock)

TIMEMORY_INSTANTIATE_EXTERN_TUPLE(tim::component::real_clock,
                                  tim::component::system_clock, tim::component::cpu_clock,
                                  tim::component::cpu_util, tim::component::peak_rss,
                                  tim::component::current_rss)

TIMEMORY_INSTANTIATE_EXTERN_TUPLE(tim::component::real_clock,
                                  tim::component::system_clock,
                                  tim::component::thread_cpu_clock,
                                  tim::component::thread_cpu_util,
                                  tim::component::process_cpu_clock,
                                  tim::component::process_cpu_util,
                                  tim::component::peak_rss, tim::component::current_rss)

TIMEMORY_INSTANTIATE_EXTERN_TUPLE(
    tim::component::peak_rss, tim::component::current_rss, tim::component::stack_rss,
    tim::component::data_rss, tim::component::num_swap, tim::component::num_io_in,
    tim::component::num_io_out, tim::component::num_minor_page_faults,
    tim::component::num_major_page_faults, tim::component::num_msg_sent,
    tim::component::num_msg_recv, tim::component::num_signals,
    tim::component::voluntary_context_switch, tim::component::priority_context_switch)

TIMEMORY_INSTANTIATE_EXTERN_TUPLE(
    tim::component::real_clock, tim::component::system_clock, tim::component::user_clock,
    tim::component::cpu_clock, tim::component::cpu_util, tim::component::thread_cpu_clock,
    tim::component::thread_cpu_util, tim::component::process_cpu_clock,
    tim::component::process_cpu_util, tim::component::monotonic_clock,
    tim::component::monotonic_raw_clock)

TIMEMORY_INSTANTIATE_EXTERN_TUPLE(tim::component::real_clock,
                                  tim::component::system_clock,
                                  tim::component::user_clock, tim::component::cpu_clock,
                                  tim::component::thread_cpu_clock,
                                  tim::component::process_cpu_clock)

TIMEMORY_INSTANTIATE_EXTERN_TUPLE(tim::component::real_clock,
                                  tim::component::thread_cpu_clock,
                                  tim::component::thread_cpu_util,
                                  tim::component::process_cpu_clock,
                                  tim::component::process_cpu_util,
                                  tim::component::peak_rss, tim::component::current_rss)

TIMEMORY_INSTANTIATE_EXTERN_TUPLE(tim::component::real_clock,
                                  tim::component::thread_cpu_clock,
                                  tim::component::process_cpu_util)

#    if defined(TIMEMORY_USE_CUDA)
TIMEMORY_INSTANTIATE_EXTERN_TUPLE(tim::component::cuda_event)

TIMEMORY_INSTANTIATE_EXTERN_TUPLE(tim::component::thread_cpu_clock,
                                  tim::component::cuda_event)

TIMEMORY_INSTANTIATE_EXTERN_TUPLE(tim::component::thread_cpu_clock,
                                  tim::component::thread_cpu_util,
                                  tim::component::cuda_event)
#    endif

#endif  // defined(TIMEMORY_BUILD_EXTERN_TEMPLATES)

//======================================================================================//
//
//                      C++ interface
//
//======================================================================================//
#if defined(TIMEMORY_BUILD_C)

extern "C" tim_api int
cxx_timemory_enabled(void)
{
    return (tim::counted_object<void>::is_enabled()) ? 1 : 0;
}

//======================================================================================//

extern "C" tim_api void*
cxx_timemory_create_auto_timer(const char* timer_tag, int lineno, int report)
{
    using namespace tim::component;
    std::string key_tag(timer_tag);
    char*       _timer_tag = (char*) timer_tag;
    free(_timer_tag);
    return (void*) new auto_timer_t(key_tag, lineno, tim::language::c(),
                                    (report > 0) ? true : false);
}

//======================================================================================//

extern "C" tim_api void*
cxx_timemory_create_auto_tuple(const char* timer_tag, int lineno, int num_components,
                               const int* components)
{
    using namespace tim::component;
    std::string key_tag(timer_tag);
    auto        obj = new auto_list_t(key_tag, lineno, tim::language::c(), false);
    obj->stop();
    obj->reset();
    for(int i = 0; i < num_components; ++i)
    {
        TIMEMORY_COMPONENT component = static_cast<TIMEMORY_COMPONENT>(components[i]);
        switch(component)
        {
            case WALL_CLOCK: obj->get<real_clock>() = new real_clock(); break;
            case SYS_CLOCK: obj->get<system_clock>() = new system_clock(); break;
            case USER_CLOCK: obj->get<user_clock>() = new user_clock(); break;
            case CPU_CLOCK: obj->get<cpu_clock>() = new cpu_clock(); break;
            case MONOTONIC_CLOCK:
                obj->get<monotonic_clock>() = new monotonic_clock();
                break;
            case MONOTONIC_RAW_CLOCK:
                obj->get<monotonic_raw_clock>() = new monotonic_raw_clock();
                break;
            case THREAD_CPU_CLOCK:
                obj->get<thread_cpu_clock>() = new thread_cpu_clock();
                break;
            case PROCESS_CPU_CLOCK:
                obj->get<process_cpu_clock>() = new process_cpu_clock();
                break;
            case CPU_UTIL: obj->get<cpu_util>() = new cpu_util(); break;
            case THREAD_CPU_UTIL:
                obj->get<thread_cpu_util>() = new thread_cpu_util();
                break;
            case PROCESS_CPU_UTIL:
                obj->get<process_cpu_util>() = new process_cpu_util();
                break;
            case CURRENT_RSS: obj->get<current_rss>() = new current_rss(); break;
            case PEAK_RSS: obj->get<peak_rss>() = new peak_rss(); break;
            case STACK_RSS: obj->get<stack_rss>() = new stack_rss(); break;
            case DATA_RSS: obj->get<data_rss>() = new data_rss(); break;
            case NUM_SWAP: obj->get<num_swap>() = new num_swap(); break;
            case NUM_IO_IN: obj->get<num_io_in>() = new num_io_in(); break;
            case NUM_IO_OUT: obj->get<num_io_out>() = new num_io_out(); break;
            case NUM_MINOR_PAGE_FAULTS:
                obj->get<num_minor_page_faults>() = new num_minor_page_faults();
                break;
            case NUM_MAJOR_PAGE_FAULTS:
                obj->get<num_major_page_faults>() = new num_major_page_faults();
                break;
            case NUM_MSG_SENT: obj->get<num_msg_sent>() = new num_msg_sent(); break;
            case NUM_MSG_RECV: obj->get<num_msg_recv>() = new num_msg_recv(); break;
            case NUM_SIGNALS: obj->get<num_signals>() = new num_signals(); break;
            case VOLUNTARY_CONTEXT_SWITCH:
                obj->get<voluntary_context_switch>() = new voluntary_context_switch();
                break;
            case PRIORITY_CONTEXT_SWITCH:
                obj->get<priority_context_switch>() = new priority_context_switch();
                break;
            case CUDA_EVENT:
#    if defined(TIMEMORY_USE_CUDA)
                obj->get<cuda_event>() = new cuda_event();
#    endif
                break;
        }
    }
    obj->push();
    obj->start();
    return static_cast<void*>(obj);
}

//======================================================================================//

extern "C" tim_api void*
cxx_timemory_delete_auto_timer(void* ctimer)
{
    auto_timer_t* cxxtimer = static_cast<auto_timer_t*>(ctimer);
    delete cxxtimer;
    ctimer = nullptr;
    return ctimer;
}

//======================================================================================//

extern "C" tim_api void*
cxx_timemory_delete_auto_tuple(void* ctuple)
{
    auto_list_t* obj = static_cast<auto_list_t*>(ctuple);
    obj->stop();
    delete obj;
    ctuple = nullptr;
    return ctuple;
}

//======================================================================================//

extern "C" tim_api const char*
cxx_timemory_string_combine(const char* _a, const char* _b)
{
    char* buff = (char*) malloc(sizeof(char) * 256);
    sprintf(buff, "%s%s", _a, _b);
    return (const char*) buff;
}

//======================================================================================//

extern "C" tim_api const char*
cxx_timemory_auto_timer_str(const char* _a, const char* _b, const char* _c, int _d)
{
    std::string _C   = std::string(_c).substr(std::string(_c).find_last_of('/') + 1);
    char*       buff = (char*) malloc(sizeof(char) * 256);
    sprintf(buff, "%s%s@'%s':%i", _a, _b, _C.c_str(), _d);
    return (const char*) buff;
}
#endif

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

//======================================================================================//
// function for storage
/*
template <typename _Tp>
details::storage_singleton_t<_Tp>&
get_storage_singleton()
{
    using _single_t                         = details::storage_singleton_t<_Tp>;
    static _single_t _instance = _single_t::instance();
    return _instance;
}
*/
}  // namespace tim

TIMEMORY_INSTANTIATE_EXTERN_STORAGE(real_clock)
TIMEMORY_INSTANTIATE_EXTERN_STORAGE(system_clock)
TIMEMORY_INSTANTIATE_EXTERN_STORAGE(user_clock)
TIMEMORY_INSTANTIATE_EXTERN_STORAGE(cpu_clock)
TIMEMORY_INSTANTIATE_EXTERN_STORAGE(monotonic_clock)
TIMEMORY_INSTANTIATE_EXTERN_STORAGE(monotonic_raw_clock)
TIMEMORY_INSTANTIATE_EXTERN_STORAGE(thread_cpu_clock)
TIMEMORY_INSTANTIATE_EXTERN_STORAGE(process_cpu_clock)
TIMEMORY_INSTANTIATE_EXTERN_STORAGE(cpu_util)
TIMEMORY_INSTANTIATE_EXTERN_STORAGE(thread_cpu_util)
TIMEMORY_INSTANTIATE_EXTERN_STORAGE(process_cpu_util)
TIMEMORY_INSTANTIATE_EXTERN_STORAGE(current_rss)
TIMEMORY_INSTANTIATE_EXTERN_STORAGE(peak_rss)
TIMEMORY_INSTANTIATE_EXTERN_STORAGE(stack_rss)
TIMEMORY_INSTANTIATE_EXTERN_STORAGE(data_rss)
TIMEMORY_INSTANTIATE_EXTERN_STORAGE(num_swap)
TIMEMORY_INSTANTIATE_EXTERN_STORAGE(num_io_in)
TIMEMORY_INSTANTIATE_EXTERN_STORAGE(num_io_out)
TIMEMORY_INSTANTIATE_EXTERN_STORAGE(num_minor_page_faults)
TIMEMORY_INSTANTIATE_EXTERN_STORAGE(num_major_page_faults)
TIMEMORY_INSTANTIATE_EXTERN_STORAGE(num_msg_sent)
TIMEMORY_INSTANTIATE_EXTERN_STORAGE(num_msg_recv)
TIMEMORY_INSTANTIATE_EXTERN_STORAGE(num_signals)
TIMEMORY_INSTANTIATE_EXTERN_STORAGE(voluntary_context_switch)
TIMEMORY_INSTANTIATE_EXTERN_STORAGE(priority_context_switch)

#    if defined(TIMEMORY_USE_CUDA)
TIMEMORY_INSTANTIATE_EXTERN_STORAGE(cuda_event)
#    endif

#endif  // defined(TIMEMORY_EXTERN_INIT
