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

using auto_list_t =
    tim::auto_list<real_clock, system_clock, user_clock, cpu_clock, monotonic_clock,
                   monotonic_raw_clock, thread_cpu_clock, process_cpu_clock, cpu_util,
                   thread_cpu_util, process_cpu_util, current_rss, peak_rss, stack_rss,
                   data_rss, num_swap, num_io_in, num_io_out, num_minor_page_faults,
                   num_major_page_faults, num_msg_sent, num_msg_recv, num_signals,
                   voluntary_context_switch, priority_context_switch>;

//======================================================================================//
//
//                      C++ interface
//
//======================================================================================//

extern "C" tim_api int
cxx_timemory_enabled(void)
{
    return (tim::counted_object<void>::is_enabled()) ? 1 : 0;
}

//======================================================================================//

extern "C" tim_api void*
cxx_timemory_create_auto_timer(const char* timer_tag, int lineno, const char* lang_tag,
                               int report)
{
    using namespace tim::component;
    std::string key_tag(timer_tag);
    char*       _timer_tag = (char*) timer_tag;
    free(_timer_tag);
    return (void*) new auto_timer_t(key_tag, lineno, lang_tag,
                                    (report > 0) ? true : false);
}

//======================================================================================//

extern "C" tim_api void*
cxx_timemory_create_auto_tuple(const char* timer_tag, int lineno, int num_components,
                               const int* components)
{
    using namespace tim::component;
    using data_type = typename auto_list_t::component_type::data_type;
    std::string key_tag(timer_tag);
    auto        lang_tag = "_c_";
    auto        obj      = new auto_list_t(key_tag, lineno, lang_tag, false);
    obj->stop();
    obj->reset();
    for(int i = 0; i < num_components; ++i)
    {
        COMPONENT component = static_cast<COMPONENT>(components[i]);
        switch(component)
        {
            case WALL_CLOCK:
                obj->get<tim::index_of<real_clock*, data_type>::value>() =
                    new real_clock();
                break;
            case SYS_CLOCK:
                obj->get<tim::index_of<system_clock*, data_type>::value>() =
                    new system_clock();
                break;
            case USER_CLOCK:
                obj->get<tim::index_of<user_clock*, data_type>::value>() =
                    new user_clock();
                break;
            case CPU_CLOCK:
                obj->get<tim::index_of<cpu_clock*, data_type>::value>() = new cpu_clock();
                break;
            case MONOTONIC_CLOCK:
                obj->get<tim::index_of<monotonic_clock*, data_type>::value>() =
                    new monotonic_clock();
                break;
            case MONOTONIC_RAW_CLOCK:
                obj->get<tim::index_of<monotonic_raw_clock*, data_type>::value>() =
                    new monotonic_raw_clock();
                break;
            case THREAD_CPU_CLOCK:
                obj->get<tim::index_of<thread_cpu_clock*, data_type>::value>() =
                    new thread_cpu_clock();
                break;
            case PROCESS_CPU_CLOCK:
                obj->get<tim::index_of<process_cpu_clock*, data_type>::value>() =
                    new process_cpu_clock();
                break;
            case CPU_UTIL:
                obj->get<tim::index_of<cpu_util*, data_type>::value>() = new cpu_util();
                break;
            case THREAD_CPU_UTIL:
                obj->get<tim::index_of<thread_cpu_util*, data_type>::value>() =
                    new thread_cpu_util();
                break;
            case PROCESS_CPU_UTIL:
                obj->get<tim::index_of<process_cpu_util*, data_type>::value>() =
                    new process_cpu_util();
                break;
            case CURRENT_RSS:
                obj->get<tim::index_of<current_rss*, data_type>::value>() =
                    new current_rss();
                break;
            case PEAK_RSS:
                obj->get<tim::index_of<peak_rss*, data_type>::value>() = new peak_rss();
                break;
            case STACK_RSS:
                obj->get<tim::index_of<stack_rss*, data_type>::value>() = new stack_rss();
                break;
            case DATA_RSS:
                obj->get<tim::index_of<data_rss*, data_type>::value>() = new data_rss();
                break;
            case NUM_SWAP:
                obj->get<tim::index_of<num_swap*, data_type>::value>() = new num_swap();
                break;
            case NUM_IO_IN:
                obj->get<tim::index_of<num_io_in*, data_type>::value>() = new num_io_in();
                break;
            case NUM_IO_OUT:
                obj->get<tim::index_of<num_io_out*, data_type>::value>() =
                    new num_io_out();
                break;
            case NUM_MINOR_PAGE_FAULTS:
                obj->get<tim::index_of<num_minor_page_faults*, data_type>::value>() =
                    new num_minor_page_faults();
                break;
            case NUM_MAJOR_PAGE_FAULTS:
                obj->get<tim::index_of<num_major_page_faults*, data_type>::value>() =
                    new num_major_page_faults();
                break;
            case NUM_MSG_SENT:
                obj->get<tim::index_of<num_msg_sent*, data_type>::value>() =
                    new num_msg_sent();
                break;
            case NUM_MSG_RECV:
                obj->get<tim::index_of<num_msg_recv*, data_type>::value>() =
                    new num_msg_recv();
                break;
            case NUM_SIGNALS:
                obj->get<tim::index_of<num_signals*, data_type>::value>() =
                    new num_signals();
                break;
            case VOLUNTARY_CONTEXT_SWITCH:
                obj->get<tim::index_of<voluntary_context_switch*, data_type>::value>() =
                    new voluntary_context_switch();
                break;
            case PRIORITY_CONTEXT_SWITCH:
                obj->get<tim::index_of<priority_context_switch*, data_type>::value>() =
                    new priority_context_switch();
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

//======================================================================================//
