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

extern "C"
{
#include "timemory/ctimemory.h"
}

using namespace tim::component;

using auto_timer_t =
    tim::auto_tuple<real_clock, system_clock, cpu_clock, cpu_util, current_rss, peak_rss>;

using auto_list_t =
    tim::auto_list<real_clock, system_clock, user_clock, cpu_clock, monotonic_clock,
                   monotonic_raw_clock, thread_cpu_clock, process_cpu_clock, cpu_util,
                   thread_cpu_util, process_cpu_util, current_rss, peak_rss, stack_rss,
                   data_rss, num_swap, num_io_in, num_io_out, num_minor_page_faults,
                   num_major_page_faults, num_msg_sent, num_msg_recv, num_signals,
                   voluntary_context_switch, priority_context_switch, cuda_event,
                   papi_array_t, cpu_roofline_dp_flops, cpu_roofline_sp_flops>;

//======================================================================================//
//
//                      C++ interface
//
//======================================================================================//

#if defined(TIMEMORY_BUILD_C)

//======================================================================================//

extern "C" tim_api void
cxx_timemory_init(int argc, char** argv, timemory_settings _settings)
{
#    define PROCESS_SETTING(variable, type)                                              \
        if(_settings.variable >= 0)                                                      \
        {                                                                                \
            tim::settings::variable() = static_cast<type>(_settings.variable);           \
        }

    PROCESS_SETTING(enabled, bool);
    PROCESS_SETTING(auto_output, bool);
    PROCESS_SETTING(file_output, bool);
    PROCESS_SETTING(text_output, bool);
    PROCESS_SETTING(json_output, bool);
    PROCESS_SETTING(cout_output, bool);
    PROCESS_SETTING(scientific, bool);
    PROCESS_SETTING(precision, int);
    PROCESS_SETTING(width, int);

    if(argv && argc > 0)
        tim::timemory_init(argc, argv);

#    undef PROCESS_SETTING
}

//======================================================================================//

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
    std::vector<int> _components(num_components);
    std::memcpy(_components.data(), components, num_components * sizeof(int));
    tim::initialize(*obj, _components);
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
    obj->pop();
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

#endif  // TIMEMORY_BUILD_C

//======================================================================================//
