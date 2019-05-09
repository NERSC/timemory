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

#include "timemory/auto_tuple.hpp"
#include "timemory/component_tuple.hpp"
#include "timemory/macros.hpp"
#include "timemory/manager.hpp"
#include "timemory/serializer.hpp"
#include "timemory/signal_detection.hpp"
#include "timemory/singleton.hpp"
#include "timemory/timemory.hpp"
#include "timemory/utility.hpp"

using namespace tim::component;
using auto_timer_t =
    tim::auto_tuple<real_clock, system_clock, cpu_clock, cpu_util, current_rss, peak_rss>;

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
    std::string cxx_timer_tag(timer_tag);
    char*       _timer_tag = (char*) timer_tag;
    free(_timer_tag);
    return (void*) new auto_timer_t(cxx_timer_tag, lineno, lang_tag,
                                    (report > 0) ? true : false);
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
