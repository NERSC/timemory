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

#include "timemory/timemory.hpp"

#include <deque>
#include <iostream>
#include <unordered_map>
#include <vector>

using namespace tim::component;
using auto_timer_t    = typename tim::auto_timer::component_type;
using complete_list_t = tim::complete_list_t;

//======================================================================================//
//
//                      C++ interface
//
//======================================================================================//

#if defined(TIMEMORY_EXTERN_INIT)
/*
//--------------------------------------------------------------------------------------//
//  construct the library at start up
//
__library_ctor__
void
timemory_library_constructor()
{
    static std::atomic<int> _once;
    if(_once++ > 0)
        return;

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
*/
#endif

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
    return (tim::settings::enabled() && tim::counted_object<void>::is_enabled()) ? 1 : 0;
}

//======================================================================================//

extern "C" tim_api void*
cxx_timemory_create_auto_timer(const char* timer_tag, int lineno)
{
    if(!tim::settings::enabled())
        return nullptr;
    using namespace tim::component;
    std::string key_tag(timer_tag);
    auto*       obj = new auto_timer_t(key_tag, true, tim::language::c(), lineno);
    obj->start();
    return (void*) obj;
}

//======================================================================================//

extern "C" tim_api void*
cxx_timemory_create_auto_tuple(const char* timer_tag, int lineno, int num_components,
                               const int* components)
{
    if(!tim::settings::enabled())
        return nullptr;
    using namespace tim::component;
    std::string key_tag(timer_tag);
    auto        obj = new complete_list_t(key_tag, true, tim::language::c(), lineno);
#    if defined(DEBUG)
    std::vector<int> _components;
    for(int i = 0; i < num_components; ++i)
    {
        if(tim::settings::debug())
            printf("[%s]> Adding component %i...\n", __FUNCTION__, components[i]);
        _components.push_back(components[i]);
    }
#    else
    std::vector<int> _components(num_components, 0);
    std::memcpy(_components.data(), components, num_components * sizeof(int));
#    endif
    tim::initialize(*obj, _components);
    obj->start();
    return static_cast<void*>(obj);
}

//======================================================================================//

extern "C" tim_api void*
cxx_timemory_delete_auto_timer(void* ctimer)
{
    if(ctimer)
    {
        auto_timer_t* obj = static_cast<auto_timer_t*>(ctimer);
        obj->stop();
        delete obj;
    }
    return nullptr;
}

//======================================================================================//

extern "C" tim_api void*
cxx_timemory_delete_auto_tuple(void* ctuple)
{
    if(ctuple)
    {
        complete_list_t* obj = static_cast<complete_list_t*>(ctuple);
        obj->stop();
        delete obj;
    }
    return nullptr;
}

//======================================================================================//

extern "C" tim_api const char*
cxx_timemory_string_combine(const char* _a, const char* _b)
{
    char* buff = (char*) malloc(sizeof(char) * 256);
    if(buff)
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
