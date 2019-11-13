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

#define TIMEMORY_BUILD_EXTERN_INIT

#include "timemory/compat/library.h"
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

#if defined(TIMEMORY_BUILD_C)

//======================================================================================//

static bool&
free_cstr()
{
    static thread_local bool _instance = false;
    return _instance;
}

//======================================================================================//

extern "C"
{
    tim_api void cxx_timemory_init(int argc, char** argv, timemory_settings _settings)
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

    tim_api int cxx_timemory_enabled(void) { return (tim::settings::enabled()) ? 1 : 0; }

    //======================================================================================//

    tim_api void* cxx_timemory_create_auto_timer(const char* timer_tag)
    {
        if(!tim::settings::enabled())
            return nullptr;
        using namespace tim::component;
        std::string key_tag(timer_tag);
        auto*       obj = new auto_timer_t(key_tag, true, tim::settings::flat_profile());
        obj->start();
        return (void*) obj;
    }

    //======================================================================================//

    tim_api void* cxx_timemory_create_auto_tuple(const char* timer_tag,
                                                 int         num_components,
                                                 const int*  components)
    {
        if(!tim::settings::enabled())
            return nullptr;
        using namespace tim::component;
        std::string key_tag(timer_tag);
        if(free_cstr())
        {
            char* _tag = (char*) timer_tag;
            free(_tag);
            free_cstr() = false;
        }
        auto obj = new complete_list_t(key_tag, true, tim::settings::flat_profile());
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

    tim_api void* cxx_timemory_delete_auto_timer(void* ctimer)
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

    tim_api void* cxx_timemory_delete_auto_tuple(void* ctuple)
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

    tim_api const char* cxx_timemory_label(int _mode, int _line, const char* _func,
                                           const char* _file, const char* _extra)
    {
        if(!tim::settings::enabled())
            return "";

        if(_mode == 0)
            return _extra;
        else if(_mode == 1 && strlen(_extra) == 0)
            return _func;

        free_cstr() = true;

        std::stringstream ss;
        if(_mode == 1)
            ss << _func << "/";
        else if(_mode == 2)
        {
            ss << _func << "/"
               << std::string(_file).substr(std::string(_file).find_last_of('/') + 1)
               << ":" << _line;
        }

        auto  len  = ss.str().length() + strlen(_extra);
        char* buff = (char*) malloc(len * sizeof(char));
        if(buff)
        {
            if(strlen(_extra) == 0)
                sprintf(buff, "%s", ss.str().c_str());
            else
                sprintf(buff, "%s/%s", ss.str().c_str(), _extra);
        }
        return (const char*) buff;
    }

}  // extern "C"

//======================================================================================//

#endif  // TIMEMORY_BUILD_C
