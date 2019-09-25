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

#include <iostream>
#include <unordered_map>

using namespace tim::component;

using auto_timer_t = tim::component_tuple<real_clock, system_clock, cpu_clock, cpu_util,
                                          page_rss, peak_rss>;

using complete_list_t = tim::complete_list_t;

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

using component_enum_t  = std::vector<TIMEMORY_COMPONENT>;
using record_map_t      = std::unordered_map<uint64_t, complete_list_t*>;
using components_keys_t = std::unordered_map<std::string, uint64_t>;

static uint64_t         uniqID = 0;
static component_enum_t components;
static std::string      spacer =
    "#-------------------------------------------------------------------------#";

//--------------------------------------------------------------------------------------//

static components_keys_t*&
get_component_keys()
{
    static thread_local components_keys_t* _instance = new components_keys_t();
    return _instance;
}

//--------------------------------------------------------------------------------------//

static record_map_t*&
get_record_map()
{
    static thread_local record_map_t* _instance = new record_map_t();
    return _instance;
}

//--------------------------------------------------------------------------------------//
// default components to record -- maybe should be empty?
//
inline std::string
get_default_components()
{
    return "real_clock, user_clock, system_clock, cpu_util, page_rss, peak_rss";
}

//--------------------------------------------------------------------------------------//
//
//      TiMemory start/stop
//
//--------------------------------------------------------------------------------------//

void
record_start(const char* name, uint64_t* kernid, const component_enum_t& types)
{
    /*
    auto itr = components_keys.find(std::string(name));
    if(itr != components_keys.end())
    {
        *kernid = itr->second;
        tim::initialize(*(record_map[*kernid]), types);
        record_map[*kernid]->start();  // start recording
    }
    else
    {*/
    *kernid = uniqID++;
    // (*get_component_keys())[name] = *kernid;
    auto obj = new complete_list_t(name, true, tim::language::cxx(), *kernid);
    tim::initialize(*obj, types);
    (*get_record_map())[*kernid] = obj;
    (*get_record_map())[*kernid]->start();
    //}
}

//--------------------------------------------------------------------------------------//

void
record_stop(uint64_t kernid)
{
    (*get_record_map())[kernid]->stop();  // stop recording
    delete(*get_record_map())[kernid];
    get_record_map()->erase(kernid);
}

//--------------------------------------------------------------------------------------//
//
//      TiMemory symbols
//
//--------------------------------------------------------------------------------------//

extern "C" void
timemory_init_library(int argc, char** argv)
{
    if(tim::settings::verbose() > 0)
    {
        printf("%s\n", spacer.c_str());
        printf("\tInitialization of timemory preload...\n");
        printf("%s\n\n", spacer.c_str());
    }

    tim::settings::auto_output() = true;   // print when destructing
    tim::settings::cout_output() = true;   // print to stdout
    tim::settings::text_output() = true;   // print text files
    tim::settings::json_output() = false;  // print to json
    tim::timemory_init(argc, argv);

    components =
        tim::enumerate_components(get_default_components(), "TIMEMORY_COMPONENTS");
}

//--------------------------------------------------------------------------------------//

extern "C" void
timemory_finalize_library()
{
    if(!get_record_map())
        return;

    if(tim::settings::verbose() > 0)
    {
        printf("\n%s\n", spacer.c_str());
        printf("\tFinalization of timemory preload...\n");
        printf("%s\n\n", spacer.c_str());
    }

    for(auto& itr : (*get_record_map()))
    {
        if(itr.second)
            itr.second->stop();
        delete itr.second;
    }
    get_record_map()->clear();

    // Compensate for Intel compiler not allowing auto output
#if defined(__INTEL_COMPILER)
    complete_list_t::print_storage();
#endif

    // PGI and Intel compilers don't respect destruction order
#if defined(__PGI) || defined(__INTEL_COMPILER)
    tim::settings::auto_output() = false;
#endif

    delete get_record_map();
    delete get_component_keys();
    get_record_map()     = nullptr;
    get_component_keys() = nullptr;
}

//--------------------------------------------------------------------------------------//

extern "C" void
timemory_begin_record(const char* name, uint64_t* kernid)
{
    record_start(name, kernid, components);
    if(tim::settings::verbose() > 2)
        printf("beginning record for '%s' (id = %lli)...\n", name,
               (long long int) *kernid);
}

//--------------------------------------------------------------------------------------//

extern "C" void
timemory_begin_record_types(const char* name, uint64_t* kernid, const char* ctypes)
{
    record_start(name, kernid, tim::enumerate_components(std::string(ctypes)));
    if(tim::settings::verbose() > 2)
        printf("beginning record for '%s' (id = %lli)...\n", name,
               (long long int) *kernid);
}

//--------------------------------------------------------------------------------------//

extern "C" void
timemory_end_record(uint64_t kernid)
{
    record_stop(kernid);
    if(tim::settings::verbose() > 2)
        printf("ending record for %lli...\n", (long long int) kernid);
}

//======================================================================================//
