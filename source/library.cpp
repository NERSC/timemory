// MIT License
//
// Copyright (c) 2019, The Regents of the University of California,
// through Lawrence Berkeley National Laboratory (subject to receipt of any
// required approvals from the U.S. Dept. of Energy).  All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "timemory/timemory.hpp"

#include <deque>
#include <iostream>
#include <unordered_map>
#include <vector>

using namespace tim::component;

using auto_timer_t = tim::component_tuple<real_clock, system_clock, cpu_clock, cpu_util,
                                          page_rss, peak_rss>;

using complete_list_t = tim::complete_list_t;

//======================================================================================//

using component_enum_t   = std::vector<TIMEMORY_COMPONENT>;
using record_map_t       = std::unordered_map<uint64_t, complete_list_t*>;
using components_keys_t  = std::unordered_map<std::string, uint64_t>;
using components_stack_t = std::deque<component_enum_t>;

static uint64_t    uniqID = 0;
static std::string spacer =
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

static components_stack_t*&
get_components_stack()
{
    static thread_local components_stack_t* _instance = new components_stack_t();
    return _instance;
}

//--------------------------------------------------------------------------------------//
// default components to record -- maybe should be empty?
//
inline std::string&
get_default_components()
{
    static thread_local std::string _instance =
        "real_clock, user_clock, system_clock, cpu_util, page_rss, peak_rss";
    return _instance;
}

//--------------------------------------------------------------------------------------//
// default components to record -- maybe should be empty?
//
inline const component_enum_t&
get_current_components()
{
    auto* _stack = get_components_stack();
    if(_stack->size() == 0)
    {
        _stack->push_back(
            tim::enumerate_components(get_default_components(), "TIMEMORY_COMPONENTS"));
    }
    return _stack->back();
}

//--------------------------------------------------------------------------------------//
//
//      TiMemory start/stop
//
//--------------------------------------------------------------------------------------//

void
record_start(const char* name, uint64_t* kernid, const component_enum_t& types)
{
    *kernid  = uniqID++;
    auto obj = new complete_list_t(name, true, tim::language::cxx(), *kernid);
    tim::initialize(*obj, types);
    (*get_record_map())[*kernid] = obj;
    (*get_record_map())[*kernid]->start();
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

void
cleanup()
{
    delete get_record_map();
    delete get_component_keys();
    delete get_components_stack();
    get_record_map()       = nullptr;
    get_component_keys()   = nullptr;
    get_components_stack() = nullptr;
}

//--------------------------------------------------------------------------------------//
//
//      TiMemory symbols
//
//--------------------------------------------------------------------------------------//

extern "C"
{
    tim_api void timemory_init_library(int argc, char** argv)
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
    }

    //--------------------------------------------------------------------------------------//

    tim_api void timemory_finalize_library()
    {
        if(tim::settings::enabled() == false)
        {
            cleanup();
            return;
        }

        if(tim::settings::verbose() > 0)
        {
            printf("\n%s\n", spacer.c_str());
            printf("\tFinalization of timemory preload...\n");
            printf("%s\n\n", spacer.c_str());
        }

        if(get_record_map())
        {
            for(auto& itr : (*get_record_map()))
            {
                if(itr.second)
                    itr.second->stop();
                delete itr.second;
            }
            get_record_map()->clear();
        }

        // Compensate for Intel compiler not allowing auto output
#if defined(__INTEL_COMPILER)
        complete_list_t::print_storage();
#endif

        // PGI and Intel compilers don't respect destruction order
#if defined(__PGI) || defined(__INTEL_COMPILER)
        tim::settings::auto_output() = false;
#endif

        cleanup();
    }

    //--------------------------------------------------------------------------------------//

    tim_api void timemory_set_default(const char* _component_string)
    {
        if(tim::settings::enabled() == false)
            return;
        get_default_components() = std::string(_component_string);
        auto* _stack             = get_components_stack();
        if(_stack->size() == 0)
        {
            _stack->push_back(tim::enumerate_components(get_default_components(),
                                                        "TIMEMORY_COMPONENTS"));
        }
    }

    //--------------------------------------------------------------------------------------//

    tim_api void timemory_push_components(const char* _component_string)
    {
        if(tim::settings::enabled() == false)
            return;
        auto* _stack = get_components_stack();
        _stack->push_back(tim::enumerate_components(_component_string));
    }

    //--------------------------------------------------------------------------------------//

    tim_api void timemory_pop_components()
    {
        if(tim::settings::enabled() == false)
            return;
        auto* _stack = get_components_stack();
        if(_stack->size() > 1)
            _stack->pop_back();
    }

    //--------------------------------------------------------------------------------------//

    tim_api void timemory_begin_record(const char* name, uint64_t* kernid)
    {
        if(tim::settings::enabled() == false)
        {
            *kernid = std::numeric_limits<uint64_t>::max();
            return;
        }
        record_start(name, kernid, get_current_components());
        if(tim::settings::verbose() > 2)
            printf("beginning record for '%s' (id = %lli)...\n", name,
                   (long long int) *kernid);
    }

    //--------------------------------------------------------------------------------------//

    tim_api void timemory_begin_record_types(const char* name, uint64_t* kernid,
                                             const char* ctypes)
    {
        if(tim::settings::enabled() == false)
        {
            *kernid = std::numeric_limits<uint64_t>::max();
            return;
        }
        record_start(name, kernid, tim::enumerate_components(std::string(ctypes)));
        if(tim::settings::verbose() > 2)
            printf("beginning record for '%s' (id = %lli)...\n", name,
                   (long long int) *kernid);
    }

    //--------------------------------------------------------------------------------------//

    tim_api uint64_t timemory_get_begin_record(const char* name)
    {
        if(tim::settings::enabled() == false)
            return std::numeric_limits<uint64_t>::max();
        uint64_t kernid = 0;
        record_start(name, &kernid, get_current_components());
        if(tim::settings::verbose() > 2)
            printf("beginning record for '%s' (id = %lli)...\n", name,
                   (long long int) kernid);
        return kernid;
    }

    //--------------------------------------------------------------------------------------//

    tim_api uint64_t timemory_get_begin_record_types(const char* name, const char* ctypes)
    {
        if(tim::settings::enabled() == false)
            return std::numeric_limits<uint64_t>::max();
        uint64_t kernid = 0;
        record_start(name, &kernid, tim::enumerate_components(std::string(ctypes)));
        if(tim::settings::verbose() > 2)
            printf("beginning record for '%s' (id = %lli)...\n", name,
                   (long long int) kernid);
        return kernid;
    }

    //--------------------------------------------------------------------------------------//

    tim_api void timemory_end_record(uint64_t kernid)
    {
        if(kernid == std::numeric_limits<uint64_t>::max())
            return;
        record_stop(kernid);
        if(tim::settings::verbose() > 2)
            printf("ending record for %lli...\n", (long long int) kernid);
    }

    //======================================================================================//

}  // extern "C"
