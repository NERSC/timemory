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

#include <cstdarg>
#include <deque>
#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace tim::component;

#if !defined(TIMEMORY_LIBRARY_TYPE)
#    define TIMEMORY_LIBRARY_TYPE tim::complete_list_t;
#endif

#if defined(__GNUC__)
#    define API tim_api __attribute__((weak))
#else
#    define API tim_api
#endif

//======================================================================================//

extern "C"
{
    typedef void (*timemory_create_func_t)(const char*, uint64_t*, int, int*);
    typedef void (*timemory_delete_func_t)(uint64_t);

    API timemory_create_func_t timemory_create_function = nullptr;
    API timemory_delete_func_t timemory_delete_function = nullptr;
}

//======================================================================================//

using toolset_t          = TIMEMORY_LIBRARY_TYPE;
using record_map_t       = std::unordered_map<uint64_t, toolset_t>;
using component_enum_t   = std::vector<TIMEMORY_COMPONENT>;
using components_stack_t = std::deque<component_enum_t>;

static std::string spacer =
    "#-------------------------------------------------------------------------#";

//--------------------------------------------------------------------------------------//

static record_map_t&
get_record_map()
{
    static thread_local record_map_t _instance;
    return _instance;
}

//--------------------------------------------------------------------------------------//

static components_stack_t&
get_components_stack()
{
    static thread_local components_stack_t _instance;
    return _instance;
}

//--------------------------------------------------------------------------------------//
// default components to record -- maybe should be empty?
//
inline std::string&
get_default_components()
{
    static thread_local std::string _instance = "real_clock,cpu_clock,cpu_util,peak_rss";
    return _instance;
}

//--------------------------------------------------------------------------------------//
// default components to record -- maybe should be empty?
//
inline const component_enum_t&
get_current_components()
{
    auto& _stack = get_components_stack();
    if(_stack.size() == 0)
    {
        _stack.push_back(
            tim::enumerate_components(get_default_components(), "TIMEMORY_COMPONENTS"));
    }
    return _stack.back();
}

//--------------------------------------------------------------------------------------//
//
//      TiMemory symbols
//
//--------------------------------------------------------------------------------------//

extern "C"
{
    //----------------------------------------------------------------------------------//
    //  get a unique id
    //
    API uint64_t timemory_get_unique_id(void)
    {
        // the maps are thread-local so no concerns for data-race here since
        // two threads updating at once and subsequently losing once of the updates
        // still results in a unique id for that thread
        static uint64_t uniqID = 0;
        return uniqID++;
    }

    //----------------------------------------------------------------------------------//
    //  create a toolset of measurements
    //
    API void timemory_create_record(const char* name, uint64_t* id, int n, int* ctypes)
    {
        if(timemory_create_function)
        {
            (*timemory_create_function)(name, id, n, ctypes);
            return;
        }
        // else: provide default behavior

        static thread_local auto& _record_map = get_record_map();
        *id                                   = timemory_get_unique_id();
        _record_map.insert({ *id, toolset_t(name, true, tim::settings::flat_profile()) });
        tim::initialize(_record_map[*id], n, ctypes);
        _record_map[*id].start();
        if(_record_map.bucket_count() > _record_map.size())
            _record_map.rehash(_record_map.size() + 10);
    }

    //----------------------------------------------------------------------------------//
    //  destroy a toolset of measurements
    //
    API void timemory_delete_record(uint64_t id)
    {
        if(timemory_delete_function)
            (*timemory_delete_function)(id);
        else if(get_record_map().find(id) != get_record_map().end())
        {
            static thread_local auto& _record_map = get_record_map();
            // stop recording, destroy objects, and erase key from map
            _record_map[id].stop();
            _record_map.erase(id);
        }
    }

    //----------------------------------------------------------------------------------//
    //  initialize the library
    //
    API void timemory_init_library(int argc, char** argv)
    {
        if(tim::settings::verbose() > 0)
        {
            printf("%s\n", spacer.c_str());
            printf("\tInitialization of timemory library...\n");
            printf("%s\n\n", spacer.c_str());
        }

        tim::settings::auto_output() = true;   // print when destructing
        tim::settings::cout_output() = true;   // print to stdout
        tim::settings::text_output() = true;   // print text files
        tim::settings::json_output() = false;  // print to json
        tim::timemory_init(argc, argv);
    }

    //----------------------------------------------------------------------------------//
    //  finalize the library
    API void timemory_finalize_library(void)
    {
        if(tim::settings::enabled() == false && get_record_map().size() == 0)
            return;

        auto& _record_map = get_record_map();

        if(tim::settings::verbose() > 0)
        {
            printf("\n%s\n", spacer.c_str());
            printf("\tFinalization of timemory library...\n");
            printf("%s\n\n", spacer.c_str());
        }

        // put keys into a set so that a potential LD_PRELOAD for timemory_delete_record
        // is called and there is not a concern for the map iterator
        std::unordered_set<uint64_t> keys;
        for(auto& itr : _record_map)
            keys.insert(itr.first);

        // delete all the records
        for(auto& itr : keys)
            timemory_delete_record(itr);

        // clear the map
        _record_map.clear();

        // Compensate for Intel compiler not allowing auto output
#if defined(__INTEL_COMPILER)
        toolset_t::print_storage();
#endif

        // PGI and Intel compilers don't respect destruction order
#if defined(__PGI) || defined(__INTEL_COMPILER)
        tim::settings::auto_output() = false;
#endif
    }

    //----------------------------------------------------------------------------------//

    API void timemory_set_default(const char* _component_string)
    {
        get_default_components()         = std::string(_component_string);
        static thread_local auto& _stack = get_components_stack();
        _stack.push_back(tim::enumerate_components(_component_string));
    }

    //----------------------------------------------------------------------------------//

    API void timemory_push_components(const char* _component_string)
    {
        static thread_local auto& _stack = get_components_stack();
        _stack.push_back(tim::enumerate_components(_component_string));
    }

    //----------------------------------------------------------------------------------//

    API void timemory_push_components_enum(int types, ...)
    {
        static thread_local auto& _stack = get_components_stack();

        component_enum_t comp({ static_cast<TIMEMORY_COMPONENT>(types) });
        va_list          args;
        va_start(args, types);
        for(int i = 0; i < TIMEMORY_COMPONENTS_END; ++i)
        {
            auto enum_arg = static_cast<TIMEMORY_COMPONENT>(va_arg(args, int));
            if(enum_arg >= TIMEMORY_COMPONENTS_END)
                break;
            comp.push_back(enum_arg);
        }
        va_end(args);

        _stack.push_back(comp);
    }

    //----------------------------------------------------------------------------------//

    API void timemory_pop_components(void)
    {
        static thread_local auto& _stack = get_components_stack();
        if(_stack.size() > 1)
            _stack.pop_back();
    }

    //----------------------------------------------------------------------------------//

    API void timemory_begin_record(const char* name, uint64_t* id)
    {
        if(tim::settings::enabled() == false)
        {
            *id = std::numeric_limits<uint64_t>::max();
            return;
        }
        auto& comp = get_current_components();
        timemory_create_record(name, id, comp.size(), (int*) (comp.data()));

#if defined(DEBUG)
        if(tim::settings::verbose() > 2)
            printf("beginning record for '%s' (id = %lli)...\n", name,
                   (long long int) *id);
#endif
    }

    //----------------------------------------------------------------------------------//

    API void timemory_begin_record_types(const char* name, uint64_t* id,
                                         const char* ctypes)
    {
        if(tim::settings::enabled() == false)
        {
            *id = std::numeric_limits<uint64_t>::max();
            return;
        }

        auto comp = tim::enumerate_components(std::string(ctypes));
        timemory_create_record(name, id, comp.size(), (int*) (comp.data()));

#if defined(DEBUG)
        if(tim::settings::verbose() > 2)
            printf("beginning record for '%s' (id = %lli)...\n", name,
                   (long long int) *id);
#endif
    }

    //----------------------------------------------------------------------------------//

    API void timemory_begin_record_enum(const char* name, uint64_t* id, ...)
    {
        if(tim::settings::enabled() == false)
        {
            *id = std::numeric_limits<uint64_t>::max();
            return;
        }

        component_enum_t comp;
        va_list          args;
        va_start(args, id);
        for(int i = 0; i < TIMEMORY_COMPONENTS_END; ++i)
        {
            auto enum_arg = static_cast<TIMEMORY_COMPONENT>(va_arg(args, int));
            if(enum_arg >= TIMEMORY_COMPONENTS_END)
                break;
            comp.push_back(enum_arg);
        }
        va_end(args);

        timemory_create_record(name, id, comp.size(), (int*) (comp.data()));

#if defined(DEBUG)
        if(tim::settings::verbose() > 2)
            printf("beginning record for '%s' (id = %lli)...\n", name,
                   (long long int) *id);
#endif
    }

    //----------------------------------------------------------------------------------//

    API uint64_t timemory_get_begin_record(const char* name)
    {
        if(tim::settings::enabled() == false)
            return std::numeric_limits<uint64_t>::max();

        uint64_t id   = 0;
        auto&    comp = get_current_components();
        timemory_create_record(name, &id, comp.size(), (int*) (comp.data()));

#if defined(DEBUG)
        if(tim::settings::verbose() > 2)
            printf("beginning record for '%s' (id = %lli)...\n", name,
                   (long long int) id);
#endif

        return id;
    }

    //----------------------------------------------------------------------------------//

    API uint64_t timemory_get_begin_record_types(const char* name, const char* ctypes)
    {
        if(tim::settings::enabled() == false)
            return std::numeric_limits<uint64_t>::max();

        uint64_t id   = 0;
        auto     comp = tim::enumerate_components(std::string(ctypes));
        timemory_create_record(name, &id, comp.size(), (int*) (comp.data()));

#if defined(DEBUG)
        if(tim::settings::verbose() > 2)
            printf("beginning record for '%s' (id = %lli)...\n", name,
                   (long long int) id);
#endif

        return id;
    }

    //----------------------------------------------------------------------------------//

    API uint64_t timemory_get_begin_record_enum(const char* name, ...)
    {
        if(tim::settings::enabled() == false)
            return std::numeric_limits<uint64_t>::max();

        uint64_t id = 0;

        component_enum_t comp;
        va_list          args;
        va_start(args, name);
        for(int i = 0; i < TIMEMORY_COMPONENTS_END; ++i)
        {
            auto enum_arg = static_cast<TIMEMORY_COMPONENT>(va_arg(args, int));
            if(enum_arg >= TIMEMORY_COMPONENTS_END)
                break;
            comp.push_back(enum_arg);
        }
        va_end(args);

        timemory_create_record(name, &id, comp.size(), (int*) (comp.data()));

#if defined(DEBUG)
        if(tim::settings::verbose() > 2)
            printf("beginning record for '%s' (id = %lli)...\n", name,
                   (long long int) id);
#endif

        return id;
    }

    //----------------------------------------------------------------------------------//

    API void timemory_end_record(uint64_t id)
    {
        if(id == std::numeric_limits<uint64_t>::max())
            return;

        timemory_delete_record(id);

#if defined(DEBUG)
        if(tim::settings::verbose() > 2)
            printf("ending record for %lli...\n", (long long int) id);
#endif
    }

    //==================================================================================//
    //
    //      Symbols for Fortran
    //
    //==================================================================================//

    void _timemory_create_record(const char* name, uint64_t* id, int n, int* ct)
    {
        timemory_create_record(name, id, n, ct);
    }

    void _timemory_delete_record(uint64_t id) { timemory_delete_record(id); }

    void _timemory_init_library(int argc, char** argv)
    {
        timemory_init_library(argc, argv);
    }

    void _timemory_finalize_library(void) { timemory_finalize_library(); }

    void _timemory_set_default(const char* components)
    {
        timemory_set_default(components);
    }

    void _timemory_push_components(const char* components)
    {
        timemory_push_components(components);
    }

    void _timemory_pop_components(void) { timemory_pop_components(); }

    void _timemory_begin_record(const char* name, uint64_t* id)
    {
        timemory_begin_record(name, id);
    }

    void _timemory_begin_record_types(const char* name, uint64_t* id, const char* ctypes)
    {
        timemory_begin_record_types(name, id, ctypes);
    }

    uint64_t _timemory_get_begin_record(const char* name)
    {
        return timemory_get_begin_record(name);
    }

    uint64_t _timemory_get_begin_record_types(const char* name, const char* ctypes)
    {
        return timemory_get_begin_record_types(name, ctypes);
    }

    void _timemory_end_record(uint64_t id) { return timemory_end_record(id); }

    //======================================================================================//

}  // extern "C"
