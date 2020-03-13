// MIT License
//
// Copyright (c) 2020, The Regents of the University of California,
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

#include "timemory/runtime/configure.hpp"
#include "timemory/timemory.hpp"

#include <cstdarg>
#include <deque>
#include <iostream>
#include <stack>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace tim::component;

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

struct timemory_trace;
using trace_bundle_t     = user_bundle<0, timemory_trace>;
using traceset_t         = tim::component_tuple<trace_bundle_t>;
using toolset_t          = TIMEMORY_LIBRARY_TYPE;
using region_map_t       = std::unordered_map<std::string, std::stack<uint64_t>>;
using record_map_t       = std::unordered_map<uint64_t, toolset_t>;
using trace_map_t        = std::unordered_map<size_t, std::vector<traceset_t*>>;
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

static trace_map_t&
get_trace_map()
{
    static thread_local trace_map_t _instance;
    return _instance;
}

//--------------------------------------------------------------------------------------//

static region_map_t&
get_region_map()
{
    static thread_local region_map_t _instance;
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
    static thread_local std::string _instance = "wall_clock";
    return _instance;
}

//--------------------------------------------------------------------------------------//
// default components to record -- maybe should be empty?
//
inline const component_enum_t&
get_current_components()
{
    auto& _stack = get_components_stack();
    if(_stack.empty())
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
        {
            (*timemory_delete_function)(id);
        }
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

        tim::settings::auto_output() = true;  // print when destructing
        tim::settings::cout_output() = true;  // print to stdout
        tim::settings::text_output() = true;  // print text files
        tim::settings::json_output() = true;  // print to json

        tim::timemory_init(argc, argv);
    }

    //----------------------------------------------------------------------------------//
    //  finalize the library
    API void timemory_finalize_library(void)
    {
        if(tim::settings::enabled() == false && get_record_map().empty())
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

        for(auto& itr : get_trace_map())
        {
            for(auto& eitr : itr.second)
            {
                eitr->stop();
                delete eitr;
            }
            // delete all the records
            itr.second.clear();
        }

        // delete all the records
        get_trace_map().clear();

        // do the finalization
        tim::timemory_finalize();

        // PGI and Intel compilers don't respect destruction order
#if defined(__PGI) || defined(__INTEL_COMPILER)
        tim::settings::auto_output() = false;
#endif
    }

    //----------------------------------------------------------------------------------//
    //  pause the collection
    //
    API void timemory_pause(void) { tim::settings::enabled() = false; }

    //----------------------------------------------------------------------------------//
    //  resume the collection
    //
    API void timemory_resume(void) { tim::settings::enabled() = true; }

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

        component_enum_t comp({ types });
        va_list          args;
        va_start(args, types);
        for(int i = 0; i < TIMEMORY_COMPONENTS_END; ++i)
        {
            auto enum_arg = va_arg(args, int);
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
            auto enum_arg = va_arg(args, int);
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
            auto enum_arg = va_arg(args, int);
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

    //----------------------------------------------------------------------------------//

    API void timemory_init_trace(uint64_t id)
    {
        PRINT_HERE("[id = %llu]", (long long unsigned) id);
        auto& comp = get_current_components();
        tim::configure<trace_bundle_t>(comp);
    }

    //----------------------------------------------------------------------------------//

    API int64_t timemory_register_trace(const char* name)
    {
        using hasher_t                       = std::hash<std::string>;
        static thread_local auto& _trace_map = get_trace_map();
        size_t                    id         = hasher_t()(std::string(name));
        int64_t                   n          = _trace_map[id].size();

#if defined(DEBUG)
        if(tim::settings::verbose() > 2)
            printf("beginning trace for '%s' (id = %llu, offset = %lli)...\n", name,
                   (long long unsigned) id, (long long int) n);
#endif

        // _trace_map[id].push_back(
        //    new toolset_t(name, true, tim::settings::flat_profile()));
        // tim::initialize(*_trace_map[id].back(), get_current_components());
        _trace_map[id].push_back(
            new traceset_t(name, true, tim::settings::flat_profile()));
        _trace_map[id].back()->start();

        return n;
    }

    //----------------------------------------------------------------------------------//

    API void timemory_deregister_trace(const char* name)
    {
        using hasher_t                       = std::hash<std::string>;
        static thread_local auto& _trace_map = get_trace_map();
        size_t                    id         = hasher_t()(std::string(name));
        int64_t                   ntotal     = _trace_map[id].size();
        int64_t                   offset     = ntotal - 1;

#if defined(DEBUG)
        if(tim::settings::verbose() > 2)
            printf("ending trace for %llu [offset = %lli]...\n", (long long unsigned) id,
                   (long long int) offset);
#endif

        if(offset >= 0 && ntotal > 0)
        {
            _trace_map[id].back()->stop();
            delete _trace_map[id].back();
            _trace_map[id].pop_back();
        }
    }

    //----------------------------------------------------------------------------------//

    API void timemory_dyninst_init(void)
    {
        PRINT_HERE("%s", "");
        auto& comp = get_current_components();
        tim::configure<trace_bundle_t>(comp);
        tim::manager::use_exit_hook(false);
        tim::settings::destructor_report() = false;
        tim::set_env<std::string>("TIMEMORY_DESTRUCTOR_REPORT", "OFF");
    }

    //----------------------------------------------------------------------------------//

    API void timemory_dyninst_finalize(void)
    {
        PRINT_HERE("%s", "");
        timemory_finalize_library();
    }

    //----------------------------------------------------------------------------------//

    API void timemory_push_region(const char* name)
    {
        auto& region_map = get_region_map();
        auto  idx        = timemory_get_begin_record(name);
        region_map[name].push(idx);
    }

    //----------------------------------------------------------------------------------//

    API void timemory_pop_region(const char* name)
    {
        auto& region_map = get_region_map();
        auto  itr        = region_map.find(name);
        if(itr == region_map.end() || (itr != region_map.end() && itr->second.empty()))
            fprintf(stderr, "Warning! region '%s' does not exist!\n", name);
        else
        {
            uint64_t idx = itr->second.top();
            timemory_end_record(idx);
            itr->second.pop();
        }
    }

    //==================================================================================//
    //
    //      Symbols for Fortran
    //
    //==================================================================================//

    void timemory_create_record_(const char* name, uint64_t* id, int n, int* ct)
    {
        timemory_create_record(name, id, n, ct);
    }

    void timemory_delete_record_(uint64_t id) { timemory_delete_record(id); }

    void timemory_init_library_(int argc, char** argv)
    {
        timemory_init_library(argc, argv);
    }

    void timemory_finalize_library_(void) { timemory_finalize_library(); }

    void timemory_set_default_(const char* components)
    {
        timemory_set_default(components);
    }

    void timemory_push_components_(const char* components)
    {
        timemory_push_components(components);
    }

    void timemory_pop_components_(void) { timemory_pop_components(); }

    void timemory_begin_record_(const char* name, uint64_t* id)
    {
        timemory_begin_record(name, id);
    }

    void timemory_begin_record_types_(const char* name, uint64_t* id, const char* ctypes)
    {
        timemory_begin_record_types(name, id, ctypes);
    }

    uint64_t timemory_get_begin_record_(const char* name)
    {
        return timemory_get_begin_record(name);
    }

    uint64_t timemory_get_begin_record_types_(const char* name, const char* ctypes)
    {
        return timemory_get_begin_record_types(name, ctypes);
    }

    void timemory_end_record_(uint64_t id) { return timemory_end_record(id); }

    void timemory_push_region_(const char* name) { return timemory_push_region(name); }

    void timemory_pop_region_(const char* name) { return timemory_pop_region(name); }

    //======================================================================================//

}  // extern "C"
