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

#include "timemory/timemory.hpp"
#include "timemory/trace.hpp"
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <limits>

extern "C"
{
    void __cyg_profile_func_enter(void* this_fn, void* call_site)
        TIMEMORY_VISIBILITY("default") TIMEMORY_NEVER_INSTRUMENT;
    void __cyg_profile_func_exit(void* this_fn, void* call_site)
        TIMEMORY_VISIBILITY("default") TIMEMORY_NEVER_INSTRUMENT;
}

//--------------------------------------------------------------------------------------//

static void
initialize() TIMEMORY_NEVER_INSTRUMENT;
static void
allocate() TIMEMORY_NEVER_INSTRUMENT;
static void
finalize() TIMEMORY_NEVER_INSTRUMENT;
static bool&
get_enabled() TIMEMORY_NEVER_INSTRUMENT;
static auto&
get_first() TIMEMORY_NEVER_INSTRUMENT;
static auto&
get_overhead() TIMEMORY_NEVER_INSTRUMENT;
static auto&
get_throttle() TIMEMORY_NEVER_INSTRUMENT;
static auto&
get_trace_map() TIMEMORY_NEVER_INSTRUMENT;
static auto&
get_label_map() TIMEMORY_NEVER_INSTRUMENT;
static auto
get_label(void*, void*) TIMEMORY_NEVER_INSTRUMENT;

//--------------------------------------------------------------------------------------//

using namespace tim::component;

template <typename Tp>
using uomap_t = std::unordered_map<void*, std::unordered_map<void*, Tp>>;

using trace_set_t =
    tim::component_bundle<TIMEMORY_API, wall_clock, cpu_clock, peak_rss, page_rss,
                          virtual_memory, read_char, written_char, read_bytes,
                          written_bytes, voluntary_context_switch, num_minor_page_faults>;
using trace_vec_t    = std::vector<trace_set_t>;
using throttle_map_t = uomap_t<bool>;
using overhead_map_t = uomap_t<std::pair<wall_clock, size_t>>;
using trace_map_t    = uomap_t<trace_vec_t>;
using label_map_t    = std::unordered_map<void*, size_t>;

//--------------------------------------------------------------------------------------//

bool m_default_enabled = (std::atexit(&finalize), true);

//--------------------------------------------------------------------------------------//

bool&
get_enabled()
{
    static auto _instance = new bool{ true };
    return *_instance;
}

//--------------------------------------------------------------------------------------//

static auto&
get_first()
{
    static auto _instance = std::pair<void*, void*>(nullptr, nullptr);
    return _instance;
}

//--------------------------------------------------------------------------------------//

static auto&
get_overhead()
{
    static thread_local auto _instance = new overhead_map_t{};
    return _instance;
}

//--------------------------------------------------------------------------------------//

static auto&
get_throttle()
{
    static thread_local auto _instance = new throttle_map_t{};
    return _instance;
}

//--------------------------------------------------------------------------------------//

static auto&
get_trace_map()
{
    static thread_local auto _instance = new trace_map_t{};
    return _instance;
}

//--------------------------------------------------------------------------------------//

static auto&
get_label_map()
{
    static thread_local auto _instance = new label_map_t{};
    return _instance;
}

//--------------------------------------------------------------------------------------//

static auto
get_label(void* this_fn, void* call_site)
{
    auto itr = get_label_map()->find(this_fn);
    if(itr != get_label_map()->end())
        return itr->second;

    Dl_info finfo;
    dladdr(this_fn, &finfo);

    if(!finfo.dli_saddr)
    {
        auto _key  = TIMEMORY_JOIN("", this_fn);
        auto _hash = tim::add_hash_id(_key);
        get_label_map()->insert({ this_fn, _hash });
        return _hash;
    }

    if(get_first().first == nullptr)
    {
        if(strcmp(finfo.dli_sname, "main") == 0)
            get_first() = { this_fn, call_site };
    }

    auto _hash = tim::add_hash_id(tim::demangle(finfo.dli_sname));
    get_label_map()->insert({ this_fn, _hash });
    return _hash;
}

//--------------------------------------------------------------------------------------//

static void
initialize()
{
    tim::set_env("TIMEMORY_COUT_OUTPUT", "OFF", 0);
    char* argv = new char[128];
    strcpy(argv, "compiler-instrumentation");
    tim::timemory_init(1, &argv);
    delete[] argv;
    tim::settings::parse();
}

//--------------------------------------------------------------------------------------//

static void
allocate()
{
    trace_set_t::get_initializer() = [](trace_set_t&) {};
}

//--------------------------------------------------------------------------------------//

void
finalize()
{
    get_enabled() = false;
    auto lk       = new tim::trace::lock<tim::trace::compiler>{};
    PRINT_HERE("%s", "finalizing compiler instrumentation");
    lk->acquire();

    // clean up trace map
    if(get_trace_map())
    {
        for(auto& sitr : *get_trace_map())
            for(auto& fitr : sitr.second)
                for(auto ritr = fitr.second.rbegin(); ritr != fitr.second.rend(); ++ritr)
                    ritr->stop();
        get_trace_map()->clear();
        delete get_trace_map();
        get_trace_map() = nullptr;
    }

    // clean up overhead map
    if(get_overhead())
        get_overhead()->clear();
    delete get_overhead();
    get_overhead() = nullptr;

    // clean up throttle map
    if(get_throttle())
        get_throttle()->clear();
    delete get_throttle();
    get_throttle() = nullptr;

    tim::timemory_finalize();
}

//--------------------------------------------------------------------------------------//
//
//      timemory symbols
//
//--------------------------------------------------------------------------------------//

extern "C"
{
    void __cyg_profile_func_enter(void* this_fn, void* call_site)
    {
        tim::trace::lock<tim::trace::compiler> lk{};
        if(!lk || !get_enabled())
            return;

        static auto _initialized = (initialize(), true);

        auto& _trace_map = get_trace_map();
        if(!_trace_map)
            return;

        auto _label = get_label(this_fn, call_site);
        if(!get_first().first)
            return;

        static auto _allocated = (allocate(), true);

        // auto&       _overhead = get_overhead();
        // const auto& _throttle = get_throttle();
        // if((*_throttle)[call_site][this_fn])
        //    return;

        (*_trace_map)[call_site][this_fn].emplace_back(trace_set_t(_label));
        (*_trace_map)[call_site][this_fn].back().start();
        //(*_overhead)[call_site][this_fn].first.start();

        tim::consume_parameters(_initialized, _allocated);
    }
    //
    //----------------------------------------------------------------------------------//
    //
    void __cyg_profile_func_exit(void* this_fn, void* call_site)
    {
        // tim::trace::lock<tim::trace::compiler> lk{};
        // if(!get_enabled() || !get_first().first)
        //    return;

        if(get_first().first == this_fn && get_first().second == call_site)
        {
            get_enabled() = false;
            finalize();
            return;
        }

        auto& _trace_map = get_trace_map();
        if(!_trace_map)
            return;

        // auto& _overhead = get_overhead();
        // auto& _throttle = get_throttle();

        // if((*_throttle)[call_site][this_fn])
        //    return;

        //(*_overhead)[call_site][this_fn].first.stop();

        if((*_trace_map)[call_site][this_fn].empty())
            return;
        else
        {
            (*_trace_map)[call_site][this_fn].back().stop();
            (*_trace_map)[call_site][this_fn].pop_back();
        }

        /*
        if(_throttle && (*_throttle)[call_site][this_fn] > 0)
            return;

        if(!_overhead)
            return;

        auto _count = ++((*_overhead)[call_site][this_fn].second);
        if(_count % tim::settings::throttle_count() == 0)
        {
            auto _accum = (*_overhead)[call_site][this_fn].first.get_accum() / _count;
            if(_accum < tim::settings::throttle_value())
                (*_throttle)[call_site][this_fn] = true;
            (*_overhead)[call_site][this_fn].first.reset();
            (*_overhead)[call_site][this_fn].second = 0;
        }*/
    }
}  // extern "C"
