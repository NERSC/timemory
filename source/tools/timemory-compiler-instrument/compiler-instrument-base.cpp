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

#define TIMEMORY_CEREAL_DLL_EXPORT
#define TIMEMORY_INTERNAL __attribute__((visibility("internal")))
#define TIMEMORY_VISIBILITY(...) TIMEMORY_INTERNAL
#define TIMEMORY_INTERNAL_NO_INSTRUMENT TIMEMORY_INTERNAL TIMEMORY_NEVER_INSTRUMENT

#include "timemory/timemory.hpp"
#include "timemory/trace.hpp"

#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <limits>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

extern "C"
{
    void timemory_profile_func_enter(void* this_fn, void* call_site)
        TIMEMORY_ATTRIBUTE(visibility("default")) TIMEMORY_NEVER_INSTRUMENT;
    void timemory_profile_func_exit(void* this_fn, void* call_site)
        TIMEMORY_ATTRIBUTE(visibility("default")) TIMEMORY_NEVER_INSTRUMENT;
}

//--------------------------------------------------------------------------------------//

using namespace tim::component;

struct pthread_gotcha;

template <typename Tp>
using uomap_t     = std::unordered_map<const void*, std::unordered_map<const void*, Tp>>;
using trace_set_t = tim::available_list_t;
using trace_vec_t = std::vector<std::unique_ptr<trace_set_t>>;
using throttle_map_t   = uomap_t<bool>;
using overhead_map_t   = uomap_t<std::pair<monotonic_clock, size_t>>;
using trace_map_t      = uomap_t<trace_vec_t>;
using label_map_t      = std::unordered_map<const void*, size_t>;
using empty_tuple_t    = tim::component_tuple<>;
using pthread_gotcha_t = tim::component::gotcha<2, empty_tuple_t, pthread_gotcha>;
using pthread_bundle_t = tim::auto_tuple<pthread_gotcha_t>;

//--------------------------------------------------------------------------------------//

template <size_t... Idx>
static auto
get_storage(tim::index_sequence<Idx...>) TIMEMORY_INTERNAL_NO_INSTRUMENT;
//
template <size_t... Idx>
static auto
init_storage(tim::index_sequence<Idx...>) TIMEMORY_INTERNAL_NO_INSTRUMENT;
//
static bool&
get_enabled() TIMEMORY_INTERNAL_NO_INSTRUMENT;
static auto&
get_first() TIMEMORY_INTERNAL_NO_INSTRUMENT;
static auto&
get_overhead() TIMEMORY_INTERNAL_NO_INSTRUMENT;
static auto&
get_throttle() TIMEMORY_INTERNAL_NO_INSTRUMENT;
static auto&
get_trace_map() TIMEMORY_INTERNAL_NO_INSTRUMENT;
static unsigned long
get_trace_size() TIMEMORY_INTERNAL_NO_INSTRUMENT;
static auto&
get_label_map() TIMEMORY_INTERNAL_NO_INSTRUMENT;
static auto
get_label(void*, void*) TIMEMORY_INTERNAL_NO_INSTRUMENT;
static auto&
get_init() TIMEMORY_INTERNAL_NO_INSTRUMENT;
//
static void
initialize() TIMEMORY_INTERNAL_NO_INSTRUMENT;
static void
allocate() TIMEMORY_INTERNAL_NO_INSTRUMENT;
static void
finalize() TIMEMORY_INTERNAL_NO_INSTRUMENT;
//
static std::shared_ptr<pthread_bundle_t>
setup_pthread_gotcha() TIMEMORY_INTERNAL_NO_INSTRUMENT;

//--------------------------------------------------------------------------------------//

namespace
{
bool        m_default_enabled     = (std::atexit(&finalize), true);
const void* null_site             = nullptr;
static auto pthread_gotcha_handle = setup_pthread_gotcha();
}  // namespace

#if !defined(CALL_SITE)
#    define CALL_SITE call_site
#endif

//--------------------------------------------------------------------------------------//

template <size_t... Idx>
auto
get_storage(tim::index_sequence<Idx...>)
{
    std::array<tim::storage_initializer, sizeof...(Idx)> _data;
    TIMEMORY_FOLD_EXPRESSION(_data[Idx] = tim::storage_initializer::get<Idx>());
    return _data;
}

//--------------------------------------------------------------------------------------//

template <size_t... Idx>
static auto
init_storage(tim::index_sequence<Idx...>)
{
    std::array<tim::storage_initializer, sizeof...(Idx)> _data;
    TIMEMORY_FOLD_EXPRESSION(_data[Idx] = tim::storage_initializer::get<Idx>());
    return _data;
}

//--------------------------------------------------------------------------------------//

bool&
get_enabled()
{
    static auto _instance = new bool{ tim::settings::enabled() };
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

static unsigned long
get_trace_size()
{
    using tuple_type = tim::convert_t<tim::available_types_t, std::tuple<>>;
    return tim::manager::get_storage<tuple_type>::size();
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
        constexpr auto _rc = std::regex_constants::optimize | std::regex_constants::egrep;
        if(std::regex_match(finfo.dli_sname, std::regex("^[_]*main$", _rc)))
        {
            printf("[%i]> timemory-compiler-instrument will close after '%s' returns\n",
                   tim::process::get_id(), finfo.dli_sname);
            get_first() = { this_fn, call_site };
        }
    }

    auto _hash = tim::add_hash_id(tim::demangle(finfo.dli_sname));
    get_label_map()->insert({ this_fn, _hash });
    return _hash;
}

//--------------------------------------------------------------------------------------//

static auto&
get_init()
{
    static auto _comp_env = tim::get_env<std::string>("TIMEMORY_COMPILER_COMPONENTS", "");
    static auto _glob_env = tim::settings::global_components();
    static auto _used_env = (_comp_env.empty()) ? _glob_env : _comp_env;
    static auto _enum_env = tim::enumerate_components(tim::delimit(_used_env, ",;: "));
    return _enum_env;
}

//--------------------------------------------------------------------------------------//

static void
initialize()
{
    // default settings
    if(tim::get_env<std::string>("TIMEMORY_COUT_OUTPUT", "").empty())
        tim::settings::cout_output() = false;
    if(tim::get_env<std::string>("TIMEMORY_GLOBAL_COMPONENTS", "").empty())
        tim::settings::global_components() = "wall_clock";

    // initialization
    char* argv = new char[128];
    strcpy(argv, "compiler-instrumentation");
    tim::timemory_init(1, &argv);
    delete[] argv;

    // parse environment
    tim::settings::parse();
    tim::settings::suppress_parsing() = true;
    tim::settings::plot_output()      = false;

    // output path
    if(tim::get_env<std::string>("TIMEMORY_OUTPUT_PATH", "").empty())
    {
        tim::settings::output_path() = tim::get_env<std::string>(
            "TIMEMORY_COMPILER_OUTPUT_PATH", "timemory-compiler-instrumentation-output");
    }
    else
    {
        tim::settings::output_prefix() =
            tim::get_env<std::string>("TIMEMORY_COMPILER_OUTPUT_PREFIX", "compiler-");
    }
    tim::consume_parameters(
        get_storage(tim::make_index_sequence<TIMEMORY_COMPONENTS_END>{}));
}

//--------------------------------------------------------------------------------------//

static void
allocate()
{}

//--------------------------------------------------------------------------------------//

void
finalize()
{
    get_enabled() = false;
    auto lk       = new tim::trace::lock<tim::trace::compiler>{};
    if(get_trace_map())
        printf("[%i]> timemory-compiler-instrument: %lu results\n",
               tim::process::get_id(), (unsigned long) get_trace_size());
    else
        printf("[%i]> timemory-compiler-instrument: finalizing...\n",
               tim::process::get_id());
    lk->acquire();

    // clean up trace map
    if(get_trace_map())
    {
        for(auto& sitr : *get_trace_map())
        {
            for(auto& fitr : sitr.second)
                for(auto ritr = fitr.second.rbegin(); ritr != fitr.second.rend(); ++ritr)
                    (*ritr)->stop();
        }
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

    auto _manager = tim::manager::instance();
    if(_manager && !_manager->is_finalized())
        tim::timemory_finalize();
}

//--------------------------------------------------------------------------------------//
//
//      timemory symbols
//
//--------------------------------------------------------------------------------------//

extern "C"
{
    void timemory_profile_func_enter(void* this_fn, void* call_site)
    {
        tim::trace::lock<tim::trace::compiler> lk{};
        if(!lk || !get_enabled())
            return;

        static auto _initialized = (initialize(), true);
        static auto _allocated   = (allocate(), true);

        const auto& _trace_map = get_trace_map();
        if(!_trace_map)
            return;

        auto _label = get_label(this_fn, call_site);
        // puts(tim::get_hash_identifier(_label).c_str());

        const auto& _overhead = get_overhead();
        const auto& _throttle = get_throttle();
        if((*_throttle)[CALL_SITE][this_fn])
            return;

        (*_trace_map)[CALL_SITE][this_fn].emplace_back(
            std::make_unique<trace_set_t>(_label));
        tim::initialize(*(*_trace_map)[CALL_SITE][this_fn].back(), get_init());
        (*_trace_map)[CALL_SITE][this_fn].back()->start();
        if((*_trace_map)[CALL_SITE][this_fn].empty())
            (*_overhead)[CALL_SITE][this_fn].first.start();

        tim::consume_parameters(call_site, null_site, _initialized, _allocated);
    }
    //
    //----------------------------------------------------------------------------------//
    //
    void timemory_profile_func_exit(void* this_fn, void* call_site)
    {
        tim::trace::lock<tim::trace::compiler> lk{};
        if(!lk || !get_enabled())
            return;

        // puts(tim::get_hash_identifier(get_label(this_fn, call_site)).c_str());

        const bool _is_first =
            (get_first().first == this_fn && get_first().second == call_site);

        if(_is_first)
            get_enabled() = false;

        const auto& _trace_map = get_trace_map();
        if(!_trace_map)
            return;

        const auto& _overhead = get_overhead();
        const auto& _throttle = get_throttle();

        if((*_throttle)[CALL_SITE][this_fn])
            return;

        if((*_trace_map)[CALL_SITE][this_fn].empty())
            return;

        if((*_trace_map)[CALL_SITE][this_fn].size() == 1)
            (*_overhead)[CALL_SITE][this_fn].first.stop();
        (*_trace_map)[CALL_SITE][this_fn].back()->stop();
        (*_trace_map)[CALL_SITE][this_fn].pop_back();

        if(_is_first)
        {
            finalize();
            return;
        }

        static auto throttle_count = tim::settings::throttle_count();
        static auto throttle_value = tim::settings::throttle_value();
        auto        _count         = ++((*_overhead)[CALL_SITE][this_fn].second);
        if(_count % throttle_count == 0)
        {
            auto _accum = (*_overhead)[CALL_SITE][this_fn].first.get_accum() / _count;
            if(_accum < throttle_value)
                (*_throttle)[CALL_SITE][this_fn] = true;
            (*_overhead)[CALL_SITE][this_fn].first.reset();
            (*_overhead)[CALL_SITE][this_fn].second = 0;
        }

        tim::consume_parameters(call_site, null_site);
    }

}  // extern "C"

//--------------------------------------------------------------------------------------//

struct pthread_gotcha : tim::component::base<pthread_gotcha, void>
{
    struct TIMEMORY_HIDDEN wrapper
    {
        typedef void* (*routine_t)(void*);

        wrapper(routine_t _routine, void* _arg, bool _debug)
        : m_routine(_routine)
        , m_arg(_arg)
        , m_debug(_debug)
        {}

        void* operator()() const { return m_routine(m_arg); }

        bool debug() const { return m_debug; }

        static void* wrap(void* _arg)
        {
            if(!_arg)
                return nullptr;

            // convert the argument
            wrapper* _wrapper = static_cast<wrapper*>(_arg);
            if(_wrapper->debug())
                PRINT_HERE("%s", "Creating timemory manager");
            // create the manager and initialize the storage
            auto _tlm = tim::manager::instance();
            assert(_tlm.get() != nullptr);
            if(_wrapper->debug())
                PRINT_HERE("%s", "Initializing timemory component storage");
            auto _ini = init_storage(tim::make_index_sequence<TIMEMORY_COMPONENTS_END>{});
            tim::consume_parameters(_tlm, _ini);
            if(_wrapper->debug())
                PRINT_HERE("%s", "Executing original function");
            // execute the original function
            return (*_wrapper)();
        }

    private:
        routine_t m_routine = nullptr;
        void*     m_arg     = nullptr;
        bool      m_debug   = false;
    };

    // pthread_create
    int operator()(pthread_t* thread, const pthread_attr_t* attr,
                   void* (*start_routine)(void*), void*     arg)
    {
        if(m_debug)
            PRINT_HERE("%s", "wrapping pthread_create");
        auto* _obj = new wrapper(start_routine, arg, m_debug);
        if(m_debug)
            PRINT_HERE("%s", "executing pthread_create");
        return pthread_create(thread, attr, &wrapper::wrap, static_cast<void*>(_obj));
    }

    // pthread_join
    int operator()(pthread_t thread, void** retval)
    {
        tim::trace::lock<tim::trace::threading> lk{};
        lk.acquire();  // ensure the lock was acquired
        if(m_debug)
            PRINT_HERE("%s", "wrapping pthread_join");
        if(m_debug)
            PRINT_HERE("%s", "finalizing timemory manager");
        tim::manager::instance()->finalize();
        if(m_debug)
            PRINT_HERE("%s", "executing pthread_join");
        return pthread_join(thread, retval);
    }

private:
    bool m_debug =
        (tim::settings::instance()) ? tim::settings::instance()->get_debug() : false;
};

//--------------------------------------------------------------------------------------//

std::shared_ptr<pthread_bundle_t>
setup_pthread_gotcha()
{
#if defined(TIMEMORY_USE_GOTCHA)
    pthread_gotcha_t::get_initializer() = []() {
        TIMEMORY_C_GOTCHA(pthread_gotcha_t, 0, pthread_create);
        TIMEMORY_C_GOTCHA(pthread_gotcha_t, 1, pthread_join);
    };
#endif
    return std::make_shared<pthread_bundle_t>("pthread");
}

//--------------------------------------------------------------------------------------//
