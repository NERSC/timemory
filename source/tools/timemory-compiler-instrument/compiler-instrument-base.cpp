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
#define TIMEMORY_EXTERNAL __attribute__((visibility("default")))
#define TIMEMORY_VISIBILITY(...) TIMEMORY_INTERNAL
#define TIMEMORY_INTERNAL_NO_INSTRUMENT TIMEMORY_INTERNAL TIMEMORY_NEVER_INSTRUMENT
//
#define TIMEMORY_USE_TIMING_MINIMAL
#define TIMEMORY_DISABLE_BANNER

#include "timemory/api/macros.hpp"

// create an API for the compiler instrumentation whose singletons will not be shared
// with the default timemory API
TIMEMORY_DEFINE_NS_API(project, compiler_instrument)

// define the API for all instantiations before including any more timemory headers
#define TIMEMORY_API ::tim::project::compiler_instrument
#define TIMEMORY_SETTINGS_PREFIX "TIMEMORY_COMPILER_"

#include "timemory/components/macros.hpp"
#include "timemory/mpl/types.hpp"

// forward declare components to disable
// NOLINTNEXTLINE
namespace tim
{
namespace component
{
struct monotonic_clock;
struct monotonic_raw_clock;
struct user_mode_time;
struct kernel_mode_time;
}  // namespace component
}  // namespace tim
// disable these components for compiler instrumentation
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::monotonic_clock, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::monotonic_raw_clock, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::user_mode_time, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::kernel_mode_time, false_type)

#include "timemory/timemory.hpp"
#include "timemory/trace.hpp"

#include <array>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <functional>
#include <limits>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

extern "C"
{
    void timemory_profile_func_enter(void* this_fn, void* call_site)
        TIMEMORY_EXTERNAL TIMEMORY_NEVER_INSTRUMENT;
    void timemory_profile_func_exit(void* this_fn, void* call_site)
        TIMEMORY_EXTERNAL TIMEMORY_NEVER_INSTRUMENT;
    int timemory_profile_thread_init(void) TIMEMORY_EXTERNAL TIMEMORY_NEVER_INSTRUMENT;
    int timemory_profile_thread_fini(void) TIMEMORY_EXTERNAL TIMEMORY_NEVER_INSTRUMENT;
}

//--------------------------------------------------------------------------------------//

using namespace tim::component;

struct pthread_gotcha;
static int64_t primary_tidx   = 0;
static size_t  throttle_count = 1000;
static size_t  throttle_value = 10000;

#if !defined(TIMEMORY_USE_GOTCHA)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, pthread_gotcha, false_type)
#endif

template <typename Tp>
using uomap_t     = std::unordered_map<const void*, std::unordered_map<const void*, Tp>>;
using trace_set_t = tim::available_list_t;
using throttle_map_t = uomap_t<bool>;
using overhead_map_t = uomap_t<std::tuple<size_t, monotonic_clock, size_t>>;
using label_map_t    = uomap_t<size_t>;
using trace_data_t =
    std::vector<std::tuple<const void*, const void*, std::unique_ptr<trace_set_t>>>;
using empty_tuple_t    = tim::component_tuple<>;
using pthread_gotcha_t = tim::component::gotcha<2, empty_tuple_t, pthread_gotcha>;
using pthread_bundle_t = tim::auto_tuple<pthread_gotcha_t>;

//--------------------------------------------------------------------------------------//

// global data

static bool&
get_enabled() TIMEMORY_INTERNAL_NO_INSTRUMENT;

static bool&
is_finalized() TIMEMORY_INTERNAL_NO_INSTRUMENT;

static bool
get_debug() TIMEMORY_INTERNAL_NO_INSTRUMENT;

static auto&
get_max_depth() TIMEMORY_INTERNAL_NO_INSTRUMENT;

static auto&
get_first() TIMEMORY_INTERNAL_NO_INSTRUMENT;

static auto
get_trace_size() TIMEMORY_INTERNAL_NO_INSTRUMENT;

// thread-local data

static bool&
get_thread_enabled() TIMEMORY_INTERNAL_NO_INSTRUMENT;

static auto&
get_depth() TIMEMORY_INTERNAL_NO_INSTRUMENT;

static auto&
get_overhead() TIMEMORY_INTERNAL_NO_INSTRUMENT;

static auto&
get_throttle() TIMEMORY_INTERNAL_NO_INSTRUMENT;

static auto&
get_trace_data() TIMEMORY_INTERNAL_NO_INSTRUMENT;

static auto&
get_labels() TIMEMORY_INTERNAL_NO_INSTRUMENT;

static auto
get_label(void*, void*) TIMEMORY_INTERNAL_NO_INSTRUMENT;

// miscellaneous functions

template <size_t... Idx>
static auto
get_storage(tim::index_sequence<Idx...>) TIMEMORY_INTERNAL_NO_INSTRUMENT;

template <size_t Idx, size_t N>
static void
get_storage_impl(std::array<std::function<void()>, N>&) TIMEMORY_INTERNAL_NO_INSTRUMENT;

static void
initialize(const char* = nullptr) TIMEMORY_INTERNAL_NO_INSTRUMENT;

static void
allocate() TIMEMORY_INTERNAL_NO_INSTRUMENT;

static void
finalize() TIMEMORY_INTERNAL_NO_INSTRUMENT;

static auto
setup_gotcha() TIMEMORY_INTERNAL_NO_INSTRUMENT;

//--------------------------------------------------------------------------------------//

namespace
{
bool m_default_enabled = (std::atexit(&finalize), true);

struct global_data;
struct thread_local_data;

global_data*&
get_global_data() TIMEMORY_INTERNAL_NO_INSTRUMENT;

thread_local_data*&
get_thread_local_data() TIMEMORY_INTERNAL_NO_INSTRUMENT;

struct global_data
{
    using void_pair_t = std::pair<void*, void*>;
    using children_t  = std::vector<thread_local_data*>;

    global_data() = default;
    ~global_data();

    bool        enabled   = tim::settings::enabled();
    bool        finalized = false;
    bool        debug     = tim::get_env(TIMEMORY_SETTINGS_KEY("DEBUG"), false);
    int64_t     max_depth = tim::settings::max_depth();
    void_pair_t first     = void_pair_t{ nullptr, nullptr };
    children_t  children  = {};
};

struct thread_local_data
{
    bool           enabled    = get_global_data()->enabled;
    int64_t        depth      = 0;
    overhead_map_t overhead   = {};
    throttle_map_t throttle   = {};
    trace_data_t   trace_data = {};
    label_map_t    labels     = {};
};

global_data::~global_data()
{
    for(auto itr : children)
        delete itr;
}

global_data*&
get_global_data()
{
    static auto _instance = new global_data{};
    return _instance;
}

thread_local_data*&
get_thread_local_data()
{
    static thread_local auto _instance = new thread_local_data{};
    return _instance;
}

}  // namespace

//--------------------------------------------------------------------------------------//

template <size_t Idx, size_t N>
void
get_storage_impl(std::array<std::function<void()>, N>& _data)
{
    static_assert(Idx < N, "Error! Expanded greater than array size");
    _data[Idx] = []() {
        tim::operation::fini_storage<tim::component::enumerator_t<Idx>>{};
    };
}

//--------------------------------------------------------------------------------------//

template <size_t... Idx>
auto
get_storage(tim::index_sequence<Idx...>)
{
    // array of finalization functions
    std::array<std::function<void()>, sizeof...(Idx)> _data{};
    // initialize the storage in the thread
    TIMEMORY_FOLD_EXPRESSION(tim::storage_initializer::get<Idx>());
    // generate a function for finalizing
    TIMEMORY_FOLD_EXPRESSION(get_storage_impl<Idx>(_data));
    // return the array of finalization functions
    return _data;
}

//--------------------------------------------------------------------------------------//

bool&
get_enabled()
{
    return get_global_data()->enabled;
}

//--------------------------------------------------------------------------------------//

bool&
is_finalized()
{
    return get_global_data()->finalized;
}

//--------------------------------------------------------------------------------------//

bool
get_debug()
{
    return get_global_data()->debug;
}

//--------------------------------------------------------------------------------------//

auto&
get_max_depth()
{
    return get_global_data()->max_depth;
}

//--------------------------------------------------------------------------------------//

auto&
get_first()
{
    return get_global_data()->first;
}

//--------------------------------------------------------------------------------------//

auto
get_trace_size()
{
    using types_type = tim::convert_t<tim::available_types_t, tim::type_list<>>;
    return tim::manager::get_storage<types_type>::size();
}

//--------------------------------------------------------------------------------------//

bool&
get_thread_enabled()
{
    return get_thread_local_data()->enabled;
}

//--------------------------------------------------------------------------------------//

auto&
get_depth()
{
    return get_thread_local_data()->depth;
}

//--------------------------------------------------------------------------------------//

auto&
get_overhead()
{
    return get_thread_local_data()->overhead;
}

//--------------------------------------------------------------------------------------//

auto&
get_throttle()
{
    return get_thread_local_data()->throttle;
}

//--------------------------------------------------------------------------------------//

auto&
get_trace_data()
{
    return get_thread_local_data()->trace_data;
}

//--------------------------------------------------------------------------------------//

auto&
get_labels()
{
    return get_thread_local_data()->labels;
}

//--------------------------------------------------------------------------------------//

auto
get_label(void* this_fn, void* call_site)
{
    auto& _label_site = get_labels()[call_site];

    auto itr = _label_site.find(this_fn);
    if(itr != _label_site.end())
        return itr->second;

    Dl_info finfo;
    dladdr(this_fn, &finfo);

    if(!finfo.dli_saddr)
    {
        auto _key  = TIMEMORY_JOIN("", '[', this_fn, ']', '[', call_site, ']');
        auto _hash = tim::add_hash_id(_key);
        _label_site.insert({ this_fn, _hash });
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

    auto _hash = tim::add_hash_id(
        TIMEMORY_JOIN("", '[', finfo.dli_sname, ']', '[', finfo.dli_fname, ']'));
    _label_site.insert({ this_fn, _hash });
    return _hash;
}

//--------------------------------------------------------------------------------------//

void
initialize(const char* _exe_name)
{
    static bool _first = true;
    if(!_first)
    {
        if(_exe_name)
        {
            auto _name = TIMEMORY_JOIN('-', "compiler-instrumentation", _exe_name);
            for(auto& itr : _name)
            {
                if(itr == '_')
                    itr = '-';
            }
            tim::settings::output_path() =
                TIMEMORY_JOIN('-', "timemory", _name, "output");
        }
        return;
    }
    _first = false;

    // ensure a static holds a reference count to the manager
    static auto _manager = tim::manager::instance();
    tim::consume_parameters(_manager);

    primary_tidx = tim::threading::get_id();

    // default settings
    if(tim::get_env<std::string>(TIMEMORY_SETTINGS_KEY("COUT_OUTPUT"), "").empty())
        tim::settings::cout_output() = false;

    // initialization
    if(_exe_name)
    {
        auto _name = TIMEMORY_JOIN('-', "compiler-instrumentation", _exe_name);
        for(auto& itr : _name)
        {
            if(itr == '_')
                itr = '-';
        }
        tim::timemory_init(_name);
    }
    else
    {
        tim::timemory_init(std::string{ "compiler-instrumentation" });
    }

    // parse environment
    tim::settings::parse();
    tim::settings::suppress_parsing() = true;
    tim::settings::plot_output()      = false;

    // throttling
    throttle_count =
        tim::get_env(TIMEMORY_SETTINGS_KEY("THROTTLE_COUNT"), throttle_count);
    throttle_value =
        tim::get_env(TIMEMORY_SETTINGS_KEY("THROTTLE_VALUE"), throttle_value);

    // output path
    if(tim::get_env<std::string>(TIMEMORY_SETTINGS_KEY("OUTPUT_PATH"), "").empty())
    {
        tim::settings::output_path() =
            tim::get_env<std::string>(TIMEMORY_SETTINGS_KEY("OUTPUT_PATH"),
                                      "timemory-compiler-instrumentation-output");
    }
    else
    {
        tim::settings::output_prefix() = tim::get_env<std::string>(
            TIMEMORY_SETTINGS_KEY("OUTPUT_PREFIX"), "compiler-");
    }

    auto _comp_env = tim::get_env<std::string>(TIMEMORY_SETTINGS_KEY("COMPONENTS"), "");
    auto _glob_env = tim::get_env<std::string>("TIMEMORY_GLOBAL_COMPONENTS", "");
    auto _used_env = (_comp_env.empty()) ? _glob_env : _comp_env;
    static auto _enum_env = tim::enumerate_components(tim::delimit(_used_env, ",;: "));
    if(_enum_env.empty())
        _enum_env.push_back(WALL_CLOCK);

    static_assert(std::is_same<trace_set_t, tim::available_list_t>::value,
                  "Error! mismatched types");

    if(get_enabled())
    {
        trace_set_t::get_initializer() = [](trace_set_t& ts) {
            tim::initialize(ts, _enum_env);
        };
    }
}

//--------------------------------------------------------------------------------------//

static void
allocate()
{
    static thread_local bool _first = true;
    if(!_first)
        return;
    _first = false;

    auto tidx = tim::threading::get_id();
    tim::consume_parameters(
        tidx, get_storage(tim::make_index_sequence<TIMEMORY_COMPONENTS_END>{}));
}

//--------------------------------------------------------------------------------------//

void
finalize()
{
    if(!get_thread_local_data() || !get_global_data())
        return;

    if(is_finalized())
        return;

    if(tim::threading::get_id() == primary_tidx)
        get_enabled() = false;

    get_thread_enabled() = false;

    // acquire a thread-local lock so that no more entries are added while finalizing
    tim::trace::lock<tim::trace::compiler> lk{};
    try
    {
        lk.acquire();
    } catch(std::system_error& e)
    {
        std::cerr << e.what() << std::endl;
        return;
    }

    // acquire global lock
    tim::auto_lock_t _projlk{ tim::type_mutex<tim::project::compiler_instrument>(),
                              std::defer_lock };
    try
    {
        if(!_projlk.owns_lock())
            _projlk.lock();
    } catch(std::system_error& e)
    {
        std::cerr << e.what() << std::endl;
        return;
    }

    bool _remove_manager = false;
    if(!get_trace_data().empty())
    {
        printf("[pid=%i][tid=%i]> timemory-compiler-instrument: %lu results\n",
               (int) tim::process::get_id(), (int) tim::threading::get_id(),
               (unsigned long) get_trace_size());
    }
    else
    {
        printf("[pid=%i][tid=%i]> timemory-compiler-instrument: finalizing...\n",
               (int) tim::process::get_id(), (int) tim::threading::get_id());
        _remove_manager = true;
    }

    if(get_thread_local_data())
    {
        // clean up trace map
        for(auto& itr : get_trace_data())
            std::get<2>(itr)->stop();

        // clean up trace map
        get_trace_data().clear();

        // clean up overhead map
        get_overhead().clear();

        // clean up throttle map
        get_throttle().clear();

        delete get_thread_local_data();
        get_thread_local_data() = nullptr;
    }

    if(tim::threading::get_id() != primary_tidx)
        return;

    auto _manager = tim::manager::instance();
    if(_manager && !_manager->is_finalized() && !_manager->is_finalizing())
    {
        is_finalized() = true;
        // must use manually bc finalization will cause the stop operation to be disabled
        wall_clock wc{};
        peak_rss   pr{};
        wc.start();
        pr.start();
        //
        tim::timemory_finalize();
        //
        wc.stop();
        pr.stop();
        std::stringstream ss;
        ss << "required " << wc.get() << " " << tim::component::wall_clock::display_unit()
           << " and " << pr.get() << " " << tim::component::peak_rss::display_unit();
        std::string msg = ss.str();
        printf("[pid=%i][tid=%i]> timemory-compiler-instrument: finalization %s\n",
               (int) tim::process::get_id(), (int) tim::threading::get_id(), msg.c_str());
#if defined(TIMEMORY_INTERNAL_TESTING)
        assert(wc.get() > 0.0);
        assert(pr.get() > 0.0);
#endif

        if(get_global_data())
        {
            delete get_global_data();
            get_global_data() = nullptr;
        }
    }

    if(_remove_manager)
        tim::manager::instance().reset();
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
        using auto_start_quirk_t = tim::quirk::config<tim::quirk::auto_start>;

        if(!get_thread_local_data())
            return;

        tim::trace::lock<tim::trace::compiler> lk{};
        if(!lk || !get_enabled() || !get_thread_enabled())
            return;

        auto _label = get_label(this_fn, call_site);
        if(get_debug())
        {
            fprintf(
                stderr, "[%i][%i][timemory-compiler-inst]> %s\n",
                (int) tim::process::get_id(), (int) tim::threading::get_id(),
                tim::operation::decode<TIMEMORY_API>{}(_label).substr(0, 120).c_str());
        }

        if(get_depth()++ > get_max_depth())
            return;

        auto& _trace_data = get_trace_data();
        auto& _overhead   = get_overhead();
        auto& _throttle   = get_throttle();
        if(_throttle[call_site][this_fn])
            return;

        if(_trace_data.size() % 100 == 0)
            _trace_data.reserve(_trace_data.size() + 100);

        _trace_data.emplace_back(
            this_fn, call_site,
            std::make_unique<trace_set_t>(_label, auto_start_quirk_t{}));

        auto& _this_over = _overhead[call_site][this_fn];
        if(std::get<0>(_this_over) == 0)
            std::get<1>(_this_over).start();
    }
    //
    //----------------------------------------------------------------------------------//
    //
    void timemory_profile_func_exit(void* this_fn, void* call_site)
    {
        if(!get_thread_local_data())
            return;

        tim::trace::lock<tim::trace::compiler> lk{};
        if(!lk || !get_enabled() || !get_thread_enabled())
            return;

        if(get_debug())
        {
            fprintf(stderr, "[%i][%i][timemory-compiler-inst]> %s\n",
                    (int) tim::process::get_id(), (int) tim::threading::get_id(),
                    tim::operation::decode<TIMEMORY_API>{}(get_label(this_fn, call_site))
                        .substr(0, 120)
                        .c_str());
        }

        const bool _is_first =
            (get_first().first == this_fn && get_first().second == call_site);

        if(_is_first)
        {
            get_thread_enabled() = false;
            if(tim::threading::get_id() == primary_tidx)
                get_enabled() = false;
        }

        if(get_depth()-- > get_max_depth())
            return;

        auto& _trace_data = get_trace_data();
        auto& _overhead   = get_overhead();
        auto& _throttle   = get_throttle();

        if(_throttle[call_site][this_fn])
            return;

        if(_trace_data.empty())
            return;

        --(std::get<0>(_overhead[call_site][this_fn]));

        std::get<2>(_trace_data.back())->stop();
        _trace_data.pop_back();

        if(_is_first)
        {
            finalize();
            return;
        }

        auto& _this_over = _overhead[call_site][this_fn];
        auto  _count     = ++(std::get<2>(_this_over));
        if(_count % throttle_count == 0)
        {
            auto& _mono = std::get<1>(_this_over);
            _mono.stop();
            auto _accum = _mono.get_accum() / _count;
            if(_accum < throttle_value)
                _throttle[call_site][this_fn] = true;
            _mono.reset();
            std::get<2>(_this_over) = 0;
        }
    }
    //
    //----------------------------------------------------------------------------------//
    //
    int timemory_profile_thread_init(void)
    {
        if(!get_thread_local_data())
            return 0;

        tim::trace::lock<tim::trace::compiler> lk{};
        if(!lk || !get_enabled() || !get_thread_enabled())
            return 0;

        static auto              _initialized = (initialize(), true);
        static thread_local auto _allocated   = (allocate(), true);
        tim::consume_parameters(_initialized, _allocated);
        return 1;
    }
    //
    //----------------------------------------------------------------------------------//
    //
    int timemory_profile_thread_fini(void)
    {
        if(!get_thread_local_data())
            return 0;

        tim::trace::lock<tim::trace::compiler> lk{};
        if(!lk || !get_enabled() || !get_thread_enabled())
            return 0;

        finalize();
        return 1;
    }

}  // extern "C"

//--------------------------------------------------------------------------------------//

struct pthread_gotcha : tim::component::base<pthread_gotcha, void>
{
    struct TIMEMORY_INTERNAL_NO_INSTRUMENT wrapper
    {
        using routine_t = void* (*) (void*);

        wrapper(routine_t _routine, void* _arg, bool _debug)
        : m_routine(_routine)
        , m_arg(_arg)
        , m_debug(_debug)
        {}

        void* operator()() const { return m_routine(m_arg); }

        TIMEMORY_NODISCARD bool debug() const { return m_debug; }

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
#if defined(TIMEMORY_INTERNAL_TESTING)
            assert(_tlm.get() != nullptr);
#endif
            if(_wrapper->debug())
                PRINT_HERE("%s", "Initializing timemory component storage");
            // initialize the storage
            tim::consume_parameters(
                get_storage(tim::make_index_sequence<TIMEMORY_COMPONENTS_END>{}));
            if(_wrapper->debug())
                PRINT_HERE("%s", "Executing original function");
            // execute the original function
            auto _ret = (*_wrapper)();
            // only child threads
            if(tim::threading::get_id() != primary_tidx)
            {
                if(_wrapper->debug())
                    PRINT_HERE("%s", "Executing finalizing");
                // disable future instrumentation for thread
                get_thread_enabled() = false;
                // finalize
                finalize();
            }
            // return the data
            return _ret;
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

private:
    bool m_debug =
        (tim::settings::instance()) ? tim::settings::instance()->get_debug() : false;
};

//--------------------------------------------------------------------------------------//

auto
setup_gotcha()
{
#if defined(TIMEMORY_USE_GOTCHA)
    auto _enable_gotcha =
        tim::get_env("TIMEMORY_COMPILER_ENABLE_PTHREAD_GOTCHA_WRAPPER", false);

    if(_enable_gotcha)
    {
        pthread_gotcha_t::get_initializer() = []() {
            TIMEMORY_C_GOTCHA(pthread_gotcha_t, 0, pthread_create);
        };
    }
#endif
    return std::make_tuple(std::make_shared<pthread_bundle_t>("pthread"));
}

namespace
{
auto internal_gotcha_handle = setup_gotcha();
}  // namespace

//--------------------------------------------------------------------------------------//
