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

#include "timemory/trace.hpp"
#include "timemory/compat/library.h"
#include "timemory/library.h"
#include "timemory/runtime/configure.hpp"
#include "timemory/timemory.hpp"
#include "timemory/utility/bits/signals.hpp"

#if defined(TIMEMORY_USE_MPI)
#    include "timemory/backends/types/mpi/extern.hpp"
#endif

#include <cstdarg>
#include <cstdlib>
#include <deque>
#include <iostream>
#include <unordered_map>

using namespace tim::component;

using string_t       = std::string;
using overhead_map_t = std::unordered_map<size_t, std::pair<wall_clock, size_t>>;
using throttle_set_t = std::set<size_t>;
using traceset_t     = tim::component_tuple<user_trace_bundle>;
using trace_map_t    = std::unordered_map<size_t, std::deque<traceset_t>>;

//======================================================================================//

static std::atomic<uint32_t> library_trace_count{ 0 };

//--------------------------------------------------------------------------------------//

static auto&
get_overhead() TIMEMORY_VISIBILITY("default");
static auto&
get_throttle() TIMEMORY_VISIBILITY("default");
static trace_map_t&
get_trace_map() TIMEMORY_VISIBILITY("default");

//--------------------------------------------------------------------------------------//

static auto&
get_overhead()
{
    static thread_local auto _instance = std::make_unique<overhead_map_t>();
    return _instance;
}

//--------------------------------------------------------------------------------------//

static auto&
get_throttle()
{
    static thread_local auto _instance = std::make_unique<throttle_set_t>();
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

extern std::array<bool, 2>&
get_library_state();

//--------------------------------------------------------------------------------------//

static bool use_mpi_gotcha = false;

//--------------------------------------------------------------------------------------//

#if defined(TIMEMORY_MPI_GOTCHA)

static bool mpi_is_attached = false;

//--------------------------------------------------------------------------------------//
// query environment setting for whether to enable finalization via MPI_Comm_create_keyval
//
extern "C" bool
timemory_mpi_finalize_comm_keyval()
{
    return tim::get_env<bool>("TIMEMORY_MPI_FINALIZE_COMM_KEYVAL", true);
}

//--------------------------------------------------------------------------------------//
// query environment setting for whether to enable finalization via gotcha wrapper
//
extern "C" bool
timemory_mpi_finalize_gotcha_wrapper()
{
    return tim::get_env<bool>("TIMEMORY_MPI_FINALIZE_GOTCHA_WRAPPER", false);
}

//--------------------------------------------------------------------------------------//

static int
timemory_trace_mpi_finalize(MPI_Comm, int, void*, void*)
{
    // only execute once
    static bool once = false;
    if(once || tim::mpi::is_finalized())
        return MPI_SUCCESS;
    once = true;
    // do finalization
    if(tim::settings::debug() || tim::settings::verbose() > 1)
        PRINT_HERE("%s", "comm keyval finalization started");
    timemory_trace_finalize();
    if(tim::settings::debug() || tim::settings::verbose() > 1)
        PRINT_HERE("%s", "comm keyval finalization complete");
    return MPI_SUCCESS;
}

//--------------------------------------------------------------------------------------//

struct mpi_trace_gotcha : tim::component::base<mpi_trace_gotcha, void>
{
    static void set_attr()
    {
        if(!timemory_mpi_finalize_comm_keyval())
            return;
        tim::trace::lock<tim::trace::library> lk{};
        auto                                  _state = tim::settings::enabled();
        tim::settings::enabled()                     = false;
        int  comm_key                                = 0;
        auto ret = MPI_Comm_create_keyval(MPI_NULL_COPY_FN, &timemory_trace_mpi_finalize,
                                          &comm_key, NULL);
        if(ret == MPI_SUCCESS)
            MPI_Comm_set_attr(MPI_COMM_SELF, comm_key, NULL);
        tim::settings::enabled() = _state;
    }

    // MPI_Init
    int operator()(int* argc, char*** argv)
    {
        if(recursive())
            return MPI_Init(argc, argv);
        recursive()              = true;
        auto ret                 = MPI_Init(argc, argv);
        tim::mpi::is_finalized() = false;
        set_attr();
        auto mode = tim::get_env<std::string>("TIMEMORY_INSTRUMENTATION_MODE", "trace");
        if(mode == "trace")
            timemory_push_trace("MPI_Init(int*, char**)");
        else if(mode == "region")
            timemory_push_region("MPI_Init(int*, char**)");
        recursive() = false;
        return ret;
    }

    // MPI_Init_thread
    int operator()(int* argc, char*** argv, int req, int* prov)
    {
        if(recursive())
            return MPI_Init_thread(argc, argv, req, prov);
        recursive()              = true;
        auto ret                 = MPI_Init_thread(argc, argv, req, prov);
        tim::mpi::is_finalized() = false;
        set_attr();
        auto mode = tim::get_env<std::string>("TIMEMORY_INSTRUMENTATION_MODE", "trace");
        if(mode == "trace")
            timemory_push_trace("MPI_Init_thread(int*, char**, int, int*)");
        else if(mode == "region")
            timemory_push_region("MPI_Init_thread(int*, char**, int, int*)");
        recursive() = false;
        return ret;
    }

    // MPI_Finalize
    int operator()()
    {
        if(recursive() || tim::mpi::is_finalized())
            return MPI_SUCCESS;
        recursive() = true;
        if(!timemory_mpi_finalize_gotcha_wrapper())
        {
            auto ret    = MPI_Finalize();
            recursive() = false;
            return ret;
        }

        if(mpi_is_attached)
        {
            auto mode =
                tim::get_env<std::string>("TIMEMORY_INSTRUMENTATION_MODE", "trace");
            if(mode == "trace")
            {
                timemory_pop_trace("MPI_Init(int*, char**)");
                timemory_pop_trace("MPI_Init_thread(int*, char**, int, int*)");
            }
            else
            {
                timemory_pop_region("MPI_Init(int*, char**)");
                timemory_pop_region("MPI_Init_thread(int*, char**, int, int*)");
            }
        }
        if(tim::settings::debug())
            PRINT_HERE("%s", "finalizing trace");
        timemory_trace_mpi_finalize(MPI_COMM_WORLD, 0, nullptr, nullptr);
        if(tim::settings::debug())
            PRINT_HERE("%s", "finalizing MPI");
        tim::mpi::is_finalized() = true;
        auto ret                 = MPI_Finalize();
        recursive()              = false;
        return ret;
    }

    static std::string& get_trace_components()
    {
        static auto _instance =
            tim::get_env<std::string>("TIMEMORY_TRACE_COMPONENTS", "");
        return _instance;
    }

    static bool& recursive()
    {
        static bool _instance = false;
        return _instance;
    }

    static bool& read_command_line()
    {
        static bool _instance = true;
        return _instance;
    }

    static std::string& get_command()
    {
        static std::string _instance = "";
        return _instance;
    }
};

//--------------------------------------------------------------------------------------//

using empty_tuple_t      = tim::component_tuple<>;
using mpi_trace_gotcha_t = tim::component::gotcha<3, empty_tuple_t, mpi_trace_gotcha>;

//--------------------------------------------------------------------------------------//

bool
setup_mpi_gotcha()
{
    mpi_trace_gotcha_t::get_initializer() = []() {
        TIMEMORY_C_GOTCHA(mpi_trace_gotcha_t, 0, MPI_Init);
        TIMEMORY_C_GOTCHA(mpi_trace_gotcha_t, 1, MPI_Init_thread);
        if(mpi_is_attached || timemory_mpi_finalize_gotcha_wrapper())
            TIMEMORY_C_GOTCHA(mpi_trace_gotcha_t, 2, MPI_Finalize);
    };
    return true;
}

//--------------------------------------------------------------------------------------//

using mpi_trace_bundle_t = tim::component_tuple<mpi_trace_gotcha_t>;

//--------------------------------------------------------------------------------------//

#else

//--------------------------------------------------------------------------------------//

struct mpi_trace_gotcha : tim::component::base<mpi_trace_gotcha, void>
{
    static void set_attr() {}

    static std::string& get_trace_components()
    {
        static auto _instance =
            tim::get_env<std::string>("TIMEMORY_TRACE_COMPONENTS", "");
        return _instance;
    }

    static bool& read_command_line()
    {
        static bool _instance = true;
        return _instance;
    }

    static std::string& get_command()
    {
        static std::string _instance{};
        return _instance;
    }
};

//--------------------------------------------------------------------------------------//

using mpi_trace_bundle_t = tim::component_tuple<>;

//--------------------------------------------------------------------------------------//

bool
setup_mpi_gotcha()
{
    return false;
}

//--------------------------------------------------------------------------------------//

#endif

//--------------------------------------------------------------------------------------//

static bool                                mpi_gotcha_configured = setup_mpi_gotcha();
static std::shared_ptr<mpi_trace_bundle_t> mpi_gotcha_handle{ nullptr };
static std::map<size_t, string_t>          master_hash_ids;

//--------------------------------------------------------------------------------------//
//
//      timemory symbols
//
//--------------------------------------------------------------------------------------//

extern "C"
{
    //
    //----------------------------------------------------------------------------------//
    //
    TIMEMORY_WEAK_PREFIX
    void timemory_mpip_library_ctor() TIMEMORY_WEAK_POSTFIX
        TIMEMORY_VISIBILITY("default");
    TIMEMORY_WEAK_PREFIX
    void timemory_register_mpip() TIMEMORY_WEAK_POSTFIX TIMEMORY_VISIBILITY("default");
    TIMEMORY_WEAK_PREFIX
    void timemory_deregister_mpip() TIMEMORY_WEAK_POSTFIX TIMEMORY_VISIBILITY("default");

    TIMEMORY_WEAK_PREFIX
    void timemory_ompt_library_ctor() TIMEMORY_WEAK_POSTFIX
        TIMEMORY_VISIBILITY("default");
    TIMEMORY_WEAK_PREFIX
    void timemory_register_ompt() TIMEMORY_WEAK_POSTFIX TIMEMORY_VISIBILITY("default");
    TIMEMORY_WEAK_PREFIX
    void timemory_deregister_ompt() TIMEMORY_WEAK_POSTFIX TIMEMORY_VISIBILITY("default");
    TIMEMORY_WEAK_PREFIX
    ompt_start_tool_result_t* ompt_start_tool(unsigned int omp_version,
                                              const char*  runtime_version)
        TIMEMORY_WEAK_POSTFIX TIMEMORY_VISIBILITY("default");

    void                      timemory_register_mpip() {}
    void                      timemory_register_ompt() {}
    void                      timemory_deregister_mpip() {}
    void                      timemory_deregister_ompt() {}
    void                      timemory_mpip_library_ctor() {}
    void                      timemory_ompt_library_ctor() {}
    ompt_start_tool_result_t* ompt_start_tool(unsigned int, const char*)
    {
        return nullptr;
    }
    //
    //----------------------------------------------------------------------------------//
    //
    bool timemory_trace_is_initialized()
    {
        return (get_library_state()[0] &&
                library_trace_count.load(std::memory_order_relaxed) > 0);
    }
    //
    //----------------------------------------------------------------------------------//
    //
    bool timemory_is_throttled(const char* name)
    {
        size_t _id = tim::get_hash_id(name);
        return (get_throttle()->count(_id) > 0);
    }
    //
    //----------------------------------------------------------------------------------//
    //
    void timemory_add_hash_id(uint64_t id, const char* name)
    {
        tim::trace::lock<tim::trace::library> lk{};
        if(tim::settings::debug())
            fprintf(stderr, "[timemory-trace]> adding '%s' with hash %lu...\n", name,
                    (unsigned long) id);
        auto _id = tim::add_hash_id(name);
        if(_id != id)
            tim::add_hash_id(_id, id);

        // master thread adds the ids
        if(tim::threading::get_id() == 0)
        {
            master_hash_ids[id] = name;
            if(_id != id)
                master_hash_ids[_id] = name;
        }
    }
    //
    //----------------------------------------------------------------------------------//
    //
    void timemory_add_hash_ids(uint64_t nentries, uint64_t* ids, const char** names)
    {
        for(uint64_t i = 0; i < nentries; ++i)
            timemory_add_hash_id(ids[i], names[i]);
    }
    //
    //----------------------------------------------------------------------------------//
    //
    void timemory_copy_hash_ids()
    {
        tim::trace::lock<tim::trace::library> lk{};
        static thread_local bool              once_per_thread = false;
        if(!once_per_thread && tim::threading::get_id() > 0)
        {
            once_per_thread  = true;
            auto _master_ids = master_hash_ids;
            for(const auto& itr : _master_ids)
                timemory_add_hash_id(itr.first, itr.second.c_str());
        }
    }
    //
    //----------------------------------------------------------------------------------//
    //
    void timemory_push_trace_hash(uint64_t id)
    {
        if(!timemory_trace_is_initialized())
            timemory_trace_init("", true, "");

        tim::trace::lock<tim::trace::library> lk{};

        if(!lk)
            return;

        if(!get_library_state()[0] || get_library_state()[1] || !tim::settings::enabled())
            return;

        if(get_throttle()->count(id) > 0)
            return;

        auto& _trace_map = get_trace_map();

        if(_trace_map.empty())
            timemory_copy_hash_ids();

        if(tim::settings::debug())
        {
            int64_t  n    = _trace_map[id].size();
            auto     itr  = tim::get_hash_ids()->find(id);
            string_t name = (itr != tim::get_hash_ids()->end()) ? itr->second : "unknown";
            fprintf(stderr,
                    "beginning trace for '%s' (id = %llu, offset = %lli, rank = %i, pid "
                    "= %i, thread = %i)...\n",
                    name.c_str(), (long long unsigned) id, (long long int) n,
                    tim::dmp::rank(), (int) tim::process::get_id(),
                    (int) tim::threading::get_id());
        }

        _trace_map[id].emplace_back(traceset_t(id));
        _trace_map[id].back().start();
        if(get_overhead())
            (*get_overhead())[id].first.start();
    }
    //
    //----------------------------------------------------------------------------------//
    //
    void timemory_pop_trace_hash(uint64_t id)
    {
        tim::trace::lock<tim::trace::library> lk{};
        if(!get_library_state()[0] || get_library_state()[1])
            return;

        auto& _trace_map = get_trace_map();
        if(!tim::settings::enabled() && _trace_map.empty())
            return;

        int64_t ntotal = _trace_map[id].size();
        int64_t offset = ntotal - 1;

        if(tim::settings::debug())
        {
            auto     itr  = tim::get_hash_ids()->find(id);
            string_t name = (itr != tim::get_hash_ids()->end()) ? itr->second : "unknown";
            fprintf(stderr,
                    "ending trace for '%s' (id = %llu, offset = %lli, rank = %i, pid = "
                    "%i, thread = %i)...\n",
                    name.c_str(), (long long unsigned) id, (long long int) offset,
                    tim::dmp::rank(), (int) tim::process::get_id(),
                    (int) tim::threading::get_id());
        }

        // if there were no entries, return (pop called without a push)
        if(offset < 0)
            return;

        if(get_overhead())
            (*get_overhead())[id].first.stop();

        if(offset >= 0 && ntotal > 0)
        {
            _trace_map[id].back().stop();
            _trace_map[id].pop_back();
        }

        if(get_throttle() && get_throttle()->count(id) > 0)
            return;

        if(!get_overhead())
            return;

        auto _count = ++(get_overhead()->at(id).second);

        if(_count % tim::settings::throttle_count() == 0)
        {
            auto _accum = get_overhead()->at(id).first.get_accum() / _count;
            if(_accum < tim::settings::throttle_value())
            {
                if(tim::settings::debug() || tim::settings::verbose() > 0)
                {
                    auto name = tim::get_hash_ids()->find(id)->second;
                    fprintf(
                        stderr,
                        "[timemory-trace]> Throttling all future calls to '%s' on rank = "
                        "%i, pid = "
                        "%i, thread = %i. avg runtime = %lu ns from %lu invocations... "
                        "Consider eliminating from instrumentation...\n",
                        name.c_str(), tim::dmp::rank(), (int) tim::process::get_id(),
                        (int) tim::threading::get_id(), (unsigned long) _accum,
                        (unsigned long) _count);
                }
                get_throttle()->insert(id);
            }
            else
            {
                if(_accum < (10 * tim::settings::throttle_value()) &&
                   (tim::settings::debug() || tim::settings::verbose() > 1))
                {
                    auto name = tim::get_hash_ids()->find(id)->second;
                    fprintf(
                        stderr,
                        "[timemory-trace]> Warning! function call '%s' within an order "
                        "of magnitude of threshold for throttling value on rank = %i, "
                        "pid = "
                        "%i, thread = %i. avg runtime = %lu ns from %lu invocations... "
                        "Consider eliminating from instrumentation...\n",
                        name.c_str(), tim::dmp::rank(), (int) tim::process::get_id(),
                        (int) tim::threading::get_id(), (unsigned long) _accum,
                        (unsigned long) _count);
                }
            }
            get_overhead()->at(id).first.reset();
            get_overhead()->at(id).second = 0;
        }
    }
    //
    //----------------------------------------------------------------------------------//
    //
    void timemory_push_trace(const char* name)
    {
        if(!timemory_trace_is_initialized())
            timemory_trace_init("", true, "");

        uint64_t _hash = std::numeric_limits<uint64_t>::max();
        {
            tim::trace::lock<tim::trace::library> lk{};
            if(tim::settings::debug())
                PRINT_HERE("rank = %i, pid = %i, thread = %i, name = %s",
                           tim::dmp::rank(), (int) tim::process::get_id(),
                           (int) tim::threading::get_id(), name);

            if(!get_library_state()[0] || get_library_state()[1] ||
               !tim::settings::enabled())
                return;

            _hash = tim::add_hash_id(name);
        }
        timemory_push_trace_hash(_hash);
    }
    //
    //----------------------------------------------------------------------------------//
    //
    void timemory_pop_trace(const char* name)
    {
        uint64_t _hash = std::numeric_limits<uint64_t>::max();
        {
            tim::trace::lock<tim::trace::library> lk{};
            if(!get_library_state()[0] || get_library_state()[1])
                return;
            _hash = tim::get_hash_id(name);
        }
        timemory_pop_trace_hash(_hash);
    }
    //
    //----------------------------------------------------------------------------------//
    //
#if defined(TIMEMORY_MPI_GOTCHA)
    //
    void timemory_trace_set_mpi(bool use, bool attached)
    {
        use_mpi_gotcha  = use;
        mpi_is_attached = attached;
    }
    //
#endif
    //
    //----------------------------------------------------------------------------------//
    //
    void timemory_trace_set_env(const char* env_var, const char* env_val)
    {
        tim::trace::lock<tim::trace::library> lk{};
        tim::set_env<std::string>(env_var, env_val, 0);
    }
    //
    //----------------------------------------------------------------------------------//
    //
    void timemory_trace_init(const char* args, bool read_command_line, const char* cmd)
    {
        tim::trace::lock<tim::trace::library> lk{};
        if(get_library_state()[0])
            return;

        if(library_trace_count++ == 0)
        {
            PRINT_HERE("rank = %i, pid = %i, thread = %i, args = %s", tim::dmp::rank(),
                       (int) tim::process::get_id(), (int) tim::threading::get_id(),
                       args);

            tim::manager::use_exit_hook(false);

            if(read_command_line)
            {
                auto _init = [](int _ac, char** _av) { timemory_init_library(_ac, _av); };
                tim::config::read_command_line(_init);
            }
            else
            {
                get_library_state()[0] = true;
                std::string exe_name   = cmd;
                while(exe_name.find('\\') != std::string::npos)
                    exe_name = exe_name.substr(exe_name.find_last_of('\\') + 1);
                while(exe_name.find('/') != std::string::npos)
                    exe_name = exe_name.substr(exe_name.find_last_of('/') + 1);

                static const std::vector<std::string> _exe_suffixes = { ".py", ".exe" };
                for(const auto& ext : _exe_suffixes)
                {
                    if(exe_name.find(ext) != std::string::npos)
                        exe_name.erase(exe_name.find(ext), ext.length() + 1);
                }

                exe_name = std::string("timemory-") + exe_name + "-output";
                for(auto& itr : exe_name)
                {
                    if(itr == '_')
                        itr = '-';
                }

                tim::settings::output_path() = exe_name;
            }

            tim::set_env<std::string>("TIMEMORY_TRACE_COMPONENTS", args, 0);

            // configure bundle
            tim::env::configure<user_trace_bundle>("TIMEMORY_TRACE_COMPONENTS", args);

            tim::settings::parse();

            std::function<void(int)> exit_func = [](int) { tim::timemory_finalize(); };
            tim::signal_settings::set_exit_action(exit_func);
            std::atexit(&tim::timemory_finalize);
#if !defined(_MACOS)
            // Apple clang version 11.0.3 (clang-1103.0.32.62) doesn't seem to have this
            // function
            std::at_quick_exit(&tim::timemory_finalize);
#endif

#if defined(TIMEMORY_MPI_GOTCHA)
            if(!mpi_gotcha_handle.get())
            {
                mpi_gotcha_handle =
                    std::make_shared<mpi_trace_bundle_t>("timemory_trace_mpi_gotcha");
                mpi_trace_gotcha::get_trace_components() = args;
                mpi_trace_gotcha::read_command_line()    = read_command_line;
                mpi_trace_gotcha::get_command()          = cmd;
                if(mpi_is_attached)
                    mpi_trace_gotcha::set_attr();
                mpi_gotcha_handle->start();
            }
            else if(mpi_gotcha_handle.get())
            {
                PRINT_HERE("rank = %i, pid = %i, thread = %i", tim::dmp::rank(),
                           (int) tim::process::get_id(), (int) tim::threading::get_id());
                tim::manager::instance()->update_metadata_prefix();
            }
#endif
        }
    }
    //
    //----------------------------------------------------------------------------------//
    //
    void timemory_trace_finalize(void)
    {
        tim::trace::lock<tim::trace::library> lk{};
        if(library_trace_count.load() == 0)
            return;

        if(tim::settings::verbose() > 1 || tim::settings::debug())
            PRINT_HERE("rank = %i, pid = %i, thread = %i", tim::dmp::rank(),
                       (int) tim::process::get_id(), (int) tim::threading::get_id());

        // do the finalization
        auto _count = --library_trace_count;

        if(_count > 0)
        {
            // have the manager finalize
            tim::manager::instance()->finalize();
            return;
        }

        tim::auto_lock_t lock(tim::type_mutex<tim::api::native_tag>());

        // tim::settings::enabled() = false;
        get_library_state()[1] = true;

        tim::mpi::barrier();

        // reset traces just in case
        user_trace_bundle::reset();

        // clean up any remaining entries
        for(auto& itr : get_trace_map())
        {
            for(auto& eitr : itr.second)
                eitr.stop();
            // delete all the records
            itr.second.clear();
        }

        // delete all the records
        get_trace_map().clear();

        // deactivate the gotcha wrappers
        if(use_mpi_gotcha)
            mpi_gotcha_handle.reset();

        // finalize the library
        timemory_finalize_library();
    }
    //
    //----------------------------------------------------------------------------------//
    //
}  // extern "C"
