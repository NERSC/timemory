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

#include "timemory/compat/library.h"
#include "timemory/library.h"
#include "timemory/runtime/configure.hpp"
#include "timemory/timemory.hpp"

#if defined(TIMEMORY_USE_MPI)
#    include "timemory/backends/types/mpi/extern.hpp"
#endif

#include <cstdarg>
#include <deque>
#include <iostream>
#include <unordered_map>

// #include <dlfcn.h>

using namespace tim::component;

CEREAL_CLASS_VERSION(tim::settings, 1)
CEREAL_CLASS_VERSION(tim::env_settings, 0)
CEREAL_CLASS_VERSION(tim::component::wall_clock, 0)
CEREAL_CLASS_VERSION(tim::statistics<double>, 0)

using string_t       = std::string;
using overhead_map_t = std::unordered_map<size_t, std::pair<wall_clock, size_t>>;
using throttle_set_t = std::set<size_t>;
using traceset_t     = tim::component_tuple<user_trace_bundle>;
using trace_map_t    = std::unordered_map<size_t, std::deque<traceset_t*>>;

//======================================================================================//

namespace
{
static auto library_manager_handle  = tim::manager::master_instance();
static auto library_settings_handle = tim::settings::shared_instance<TIMEMORY_API>();
static std::atomic<uint32_t> library_trace_count{ 0 };
}  // namespace

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

#if defined(TIMEMORY_MPI_GOTCHA)

static int
timemory_trace_mpi_finalize(MPI_Comm, int, void*, void*)
{
    timemory_trace_finalize();
    return MPI_SUCCESS;
}

struct mpi_trace_gotcha : tim::component::base<mpi_trace_gotcha, void>
{
    static void set_attr()
    {
        int comm_key = 0;
        MPI_Comm_create_keyval(MPI_NULL_COPY_FN, &timemory_trace_mpi_finalize, &comm_key,
                               NULL);
        MPI_Comm_set_attr(MPI_COMM_SELF, comm_key, NULL);
    }

    // MPI_Init
    int operator()(int* argc, char*** argv)
    {
        auto ret = MPI_Init(argc, argv);
        timemory_trace_init(get_trace_components().c_str(), read_command_line(),
                            get_command().c_str());
        set_attr();
        return ret;
    }

    // MPI_Init_thread
    int operator()(int* argc, char*** argv, int req, int* prov)
    {
        auto ret = MPI_Init_thread(argc, argv, req, prov);
        timemory_trace_init(get_trace_components().c_str(), read_command_line(),
                            get_command().c_str());
        set_attr();
        return ret;
    }

    // MPI_Finalize
    int operator()()
    {
        timemory_trace_finalize();
        tim::mpi::is_finalized() = true;
        auto ret                 = PMPI_Finalize();
        return ret;
    }

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
        // TIMEMORY_C_GOTCHA(mpi_trace_gotcha_t, 2, MPI_Finalize);
    };
    return true;
}

//--------------------------------------------------------------------------------------//

using mpi_trace_bundle_t = tim::auto_tuple<mpi_trace_gotcha_t>;

//--------------------------------------------------------------------------------------//

#else

struct mpi_trace_gotcha : tim::component::base<mpi_trace_gotcha, void>
{
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
        static std::string _instance = "";
        return _instance;
    }
};

using mpi_trace_bundle_t = tim::auto_tuple<>;

bool
setup_mpi_gotcha()
{
    return false;
}

#endif

//--------------------------------------------------------------------------------------//

static bool                                use_mpi_gotcha        = false;
static bool                                mpi_gotcha_configured = setup_mpi_gotcha();
static std::shared_ptr<mpi_trace_bundle_t> mpi_gotcha_handle{ nullptr };

//--------------------------------------------------------------------------------------//
//
//      TiMemory symbols
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
        if(tim::settings::debug())
            fprintf(stderr, "[timemory-trace]> adding '%s' with hash %lu...\n", name,
                    (unsigned long) id);
        auto _id = tim::add_hash_id(name);
        if(_id != id)
            tim::add_hash_id(_id, id);
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
    void timemory_push_trace_hash(uint64_t id)
    {
        if(!get_library_state()[0] || get_library_state()[1])
            return;

        if(!tim::settings::enabled())
            return;

        if(get_throttle()->count(id) > 0)
            return;

        auto& _trace_map = get_trace_map();

#if defined(DEBUG)
        if(tim::settings::verbose() > 2)
        {
            int64_t n = _trace_map[id].size();
            printf("beginning trace for '%s' (id = %llu, offset = %lli)...\n",
                   tim::get_hash_ids()->find(id)->second.c_str(), (long long unsigned) id,
                   (long long int) n);
        }
#endif

        _trace_map[id].push_back(new traceset_t(id));
        _trace_map[id].back()->start();
        (*get_overhead())[id].first.start();
    }
    //
    //----------------------------------------------------------------------------------//
    //
    void timemory_pop_trace_hash(uint64_t id)
    {
        if(!get_library_state()[0] || get_library_state()[1])
            return;

        auto& _trace_map = get_trace_map();
        if(!tim::settings::enabled() && _trace_map.empty())
            return;

        int64_t ntotal = _trace_map[id].size();
        int64_t offset = ntotal - 1;

        (*get_overhead())[id].first.stop();

#if defined(DEBUG)
        if(tim::settings::verbose() > 2)
            printf("ending trace for %llu [offset = %lli]...\n", (long long unsigned) id,
                   (long long int) offset);
#endif

        if(offset >= 0 && ntotal > 0)
        {
            if(_trace_map[id].back())
            {
                _trace_map[id].back()->stop();
                delete _trace_map[id].back();
            }
            _trace_map[id].pop_back();
        }

        if(get_throttle()->count(id) > 0)
            return;

        auto _count = ++((*get_overhead())[id].second);

        if(_count >= tim::settings::throttle_count())
        {
            auto _accum = (*get_overhead())[id].first.get_accum() / _count;
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
                if(_count % tim::settings::throttle_count() == 0 &&
                   _accum < (10 * tim::settings::throttle_value()) &&
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
                // (*get_overhead())[id].first.reset();
                // (*get_overhead())[id].second = 0;
            }
        }
    }
    //
    //----------------------------------------------------------------------------------//
    //
    void timemory_push_trace(const char* name)
    {
        if(!get_library_state()[0] || get_library_state()[1])
            return;

        if(!tim::settings::enabled())
            return;
        timemory_push_trace_hash(tim::add_hash_id(name));
    }
    //
    //----------------------------------------------------------------------------------//
    //
    void timemory_pop_trace(const char* name)
    {
        if(!get_library_state()[0] || get_library_state()[1])
            return;

        timemory_pop_trace_hash(tim::get_hash_id(name));
    }
    //
    //----------------------------------------------------------------------------------//
    //
#if defined(TIMEMORY_MPI_GOTCHA)
    //
    void timemory_trace_set_mpi(bool use) { use_mpi_gotcha = use; }
    //
#endif
    //
    //----------------------------------------------------------------------------------//
    //
    void timemory_trace_set_env(const char* env_var, const char* env_val)
    {
        tim::set_env<std::string>(env_var, env_val, 0);
    }
    //
    //----------------------------------------------------------------------------------//
    //
    void timemory_trace_init(const char* args, bool read_command_line, const char* cmd)
    {
        if(get_library_state()[0])
            return;

        if(use_mpi_gotcha && !mpi_gotcha_handle.get())
        {
            PRINT_HERE("rank = %i, pid = %i, thread = %i", tim::dmp::rank(),
                       (int) tim::process::get_id(), (int) tim::threading::get_id());
            mpi_gotcha_handle =
                std::make_shared<mpi_trace_bundle_t>("timemory_trace_mpi_gotcha");
            mpi_trace_gotcha::get_trace_components() = args;
            mpi_trace_gotcha::read_command_line()    = read_command_line;
            mpi_trace_gotcha::get_command()          = cmd;
            return;
        }

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

            tim::set_env<std::string>("TIMEMORY_TRACE_COMPONENTS", args, 0);

            // reset traces just in case
            user_trace_bundle::reset();

            // configure bundle
            user_trace_bundle::global_init(nullptr);

            /*
            if(read_command_line)
            {
                auto _init = [](int _ac, char** _av) { timemory_init_library(_ac, _av); };
                tim::config::read_command_line(_init);
            }
            else if(strlen(cmd) > 0)
            {
                PRINT_HERE("rank = %i, pid = %i, thread = %i", tim::dmp::rank(),
                           (int) tim::process::get_id(), (int) tim::threading::get_id());
                char* _cmd = new char[strlen(cmd) + 1];
                strcpy(_cmd, cmd);
                timemory_init_library(1, &_cmd);
                delete[] _cmd;
            }*/

            tim::settings::parse();
        }
        else
        {
            PRINT_HERE("rank = %i, pid = %i, thread = %i, args = %s", tim::dmp::rank(),
                       (int) tim::process::get_id(), (int) tim::threading::get_id(),
                       args);

            if(read_command_line)
            {
                auto _init = [](int _ac, char** _av) { timemory_init_library(_ac, _av); };
                tim::config::read_command_line(_init);
            }

            auto manager = tim::manager::instance();
            tim::consume_parameters(manager);
        }
    }
    //
    //----------------------------------------------------------------------------------//
    //
    void timemory_trace_finalize(void)
    {
        if(get_library_state()[1] || library_trace_count.load() == 0)
            return;

        if(tim::settings::verbose() > 1 || tim::settings::debug())
            PRINT_HERE("rank = %i, pid = %i, thread = %i", tim::dmp::rank(),
                       (int) tim::process::get_id(), (int) tim::threading::get_id());

        // do the finalization
        auto _count = --library_trace_count;

        tim::auto_lock_t lk(tim::type_mutex<tim::api::native_tag>());

        if(_count > 0)
        {
            // have the manager finalize
            tim::manager::instance()->finalize();
            return;
        }

        tim::mpi::barrier();

        // reset traces just in case
        user_trace_bundle::reset();

        // clean up any remaining entries
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
