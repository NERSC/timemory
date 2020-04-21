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

//======================================================================================//

namespace
{
static auto library_manager_handle  = tim::manager::master_instance();
static auto library_settings_handle = tim::settings::shared_instance<TIMEMORY_API>();
static thread_local bool library_dyninst_init = false;
}  // namespace

//======================================================================================//

using string_t    = std::string;
using traceset_t  = tim::component_tuple<user_trace_bundle>;
using trace_map_t = std::unordered_map<size_t, std::deque<traceset_t*>>;

//--------------------------------------------------------------------------------------//

static trace_map_t&
get_trace_map()
{
    static thread_local trace_map_t _instance;
    return _instance;
}

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
        timemory_trace_init(get_trace_components().c_str());
        set_attr();
        return ret;
    }

    // MPI_Init_thread
    int operator()(int* argc, char*** argv, int req, int* prov)
    {
        auto ret = MPI_Init_thread(argc, argv, req, prov);
        timemory_trace_init(get_trace_components().c_str());
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
    void timemory_push_trace(const char* name)
    {
        if(tim::settings::verbose() > 2 || tim::settings::debug())
            PRINT_HERE("%s", name);

        static thread_local auto& _trace_map = get_trace_map();
        size_t                    id         = tim::add_hash_id(name);

#if defined(DEBUG)
        if(tim::settings::verbose() > 2)
        {
            int64_t n = _trace_map[id].size();
            printf("beginning trace for '%s' (id = %llu, offset = %lli)...\n", name,
                   (long long unsigned) id, (long long int) n);
        }
#endif

        _trace_map[id].push_back(new traceset_t(name));
        _trace_map[id].back()->start();

        if(tim::settings::verbose() > 3)
            PRINT_HERE("%s", name);
    }
    //
    //----------------------------------------------------------------------------------//
    //
    void timemory_pop_trace(const char* name)
    {
        if(tim::settings::verbose() > 2 || tim::settings::debug())
            PRINT_HERE("%s", name);

        static thread_local auto& _trace_map = get_trace_map();
        size_t                    id         = tim::add_hash_id(name);
        int64_t                   ntotal     = _trace_map[id].size();
        int64_t                   offset     = ntotal - 1;

        if(tim::settings::verbose() > 2)
            printf("ending trace for %llu [offset = %lli]...\n", (long long unsigned) id,
                   (long long int) offset);

        if(offset >= 0 && ntotal > 0)
        {
            if(_trace_map[id].back())
            {
                _trace_map[id].back()->stop();
                delete _trace_map[id].back();
            }
            _trace_map[id].pop_back();
        }

        if(tim::settings::verbose() > 3)
            PRINT_HERE("%s", name);
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
    void timemory_trace_init(const char* args)
    {
        auto tid = tim::threading::get_id();

        if(use_mpi_gotcha && !mpi_gotcha_handle.get())
        {
            PRINT_HERE("rank = %i, thread = %i", tim::dmp::rank(), (int) tid);
            mpi_gotcha_handle =
                std::make_shared<mpi_trace_bundle_t>("timemory_trace_mpi_gotcha");
            mpi_trace_gotcha::get_trace_components() = args;
            return;
        }

        PRINT_HERE("rank = %i, thread = %i", tim::dmp::rank(), (int) tid);

        auto settings_instance = tim::settings::instance<tim::api::native_tag>();
        tim::consume_parameters(settings_instance);

        if(tid == 0)
        {
            library_dyninst_init = true;
            tim::manager::use_exit_hook(false);
            tim::set_env<std::string>("TIMEMORY_TRACE_COMPONENTS", args, 0);
            // reset traces just in case
            user_trace_bundle::reset();
            // configure bundle
            user_trace_bundle::global_init(nullptr);
            // add default
            // if(user_trace_bundle::bundle_size() == 0)
            //    user_trace_bundle::configure<wall_clock>();

            auto init_func = [](int _ac, char** _av) { timemory_init_library(_ac, _av); };
            tim::config::read_command_line(init_func);
            tim::settings::parse();
        }
        else
        {
            auto manager = tim::manager::instance();
            tim::consume_parameters(manager);
        }
    }
    //
    //----------------------------------------------------------------------------------//
    //
    void timemory_trace_finalize(void)
    {
        tim::auto_lock_t lk(tim::type_mutex<tim::api::native_tag>());

        if(tim::threading::get_id() > 0)
        {
            // have the manager finalize
            tim::manager::instance()->finalize();
            return;
        }

        if(!library_dyninst_init)
            return;

        if(tim::settings::verbose() > 1 || tim::settings::debug())
            PRINT_HERE("pid = %i, tid = %i", (int) tim::process::get_id(),
                       (int) tim::threading::get_id());

        tim::mpi::barrier();

        // do the finalization
        library_dyninst_init = false;

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
