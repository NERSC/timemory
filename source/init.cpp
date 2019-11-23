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

/** \file init.cpp
 * This file defined the extern init
 *
 */

#define TIMEMORY_BUILD_EXTERN_INIT

#include "timemory/components.hpp"
#include "timemory/manager.hpp"
#include "timemory/timemory.hpp"
#include "timemory/utility/macros.hpp"
#include "timemory/utility/serializer.hpp"
#include "timemory/utility/singleton.hpp"
#include "timemory/utility/utility.hpp"

using namespace tim::component;

#if defined(TIMEMORY_EXTERN_INIT)

//======================================================================================//
#    if defined(TIMEMORY_USE_MPI)
::tim::manager*
timemory_mpi_manager_master_instance()
{
    using manager_t     = tim::manager;
    static auto& _pinst = tim::get_shared_ptr_pair<manager_t>();
    return _pinst.first.get();
}
#    endif

extern "C"
{
    __library_ctor__ void timemory_library_constructor()
    {
#    if defined(DEBUG)
        auto _debug   = tim::settings::debug();
        auto _verbose = tim::settings::verbose();
#    endif

#    if defined(DEBUG)
        if(_debug || _verbose > 3)
            printf("[%s]> initializing manager...\n", __FUNCTION__);
#    endif

        // fully initialize manager
        static thread_local auto _instance = tim::manager::instance();
        static auto              _master   = tim::manager::master_instance();

        if(_instance != _master)
        {
            printf("[%s]> master_instance() != instance() : %p vs. %p\n", __FUNCTION__,
                   (void*) _instance.get(), (void*) _master.get());
        }

#    if defined(DEBUG)
        if(_debug || _verbose > 3)
            printf("[%s]> initializing storage...\n", __FUNCTION__);
#    endif

        // initialize storage
        using tuple_type = tim::available_tuple<tim::complete_tuple_t>;
        tim::manager::get_storage<tuple_type>::initialize(_master);
    }

#    if defined(TIMEMORY_USE_MPI)
    int MPI_Init(int* argc, char*** argv)
    {
        static auto _manager = timemory_mpi_manager_master_instance();
        tim::consume_parameters(_manager);
        if(tim::settings::debug())
        {
            printf("[%s@%s:%i]> timemory intercepted MPI_Init!\n", __FUNCTION__, __FILE__,
                   __LINE__);
        }
        ::tim::timemory_init(*argc, *argv);
        return PMPI_Init(argc, argv);
    }

    int MPI_Init_thread(int* argc, char*** argv, int req, int* prov)
    {
        if(req != MPI_THREAD_MULTIPLE)
            throw std::runtime_error(
                "Error! Invalid call to MPI_Init_thread(...)! timemory requires "
                "MPI_Init_thread(int*, char***, MPI_THREAD_MULTIPLE, int*)");

        static auto _manager = timemory_mpi_manager_master_instance();
        tim::consume_parameters(_manager);
        if(tim::settings::debug())
        {
            printf("[%s@%s:%i]> timemory intercepted MPI_Init_thread!\n", __FUNCTION__,
                   __FILE__, __LINE__);
        }
        ::tim::timemory_init(*argc, *argv);
        return PMPI_Init_thread(argc, argv, req, prov);
    }

    int MPI_Finalize()
    {
        if(tim::settings::debug())
        {
            printf("[%s@%s:%i]> timemory intercepted MPI_Finalize!\n", __FUNCTION__,
                   __FILE__, __LINE__);
        }
        auto manager = timemory_mpi_manager_master_instance();
        if(manager)
            manager->finalize();
        ::tim::mpi::is_finalized() = true;
        return PMPI_Finalize();
    }
#    endif
}
//======================================================================================//

namespace tim
{
//======================================================================================//

env_settings*
env_settings::instance()
{
    static env_settings* _instance = new env_settings();
    return _instance;
}

//======================================================================================//

std::atomic<int32_t>&
manager::f_manager_instance_count()
{
    static std::atomic<int32_t> _instance;
    return _instance;
}

//======================================================================================//
// number of threads counter
//
std::atomic<int32_t>&
manager::f_thread_counter()
{
    static std::atomic<int32_t> _instance;
    return _instance;
}

//======================================================================================//
// get either master or thread-local instance
//
manager::pointer_t
manager::instance()
{
    static thread_local auto _inst = get_shared_ptr_pair_instance<manager>();
    return _inst;
}

//======================================================================================//
// get master instance
//
manager::pointer_t
manager::master_instance()
{
    static auto _pinst = get_shared_ptr_pair_master_instance<manager>();
    return _pinst;
}

//--------------------------------------------------------------------------------------//
//
//
graph_hash_map_ptr_t
get_hash_ids()
{
    static thread_local auto _inst = get_shared_ptr_pair_instance<graph_hash_map_t>();
    return _inst;
}

//--------------------------------------------------------------------------------------//
//
//
graph_hash_alias_ptr_t
get_hash_aliases()
{
    static thread_local auto _inst = get_shared_ptr_pair_instance<graph_hash_alias_t>();
    return _inst;
}

}  // namespace tim

#endif  // defined(TIMEMORY_EXTERN_INIT)
