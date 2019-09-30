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
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//

/** \file manager.hpp
 * \headerfile manager.hpp "timemory/details/manager.hpp"
 * Provides inline implementation of manager functions
 *
 */

#include "timemory/settings.hpp"
#include "timemory/utility/macros.hpp"
#include "timemory/utility/singleton.hpp"
#include "timemory/utility/utility.hpp"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <mutex>
#include <sstream>
#include <thread>

//======================================================================================//

namespace tim
{
//======================================================================================//
#if !defined(TIMEMORY_EXTERN_INIT)
inline std::atomic<int32_t>&
manager::f_manager_instance_count()
{
    static std::atomic<int32_t> instance;
    return instance;
}

//======================================================================================//
// get either master or thread-local instance
//
inline manager::pointer
manager::instance()
{
    return details::manager_singleton().instance();
}

//======================================================================================//
// get master instance
//
inline manager::pointer
manager::master_instance()
{
    return details::manager_singleton().master_instance();
}

//======================================================================================//
// static function
inline manager::pointer
manager::noninit_instance()
{
    return details::manager_singleton().instance_ptr();
}

//======================================================================================//
// static function
inline manager::pointer
manager::noninit_master_instance()
{
    return details::manager_singleton().master_instance_ptr();
}
#endif
//======================================================================================//

inline manager::manager()
: m_instance_count(f_manager_instance_count()++)
{
    f_thread_counter()++;
    static std::atomic<int> _once(0);

    if(_once++ == 0)
    {
        settings::parse();
        std::atexit(&exit_hook);
    }

    if(m_instance_count == 0)
    {
        if(settings::banner())
            printf(
                "#--------------------- tim::manager initialized [%i] "
                "---------------------#\n\n",
                m_instance_count);
    }

    if(singleton_t::master_instance_ptr() && singleton_t::instance_ptr())
    {
        std::ostringstream errss;
        errss << "manager singleton has already been created";
        throw std::runtime_error(errss.str().c_str());
    }
}

//======================================================================================//

inline manager::~manager()
{
    if(m_instance_count > 0)
    {
        f_thread_counter().store(0, std::memory_order_relaxed);
    }

    --f_manager_instance_count();
}

//======================================================================================//

inline void
manager::exit_hook()
{
    auto*   ptr   = noninit_master_instance();
    int32_t count = 0;
    if(ptr)
    {
        count = ptr->instance_count();
        if(settings::banner())
            printf(
                "\n\n#---------------------- tim::manager destroyed [%i] "
                "----------------------#\n",
                count);
        delete ptr;
    }
    papi::shutdown();
    mpi::finalize();
}

//======================================================================================//
// static function
inline manager::comm_group_t
manager::get_communicator_group()
{
    int32_t max_concurrency = std::thread::hardware_concurrency();
    // We want on-node communication only
    int32_t nthreads         = f_thread_counter().load();
    int32_t max_processes    = max_concurrency / nthreads;
    int32_t mpi_node_default = mpi::size() / max_processes;
    if(mpi_node_default < 1)
        mpi_node_default = 1;
    int32_t mpi_node_count = get_env<int32_t>("TIMEMORY_NODE_COUNT", mpi_node_default);
    int32_t mpi_split_size = mpi::rank() / (mpi::size() / mpi_node_count);

    // Split the communicator based on the number of nodes and use the
    // original rank for ordering
    mpi::comm_t local_mpi_comm;
    mpi::comm_split(mpi::comm_world_v, mpi_split_size, mpi::rank(), &local_mpi_comm);

#if defined(DEBUG)
    if(settings::verbose() > 1 || settings::debug())
    {
        int32_t local_mpi_rank = mpi::rank(local_mpi_comm);
        int32_t local_mpi_size = mpi::size(local_mpi_comm);
        int32_t local_mpi_file = mpi::rank() / local_mpi_size;

        std::stringstream _info;
        _info << "\t" << mpi::rank() << " Rank      : " << mpi::rank() << std::endl;
        _info << "\t" << mpi::rank() << " Size      : " << mpi::size() << std::endl;
        _info << "\t" << mpi::rank() << " Node      : " << mpi_node_count << std::endl;
        _info << "\t" << mpi::rank() << " Local Size: " << local_mpi_size << std::endl;
        _info << "\t" << mpi::rank() << " Local Rank: " << local_mpi_rank << std::endl;
        _info << "\t" << mpi::rank() << " Local File: " << local_mpi_file << std::endl;
        std::cout << "tim::manager::" << __FUNCTION__ << "\n" << _info.str();
    }
#endif

    auto local_rank = mpi::rank() / mpi::size(local_mpi_comm);
    // check
    assert(local_rank == mpi::get_node_index());

    return comm_group_t(local_mpi_comm, local_rank);
}

//======================================================================================//

}  // namespace tim

//======================================================================================//

#include "timemory/settings.hpp"
#include "timemory/utility/storage.hpp"
#include "timemory/variadic/component_tuple.hpp"

//======================================================================================//

template <typename _Tuple>
void
tim::settings::initialize_storage()
{
    manager::get_storage<_Tuple>::initialize();
}

namespace tim
{
//--------------------------------------------------------------------------------------//
// extra variadic initialization
//
template <typename... _Types>
inline void
timemory_init()
{
    using tuple_type = tuple_concat_t<_Types...>;
    settings::initialize_storage<tuple_type>();
}

//--------------------------------------------------------------------------------------//

}  // namespace tim

//======================================================================================//
