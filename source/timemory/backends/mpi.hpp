//  MIT License
//
//  Copyright (c) 2020, The Regents of the University of California,
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
//

/** \file backends/mpi.hpp
 * \headerfile backends/mpi.hpp "timemory/backends/mpi.hpp"
 * Defines mpi functions and dummy functions when compiled without MPI
 *
 */

#pragma once

#include "timemory/settings/declaration.hpp"
#include "timemory/utility/macros.hpp"  // macro definitions w/ no internal deps
#include "timemory/utility/types.hpp"
#include "timemory/utility/utility.hpp"  // generic functions w/ no internal deps

#include <cstdint>
#include <unordered_map>

#if defined(TIMEMORY_USE_MPI)
#    include <mpi.h>
#endif

namespace tim
{
namespace mpi
{
//--------------------------------------------------------------------------------------//
#if defined(TIMEMORY_USE_MPI)
using comm_t                            = MPI_Comm;
using info_t                            = MPI_Info;
static const comm_t  comm_world_v       = MPI_COMM_WORLD;
static const info_t  info_null_v        = MPI_INFO_NULL;
static const int32_t comm_type_shared_v = MPI_COMM_TYPE_SHARED;
namespace threading
{
enum : int
{
    /// Only one thread will execute.
    single = MPI_THREAD_SINGLE,
    /// Only main thread will do MPI calls. The process may be multi-threaded, but only
    /// the main thread will make MPI calls (all MPI calls are funneled to the main
    /// thread)
    funneled = MPI_THREAD_FUNNELED,
    /// Only one thread at the time do MPI calls. The process may be multi-threaded, and
    /// multiple threads may make MPI calls, but only one at a time: MPI calls are not
    /// made concurrently from two distinct threads (all MPI calls are serialized).
    serialized = MPI_THREAD_SERIALIZED,
    /// Multiple thread may do MPI calls with no restrictions.
    multiple = MPI_THREAD_MULTIPLE
};
}  // namespace threading
#else
// dummy MPI types
using comm_t                            = int32_t;
using info_t                            = int32_t;
static const comm_t  comm_world_v       = 0;
static const info_t  info_null_v        = 0;
static const int32_t comm_type_shared_v = 0;
namespace threading
{
enum : int
{
    /// Only one thread will execute.
    single = 0,
    /// Only main thread will do MPI calls. The process may be multi-threaded, but only
    /// the main thread will make MPI calls (all MPI calls are funneled to the main
    /// thread)
    funneled = 1,
    /// Only one thread at the time do MPI calls. The process may be multi-threaded, and
    /// multiple threads may make MPI calls, but only one at a time: MPI calls are not
    /// made concurrently from two distinct threads (all MPI calls are serialized).
    serialized = 2,
    /// Multiple thread may do MPI calls with no restrictions.
    multiple = 3
};
}  // namespace threading
#endif

template <typename Tp>
using communicator_map_t = std::unordered_map<comm_t, Tp>;

inline int32_t
rank(comm_t comm = comm_world_v);

//--------------------------------------------------------------------------------------//

inline bool
check_error(int err_code)
{
#if defined(TIMEMORY_USE_MPI)
    if(err_code != MPI_SUCCESS)
    {
        int  len = 0;
        char msg[1024];
        MPI_Error_string(err_code, msg, &len);
        int idx   = (len < 1023) ? len + 1 : 1023;
        msg[idx]  = '\0';
        int _rank = rank();
        printf("[%i]> Error code (%i): %s\n", _rank, err_code, msg);
    }
    return (err_code == MPI_SUCCESS);
#else
    consume_parameters(err_code);
    return false;
#endif
}
//--------------------------------------------------------------------------------------//

inline void
barrier(comm_t comm = comm_world_v);

inline bool
is_supported()
{
#if defined(TIMEMORY_USE_MPI)
    return true;
#else
    return false;
#endif
}

//--------------------------------------------------------------------------------------//

inline bool&
is_finalized()
{
#if defined(TIMEMORY_USE_MPI)
    static bool _instance = false;
#else
    static bool _instance = true;
#endif
    return _instance;
}

//--------------------------------------------------------------------------------------//

inline bool
is_initialized()
{
    int32_t _init = 0;
#if defined(TIMEMORY_USE_MPI)
    if(!is_finalized())
        MPI_Initialized(&_init);
#endif
    return (_init != 0) ? true : false;
}

//--------------------------------------------------------------------------------------//

inline void
initialize(int& argc, char**& argv)
{
#if defined(TIMEMORY_USE_MPI)
    if(!is_initialized())
    {
        using namespace threading;
        bool success_v = false;
        if(settings::mpi_thread())
        {
            auto _init = [&](int itr) {
                int  _actual = -1;
                auto ret     = MPI_Init_thread(&argc, &argv, itr, &_actual);
                if(_actual != itr)
                {
                    std::stringstream ss;
                    ss << "Warning! MPI_Init_thread does not support " << itr;
                    std::cerr << ss.str() << std::flush;
                    throw std::runtime_error(ss.str().c_str());
                }
                return check_error(ret);
            };

            // check_error(MPI_Init(&argc, &argv));
            // int _provided = 0;
            // MPI_Query_thread(&_provided);

            auto _mpi_type = settings::mpi_thread_type();
            if(_mpi_type == "single")
                success_v = _init(single);
            else if(_mpi_type == "serialized")
                success_v = _init(serialized);
            else if(_mpi_type == "funneled")
                success_v = _init(funneled);
            else if(_mpi_type == "multiple")
                success_v = _init(multiple);
            else
                success_v = _init(multiple);
        }

        if(!success_v)
            check_error(MPI_Init(&argc, &argv));
    }
#else
    consume_parameters(argc, argv);
#endif
}

//--------------------------------------------------------------------------------------//

inline void
initialize(int* argc, char*** argv)
{
    initialize(*argc, *argv);
}

//--------------------------------------------------------------------------------------//

inline void
finalize()
{
    // is_initialized has a check against is_finalized(), if manually invoking
    // MPI_Finalize() [not recommended bc timemory will do it when MPI support is enabled]
    // then set that value to true, e.g.
    //          tim::mpi::is_finalized() = true;
#if defined(TIMEMORY_USE_MPI)
    if(is_initialized())
    {
        // barrier();
        MPI_Finalize();
        is_finalized() = true;  // to try to avoid calling MPI_Initialized(...) after
        // finalized
    }
#endif
}

//--------------------------------------------------------------------------------------//

inline int32_t
rank(comm_t comm)
{
    int32_t _rank = 0;
#if defined(TIMEMORY_USE_MPI)
    if(is_initialized())
    {
        // this is used to guard against the queries that might happen after an
        // application calls MPI_Finalize() directly
        static communicator_map_t<int32_t>* _instance = new communicator_map_t<int32_t>();
        if(_instance->find(comm) == _instance->end())
        {
            MPI_Comm_rank(comm, &_rank);
            (*_instance)[comm] = _rank;
        }
        else
        {
            _rank = (*_instance)[comm];
        }
    }
#else
    consume_parameters(comm);
#endif
    return std::max(_rank, (int32_t) 0);
}

//--------------------------------------------------------------------------------------//

inline int32_t
size(comm_t comm = comm_world_v)
{
    int32_t _size = 1;
#if defined(TIMEMORY_USE_MPI)
    if(is_initialized())
    {
        // this is used to guard against the queries that might happen after an
        // application calls MPI_Finalize() directly
        static communicator_map_t<int32_t>* _instance = new communicator_map_t<int32_t>();
        if(_instance->find(comm) == _instance->end())
        {
            MPI_Comm_size(comm, &_size);
            (*_instance)[comm] = _size;
        }
        else
        {
            _size = (*_instance)[comm];
        }
    }
#else
    consume_parameters(comm);
#endif
    return std::max(_size, (int32_t) 1);
}

//--------------------------------------------------------------------------------------//

inline void
barrier(comm_t comm)
{
#if defined(TIMEMORY_USE_MPI)
    if(is_initialized())
        MPI_Barrier(comm);
#else
    consume_parameters(comm);
#endif
}

//--------------------------------------------------------------------------------------//

inline void
comm_split(comm_t comm, int split_size, int rank, comm_t* local_comm)
{
#if defined(TIMEMORY_USE_MPI)
    if(is_initialized())
        MPI_Comm_split(comm, split_size, rank, local_comm);
#else
    consume_parameters(comm, split_size, rank, local_comm);
#endif
}

//--------------------------------------------------------------------------------------//

inline void
comm_split_type(comm_t comm, int split_size, int key, info_t info, comm_t* local_comm)
{
#if defined(TIMEMORY_USE_MPI)
    if(is_initialized())
        MPI_Comm_split_type(comm, split_size, key, info, local_comm);
#else
    consume_parameters(comm, split_size, key, info, local_comm);
#endif
}

//--------------------------------------------------------------------------------------//
/// returns the communicator for the node
inline comm_t
get_node_comm()
{
    if(!is_initialized())
        return comm_world_v;
    auto _get_node_comm = []() {
        comm_t local_comm;
        comm_split_type(mpi::comm_world_v, mpi::comm_type_shared_v, 0, mpi::info_null_v,
                        &local_comm);
        return local_comm;
    };
    static comm_t _instance = _get_node_comm();
    return _instance;
}

//--------------------------------------------------------------------------------------//
/// returns the number of ranks on a node
inline int32_t
get_num_ranks_per_node()
{
    if(!is_initialized())
        return 1;
    return size(get_node_comm());
}

//--------------------------------------------------------------------------------------//

inline int32_t
get_num_nodes()
{
    if(!is_initialized())
        return 1;
    auto _world_size = size(comm_world_v);
    auto _ncomm_size = get_num_ranks_per_node();
    return (_world_size >= _ncomm_size) ? (_world_size / _ncomm_size) : 1;
}

//--------------------------------------------------------------------------------------//

inline int32_t
get_node_index()
{
    if(!is_initialized())
        return 0;
    return rank() / get_num_ranks_per_node();
}

//--------------------------------------------------------------------------------------//

inline void
send(const std::string& str, int dest, int tag, comm_t comm)
{
#if defined(TIMEMORY_USE_MPI)
    unsigned long long len = str.size();
    MPI_Send(&len, 1, MPI_UNSIGNED_LONG_LONG, dest, tag, comm);
    if(len != 0)
        MPI_Send(const_cast<char*>(str.data()), len, MPI_CHAR, dest, tag, comm);
#else
    consume_parameters(str, dest, tag, comm);
#endif
}

//--------------------------------------------------------------------------------------//

inline void
recv(std::string& str, int src, int tag, comm_t comm)
{
#if defined(TIMEMORY_USE_MPI)
    unsigned long long len;
    MPI_Status         s;
    MPI_Recv(&len, 1, MPI_UNSIGNED_LONG_LONG, src, tag, comm, &s);
    if(len != 0)
    {
        std::vector<char> tmp(len);
        MPI_Recv(tmp.data(), len, MPI_CHAR, src, tag, comm, &s);
        str.assign(tmp.begin(), tmp.end());
    }
    else
    {
        str.clear();
    }
#else
    consume_parameters(str, src, tag, comm);
#endif
}

//--------------------------------------------------------------------------------------//

}  // namespace mpi

}  // namespace tim

//--------------------------------------------------------------------------------------//
