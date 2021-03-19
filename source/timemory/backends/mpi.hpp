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

#include "timemory/backends/process.hpp"
#include "timemory/backends/threading.hpp"
#include "timemory/environment/declaration.hpp"
#include "timemory/utility/macros.hpp"
#include "timemory/utility/types.hpp"
#include "timemory/utility/utility.hpp"

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

#if !defined(TIMEMORY_USE_MPI)
struct dummy_data_type
{
    enum type
    {
        int_t,
        float_t,
        double_t
    };
};
#endif

//--------------------------------------------------------------------------------------//

#if !defined(TIMEMORY_USE_MPI) && !defined(MPI_INT)
#    define MPI_INT ::tim::mpi::dummy_data_type::int_t
#endif

#if !defined(TIMEMORY_USE_MPI) && !defined(MPI_FLOAT)
#    define MPI_FLOAT ::tim::mpi::dummy_data_type::float_t
#endif

#if !defined(TIMEMORY_USE_MPI) && !defined(MPI_DOUBLE)
#    define MPI_DOUBLE ::tim::mpi::dummy_data_type::double_t
#endif

//--------------------------------------------------------------------------------------//
#if defined(TIMEMORY_USE_MPI)
using comm_t                            = MPI_Comm;
using info_t                            = MPI_Info;
using data_type_t                       = MPI_Datatype;
using status_t                          = MPI_Status;
static const comm_t  comm_world_v       = MPI_COMM_WORLD;
static const info_t  info_null_v        = MPI_INFO_NULL;
static const int32_t comm_type_shared_v = MPI_COMM_TYPE_SHARED;
namespace threading
{
inline auto
get_id()
{
    return ::tim::threading::get_id();
}

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
using data_type_t                       = int32_t;
using status_t                          = int32_t;
static const comm_t  comm_world_v       = 0;
static const info_t  info_null_v        = 0;
static const int32_t comm_type_shared_v = 0;
namespace threading
{
inline auto
get_id()
{
    return ::tim::threading::get_id();
}

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

static inline bool&
use_mpi_thread()
{
    static bool _instance = tim::get_env("TIMEMORY_MPI_THREAD", true);
    return _instance;
}

//--------------------------------------------------------------------------------------//

inline std::string&
use_mpi_thread_type()
{
    static std::string _instance =
        tim::get_env<std::string>("TIMEMORY_MPI_THREAD_TYPE", "");
    return _instance;
}

//--------------------------------------------------------------------------------------//

inline bool&
fail_on_error()
{
    static bool _instance = false;
    return _instance;
}

//--------------------------------------------------------------------------------------//

inline bool&
quiet()
{
    static bool _instance = false;
    return _instance;
}

//--------------------------------------------------------------------------------------//

#if !defined(TIMEMORY_MPI_ERROR_FUNCTION)
#    define TIMEMORY_MPI_ERROR_FUNCTION(FUNC, ...) #    FUNC
#endif

#if !defined(TIMEMORY_MPI_ERROR_CHECK)
#    define TIMEMORY_MPI_ERROR_CHECK(...)                                                \
        ::tim::mpi::check_error(TIMEMORY_MPI_ERROR_FUNCTION(__VA_ARGS__, ""), __VA_ARGS__)
#endif

//--------------------------------------------------------------------------------------//

inline bool
check_error(const char* _func, int err_code, comm_t _comm = mpi::comm_world_v)
{
#if defined(TIMEMORY_USE_MPI)
    bool _success = (err_code == MPI_SUCCESS);
    if(!_success && !mpi::quiet())
    {
        int  len = 0;
        char msg[1024];
        MPI_Error_string(err_code, msg, &len);
        int idx   = (len < 1023) ? len + 1 : 1023;
        msg[idx]  = '\0';
        int _rank = rank();
        fprintf(stderr, "[rank=%i][pid=%i][tid=%i][%s]> Error code (%i): %s\n", _rank,
                (int) process::get_id(), (int) threading::get_id(), _func, err_code, msg);
    }
    if(!_success && fail_on_error())
        MPI_Abort(_comm, err_code);
    return (err_code == MPI_SUCCESS);
#else
    consume_parameters(_func, err_code, _comm);
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
    int32_t _fini = 0;
    MPI_Finalized(&_fini);
    static bool _instance = static_cast<bool>(_fini);
    if(!_instance)
        _instance = static_cast<bool>(_fini);
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
        if(use_mpi_thread())
        {
            auto _init = [&argc, &argv](int itr, const std::string& _type) {
                int  _actual = -1;
                auto ret     = MPI_Init_thread(&argc, &argv, itr, &_actual);
                if(_actual != itr)
                {
                    fprintf(stderr, "Warning! MPI_Init_thread does not support: %s\n",
                            _type.c_str());
                }
                return TIMEMORY_MPI_ERROR_CHECK(ret);
            };

            // TIMEMORY_MPI_ERROR_CHECK(MPI_Init(&argc, &argv));
            // int _provided = 0;
            // MPI_Query_thread(&_provided);

            auto _mpi_type = use_mpi_thread_type();
            if(_mpi_type == "single")
            {
                success_v = _init(single, _mpi_type);
            }
            else if(_mpi_type == "serialized")
            {
                success_v = _init(serialized, _mpi_type);
            }
            else if(_mpi_type == "funneled")
            {
                success_v = _init(funneled, _mpi_type);
            }
            else if(_mpi_type == "multiple")
            {
                success_v = _init(multiple, _mpi_type);
            }
            else
            {
                success_v = _init(multiple, "multiple");
            }
        }

        if(!success_v)
            TIMEMORY_MPI_ERROR_CHECK(MPI_Init(&argc, &argv));
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
#if defined(TIMEMORY_USE_MPI)
    if(is_initialized())
    {
        // barrier();
        MPI_Finalize();
        is_finalized() = true;
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
        TIMEMORY_MPI_ERROR_CHECK(MPI_Comm_split(comm, split_size, rank, local_comm));
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
    {
        TIMEMORY_MPI_ERROR_CHECK(
            MPI_Comm_split_type(comm, split_size, key, info, local_comm));
    }
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
send(const std::string& str, int dest, int tag, comm_t comm = mpi::comm_world_v)
{
#if defined(TIMEMORY_USE_MPI)
    unsigned long long len = str.size();
    TIMEMORY_MPI_ERROR_CHECK(MPI_Send(&len, 1, MPI_UNSIGNED_LONG_LONG, dest, tag, comm));
    if(len != 0)
    {
        TIMEMORY_MPI_ERROR_CHECK(
            MPI_Send(const_cast<char*>(str.data()), len, MPI_CHAR, dest, tag, comm));
    }
#else
    consume_parameters(str, dest, tag, comm);
#endif
}

//--------------------------------------------------------------------------------------//

inline void
recv(std::string& str, int src, int tag, comm_t comm = mpi::comm_world_v)
{
#if defined(TIMEMORY_USE_MPI)
    unsigned long long len;
    MPI_Status         s;
    TIMEMORY_MPI_ERROR_CHECK(
        MPI_Recv(&len, 1, MPI_UNSIGNED_LONG_LONG, src, tag, comm, &s));
    if(len != 0)
    {
        std::vector<char> tmp(len);
        TIMEMORY_MPI_ERROR_CHECK(MPI_Recv(tmp.data(), len, MPI_CHAR, src, tag, comm, &s));
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

inline void
gather(const void* sendbuf, int sendcount, data_type_t sendtype, void* recvbuf,
       int recvcount, data_type_t recvtype, int root, comm_t comm = mpi::comm_world_v)
{
#if defined(TIMEMORY_USE_MPI)
    if(is_initialized())
    {
        TIMEMORY_MPI_ERROR_CHECK(MPI_Gather(sendbuf, sendcount, sendtype, recvbuf,
                                            recvcount, recvtype, root, comm));
    }
#else
    consume_parameters(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root,
                       comm);
#endif
}

//--------------------------------------------------------------------------------------//

inline void
comm_spawn_multiple(int count, char** commands, char*** argv, const int* maxprocs,
                    const info_t* info, int root, comm_t comm, comm_t* intercomm,
                    int* errcodes)
{
#if defined(TIMEMORY_USE_MPI)
    if(is_initialized())
    {
        TIMEMORY_MPI_ERROR_CHECK(MPI_Comm_spawn_multiple(
            count, commands, argv, maxprocs, info, root, comm, intercomm, errcodes));
    }
#else
    consume_parameters(count, commands, argv, maxprocs, info, root, comm, intercomm,
                       errcodes);
#endif
}

//--------------------------------------------------------------------------------------//

}  // namespace mpi

}  // namespace tim

//--------------------------------------------------------------------------------------//
