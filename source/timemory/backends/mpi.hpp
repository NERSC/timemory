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
//

/** \file mpi.hpp
 * \headerfile mpi.hpp "timemory/mpi.hpp"
 * Defines mpi functions and dummy functions when compiled without MPI
 *
 */

#pragma once

#include "timemory/utility/macros.hpp"
#include "timemory/utility/utility.hpp"

#include <algorithm>
#include <cstdint>

#if defined(TIMEMORY_USE_MPI)
#    include <mpi.h>
#else
// dummy MPI types
#    define MPI_Comm int32_t
#    define MPI_COMM_WORLD 0
#    define MPI_INFO_NULL 0
#    define MPI_COMM_TYPE_SHARED 0
#    define MPI_INT int32_t
#    define MPI_CHAR char
#    define MPI_Gather(...)
#    define MPI_Gatherv(...)
#    define MPI_Comm_free(...)
#    define MPI_Comm_split(...)
#    define MPI_Comm_split_type(...)
#endif

#include "timemory/utility/utility.hpp"

namespace tim
{
namespace mpi
{
//--------------------------------------------------------------------------------------//
#if defined(TIMEMORY_USE_MPI)
using comm_t = MPI_Comm;
using info_t = MPI_Info;
#else
// dummy MPI types
using comm_t = int32_t;
using info_t = int32_t;
#endif

static const comm_t  comm_world_v       = MPI_COMM_WORLD;
static const info_t  info_null_v        = MPI_INFO_NULL;
static const int32_t comm_type_shared_v = MPI_COMM_TYPE_SHARED;

//--------------------------------------------------------------------------------------//

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

inline bool
is_initialized()
{
    int32_t _init = 0;
#if defined(TIMEMORY_USE_MPI)
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
        MPI_Init(&argc, &argv);
#else
    consume_parameters(argc, argv);
#endif
}

//--------------------------------------------------------------------------------------//

inline void
finalize()
{
#if defined(TIMEMORY_USE_MPI)
    MPI_Finalize();
#endif
}

//--------------------------------------------------------------------------------------//

inline int32_t
rank(comm_t comm = comm_world_v)
{
    int32_t _rank = 0;
#if defined(TIMEMORY_USE_MPI)
    if(is_initialized())
        MPI_Comm_rank(comm, &_rank);
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
        MPI_Comm_size(comm, &_size);
#else
    consume_parameters(comm);
#endif
    return std::max(_size, (int32_t) 1);
}

//--------------------------------------------------------------------------------------//

inline void
barrier(comm_t comm = comm_world_v)
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

}  // namespace mpi

}  // namespace tim

//--------------------------------------------------------------------------------------//
