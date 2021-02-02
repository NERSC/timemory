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

#pragma once

#include "timemory/defines.h"

#if defined(TIMEMORY_USE_EXTERN) && !defined(TIMEMORY_USE_CORE_EXTERN)
#    define TIMEMORY_USE_CORE_EXTERN
#endif

#include "timemory/utility/macros.hpp"
#include "timemory/utility/types.hpp"
#include "timemory/utility/utility.hpp"

#include <cstdint>
#include <string>
#include <unordered_map>

#if defined(TIMEMORY_USE_MPI)
#    include <mpi.h>
#endif

namespace tim
{
namespace mpi
{
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

#    if !defined(MPI_INT)
#        define MPI_INT ::tim::mpi::dummy_data_type::int_t
#    endif

#    if !defined(MPI_FLOAT)
#        define MPI_FLOAT ::tim::mpi::dummy_data_type::float_t
#    endif

#    if !defined(MPI_DOUBLE)
#        define MPI_DOUBLE ::tim::mpi::dummy_data_type::double_t
#    endif

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
int64_t
get_id();
//
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

#else

using comm_t                            = MPI_Comm;
using info_t                            = MPI_Info;
using data_type_t                       = MPI_Datatype;
using status_t                          = MPI_Status;
static const comm_t  comm_world_v       = MPI_COMM_WORLD;
static const info_t  info_null_v        = MPI_INFO_NULL;
static const int32_t comm_type_shared_v = MPI_COMM_TYPE_SHARED;

namespace threading
{
int64_t
get_id();
//
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

#endif  // TIMEMORY_USE_MPI

template <typename Tp>
using communicator_map_t = std::unordered_map<comm_t, Tp>;

int32_t
rank(comm_t comm = mpi::comm_world_v);

bool&
use_mpi_thread();

std::string&
use_mpi_thread_type();

bool&
fail_on_error();

bool&
quiet();

bool
check_error(const char* _func, int err_code, comm_t _comm = mpi::comm_world_v);

void
barrier(comm_t comm = mpi::comm_world_v);

bool
is_supported();

bool&
is_finalized();

bool
is_initialized();

void
initialize(int& argc, char**& argv);

void
initialize(int* argc, char*** argv);

void
finalize();

int32_t
rank(comm_t comm);

int32_t
size(comm_t comm = mpi::comm_world_v);

void
barrier(comm_t comm);

void
comm_split(comm_t comm, int split_size, int rank, comm_t* local_comm);

void
comm_split_type(comm_t comm, int split_size, int key, info_t info, comm_t* local_comm);

comm_t
get_node_comm();

int32_t
get_num_ranks_per_node();

int32_t
get_num_nodes();

int32_t
get_node_index();

void
send(const std::string& str, int dest, int tag, comm_t comm = mpi::comm_world_v);

void
recv(std::string& str, int src, int tag, comm_t comm = mpi::comm_world_v);

void
gather(const void* sendbuf, int sendcount, data_type_t sendtype, void* recvbuf,
       int recvcount, data_type_t recvtype, int root, comm_t comm = mpi::comm_world_v);

void
comm_spawn_multiple(int count, char** commands, char*** argv, const int* maxprocs,
                    const info_t* info, int root, comm_t comm, comm_t* intercomm,
                    int* errcodes);

}  // namespace mpi
}  // namespace tim

#if !defined(TIMEMORY_CORE_SOURCE) && !defined(TIMEMORY_USE_CORE_EXTERN)
#    include "timemory/backends/mpi.cpp"
#endif
