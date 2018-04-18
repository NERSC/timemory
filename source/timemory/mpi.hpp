//  MIT License
//  
//  Copyright (c) 2018, The Regents of the University of California, 
// through Lawrence Berkeley National Laboratory (subject to receipt of any 
// required approvals from the U.S. Dept. of Energy).  All rights reserved.
//  
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to deal
//  in the Software without restriction, including without limitation the rights
//  to use, copy, modify, merge, publish, distribute, sublicense, and
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//  
//  The above copyright notice and this permission notice shall be included in all
//  copies or substantial portions of the Software.
//  
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//  SOFTWARE.
//

/** \file mpi.hpp
 * \headerfile mpi.hpp "timemory/mpi.hpp"
 * Defines mpi functions and dummy functions when compiled without MPI
 *
 */

#ifndef mpi_hpp_
#define mpi_hpp_

#include "timemory/macros.hpp"
#include "timemory/utility.hpp"

#include <cstdint>
#include <algorithm>

#if defined(TIMEMORY_USE_MPI)
#   include <mpi.h>
#else
// dummy MPI types
#   define MPI_Comm int32_t
#   define MPI_COMM_WORLD 0
#endif

#include "timemory/utility.hpp"

namespace tim
{

//----------------------------------------------------------------------------//

inline bool mpi_is_initialized()
{
    int32_t _init = 0;
#if defined(TIMEMORY_USE_MPI)
    MPI_Initialized(&_init);
#endif
    return (_init != 0) ? true : false;
}

//----------------------------------------------------------------------------//

inline int32_t mpi_rank(MPI_Comm comm = MPI_COMM_WORLD)
{
    int32_t _rank = 0;
#if defined(TIMEMORY_USE_MPI)
    if(mpi_is_initialized())
        MPI_Comm_rank(comm, &_rank);
#else
    consume_parameters(comm);
#endif
    return std::max(_rank, (int32_t) 0);
}

//----------------------------------------------------------------------------//

inline int32_t mpi_size(MPI_Comm comm = MPI_COMM_WORLD)
{
    int32_t _size = 1;
#if defined(TIMEMORY_USE_MPI)
    if(mpi_is_initialized())
        MPI_Comm_size(comm, &_size);
#else
    consume_parameters(comm);
#endif
    return std::max(_size, (int32_t) 1);
}

//----------------------------------------------------------------------------//

inline void mpi_barrier(MPI_Comm comm = MPI_COMM_WORLD)
{
#if defined(TIMEMORY_USE_MPI)
    if(mpi_is_initialized())
        MPI_Barrier(comm);
#else
    consume_parameters(comm);
#endif
}

//----------------------------------------------------------------------------//

inline bool has_mpi_support()
{
#if defined(TIMEMORY_USE_MPI)
    return true;
#else
    return false;
#endif
}

//----------------------------------------------------------------------------//

} // namespace tim

//----------------------------------------------------------------------------//

#if defined(TIMEMORY_USE_MPI)
#else

#   define MPI_INT int32_t
#   define MPI_CHAR char
#   define MPI_Gather(send_data, send_count, send_type, \
                      recv_data, recv_count, recv_type, \
                      root, communicator) \
    { tim::consume_parameters(send_data, send_count, \
                              recv_data, recv_count,  \
                              root, communicator); }

#   define MPI_Gatherv(send_buf, send_count, send_type, \
                       recv_buf, recv_count, group_size, \
                       recv_type, root, communicator) \
    { tim::consume_parameters(send_buf, send_count, \
                              recv_buf, recv_count,  \
                              root, communicator); }
#   define MPI_Comm_free(comm) { tim::consume_parameters(comm); }
#   define MPI_Comm_split(comm, color, key, new_comm) { \
    tim::consume_parameters(color, key, new_comm); }

#endif

//----------------------------------------------------------------------------//

#endif
