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
//

#include <timemory/timemory.hpp>

#include "gotcha/gotcha.h"
#include "gotcha/gotcha_types.h"

#include <array>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <set>
#include <string>
#include <unistd.h>
#include <vector>

//--------------------------------------------------------------------------------------//
// make the namespace usage a little clearer
//
namespace settings {
using namespace tim::settings;
}
namespace mpi {
using namespace tim::mpi;
}

void
init()
{
    using ftype = decltype(MPI_Init);
    std::cout << "function type: " << tim::demangle(typeid(ftype).name()) << std::endl;

    using auto_timer_t   = tim::auto_timer;
    constexpr size_t _Pn = 1;
    constexpr size_t _Mn = 18;

    using put_gotcha_t = tim::component::gotcha<_Pn, auto_timer_t>;
    using mpi_gotcha_t = tim::component::gotcha<_Mn, auto_timer_t>;

    put_gotcha_t::configure<0, int, const char*>("puts");

    // mpi_gotcha_t::instrument<0, int, int*, char***>("MPI_Init");
    TIMEMORY_GOTCHA(mpi_gotcha_t, 0, MPI_Init);
    TIMEMORY_GOTCHA(mpi_gotcha_t, 1, MPI_Barrier);
    mpi_gotcha_t::configure<2, int, MPI_Comm, int*>("MPI_Comm_rank");
    mpi_gotcha_t::configure<3, int, MPI_Comm, int*>("MPI_Comm_size");
    mpi_gotcha_t::configure<4, int>("MPI_Finalize");
    TIMEMORY_GOTCHA(mpi_gotcha_t, 5, MPI_Bcast);
    TIMEMORY_GOTCHA(mpi_gotcha_t, 6, MPI_Scan);
    TIMEMORY_GOTCHA(mpi_gotcha_t, 7, MPI_Allreduce);
    TIMEMORY_GOTCHA(mpi_gotcha_t, 8, MPI_Reduce);
    TIMEMORY_GOTCHA(mpi_gotcha_t, 9, MPI_Alltoall);
    TIMEMORY_GOTCHA(mpi_gotcha_t, 10, MPI_Allgather);
    TIMEMORY_GOTCHA(mpi_gotcha_t, 11, MPI_Gather);
    TIMEMORY_GOTCHA(mpi_gotcha_t, 12, MPI_Scatter);
}

//======================================================================================//

int
main(int argc, char** argv)
{
    settings::verbose() = 1;
    tim::timemory_init(argc, argv);  // parses environment, sets output paths

    init();

    mpi::initialize(argc, argv);
    mpi::barrier();
    auto sz      = mpi::size();
    auto rank    = mpi::rank();
    auto rank_sz = mpi::rank() * mpi::size();
    mpi::barrier();
    printf("mpi size = %i\n", (int) sz);
    printf("mpi rank = %i\n", (int) rank);
    printf("mpi (rank * size) = %i\n", (int) rank_sz);
}

//======================================================================================//
