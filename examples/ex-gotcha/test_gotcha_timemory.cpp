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
namespace settings
{
using namespace tim::settings;
}
namespace mpi
{
using namespace tim::mpi;
}

//--------------------------------------------------------------------------------------//

void
init()
{
    // enum gotcha_error_t result;
    // gotcha_set_priority("timemory", 0);

    constexpr size_t _Nt = 18;
    using gotcha_t       = tim::component::gotcha<_Nt, tim::auto_timer>;
    gotcha_t::configure<0, int>("MPI_Finalize");
    gotcha_t::configure<1, int, const char*>("puts");
    gotcha_t::configure<2, int, MPI_Comm>("MPI_Barrier");
    gotcha_t::configure<3, int, int*, char***>("MPI_Init");
    gotcha_t::configure<4, int, MPI_Comm, int*>("MPI_Comm_rank");
    gotcha_t::configure<5, int, MPI_Comm, int*>("MPI_Comm_size");
    gotcha_t::configure<6, int, int*, char***, int, int*>("MPI_Init_thread");
    gotcha_t::configure<7, int, void*, int, MPI_Datatype, int, MPI_Comm>("MPI_Bcast");
    gotcha_t::configure<8, int, void*, void*, int, MPI_Datatype, MPI_Op, MPI_Comm>(
        "MPI_Scan");
    gotcha_t::configure<9, int, void*, void*, int, MPI_Datatype, MPI_Op, MPI_Comm>(
        "MPI_Allreduce");
    gotcha_t::configure<10, int, void*, void*, int, MPI_Datatype, MPI_Op, int, MPI_Comm>(
        "MPI_Reduce");
    gotcha_t::configure<11, int, void*, int, MPI_Datatype, void*, int, MPI_Datatype,
                        MPI_Comm>("MPI_Alltoall");
    gotcha_t::configure<12, int, void*, int, MPI_Datatype, void*, int, MPI_Datatype,
                        MPI_Comm>("MPI_Allgather");
    gotcha_t::configure<13, int, void*, int, MPI_Datatype, void*, int, MPI_Datatype, int,
                        MPI_Comm>("MPI_Gather");
    gotcha_t::configure<14, int, void*, int, MPI_Datatype, void*, int, MPI_Datatype, int,
                        MPI_Comm>("MPI_Scatter");
    gotcha_t::configure<15, int, void*, int, MPI_Datatype, void*, int*, int*,
                        MPI_Datatype, MPI_Comm>("MPI_Allgatherv");
    gotcha_t::configure<16, int, void*, int, MPI_Datatype, void*, int*, int*,
                        MPI_Datatype, int, MPI_Comm>("MPI_Gatherv");
    gotcha_t::configure<17, int, void*, int*, int*, MPI_Datatype, void*, int,
                        MPI_Datatype, int, MPI_Comm>("MPI_Scatterv");

    gotcha_t::configure();
}

//--------------------------------------------------------------------------------------//

long
fibonacci(long n)
{
    return (n < 2) ? n : (fibonacci(n - 1) + fibonacci(n - 2));
}

//======================================================================================//

int
main(int argc, char** argv)
{
    settings::verbose() = 1;
    tim::timemory_init(argc, argv);  // parses environment, sets output paths

    init();

    mpi::initialize(argc, argv);
    const int n = 10;
    puts("Test");
    auto ret = fibonacci(n);
    printf("fibonacci(%i) = %li\n", n, ret);
}

//======================================================================================//
