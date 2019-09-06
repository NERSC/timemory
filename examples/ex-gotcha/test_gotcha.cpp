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

#if !defined(DEBUG)
#    define DEBUG
#endif

#include "test_gotcha_lib.hpp"
#include <timemory/timemory.hpp>

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
namespace trait
{
using namespace tim::trait;
}

//======================================================================================//

using namespace tim::component;

using auto_timer_t = tim::auto_timer;
// this should fail
// using auto_tuple_t = tim::component_tuple<real_clock, cpu_clock, cpu_util, gotcha<1,
// auto_timer_t, int>>;
using auto_tuple_t =
    tim::component_tuple<real_clock, cpu_clock, cpu_util, peak_rss, current_rss>;

constexpr size_t _Pn = 3;
constexpr size_t _Mn = 15;
constexpr size_t _Sn = 8;

using put_gotcha_t = tim::component::gotcha<_Pn, auto_timer_t>;
using std_gotcha_t = tim::component::gotcha<_Sn, auto_timer_t, int>;
using mpi_gotcha_t = tim::component::gotcha<_Mn, auto_tuple_t>;

//======================================================================================//

void
init()
{
    put_gotcha_t::configure<0, int, const char*>("puts");

    // TIMEMORY_GOTCHA(std_gotcha_t, 0, sinf);
    // TIMEMORY_GOTCHA(std_gotcha_t, 2, expf);
    std_gotcha_t::configure<1, double, double>("sin");
    std_gotcha_t::configure<3, double, double>("exp");

    std_gotcha_t::configure<4, ext::tuple_t, int>("_ZN3ext7do_workEi");

    // mpi_gotcha_t::configure<0, int, int*, char***>("MPI_Init");

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

    printf("put gotcha is available: %s\n",
           trait::as_string<trait::is_available<put_gotcha_t>>().c_str());
    printf("std gotcha is available: %s\n",
           trait::as_string<trait::is_available<std_gotcha_t>>().c_str());
    printf("mpi gotcha is available: %s\n",
           trait::as_string<trait::is_available<mpi_gotcha_t>>().c_str());
}

//======================================================================================//

int
main(int argc, char** argv)
{
    settings::verbose() = 1;
    tim::timemory_init(argc, argv);  // parses environment, sets output paths

    init();

    PRINT_HERE("backend MPI_Init");
    mpi::initialize(argc, argv);

    PRINT_HERE("direct  MPI_Barrier");
    MPI_Barrier(MPI_COMM_WORLD);

    PRINT_HERE("backend MPI_Barrier");
    mpi::barrier();

    int size = 1;
    int rank = 0;

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    size = std::max<int>(size, mpi::size());
    rank = std::max<int>(rank, mpi::rank());

    auto rank_size = (rank + 1) * (size + 1);

    MPI_Barrier(MPI_COMM_WORLD);
    mpi::barrier();

    printf("mpi size = %i\n", (int) size);
    printf("mpi rank = %i\n", (int) rank);
    printf("mpi ((rank + 1) * (size + 1)) = %i\n", (int) rank_size);

    int nitr = 10;
    if(argc > 1) nitr = atoi(argv[1]);

    auto _sqrt = ext::do_work(nitr);

    printf("[iterations=%i]> single-precision work = %f\n", nitr, std::get<0>(_sqrt));
    printf("[iterations=%i]> double-precision work = %f\n", nitr, std::get<1>(_sqrt));
}

//======================================================================================//
