// MIT License
//
// Copyright (c) 2020, The Regents of the University of California,
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

#include "timemory/timemory.hpp"

#include <cstdio>
#include <cstdlib>
#include <thread>
#include <omp.h>

//
// shorthand
//
using namespace tim::component;

//
// scorep
//
using tuple_t = tim::auto_tuple<scorep>;

//--------------------------------------------------------------------------------------//

long
fib(long n)
{
    return (n < 2) ? n : (fib(n - 1) + fib(n - 2));
}

//--------------------------------------------------------------------------------------//

void 
consume(long nfib, int nitr)
{
    auto id = std::this_thread::get_id();
    std::stringstream tid;
    tid << id;

    for(int i = 0; i < nitr; ++i)
    {
        TIMEMORY_MARKER(tuple_t, tid.str() + ":total");
        long ans = fib(nfib);
            
        TIMEMORY_BLANK_MARKER(tuple_t, tid.str() + ":nested/", i % 2);
        ans += fib(nfib - 2);

        TIMEMORY_CONDITIONAL_BASIC_MARKER(i % 3 == 1, tuple_t, tid.str() + ":occasional/", i);
        ans += fib(nfib);

        printf("%d: Answer = %li\n", i, ans);
    }
}

//--------------------------------------------------------------------------------------//

void 
consume_omp(long nfib, int nitr)
{
#pragma omp parallel for num_threads(2) schedule(static, nitr/2)
    for(int i = 0; i < nitr; ++i)
    {
        TIMEMORY_MARKER(tuple_t, "total");
        long ans = fib(nfib);

        TIMEMORY_BLANK_MARKER(tuple_t, "nested/", i % 2);
        ans += fib(nfib - 2);

        TIMEMORY_CONDITIONAL_BASIC_MARKER(i % 3 == 1, tuple_t, "occasional/", i);
        ans += fib(nfib);

        printf("%d: Answer = %li\n", i, ans);
    }
}

//--------------------------------------------------------------------------------------//

int
main(int argc, char** argv)
{
    tim::settings::file_output() = false;
    tim::settings::text_output() = false;

    tim::mpi::initialize(argc, argv);
    tim::timemory_init(argc, argv);

    long nfib = (argc > 1) ? atol(argv[1]) : 30;
    int  nitr = (argc > 2) ? atoi(argv[2]) : 3;

    // init scorep profiling from the main thread
    TIMEMORY_MARKER(tuple_t, "main");

#if !defined (USE_MPI)
#   if !defined(USE_OPENMP)
    std::cout << "Using STL threading \n";

    // do work from a thread and measure with Score-P
    std::thread t1(consume, nfib, nitr);
    std::thread t2(consume, nfib, nitr);

    t1.join();
    t2.join();

#   else
    std::cout << "Using OpenMP threading \n";
    consume_omp(nfib, nitr * 2);
#   endif // USE_OPENMP

#else
    // do work from a thread and measure with Score-P
    std::thread t1(consume, nfib, nitr);
    std::thread t2(consume, nfib, nitr);

    t1.join();
    t2.join();

#   if defined(USE_OPENMP)
    tim::mpi::barrier();
    consume_omp(nfib, nitr * 2);
#   endif // USE_OPENMP

#endif // USE_MPI

    tim::mpi::barrier();
    tim::timemory_finalize();

    return EXIT_SUCCESS;
}

//--------------------------------------------------------------------------------------//
