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
//

#include "ex_optional.hpp"  // file that includes optional usage
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

//--------------------------------------------------------------------------------------//
//
//      Declarations
//
//--------------------------------------------------------------------------------------//

template <typename Tp> using vector_t = std::vector<Tp>;

namespace impl
{
static long fibonacci(long n);
}
long fibonacci(long n);
void status();
void write(long, long);
void allreduce(const vector_t<long>& send, vector_t<long>& recv);

//--------------------------------------------------------------------------------------//
//
//      Construct a main that uses macro tricks to avoid:
//
//      #ifdef USE_TIMEMORY
//          ....
//      #endif
//
//--------------------------------------------------------------------------------------//

int main(int argc, char** argv)
{
    // setenv when available
#if(_POSIX_C_SOURCE >= 200112L) || defined(_BSD_SOURCE) || defined(_UNIX)
    setenv("TIMEMORY_TIMING_UNITS", "ms", 0);
    setenv("TIMEMORY_MEMORY_UNITS", "kb", 0);
    setenv("TIMEMORY_TIMING_WIDTH", "14", 0);
    setenv("TIMEMORY_MEMORY_WIDTH", "12", 0);
    setenv("TIMEMORY_TIMING_PRECISION", "6", 0);
    setenv("TIMEMORY_MEMORY_PRECISION", "3", 0);
    setenv("TIMEMORY_TIMING_SCIENTIFIC", "OFF", 0);
    setenv("TIMEMORY_MEMORY_SCIENTIFIC", "OFF", 0);
#endif

    //
    //  Dummy functions when USE_TIMEMORY not defined
    //
    tim::mpi::initialize(argc, argv);
    status();
    tim::timemory_init(argc, argv);
    //
    //  Provide some work
    //
    std::vector<long> fibvalues;
    for(int i = 1; i < argc; ++i) fibvalues.push_back(atol(argv[i]));
    if(fibvalues.empty())
    {
        fibvalues.resize(44);
        long n = 0;
        std::generate_n(fibvalues.data(), fibvalues.size(), [&]() { return ++n; });
    }

    //
    // create an auto tuple accessible via a caliper integer or expand to nothing
    //
    auto main = TIMEMORY_HANDLE(auto_hybrid_t, "");
    TIMEMORY_CALIPER(0, auto_hybrid_t, "");

    //
    // call <auto_tuple_t>.report_at_exit(true) or expand to nothing
    //
    main.report_at_exit(true);
    TIMEMORY_CALIPER_APPLY(0, report_at_exit, true);

    //
    //  Execute the work
    //
    std::atomic<long> ret_sum;
#pragma omp parallel for
    for(int64_t i = 0; i < (int64_t) fibvalues.size(); ++i)
    {
        auto itr = fibvalues.at(i);
        auto ret = fibonacci(itr);
        write(itr, ret);
        ret_sum += ret;
    }

    std::vector<long> ret_reduce;
    std::vector<long> ret_send;
    for(size_t i = 0; i < fibvalues.size(); ++i)
        ret_send.push_back(fibonacci(fibvalues.at(i) % 20 + 10));
    allreduce(ret_send, ret_reduce);
    for(size_t i = 0; i < fibvalues.size(); ++i) write(fibvalues.at(i), ret_reduce.at(i));

    //
    // call <auto_tuple_t>.stop() or expand to nothing
    //
    TIMEMORY_CALIPER_APPLY(0, stop);

    //
    // sleep for 1 second so difference between two calipers
    //
    std::this_thread::sleep_for(std::chrono::seconds(1));

    //
    // call <auto_tuple_t>.stop() or expand to nothing
    //
    main.stop();

    status();
    tim::timemory_finalize();
    tim::mpi::finalize();

    return EXIT_SUCCESS;
}

//--------------------------------------------------------------------------------------//
//
//      Implementations
//
//--------------------------------------------------------------------------------------//

long impl::fibonacci(long n)
{
    return (n < 2) ? n : (impl::fibonacci(n - 1) + impl::fibonacci(n - 2));
}

long fibonacci(long n)
{
    TIMEMORY_BASIC_MARKER(auto_hybrid_t, "(", n, ")");
    return impl::fibonacci(n);
}

void write(long nfib, long answer)
{
#pragma omp critical
    printf("fibonacci(%li) = %li\n", nfib, answer);
}

void status()
{
#if defined(USE_TIMEMORY)
    printf("\n#----------------- TIMEMORY is enabled  ----------------#\n\n");
#else
    printf("\n#----------------- TIMEMORY is disabled ----------------#\n\n");
#endif
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

//--------------------------------------------------------------------------------------//

void allreduce(const vector_t<long>& sendbuf, vector_t<long>& recvbuf)
{
    recvbuf.resize(sendbuf.size(), 0L);
#if defined(TIMEMORY_USE_MPI)
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Allreduce((long*) sendbuf.data(), recvbuf.data(), sendbuf.size(), MPI_LONG,
                  MPI_SUM, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
#else
    std::copy(sendbuf.begin(), sendbuf.end(), recvbuf.begin());
#endif
}
