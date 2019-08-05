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

#include "gtest/gtest.h"

#include <atomic>
#include <chrono>
#include <mutex>
#include <random>
#include <thread>

//--------------------------------------------------------------------------------------//
//      Declarations
//--------------------------------------------------------------------------------------//

extern "C" void
timemory_init_library(int argc, char** argv);
extern "C" void
timemory_finalize_library();
extern "C" void
timemory_begin_record(const char* name, uint64_t* kernid);
extern "C" void
timemory_end_record(uint64_t kernid);

//--------------------------------------------------------------------------------------//

using mutex_t   = std::mutex;
using lock_t    = std::unique_lock<mutex_t>;
using condvar_t = std::condition_variable;

static const int64_t niter     = 20;
static const int64_t nelements = 500000;

//--------------------------------------------------------------------------------------//
namespace details
{
// this function consumes approximately "n" milliseconds of real time
void
do_sleep(long n)
{
    std::this_thread::sleep_for(std::chrono::milliseconds(n));
}

// this function consumes an unknown number of cpu resources
long
fibonacci(long n)
{
    return (n < 2) ? n : (fibonacci(n - 1) + fibonacci(n - 2));
}

// this function consumes approximately "t" milliseconds of cpu time
void
consume(long n)
{
    // a mutex held by one lock
    mutex_t mutex;
    // acquire lock
    lock_t hold_lk(mutex);
    // associate but defer
    lock_t try_lk(mutex, std::defer_lock);
    // get current time
    auto now = std::chrono::system_clock::now();
    // get elapsed
    auto until = now + std::chrono::milliseconds(n);
    // try until time point
    while(std::chrono::system_clock::now() < until)
        try_lk.try_lock();
}

// this function ensures an allocation cannot be optimized
template <typename _Tp>
size_t
random_entry(const std::vector<_Tp>& v)
{
    std::mt19937 rng;
    rng.seed(std::random_device()());
    std::uniform_int_distribution<std::mt19937::result_type> dist(0, v.size() - 1);
    return v.at(dist(rng));
}
void
allocate()
{
    std::vector<int64_t> v(nelements, 15);
    auto                 ret  = fibonacci(0);
    long                 nfib = details::random_entry(v);
    for(int64_t i = 0; i < niter; ++i)
    {
        nfib = details::random_entry(v);
        ret += details::fibonacci(nfib);
    }
    printf("fibonacci(%li) * %li = %li\n", (long) nfib, (long) niter, ret);
}
}  // namespace details

//--------------------------------------------------------------------------------------//

class preload_tests : public ::testing::Test
{
};

//--------------------------------------------------------------------------------------//

TEST_F(preload_tests, timers)
{
    uint64_t id;
    timemory_begin_record("preload_tests.timers", &id);
    details::do_sleep(250);
    details::consume(750);
    timemory_end_record(id);
}

//--------------------------------------------------------------------------------------//

TEST_F(preload_tests, rusage)
{
    uint64_t id;
    timemory_begin_record("preload_tests.rusage", &id);
    details::do_sleep(250);
    details::allocate();
    timemory_end_record(id);
}

//--------------------------------------------------------------------------------------//

int
main(int argc, char** argv)
{
    timemory_init_library(argc, argv);
    // uint64_t id;
    // timemory_begin_record("preload_tests", &id);
    ::testing::InitGoogleTest(&argc, argv);
    auto ret = RUN_ALL_TESTS();
    // timemory_finalize_library();
    return ret;
}

//--------------------------------------------------------------------------------------//
