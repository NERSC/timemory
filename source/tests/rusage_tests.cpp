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

#include <chrono>
#include <iostream>
#include <random>
#include <thread>
#include <timemory/timemory.hpp>
#include <vector>

using namespace tim::component;
using mutex_t   = std::mutex;
using lock_t    = std::unique_lock<mutex_t>;
using condvar_t = std::condition_variable;

static const float   peak_tolerance = 5;
static const float   curr_tolerance = 5;
static const int64_t nelements      = 5000000;

#define CHECK_AVAILABLE(type) \
    if(!tim::trait::impl_available< type >::value)\
        return;

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
}  // namespace details

//--------------------------------------------------------------------------------------//

class rusage_tests : public ::testing::Test
{
protected:
    void SetUp() override
    {
        tim::settings::precision()    = 9;
        tim::settings::memory_units() = "MB";
        tim::settings::process();
    }
};

//--------------------------------------------------------------------------------------//

TEST_F(rusage_tests, peak_rss)
{
    CHECK_AVAILABLE(peak_rss);
    peak_rss peak;
    peak.start();
    std::vector<int64_t> v(nelements, 42);
    long                 nfib = details::random_entry(v);
    auto                 ret  = details::fibonacci(nfib);
    peak.stop();

    auto tot_size = nelements * sizeof(int64_t) / tim::units::megabyte;

    printf("fibonacci(%li) = %li\n", nfib, ret);
    std::cout << "[" << __FUNCTION__ << "]> peak:     " << peak << std::endl;
    std::cout << "[" << __FUNCTION__ << "]> expected: " << tot_size << " MB" << std::endl;

    ASSERT_NEAR(tot_size, peak.get(), peak_tolerance);
}

//--------------------------------------------------------------------------------------//

TEST_F(rusage_tests, current_rss)
{
    CHECK_AVAILABLE(current_rss);
    current_rss curr;
    curr.start();
    std::vector<int64_t> v(nelements, 42);
    long                 nfib = details::random_entry(v);
    auto                 ret  = details::fibonacci(nfib);
    curr.stop();

    auto tot_size = nelements * sizeof(int64_t) / tim::units::megabyte;

    printf("fibonacci(%li) = %li\n", nfib, ret);
    std::cout << "[" << __FUNCTION__ << "]> current:  " << curr << std::endl;
    std::cout << "[" << __FUNCTION__ << "]> expected: " << tot_size << " MB" << std::endl;

    ASSERT_NEAR(tot_size, curr.get(), curr_tolerance);
}

//--------------------------------------------------------------------------------------//

int
main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

//--------------------------------------------------------------------------------------//
