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

#include "gtest/gtest.h"

#include <chrono>
#include <iostream>
#include <random>
#include <thread>
#include <vector>

#include "timemory/timemory.hpp"

#if defined(_OPENMP)
#    include "timemory/components/ompt.hpp"
#    include <omp.h>
#endif

using namespace tim::component;
using tuple_t = tim::auto_tuple_t<wall_clock, cpu_clock, cpu_util, thread_cpu_clock>;

#if defined(_OPENMP)
static const double pi_epsilon = std::numeric_limits<float>::epsilon();
#endif

static const long nfib = 39;
static const long noff = 4;

//--------------------------------------------------------------------------------------//

namespace details
{
//--------------------------------------------------------------------------------------//
//  Get the current tests name
//
inline std::string
get_test_name()
{
    return ::testing::UnitTest::GetInstance()->current_test_info()->name();
}

// this function consumes approximately "n" milliseconds of real time
inline void
do_sleep(long n)
{
    std::this_thread::sleep_for(std::chrono::milliseconds(n));
}

// this function consumes an unknown number of cpu resources
inline long
fibonacci(long n)
{
    return (n < 2) ? n : (fibonacci(n - 1) + fibonacci(n - 2));
}

#if defined(_OPENMP)
double
compute_pi(uint64_t nstart, uint64_t nstop, double step, uint64_t nblock)
{
    double sum  = 0.0;
    double sum1 = 0.0;
    double sum2 = 0.0;

    if(nstop - nstart < nblock)
    {
        for(uint64_t i = nstart; i < nstop; ++i)
        {
            double x = (i + 0.5) * step;
            sum      = sum + 4.0 / (1.0 + x * x);
        }
    }
    else
    {
        uint64_t iblk = nstop - nstart;
#    pragma omp task shared(sum1)
        sum1 = compute_pi(nstart, nstop - iblk / 2, step, nblock);
#    pragma omp task shared(sum2)
        sum2 = compute_pi(nstop - iblk / 2, nstop, step, nblock);
#    pragma omp taskwait
        sum = sum1 + sum2;
    }
    return sum;
}
#endif

}  // namespace details

//--------------------------------------------------------------------------------------//

class threading_tests : public ::testing::Test
{};

static std::atomic<int> thread_count;
int&
get_tid()
{
    static thread_local int _id = thread_count++;
    return _id;
}

//--------------------------------------------------------------------------------------//
#if defined(_OPENMP)

TEST_F(threading_tests, openmp)
{
    tim::trait::runtime_enabled<ompt_native_handle>::set(false);
    user_ompt_bundle::configure<wall_clock, cpu_clock, cpu_util, thread_cpu_clock>();

    TIMEMORY_BLANK_MARKER(tuple_t, details::get_test_name());

    omp_set_num_threads(2);

    std::atomic<uint64_t> sum(0);
    {
        std::string region = "AAAAA";
        TIMEMORY_BLANK_MARKER(tuple_t, details::get_test_name(), "/master/0");
#    pragma omp parallel for
        for(int i = 0; i < 4; i++)
        {
            TIMEMORY_BLANK_MARKER(tuple_t, details::get_test_name(), "/worker/", region,
                                  "/", omp_get_thread_num());
            sum += details::fibonacci(nfib);
            details::do_sleep(500);
        }
    }

    {
        std::string region = "BBBBB";
        TIMEMORY_BLANK_MARKER(tuple_t, details::get_test_name(), "/master/1");
#    pragma omp parallel for
        for(int i = 0; i < 4; i++)
        {
            TIMEMORY_BLANK_MARKER(tuple_t, details::get_test_name(), "/worker/", region,
                                  "/", omp_get_thread_num());
            sum += details::fibonacci(nfib + noff);
            details::do_sleep(500);
        }
    }

    printf("[%s]> sum: %lu\n", details::get_test_name().c_str(),
           static_cast<unsigned long>(sum));

    auto wc = tim::storage<wall_clock>::instance()->get();
    auto tc = tim::storage<thread_cpu_clock>::instance()->get();

    int64_t wc_nlaps = 0;
    int64_t tc_nlaps = 0;
    for(auto& itr : wc)
        wc_nlaps += itr.data().get_laps();
    for(auto& itr : tc)
        tc_nlaps += itr.data().get_laps();

    ASSERT_EQ(wc_nlaps, tc_nlaps);
}

//--------------------------------------------------------------------------------------//

TEST_F(threading_tests, openmp_ompt)
{
    tim::trait::runtime_enabled<ompt_native_handle>::set(true);
    user_ompt_bundle::configure<wall_clock, cpu_clock, cpu_util, thread_cpu_clock>();

    omp_set_num_threads(2);

    std::atomic<uint64_t> sum(0);
    {
#    pragma omp parallel for
        for(int i = 0; i < 4; i++)
        {
            sum += details::fibonacci(nfib);
        }
    }

    {
#    pragma omp parallel for
        for(int i = 0; i < 4; i++)
        {
            sum += details::fibonacci(nfib + noff);
        }
    }

    printf("[%s]> sum: %lu\n", details::get_test_name().c_str(),
           static_cast<unsigned long>(sum));

    auto wc = tim::storage<wall_clock>::instance()->get();
    auto tc = tim::storage<thread_cpu_clock>::instance()->get();

    int64_t wc_nlaps = 0;
    int64_t tc_nlaps = 0;
    for(auto& itr : wc)
        wc_nlaps += itr.data().get_laps();
    for(auto& itr : tc)
        tc_nlaps += itr.data().get_laps();

    ASSERT_EQ(wc_nlaps, tc_nlaps);
}

//--------------------------------------------------------------------------------------//

TEST_F(threading_tests, openmp_task)
{
    tim::trait::runtime_enabled<ompt_native_handle>::set(true);
    user_ompt_bundle::configure<wall_clock, cpu_clock, cpu_util, thread_cpu_clock>();

    TIMEMORY_BLANK_MARKER(tuple_t, details::get_test_name());

    uint64_t num_threads = tim::get_env<uint64_t>("NUM_THREADS", 4);
    uint64_t num_steps   = tim::get_env<uint64_t>("NUM_STEPS", 500000000);

    omp_set_num_threads(num_threads);

    double sum    = 0.0;
    double step   = 1.0 / static_cast<double>(num_steps);
    auto   nblock = num_steps / num_threads;

#    pragma omp parallel
    {
#    pragma omp single
        {
            sum = details::compute_pi(0, num_steps, step, nblock);
        }
    }

    double pi = step * sum;
    printf("[%s]> pi: %f\n", details::get_test_name().c_str(), pi);

    ASSERT_NEAR(pi, M_PI, pi_epsilon);
}

#endif
//--------------------------------------------------------------------------------------//

TEST_F(threading_tests, stl)
{
    TIMEMORY_BLANK_MARKER(tuple_t, details::get_test_name());

    auto run_fib = [&](uint64_t n) { return details::fibonacci(n); };

    std::atomic<uint64_t> sum(0);

    auto run_a = [&]() {
        for(int i = 0; i < 2; ++i)
        {
            TIMEMORY_BLANK_MARKER(tuple_t, details::get_test_name(), "/worker/AAAAA/",
                                  get_tid());
            sum += run_fib(nfib);
            details::do_sleep(500);
        }
        if(get_tid() > 0)
            thread_count--;
    };

    auto run_b = [&]() {
        for(int i = 0; i < 2; ++i)
        {
            TIMEMORY_BLANK_MARKER(tuple_t, details::get_test_name(), "/worker/BBBBB/",
                                  get_tid());
            sum += run_fib(45);
            details::do_sleep(nfib + noff);
        }
        if(get_tid() > 0)
            thread_count--;
    };

    {
        TIMEMORY_BLANK_MARKER(tuple_t, details::get_test_name(), "/master/", get_tid());
        std::thread t(run_a);
        run_a();
        t.join();
    }

    {
        TIMEMORY_BLANK_MARKER(tuple_t, details::get_test_name(), "/master/1");
        std::thread t(run_b);
        run_b();
        t.join();
    };

    printf("[%s]> sum: %lu\n", details::get_test_name().c_str(),
           static_cast<unsigned long>(sum));

    auto wc = tim::storage<wall_clock>::instance()->get();
    auto tc = tim::storage<thread_cpu_clock>::instance()->get();

    int64_t wc_nlaps = 0;
    int64_t tc_nlaps = 0;
    for(auto& itr : wc)
        wc_nlaps += itr.data().get_laps();
    for(auto& itr : tc)
        tc_nlaps += itr.data().get_laps();

    ASSERT_EQ(wc_nlaps, tc_nlaps);
}

//--------------------------------------------------------------------------------------//

int
main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    tim::settings::verbose()     = 0;
    tim::settings::debug()       = false;
    tim::settings::json_output() = true;
    tim::timemory_init(&argc, &argv);
    tim::settings::dart_output() = false;
    tim::settings::dart_count()  = 1;
    tim::settings::banner()      = false;

    auto ret = RUN_ALL_TESTS();

    tim::timemory_finalize();

    return ret;
}

//--------------------------------------------------------------------------------------//
