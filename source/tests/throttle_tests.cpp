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
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

#include "timemory/library.h"
#include "timemory/timemory.hpp"

static int    _argc = 0;
static char** _argv = nullptr;

using mutex_t = std::mutex;
using lock_t  = std::unique_lock<mutex_t>;

//--------------------------------------------------------------------------------------//

namespace details
{
//--------------------------------------------------------------------------------------//
//  Get the current tests name
//
inline std::string
get_test_name()
{
    return std::string(::testing::UnitTest::GetInstance()->current_test_suite()->name()) +
           "." + ::testing::UnitTest::GetInstance()->current_test_info()->name();
}

// this function consumes approximately "n" milliseconds of real time
inline void
do_sleep(long n)
{
    std::this_thread::sleep_for(std::chrono::nanoseconds(n));
}

inline void
consume(long n)
{
    auto now = std::chrono::steady_clock::now();
    // try until time point
    while(std::chrono::steady_clock::now() < (now + std::chrono::nanoseconds(n)))
    {
    }
}
}  // namespace details

//--------------------------------------------------------------------------------------//

class throttle_tests : public ::testing::Test
{
protected:
    void SetUp() override
    {
        static bool configured = false;
        if(!configured)
        {
            tim::set_env("TIMEMORY_VERBOSE", "1", 1);
            tim::set_env("TIMEMORY_COLLAPSE_THREADS", "OFF", 0);
            configured                   = true;
            tim::settings::debug()       = false;
            tim::settings::json_output() = true;
            tim::settings::mpi_thread()  = false;
            tim::dmp::initialize(_argc, _argv);
            tim::timemory_init(_argc, _argv);
            tim::settings::dart_output() = true;
            tim::settings::dart_count()  = 1;
            tim::settings::banner()      = false;
            timemory_trace_init("wall_clock", false, "throttle_tests");
            tim::settings::verbose() = 1;
        }
    }

    static constexpr uint64_t nthreads = 4;
};

//--------------------------------------------------------------------------------------//

TEST_F(throttle_tests, expect_true)
{
    auto name = details::get_test_name();
    auto n    = 2 * tim::settings::throttle_count();

    for(size_t i = 0; i < n; ++i)
    {
        timemory_push_trace(name.c_str());
        timemory_pop_trace(name.c_str());
    }

    EXPECT_TRUE(timemory_is_throttled(name.c_str()));
}

//--------------------------------------------------------------------------------------//

TEST_F(throttle_tests, expect_false)
{
    auto name = details::get_test_name();
    auto n    = 2 * tim::settings::throttle_count();
    auto v    = 2 * tim::settings::throttle_value();

    for(size_t i = 0; i < n; ++i)
    {
        timemory_push_trace(name.c_str());
        // details::do_sleep(v);
        details::consume(v);
        timemory_pop_trace(name.c_str());
    }

    EXPECT_FALSE(timemory_is_throttled(name.c_str()));
}

//--------------------------------------------------------------------------------------//

TEST_F(throttle_tests, multithreaded)
{
    using tuple_t = tim::auto_tuple<tim::component::wall_clock>;
    std::array<bool, nthreads> is_throttled;
    is_throttled.fill(false);

    auto _run = [&is_throttled](uint64_t idx) {
        timemory_push_trace("thread");
        auto name = details::get_test_name();
        auto n    = 2 * tim::settings::throttle_count();
        auto v    = 2 * tim::settings::throttle_value();
        if(idx % 2 == 1)
        {
            for(size_t i = 0; i < n; ++i)
            {
                timemory_push_trace(name.c_str());
                // details::do_sleep(v);
                details::consume(v);
                timemory_pop_trace(name.c_str());
            }
        }
        else
        {
            for(size_t i = 0; i < n; ++i)
            {
                timemory_push_trace(name.c_str());
                timemory_pop_trace(name.c_str());
            }
        }
        timemory_pop_trace("thread");
        is_throttled.at(idx) = timemory_is_throttled(name.c_str());
    };

    std::vector<std::thread> threads;
    for(uint64_t i = 0; i < nthreads; ++i)
        threads.push_back(std::thread(_run, i));
    for(auto& itr : threads)
        itr.join();

    for(uint64_t i = 0; i < nthreads; ++i)
    {
        bool _answer = (i % 2 == 1) ? false : true;
        std::cout << "thread " << i << " throttling: " << std::boolalpha
                  << is_throttled[i] << std::endl;
        EXPECT_TRUE(is_throttled[i] == _answer);
    }
}

//--------------------------------------------------------------------------------------//

TEST_F(throttle_tests, do_nothing)
{
    auto n = tim::settings::throttle_count();
    for(size_t i = 0; i < n; ++i)
        details::do_sleep(10);
}

//--------------------------------------------------------------------------------------//

TEST_F(throttle_tests, region_serial)
{
    auto _run = []() {
        timemory_push_region("thread");
        auto name = details::get_test_name();
        auto n    = 8 * tim::settings::throttle_count();
        for(size_t i = 0; i < n; ++i)
        {
            timemory_push_region(name.c_str());
            timemory_pop_region(name.c_str());
        }
        timemory_pop_region("thread");
    };

    for(uint64_t i = 0; i < nthreads; ++i)
        _run();
}

//--------------------------------------------------------------------------------------//

TEST_F(throttle_tests, region_multithreaded)
{
    auto _run = []() {
        timemory_push_region("thread");
        auto name = details::get_test_name();
        auto n    = 8 * tim::settings::throttle_count();
        for(size_t i = 0; i < n; ++i)
        {
            timemory_push_region(name.c_str());
            timemory_pop_region(name.c_str());
        }
        timemory_pop_region("thread");
    };

    std::vector<std::thread> threads;
    for(uint64_t i = 0; i < nthreads; ++i)
        threads.push_back(std::thread(_run));
    for(auto& itr : threads)
        itr.join();
}

//--------------------------------------------------------------------------------------//

TEST_F(throttle_tests, tuple_serial)
{
    using tuple_t = tim::auto_tuple<tim::component::wall_clock>;
    auto _run     = []() {
        TIMEMORY_BLANK_MARKER(tuple_t, "thread");
        auto name = details::get_test_name();
        auto n    = 8 * tim::settings::throttle_count();
        for(size_t i = 0; i < n; ++i)
        {
            TIMEMORY_BLANK_MARKER(tuple_t, name);
        }
    };

    for(uint64_t i = 0; i < nthreads; ++i)
        _run();
}

//--------------------------------------------------------------------------------------//

TEST_F(throttle_tests, tuple_multithreaded)
{
    using tuple_t = tim::auto_tuple<tim::component::wall_clock>;
    auto _run     = []() {
        TIMEMORY_BLANK_MARKER(tuple_t, "thread");
        auto name = details::get_test_name();
        auto n    = 8 * tim::settings::throttle_count();
        for(size_t i = 0; i < n; ++i)
        {
            TIMEMORY_BLANK_MARKER(tuple_t, name);
        }
    };

    std::vector<std::thread> threads;
    for(uint64_t i = 0; i < nthreads; ++i)
        threads.push_back(std::thread(_run));
    for(auto& itr : threads)
        itr.join();
}

//--------------------------------------------------------------------------------------//

int
main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    _argc = argc;
    _argv = argv;

    auto ret = RUN_ALL_TESTS();

    tim::timemory_finalize();
    tim::dmp::finalize();
    return ret;
}

//--------------------------------------------------------------------------------------//
