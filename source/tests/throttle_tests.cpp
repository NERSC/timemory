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

#include "test_macros.hpp"

TIMEMORY_TEST_DEFAULT_MAIN

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
    using wc_t = tim::component::wall_clock;

protected:
    static void SetUpTestSuite()
    {
        puts("[SetupTestSuite] setup starting");
        tim::set_env("TIMEMORY_VERBOSE", "1", 1);
        tim::set_env("TIMEMORY_COLLAPSE_THREADS", "OFF", 0);
        tim::settings::debug()       = false;
        tim::settings::json_output() = true;
        tim::settings::mpi_thread()  = false;
        puts("[SetupTestSuite] initializing dmp");
        tim::dmp::initialize(_argc, _argv);
        puts("[SetupTestSuite] initializing timemory");
        tim::timemory_init(_argc, _argv);
        tim::settings::dart_output() = true;
        tim::settings::dart_count()  = 1;
        tim::settings::banner()      = false;
        timemory_trace_init("wall_clock", false, "throttle_tests");
        puts("[SetupTestSuite] timemory initialized");
        tim::settings::verbose() = 1;
        tim::settings::debug()   = false;
        puts("[SetupTestSuite] setup completed");
        metric().start();
    }

    // always disable debug
    void SetUp() override
    {
        tim::settings::debug() = false;
        wc_init                = tim::storage<wc_t>::instance()->get();
        if(!wc_init.empty())
            offset = 0;
    }

    static void TearDownTestSuite()
    {
        metric().stop();
        print_dart(metric());
        timemory_trace_finalize();
        tim::timemory_finalize();
        if(tim::dmp::rank() == 0)
            tim::enable_signal_detection(tim::signal_settings::get_default());
        tim::dmp::finalize();
    }

    auto get_size_delta()
    {
        auto wc_data = tim::storage<wc_t>::instance()->get();
        return (wc_data.size() - wc_init.size());
    }

    auto get_count_delta(const std::string& _id)
    {
        auto _get_count = [&_id](auto&& _data) {
            int64_t _n = 0;
            for(auto&& itr : _data)
            {
                auto _lbl = tim::operation::decode<TIMEMORY_API>{}(
                    ::tim::get_hash_identifier(itr.hash()));
                if(_lbl == _id)
                {
                    _n += itr.data().get_laps();
                }
            }
            return _n;
        };

        return (_get_count(tim::storage<wc_t>::instance()->get()) - _get_count(wc_init));
    }

    std::string write_data()
    {
        if(tim::dmp::rank() > 0)
            return std::string{};

        std::ostringstream _oss;
        auto               _write_data = [&_oss](const auto& _data) {
            for(size_t i = 0; i < _data.size(); ++i)
                _oss << "    " << i << "/" << _data.size()
                     << " :: " << as_string(_data.at(i)) << "\n";
        };

        _oss << "Initial data:\n";
        _write_data(wc_init);
        _oss << "Current data:\n";
        _write_data(tim::storage<wc_t>::instance()->get());

        return _oss.str();
    }

    static constexpr uint64_t            nthreads = 4;
    size_t                               offset   = 1;
    std::vector<tim::node::result<wc_t>> wc_init{};
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

    std::cout << std::boolalpha << "is_throttled(" << name
              << ") == " << timemory_is_throttled(name.c_str()) << std::endl;
    EXPECT_EQ(get_count_delta(name), tim::settings::throttle_count()) << write_data();
#if !defined(TIMEMORY_RELAXED_TESTING)
    EXPECT_TRUE(timemory_is_throttled(name.c_str())) << write_data();
#endif
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

    std::cout << std::boolalpha << "is_throttled(" << name
              << ") == " << timemory_is_throttled(name.c_str()) << std::endl;
    EXPECT_EQ(get_count_delta(name), 2 * tim::settings::throttle_count()) << write_data();
#if !defined(TIMEMORY_RELAXED_TESTING)
    EXPECT_FALSE(timemory_is_throttled(name.c_str())) << write_data();
#endif
}

//--------------------------------------------------------------------------------------//

TEST_F(throttle_tests, multithreaded)
{
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
        threads.emplace_back(_run, i);
    for(auto& itr : threads)
        itr.join();

    for(uint64_t i = 0; i < nthreads; ++i)
    {
        bool _answer = (i % 2 == 1) ? false : true;
        std::cout << "thread " << i << " throttling: " << std::boolalpha
                  << is_throttled[i] << ". expected: " << _answer << "\n";
        EXPECT_EQ(get_size_delta(), 2 * nthreads) << write_data();
#if !defined(TIMEMORY_RELAXED_TESTING)
        EXPECT_TRUE(is_throttled[i] == _answer) << write_data();
#endif
    }
}

//--------------------------------------------------------------------------------------//

TEST_F(throttle_tests, do_nothing)
{
    auto n = tim::settings::throttle_count();
    for(size_t i = 0; i < n; ++i)
        details::do_sleep(10);
    EXPECT_EQ(get_size_delta(), 0) << write_data();
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

    EXPECT_EQ(get_size_delta(), 1 + offset) << write_data();
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
        threads.emplace_back(_run);
    for(auto& itr : threads)
        itr.join();
    EXPECT_EQ(get_size_delta(), 2 * nthreads) << write_data();
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
    EXPECT_EQ(get_size_delta(), 1 + offset) << write_data();
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
        threads.emplace_back(_run);
    for(auto& itr : threads)
        itr.join();
    EXPECT_EQ(get_size_delta(), 2 * nthreads) << write_data();
}

//--------------------------------------------------------------------------------------//
