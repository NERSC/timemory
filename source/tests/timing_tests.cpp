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

#include "timemory/timemory.hpp"

#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>

using namespace tim::component;
using mutex_t = std::mutex;
using lock_t  = std::unique_lock<mutex_t>;

static const double util_tolerance  = 2.5;
static const double timer_tolerance = 0.015;

#define CHECK_AVAILABLE(type)                                                            \
    if(!tim::trait::is_available<type>::value)                                           \
        return;

//--------------------------------------------------------------------------------------//
namespace details
{
inline std::string
get_test_name()
{
    return ::testing::UnitTest::GetInstance()->current_test_info()->name();
}

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
    auto now = std::chrono::steady_clock::now();
    // try until time point
    while(std::chrono::steady_clock::now() < (now + std::chrono::milliseconds(n)))
        try_lk.try_lock();
}
}  // namespace details

//--------------------------------------------------------------------------------------//

class timing_tests : public ::testing::Test
{
protected:
    void SetUp() override
    {
        tim::settings::timing_units() = "sec";
        tim::settings::precision()    = 9;
    }
};

//--------------------------------------------------------------------------------------//

TEST_F(timing_tests, wall_timer)
{
    CHECK_AVAILABLE(wall_clock);
    wall_clock obj;
    obj.start();
    details::do_sleep(1000);
    obj.stop();
    std::cout << "\n[" << details::get_test_name() << "]> result: " << obj << "\n"
              << std::endl;
    ASSERT_NEAR(1.0, obj.get(), timer_tolerance);
}

//--------------------------------------------------------------------------------------//

TEST_F(timing_tests, monotonic_timer)
{
    CHECK_AVAILABLE(monotonic_clock);
    monotonic_clock obj;
    obj.start();
    details::do_sleep(1000);
    obj.stop();
    std::cout << "\n[" << details::get_test_name() << "]> result: " << obj << "\n"
              << std::endl;
    ASSERT_NEAR(1.0, obj.get(), timer_tolerance);
}

//--------------------------------------------------------------------------------------//

TEST_F(timing_tests, monotonic_raw_timer)
{
    CHECK_AVAILABLE(monotonic_raw_clock);
    monotonic_raw_clock obj;
    obj.start();
    details::do_sleep(1000);
    obj.stop();
    std::cout << "\n[" << details::get_test_name() << "]> result: " << obj << "\n"
              << std::endl;
    ASSERT_NEAR(1.0, obj.get(), timer_tolerance);
}

//--------------------------------------------------------------------------------------//

TEST_F(timing_tests, system_timer)
{
    CHECK_AVAILABLE(system_clock);
    system_clock obj;
    obj.start();
    std::thread t(details::do_sleep, 1000);
    t.join();
    obj.stop();
    std::cout << "\n[" << details::get_test_name() << "]> result: " << obj << "\n"
              << std::endl;
    ASSERT_NEAR(0.0, obj.get(), timer_tolerance);
}

//--------------------------------------------------------------------------------------//

TEST_F(timing_tests, user_timer)
{
    CHECK_AVAILABLE(user_clock);
    user_clock obj;
    obj.start();
    std::thread t(details::do_sleep, 1000);
    t.join();
    obj.stop();
    std::cout << "\n[" << details::get_test_name() << "]> result: " << obj << "\n"
              << std::endl;
    ASSERT_NEAR(0.0, obj.get(), timer_tolerance);
}

//--------------------------------------------------------------------------------------//

TEST_F(timing_tests, cpu_timer)
{
    CHECK_AVAILABLE(cpu_clock);
    cpu_clock obj;
    obj.start();
    details::consume(1000);
    obj.stop();
    std::cout << "\n[" << details::get_test_name() << "]> result: " << obj << "\n"
              << std::endl;
    ASSERT_NEAR(1.0, obj.get(), timer_tolerance);
}

//--------------------------------------------------------------------------------------//

TEST_F(timing_tests, cpu_utilization)
{
    CHECK_AVAILABLE(cpu_util);
    cpu_util obj;
    obj.start();
    details::consume(750);
    details::do_sleep(250);
    obj.stop();
    std::cout << "\n[" << details::get_test_name() << "]> result: " << obj << "\n"
              << std::endl;
    ASSERT_NEAR(75.0, obj.get(), util_tolerance);
}

//--------------------------------------------------------------------------------------//

TEST_F(timing_tests, thread_cpu_timer)
{
    CHECK_AVAILABLE(thread_cpu_clock);
    thread_cpu_clock obj;
    obj.start();
    std::thread t(details::fibonacci, 43);
    t.join();
    obj.stop();
    std::cout << "\n[" << details::get_test_name() << "]> result: " << obj << "\n"
              << std::endl;
    ASSERT_NEAR(0.0, obj.get(), timer_tolerance);
}

//--------------------------------------------------------------------------------------//

TEST_F(timing_tests, thread_cpu_utilization)
{
    CHECK_AVAILABLE(thread_cpu_util);
    thread_cpu_util obj;
    obj.start();
    std::thread t(details::consume, 2000);
    details::consume(1000);
    details::do_sleep(1000);
    t.join();
    obj.stop();
    std::cout << "\n[" << details::get_test_name() << "]> result: " << obj << "\n"
              << std::endl;
    ASSERT_NEAR(50.0, obj.get(), util_tolerance);
}

//--------------------------------------------------------------------------------------//

TEST_F(timing_tests, process_cpu_timer)
{
    CHECK_AVAILABLE(process_cpu_clock);
    process_cpu_clock obj;
    obj.start();
    details::consume(1000);
    obj.stop();
    std::cout << "\n[" << details::get_test_name() << "]> result: " << obj << "\n"
              << std::endl;
    // this test seems to fail sporadically
    ASSERT_NEAR(1.0, obj.get(), 2.5 * timer_tolerance);
}

//--------------------------------------------------------------------------------------//

TEST_F(timing_tests, process_cpu_utilization)
{
    CHECK_AVAILABLE(process_cpu_util);
    process_cpu_util obj;
    obj.start();
    std::thread t(details::consume, 2000);
    details::consume(1000);
    details::do_sleep(1000);
    t.join();
    obj.stop();
    std::cout << "\n[" << details::get_test_name() << "]> result: " << obj << "\n"
              << std::endl;
    ASSERT_NEAR(150.0, obj.get(), util_tolerance);
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
    tim::settings::dart_output() = true;
    tim::settings::dart_count()  = 1;
    tim::settings::banner()      = false;

    tim::settings::dart_type() = "peak_rss";
    // TIMEMORY_VARIADIC_BLANK_AUTO_TUPLE("PEAK_RSS", ::tim::component::peak_rss);
    auto ret = RUN_ALL_TESTS();

    tim::timemory_finalize();
    tim::dmp::finalize();
    return ret;
}

//--------------------------------------------------------------------------------------//
