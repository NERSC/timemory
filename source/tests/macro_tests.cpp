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

#include <timemory/timemory.hpp>

#include <chrono>
#include <iostream>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

using namespace tim::component;
using mutex_t   = std::mutex;
using lock_t    = std::unique_lock<mutex_t>;
using condvar_t = std::condition_variable;

using auto_tuple_t      = tim::auto_tuple<real_clock, cpu_clock, cpu_util>;
using component_tuple_t = typename auto_tuple_t::component_type;

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

}  // namespace details

//--------------------------------------------------------------------------------------//

class macro_tests : public ::testing::Test
{
};

//--------------------------------------------------------------------------------------//

TEST_F(macro_tests, blank_object)
{
    TIMEMORY_BLANK_MARKER(auto_tuple_t, details::get_test_name());
    details::do_sleep(25);
    details::consume(75);
    timemory_variable_101.stop();
    auto              key = timemory_variable_101.key();
    std::stringstream expected;
    expected << details::get_test_name();
    if(key != expected.str())
    {
        std::cout << std::endl;
        std::cout << std::setw(12) << "key"
                  << ": " << key << std::endl;
        std::cout << std::setw(12) << "expected"
                  << ": " << expected.str() << std::endl;
        std::cout << std::endl;
        FAIL();
    }
    else
        SUCCEED();
}

//--------------------------------------------------------------------------------------//

TEST_F(macro_tests, basic_object)
{
    TIMEMORY_BASIC_MARKER(auto_tuple_t, "_", details::get_test_name());
    details::do_sleep(25);
    details::consume(75);
    timemory_variable_126.stop();
    auto              key = timemory_variable_126.key();
    std::stringstream expected;
    expected << __FUNCTION__ << "_" << details::get_test_name();
    if(key != expected.str())
    {
        std::cout << std::endl;
        std::cout << std::setw(12) << "key"
                  << ": " << key << std::endl;
        std::cout << std::setw(12) << "expected"
                  << ": " << expected.str() << std::endl;
        std::cout << std::endl;
        FAIL();
    }
    else
        SUCCEED();
}

//--------------------------------------------------------------------------------------//

int
main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    tim::timemory_init(argc, argv);
    return RUN_ALL_TESTS();
}

//--------------------------------------------------------------------------------------//
