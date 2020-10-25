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
// clang-format off
#include "gtest/gtest.h"

#include "timemory/timemory.hpp"

#include <chrono>
#include <iostream>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

using namespace tim::component;
using mutex_t = std::mutex;
using lock_t  = std::unique_lock<mutex_t>;

using auto_tuple_t = tim::auto_tuple_t<wall_clock, cpu_clock, cpu_util>;
using comp_tuple_t = typename auto_tuple_t::component_type;

//--------------------------------------------------------------------------------------//
//
namespace details
{
//--------------------------------------------------------------------------------------//
//  Get the current tests name
//
inline std::string
get_test_name()
{
    return std::string(::testing::UnitTest::GetInstance()->current_test_suite()->name()) + "." + ::testing::UnitTest::GetInstance()->current_test_info()->name();
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
    // wait time
    auto wait = std::chrono::milliseconds(n);
    // get current time
    auto now = std::chrono::steady_clock::now();
    // try until time point
    while(std::chrono::steady_clock::now() < (now + wait))
    {
    }
}

}  // namespace details

//--------------------------------------------------------------------------------------//

class macro_tests : public ::testing::Test
{};

//--------------------------------------------------------------------------------------//

TEST_F(macro_tests, blank_marker)
{
    TIMEMORY_BLANK_MARKER(auto_tuple_t, details::get_test_name());
    details::do_sleep(25);
    details::consume(75);
    timemory_variable_94.stop();
    auto              key = timemory_variable_94.key();
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
    {
        SUCCEED();
    }
}

//--------------------------------------------------------------------------------------//

TEST_F(macro_tests, basic_marker)
{
    TIMEMORY_BASIC_MARKER(auto_tuple_t, details::get_test_name());
    details::do_sleep(25);
    details::consume(75);
    timemory_variable_121.stop();
    auto              key = timemory_variable_121.key();
    std::stringstream expected;
    expected << __FUNCTION__ << "/" << details::get_test_name();
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
    {
        SUCCEED();
    }
}

//--------------------------------------------------------------------------------------//

TEST_F(macro_tests, marker)
{
    TIMEMORY_MARKER(auto_tuple_t, details::get_test_name());
    auto line = __LINE__ - 1;
    details::do_sleep(25);
    details::consume(75);
    timemory_variable_148.stop();
    auto              key = timemory_variable_148.key();
    std::stringstream expected;
    std::string       file = __FILE__;
    file = std::string(file).substr(std::string(file).find_last_of('/') + 1);
    expected << __FUNCTION__ << "@" << file << ":" << line << "/"
             << details::get_test_name();
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
    {
        SUCCEED();
    }
}

//--------------------------------------------------------------------------------------//

TEST_F(macro_tests, marker_auto_type)
{
    TIMEMORY_BLANK_MARKER(comp_tuple_t, details::get_test_name());
    details::do_sleep(250);
    details::consume(750);
    timemory_variable_179.stop();
    auto              key = timemory_variable_179.key();
    std::stringstream expected;
    expected << details::get_test_name();

    using check_tuple_t = std::decay_t<decltype(timemory_variable_179)>;

    std::cout << "\n";
    std::cout << "comp_tuple  :: " << tim::demangle<comp_tuple_t>() << "\n";
    std::cout << "auto_tuple  :: " << tim::demangle<auto_tuple_t>() << "\n";
    std::cout << "check_tuple :: " << tim::demangle<check_tuple_t>() << "\n";
    std::cout << "\n";
    std::cout << timemory_variable_179 << std::endl;
    std::cout << "\n";

    if(!std::is_same<auto_tuple_t, check_tuple_t>::value)
    {
        FAIL();
    }

    if(timemory_variable_179.get<wall_clock>()->get() < 0.9)
    {
        FAIL();
    }

    if(timemory_variable_179.get<cpu_clock>()->get() < 0.7)
    {
        FAIL();
    }

    if(timemory_variable_179.get<cpu_util>()->get() < 70.0)
    {
        FAIL();
    }

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
    {
        SUCCEED();
    }
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
// clang-format on
