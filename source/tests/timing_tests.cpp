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
#include <thread>
#include <timemory/timemory.hpp>

using namespace tim::component;

//--------------------------------------------------------------------------------------//
namespace details
{
void
do_sleep(int n)
{
    std::this_thread::sleep_for(std::chrono::seconds(n));
}
long
fibonacci(long n)
{
    return (n < 2) ? n : (fibonacci(n - 1) + fibonacci(n - 2));
}
}  // namespace details

//--------------------------------------------------------------------------------------//

class timing_tests : public ::testing::Test
{
};

//--------------------------------------------------------------------------------------//

TEST_F(timing_tests, wall_timer)
{
    wall_clock obj;
    obj.start();
    details::do_sleep(1);
    obj.stop();
    ASSERT_NEAR(1.0f, obj.get(), 1.0e-2);
}

//--------------------------------------------------------------------------------------//

TEST_F(timing_tests, system_timer)
{
    system_clock obj;
    obj.start();
    std::thread t(details::do_sleep, 1);
    t.join();
    obj.stop();
    ASSERT_NEAR(0.0f, obj.get(), 1.0e-2);
}

//--------------------------------------------------------------------------------------//

TEST_F(timing_tests, user_timer)
{
    user_clock obj;
    obj.start();
    std::thread t(details::do_sleep, 1);
    t.join();
    obj.stop();
    ASSERT_NEAR(0.0f, obj.get(), 1.0e-2);
}

//--------------------------------------------------------------------------------------//

TEST_F(timing_tests, cpu_timer)
{
    cpu_clock obj;
    obj.start();
    std::thread t(details::do_sleep, 1);
    t.join();
    obj.stop();
    ASSERT_NEAR(0.0f, obj.get(), 1.0e-2);
}

//--------------------------------------------------------------------------------------//

TEST_F(timing_tests, thread_cpu_timer)
{
    thread_cpu_clock obj;
    obj.start();
    std::thread t(details::fibonacci, 40);
    t.join();
    obj.stop();
    ASSERT_NEAR(0.0f, obj.get(), 1.0e-2);
}

//--------------------------------------------------------------------------------------//

int
main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

//--------------------------------------------------------------------------------------//
