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
#include <condition_variable>
#include <functional>
#include <iostream>
#include <limits>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using namespace tim::component;
using mutex_t        = std::mutex;
using lock_t         = std::unique_lock<mutex_t>;
using string_t       = std::string;
using stringstream_t = std::stringstream;

using user_bundle_t = tim::auto_tuple<user_bundle_0, user_bundle_1>;

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

// this function consumes approximately "t" milliseconds of cpu time
inline void
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

class user_bundle_tests : public ::testing::Test
{
protected:
    void SetUp() override
    {
        using bundle0_t = tim::auto_tuple<wall_clock, cpu_util>;
        using bundle1_t = tim::auto_list<cpu_clock, peak_rss>;

        auto bundle1_init = [](bundle1_t& _bundle) {
            if(details::get_test_name() == "bundle_1")
            {
                PRINT_HERE("%s", details::get_test_name().c_str());
                _bundle.initialize<cpu_clock, peak_rss>();
            }
        };

        user_bundle_0::configure<bundle0_t>();
        user_bundle_1::configure<bundle1_t>(bundle1_init);
    }
};

//--------------------------------------------------------------------------------------//

TEST_F(user_bundle_tests, bundle_0)
{
    {
        TIMEMORY_BLANK_MARKER(user_bundle_t, details::get_test_name().c_str());
        long ret = details::fibonacci(35);
        printf("fibonacci(35) = %li\n", ret);
    }
    ASSERT_EQ(tim::storage<wall_clock>::instance()->size(), 1);
    ASSERT_EQ(tim::storage<cpu_util>::instance()->size(), 1);
    ASSERT_EQ(tim::storage<cpu_clock>::instance()->size(), 0);
    ASSERT_EQ(tim::storage<peak_rss>::instance()->size(), 0);
}

//--------------------------------------------------------------------------------------//

TEST_F(user_bundle_tests, bundle_1)
{
    auto wc_size_orig = tim::storage<wall_clock>::instance()->size();
    auto cu_size_orig = tim::storage<cpu_util>::instance()->size();
    {
        TIMEMORY_BLANK_MARKER(user_bundle_t, details::get_test_name().c_str());
        long ret = details::fibonacci(35);
        printf("fibonacci(35) = %li\n", ret);
    }
    ASSERT_EQ(tim::storage<wall_clock>::instance()->size(), wc_size_orig + 1);
    ASSERT_EQ(tim::storage<cpu_util>::instance()->size(), cu_size_orig + 1);
    ASSERT_EQ(tim::storage<cpu_clock>::instance()->size(), 1);
    ASSERT_EQ(tim::storage<peak_rss>::instance()->size(), 1);
}

//--------------------------------------------------------------------------------------//

int
main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    tim::settings::verbose()     = 0;
    tim::settings::debug()       = false;
    tim::settings::json_output() = true;
    tim::timemory_init(&argc, &argv);  // parses environment, sets output paths
    tim::settings::dart_output() = false;
    tim::settings::dart_count()  = 1;
    tim::settings::banner()      = false;

    auto ret = RUN_ALL_TESTS();

    tim::timemory_finalize();

    return ret;
}

//--------------------------------------------------------------------------------------//
