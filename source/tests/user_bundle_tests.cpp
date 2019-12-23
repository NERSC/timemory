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

#include <timemory/runtime/configure.hpp>
#include <timemory/runtime/insert.hpp>
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

using custom_bundle_t = user_bundle<0, native_tag>;
using auto_bundle_t   = tim::auto_tuple<user_tuple_bundle, user_list_bundle>;
using comp_bundle_t   = typename auto_bundle_t::component_type;
using bundle0_t       = tim::auto_tuple<wall_clock, cpu_util>;
using bundle1_t       = tim::auto_list<cpu_clock, peak_rss>;
using bundle2_t       = tim::auto_tuple<custom_bundle_t>;

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
        wc_size_orig = tim::storage<wall_clock>::instance()->size();
        cu_size_orig = tim::storage<cpu_util>::instance()->size();
        cc_size_orig = tim::storage<cpu_clock>::instance()->size();
        pr_size_orig = tim::storage<peak_rss>::instance()->size();
        ret          = 0;

        user_list_bundle::reset();

        auto bundle1_init = [](bundle1_t& _bundle) {
            if(details::get_test_name() != "bundle_0" &&
               details::get_test_name() != "bundle_3")
                _bundle.initialize<cpu_clock, peak_rss>();
        };

        user_tuple_bundle::configure<bundle0_t>();
        user_list_bundle::configure<bundle1_t>(bundle1_init);
    }

protected:
    size_t wc_size_orig;
    size_t cu_size_orig;
    size_t cc_size_orig;
    size_t pr_size_orig;
    long   ret;
};

//--------------------------------------------------------------------------------------//

TEST_F(user_bundle_tests, bundle_0)
{
    {
        TIMEMORY_BLANK_MARKER(auto_bundle_t, details::get_test_name().c_str());
        ret += details::fibonacci(35);
    }

    {
        TIMEMORY_BLANK_MARKER(auto_bundle_t, details::get_test_name().c_str());
        ret += details::fibonacci(35);
    }

    printf("fibonacci(35) = %li\n", ret);

    auto wc_n = wc_size_orig + 1;
    auto cu_n = cu_size_orig + 1;
    auto cc_n = cc_size_orig + 0;
    auto pr_n = pr_size_orig + 0;

    ASSERT_EQ(tim::storage<wall_clock>::instance()->size(), wc_n);
    ASSERT_EQ(tim::storage<cpu_util>::instance()->size(), cu_n);
    ASSERT_EQ(tim::storage<cpu_clock>::instance()->size(), cc_n);
    ASSERT_EQ(tim::storage<peak_rss>::instance()->size(), pr_n);
}

//--------------------------------------------------------------------------------------//

TEST_F(user_bundle_tests, bundle_1)
{
    {
        TIMEMORY_BLANK_MARKER(auto_bundle_t, details::get_test_name().c_str());
        ret += details::fibonacci(35);
    }

    printf("fibonacci(35) = %li\n", ret);

    auto wc_n = wc_size_orig + 1;
    auto cu_n = cu_size_orig + 1;
    auto cc_n = cc_size_orig + 1;
    auto pr_n = pr_size_orig + 1;

    ASSERT_EQ(tim::storage<wall_clock>::instance()->size(), wc_n);
    ASSERT_EQ(tim::storage<cpu_util>::instance()->size(), cu_n);
    ASSERT_EQ(tim::storage<cpu_clock>::instance()->size(), cc_n);
    ASSERT_EQ(tim::storage<peak_rss>::instance()->size(), pr_n);
}

//--------------------------------------------------------------------------------------//

TEST_F(user_bundle_tests, bundle_2)
{
    {
        comp_bundle_t _instance(details::get_test_name(), true);
        _instance.get<user_tuple_bundle>().clear();

        _instance.start();
        ret += details::fibonacci(35);
        _instance.stop();

        _instance.start();
        ret += details::fibonacci(35);
        _instance.stop();
    }

    printf("fibonacci(35) = %li\n", ret);

    auto wc_n = wc_size_orig + 0;
    auto cu_n = cu_size_orig + 0;
    auto cc_n = cc_size_orig + 1;
    auto pr_n = pr_size_orig + 1;

    ASSERT_EQ(tim::storage<wall_clock>::instance()->size(), wc_n);
    ASSERT_EQ(tim::storage<cpu_util>::instance()->size(), cu_n);
    ASSERT_EQ(tim::storage<cpu_clock>::instance()->size(), cc_n);
    ASSERT_EQ(tim::storage<peak_rss>::instance()->size(), pr_n);
}

//--------------------------------------------------------------------------------------//

TEST_F(user_bundle_tests, bundle_init_func)
{
    printf("TEST_NAME: %s\n", details::get_test_name().c_str());

    using auto_hybrid_t = tim::auto_hybrid<bundle0_t, bundle1_t>;

    auto init_func = [](auto_hybrid_t& ah) { ah.get_list().initialize<cpu_clock>(); };

    {
        auto_hybrid_t _bundle(details::get_test_name(), false, false, init_func);
        ret += details::fibonacci(35);
    }

    {
        auto_hybrid_t _bundle(details::get_test_name());
        ret += details::fibonacci(35);
    }

    printf("fibonacci(35) = %li\n", ret);

    auto wc_n = wc_size_orig + 1;
    auto cu_n = cu_size_orig + 1;
    auto cc_n = cc_size_orig + 1;
    auto pr_n = pr_size_orig + 0;

    ASSERT_EQ(tim::storage<wall_clock>::instance()->size(), wc_n);
    ASSERT_EQ(tim::storage<cpu_util>::instance()->size(), cu_n);
    ASSERT_EQ(tim::storage<cpu_clock>::instance()->size(), cc_n);
    ASSERT_EQ(tim::storage<peak_rss>::instance()->size(), pr_n);
}

//--------------------------------------------------------------------------------------//

TEST_F(user_bundle_tests, bundle_insert)
{
    auto init_func = [](bundle2_t& al) {
        std::vector<std::string> _init   = { "wall_clock", "cpu_clock" };
        auto&                    _bundle = al.get<custom_bundle_t>();
        tim::insert(_bundle, _init);
        tim::insert(_bundle, { CPU_UTIL, PEAK_RSS });
    };

    {
        bundle2_t _one(details::get_test_name(), false, false, init_func);
        ret += details::fibonacci(35);
    }

    {
        bundle2_t _two(details::get_test_name(), false, false, init_func);
        ret += details::fibonacci(35);
    }

    printf("fibonacci(35) = %li\n", ret);

    auto wc_n = wc_size_orig + 1;
    auto cu_n = cu_size_orig + 1;
    auto cc_n = cc_size_orig + 1;
    auto pr_n = pr_size_orig + 1;

    ASSERT_EQ(tim::storage<wall_clock>::instance()->size(), wc_n);
    ASSERT_EQ(tim::storage<cpu_util>::instance()->size(), cu_n);
    ASSERT_EQ(tim::storage<cpu_clock>::instance()->size(), cc_n);
    ASSERT_EQ(tim::storage<peak_rss>::instance()->size(), pr_n);
}

//--------------------------------------------------------------------------------------//

TEST_F(user_bundle_tests, bundle_configure)
{
    std::initializer_list<std::string> _init = { "wall_clock", "cpu_clock" };
    tim::configure<custom_bundle_t>(_init);
    tim::configure<custom_bundle_t>({ CPU_UTIL, PEAK_RSS });

    {
        bundle2_t _one(details::get_test_name());
        ret += details::fibonacci(35);
        bundle2_t _two(details::get_test_name());
        ret += details::fibonacci(35);
    }

    printf("fibonacci(35) = %li\n", ret);

    auto wc_n = wc_size_orig + 2;
    auto cu_n = cu_size_orig + 2;
    auto cc_n = cc_size_orig + 2;
    auto pr_n = pr_size_orig + 2;

    ASSERT_EQ(tim::storage<wall_clock>::instance()->size(), wc_n);
    ASSERT_EQ(tim::storage<cpu_util>::instance()->size(), cu_n);
    ASSERT_EQ(tim::storage<cpu_clock>::instance()->size(), cc_n);
    ASSERT_EQ(tim::storage<peak_rss>::instance()->size(), pr_n);
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

    tim::settings::dart_type() = "peak_rss";
    // TIMEMORY_VARIADIC_BLANK_AUTO_TUPLE("PEAK_RSS", ::tim::component::peak_rss);
    auto ret = RUN_ALL_TESTS();

    tim::timemory_finalize();

    return ret;
}

//--------------------------------------------------------------------------------------//
