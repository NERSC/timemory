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

#include "timemory/timemory.hpp"

using namespace tim::component;

static int    _argc = 0;
static char** _argv = nullptr;

using mutex_t = std::mutex;
using lock_t  = std::unique_lock<mutex_t>;

using toolset_t = tim::auto_tuple<wall_clock>;

TIMEMORY_DECLARE_EXTERN_STORAGE(component::wall_clock, wc)
TIMEMORY_DECLARE_EXTERN_OPERATIONS(component::wall_clock, true)

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
    std::this_thread::sleep_for(std::chrono::milliseconds(n));
}

// this function consumes an unknown number of cpu resources
inline long
fibonacci(long n)
{
    return (n < 2) ? n : (fibonacci(n - 1) + fibonacci(n - 2));
}

// this function consumes an unknown number of cpu resources
inline long
fibonacci(long n, bool instr)
{
    if(instr)
    {
        TIMEMORY_BASIC_MARKER(toolset_t, "");
        return (n < 2) ? n : (fibonacci(n - 1, true) + fibonacci(n - 2, false));
    }
    else
    {
        return (n < 2) ? n : (fibonacci(n - 1) + fibonacci(n - 2));
    }
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

// get a random entry from vector
template <typename Tp>
size_t
random_entry(const std::vector<Tp>& v)
{
    std::mt19937 rng;
    rng.seed(std::random_device()());
    std::uniform_int_distribution<std::mt19937::result_type> dist(0, v.size() - 1);
    return v.at(dist(rng));
}

}  // namespace details

//--------------------------------------------------------------------------------------//

class timeline_tests : public ::testing::Test
{
protected:
    void SetUp() override
    {
        static bool configured = false;
        if(!configured)
        {
            tim::set_env("TIMEMORY_TIMELINE_PROFILE", "ON", 1);
            configured                   = true;
            tim::settings::verbose()     = 0;
            tim::settings::debug()       = false;
            tim::settings::json_output() = true;
            tim::settings::mpi_thread()  = false;
            tim::dmp::initialize(_argc, _argv);
            tim::timemory_init(_argc, _argv);
            tim::settings::dart_output() = true;
            tim::settings::dart_count()  = 1;
            tim::settings::banner()      = false;
        }
    }
};

//--------------------------------------------------------------------------------------//

TEST_F(timeline_tests, parse)
{
    tim::settings::timeline_profile() = false;
    tim::set_env("TIMEMORY_TIMELINE_PROFILE", "ON", 1);
    tim::settings::parse();
    std::cout << "\ntimeline_profile() = " << std::boolalpha
              << tim::settings::timeline_profile() << '\n'
              << std::endl;
    ASSERT_TRUE(tim::settings::timeline_profile());
}

//--------------------------------------------------------------------------------------//

TEST_F(timeline_tests, general)
{
    auto bsize = tim::storage<wall_clock>::instance()->size();
    // tim::settings::timeline_profile() = true;

    long n = 10;
    for(long i = 0; i < n; ++i)
    {
        TIMEMORY_BLANK_MARKER(toolset_t, details::get_test_name());
        auto ret = details::fibonacci(n, false);
        printf("\nfibonacci(%li) = %li\n", n, ret);
    }

    auto esize = tim::storage<wall_clock>::instance()->size();
    printf("\nbsize = %lu\n", (unsigned long) bsize);
    printf("esize = %lu\n\n", (unsigned long) esize);
    auto data = tim::storage<wall_clock>::instance()->get();

    EXPECT_EQ(esize - bsize, 10);
    EXPECT_EQ(data.at(bsize + 1).depth(), 0);
}

//--------------------------------------------------------------------------------------//

TEST_F(timeline_tests, timeline_quirk)
{
    using bundle_t = tim::component_bundle<TIMEMORY_API, wall_clock>;
    bundle_t _bundle(details::get_test_name(),
                     tim::quirk::config<tim::quirk::timeline_scope>{});
    auto     _scope = _bundle.get_scope();
    std::cout << "\nscope: " << _scope << '\n' << std::endl;
    EXPECT_TRUE(_scope.is_timeline());
    EXPECT_FALSE(_scope.is_tree());
    EXPECT_FALSE(_scope.is_flat());
    EXPECT_FALSE(_scope.is_flat_timeline());
    EXPECT_FALSE(_scope.is_tree_timeline());
}

//--------------------------------------------------------------------------------------//

TEST_F(timeline_tests, general_quirk)
{
    using bundle_t = tim::component_bundle<TIMEMORY_API, wall_clock>;
    bundle_t _bundle(
        details::get_test_name(),
        tim::quirk::config<tim::quirk::timeline_scope, tim::quirk::no_store>{});
    auto _scope1 = _bundle.get_scope();
    auto _store1 = _bundle.get_store();
    auto _scope2 = bundle_t::get_scope_config<tim::quirk::timeline_scope>();
    auto _store2 = bundle_t::get_store_config<tim::quirk::config<tim::quirk::no_store>>();

    std::cout << "\n[1] scope config : " << std::boolalpha << _scope1 << '\n';
    std::cout << "[1] store config : " << std::boolalpha << _store1 << '\n';

    EXPECT_TRUE(_scope1.is_timeline());
    EXPECT_FALSE(_scope1.is_tree());
    EXPECT_FALSE(_scope1.is_flat());
    EXPECT_FALSE(_scope1.is_flat_timeline());
    EXPECT_FALSE(_scope1.is_tree_timeline());
    EXPECT_FALSE(_store1);

    std::cout << "\n[2] scope config : " << std::boolalpha << _scope2 << '\n';
    std::cout << "[2] store config : " << std::boolalpha << _store2 << '\n';

    EXPECT_TRUE(_scope2.is_timeline());
    EXPECT_FALSE(_scope2.is_tree());
    EXPECT_FALSE(_scope2.is_flat());
    EXPECT_FALSE(_scope2.is_flat_timeline());
    EXPECT_FALSE(_scope2.is_tree_timeline());
    EXPECT_FALSE(_store2);
}

//--------------------------------------------------------------------------------------//

TEST_F(timeline_tests, nested)
{
    auto bsize = tim::storage<wall_clock>::instance()->size();
    // tim::settings::timeline_profile() = true;

    long n = 5;
    {
        toolset_t t(details::get_test_name(), tim::scope::config{ false, false, false });
        long      ret = 0;
        for(int i = 0; i < 5; ++i)
            ret += details::fibonacci(n, true);
        printf("\nfibonacci(%li) * 5 = %li\n", n, ret);
    }

    auto esize = tim::storage<wall_clock>::instance()->size();
    printf("\nbsize = %lu\n", (unsigned long) bsize);
    printf("esize = %lu\n\n", (unsigned long) esize);
    auto data = tim::storage<wall_clock>::instance()->get();

    EXPECT_EQ(esize - bsize, 26);
    EXPECT_EQ(data.at(bsize + 1).depth(), 1);
    EXPECT_EQ(data.at(bsize + 2).depth(), 2);
    EXPECT_EQ(data.at(bsize + 6).depth(), 1);
    EXPECT_EQ(data.at(bsize + 7).depth(), 2);
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
