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

extern template struct tim::component_tuple<wall_clock>;
extern template struct tim::auto_tuple<wall_clock>;
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

class flat_tests : public ::testing::Test
{
protected:
    void SetUp() override
    {
        tim::set_env("TIMEMORY_FLAT_PROFILE", "ON", 1);
        static bool configured = false;
        if(!configured)
        {
            configured                   = true;
            tim::settings::verbose()     = 0;
            tim::settings::debug()       = false;
            tim::settings::json_output() = true;
            tim::settings::mpi_thread()  = false;
            tim::mpi::initialize(_argc, _argv);
            tim::timemory_init(_argc, _argv);
            tim::settings::dart_output() = true;
            tim::settings::dart_count()  = 1;
            tim::settings::banner()      = false;
        }
        tim::settings::parse();
    }
};

//--------------------------------------------------------------------------------------//

TEST_F(flat_tests, parse)
{
    tim::settings::flat_profile() = false;
    tim::set_env("TIMEMORY_FLAT_PROFILE", "ON", 1);
    tim::settings::parse();
    std::cout << "\nflat_profile() = " << std::boolalpha << tim::settings::flat_profile()
              << std::endl;
    auto ret = tim::get_env<bool>("TIMEMORY_FLAT_PROFILE", false);
    std::cout << "environment = " << std::boolalpha << ret << '\n' << std::endl;
    ASSERT_TRUE(ret);
    ASSERT_TRUE(tim::settings::flat_profile());
}

//--------------------------------------------------------------------------------------//

TEST_F(flat_tests, get_default)
{
    tim::settings::flat_profile() = false;
    tim::set_env("TIMEMORY_FLAT_PROFILE", "ON", 1);
    tim::settings::parse();
    auto _scope = tim::scope::config{};
    std::cout << "\nscope: " << _scope << '\n' << std::endl;
    ASSERT_TRUE(_scope.is_flat());
    ASSERT_FALSE(_scope.is_tree());
    ASSERT_FALSE(_scope.is_timeline());
    ASSERT_FALSE(_scope.is_flat_timeline());
    ASSERT_FALSE(_scope.is_tree_timeline());
}

//--------------------------------------------------------------------------------------//

TEST_F(flat_tests, flat)
{
    tim::settings::flat_profile() = false;
    auto _scope                   = tim::scope::config(tim::scope::flat{});
    std::cout << "\nscope: " << _scope << '\n' << std::endl;
    ASSERT_TRUE(_scope.is_flat());
    ASSERT_FALSE(_scope.is_tree());
    ASSERT_FALSE(_scope.is_timeline());
    ASSERT_FALSE(_scope.is_flat_timeline());
    ASSERT_FALSE(_scope.is_tree_timeline());
}

//--------------------------------------------------------------------------------------//

TEST_F(flat_tests, general)
{
    auto bsize = tim::storage<wall_clock>::instance()->size();

    {
        long n = 25;
        TIMEMORY_BLANK_MARKER(toolset_t, details::get_test_name());
        auto ret = details::fibonacci(n, true);
        printf("\nfibonacci(%li) = %li\n", n, ret);
    }

    auto esize = tim::storage<wall_clock>::instance()->size();
    printf("\nbsize = %lu\n", (unsigned long) bsize);
    printf("esize = %lu\n\n", (unsigned long) esize);
    auto data = tim::storage<wall_clock>::instance()->get();

    EXPECT_EQ(esize - bsize, 2);
    EXPECT_EQ(data.at(bsize + 1).depth(), 0);
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
