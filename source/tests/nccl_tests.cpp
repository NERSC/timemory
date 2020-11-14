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

#include "nccl_test/all_gather.h"
#include "nccl_test/all_reduce.h"
#include "nccl_test/alltoall.h"
#include "nccl_test/broadcast.h"
#include "nccl_test/reduce.h"
#include "nccl_test/reduce_scatter.h"

static int    _argc = 0;
static char** _argv = nullptr;

using mutex_t = std::mutex;
using lock_t  = std::unique_lock<mutex_t>;

extern "C"
{
    extern void     timemory_register_ncclp();
    extern void     timemory_deregister_ncclp();
    extern uint64_t timemory_start_ncclp();
    extern uint64_t timemory_stop_ncclp(uint64_t);
}

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

template <typename Tp = tim::component::wall_clock>
auto
get_size()
{
    return tim::storage<Tp>::instance()->size();
}
}  // namespace details

//--------------------------------------------------------------------------------------//

class nccl_tests : public ::testing::Test
{
protected:
    void SetUp() override
    {
        static bool configured = false;
        if(!configured)
        {
            configured                   = true;
            tim::settings::verbose()     = 0;
            tim::settings::debug()       = false;
            tim::settings::json_output() = true;
            tim::settings::mpi_thread()  = false;
            tim::dmp::initialize(_argc, _argv);
            tim::timemory_init(_argc, _argv);
            tim::settings::dart_output()      = true;
            tim::settings::dart_count()       = 1;
            tim::settings::banner()           = false;
            tim::settings::ncclp_components() = "wall_clock";
        }
    }

    void TearDown() override {}
};

//--------------------------------------------------------------------------------------//

TEST_F(nccl_tests, all_gather)
{
    auto beg_sz = details::get_size();
    int  ret    = test_main(allGatherEngine, _argc, _argv);
    auto end_sz = details::get_size();
    EXPECT_EQ(ret, 0);
    EXPECT_EQ(end_sz - beg_sz, 1);
}

//--------------------------------------------------------------------------------------//

TEST_F(nccl_tests, all_reduce)
{
    auto beg_sz = details::get_size();
    int  ret    = test_main(allReduceEngine, _argc, _argv);
    auto end_sz = details::get_size();
    EXPECT_EQ(ret, 0);
    EXPECT_EQ(end_sz - beg_sz, 1);
}

//--------------------------------------------------------------------------------------//

TEST_F(nccl_tests, all_to_all)
{
    auto beg_sz = details::get_size();
    int  ret    = test_main(alltoAllEngine, _argc, _argv);
    auto end_sz = details::get_size();
    EXPECT_EQ(ret, 0);
    EXPECT_EQ(end_sz - beg_sz, 2);
}

//--------------------------------------------------------------------------------------//

TEST_F(nccl_tests, broadcast)
{
    auto beg_sz = details::get_size();
    int  ret    = test_main(broadcastEngine, _argc, _argv);
    auto end_sz = details::get_size();
    EXPECT_EQ(ret, 0);
    EXPECT_EQ(end_sz - beg_sz, 1);
}

//--------------------------------------------------------------------------------------//

TEST_F(nccl_tests, reduce)
{
    auto beg_sz = details::get_size();
    int  ret    = test_main(reduceEngine, _argc, _argv);
    auto end_sz = details::get_size();
    EXPECT_EQ(ret, 0);
    EXPECT_EQ(end_sz - beg_sz, 1);
}

//--------------------------------------------------------------------------------------//

TEST_F(nccl_tests, reduce_scatter)
{
    auto beg_sz = details::get_size();
    int  ret    = test_main(reduceScatterEngine, _argc, _argv);
    auto end_sz = details::get_size();
    EXPECT_EQ(ret, 0);
    EXPECT_EQ(end_sz - beg_sz, 1);
}

//--------------------------------------------------------------------------------------//

int
main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    _argc = argc;
    _argv = argv;

    tim::set_env("TIMEMORY_NCCLP_REJECT_LIST",
                 "ncclGroupStart, ncclGroupEnd, ncclCommCuDevice, ncclCommUserRank", 0);
    timemory_register_ncclp();
    auto ret = RUN_ALL_TESTS();
    timemory_deregister_ncclp();

    tim::timemory_finalize();
    return ret;
}

//--------------------------------------------------------------------------------------//
