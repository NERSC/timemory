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

// random number generator
template <typename T = std::mt19937>
T&
get_rng(size_t initial_seed = 0)
{
    static T _instance = [=]() {
        T _rng;
        _rng.seed((initial_seed == 0) ? std::random_device()() : initial_seed);
        return _rng;
    }();
    return _instance;
}

// random integer
template <typename T, std::enable_if_t<(std::is_integral<T>::value), int> = 0>
T
get_random_value(T beg, T end)
{
    std::uniform_int_distribution<T> dist(beg, end);
    return dist(get_rng());
}

template <typename T>
struct identity
{
    using type = T;
};

template <typename T>
using identity_t = typename identity<T>::type;

template <typename T, std::enable_if_t<(std::is_floating_point<T>::value), int> = 0>
T
get_random_value(identity_t<T> beg, T end)
{
    std::uniform_real_distribution<T> dist(beg, end);
    return dist(get_rng());
}

// get a random entry from vector
template <typename Tp>
Tp
random_entry(const std::vector<Tp>& v)
{
    return v.at(get_random_value(0, v.size() - 1));
}

}  // namespace details

//--------------------------------------------------------------------------------------//

class empty_tests : public ::testing::Test
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
            tim::settings::scientific()  = true;
            tim::mpi::initialize(_argc, _argv);
            tim::timemory_init(_argc, _argv);
            tim::settings::dart_output() = true;
            tim::settings::dart_count()  = 1;
            tim::settings::banner()      = false;
        }
    }
};

//--------------------------------------------------------------------------------------//

TEST_F(empty_tests, iteration_tracker)
{
    struct iteration_tag
    {};

    using iteration_value_tracker_t = data_tracker<double, iteration_tag>;
    using iteration_count_tracker_t = data_tracker<uint64_t, iteration_tag>;
    using iteration_value_handle_t  = data_handler_t<iteration_value_tracker_t>;

    iteration_value_tracker_t::label()       = "iteration_value";
    iteration_value_tracker_t::description() = "Iteration value tracker";
    iteration_count_tracker_t::label()       = "iteration_count";
    iteration_count_tracker_t::description() = "Iteration count tracker";

    using tuple_t =
        tim::auto_tuple<wall_clock, iteration_count_tracker_t, iteration_value_tracker_t>;

    double       err      = std::numeric_limits<double>::max();
    const double tol      = 1.0e-3;
    uint64_t     num_iter = 0;

    tuple_t t(details::get_test_name());
    while(err > tol)
    {
        err = details::get_random_value<double>(0.0, 10.0);
        t.store(std::plus<uint64_t>{}, 1);
        t.store(iteration_value_handle_t{}, err);
        ++num_iter;
    }
    t.stop();

    std::cout << "\n" << t << std::endl;
    std::cout << "num_iter : " << num_iter << std::endl;
    std::cout << "error    : " << err << "\n" << std::endl;

    ASSERT_TRUE(num_iter != 0);
    ASSERT_TRUE(t.get<iteration_count_tracker_t>()->get() == num_iter);
    ASSERT_TRUE(t.get<iteration_value_tracker_t>()->get() < tol);
    EXPECT_NEAR(t.get<iteration_value_tracker_t>()->get(), err, 1.0e-6);
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
