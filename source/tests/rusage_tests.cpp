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
#include <iostream>
#include <random>
#include <thread>
#include <timemory/timemory.hpp>
#include <vector>

using namespace tim::component;
using mutex_t   = std::mutex;
using lock_t    = std::unique_lock<mutex_t>;
using condvar_t = std::condition_variable;

static const float   peak_tolerance = 5;
static const float   curr_tolerance = 5;
static const int64_t niter          = 20;
static const int64_t nelements      = 0.95 * (tim::units::get_page_size() * 500);
static const auto  memory_unit = std::pair<int64_t, std::string>(tim::units::MiB, "MiB");
static peak_rss    peak;
static current_rss curr;
static auto        tot_size = nelements * sizeof(int64_t) / memory_unit.first;

#define CHECK_AVAILABLE(type)                                                            \
    if(!tim::trait::is_available<type>::value)                                           \
        return;

//--------------------------------------------------------------------------------------//
namespace details
{
// this function consumes an unknown number of cpu resources
long
fibonacci(long n)
{
    return (n < 2) ? n : (fibonacci(n - 1) + fibonacci(n - 2));
}

// this function ensures an allocation cannot be optimized
template <typename _Tp>
size_t
random_entry(const std::vector<_Tp>& v)
{
    std::mt19937 rng;
    rng.seed(std::random_device()());
    std::uniform_int_distribution<std::mt19937::result_type> dist(0, v.size() - 1);
    return v.at(dist(rng));
}
void
allocate()
{
    peak.reset();
    curr.reset();

    curr.start();
    peak.start();

    std::vector<int64_t> v(nelements, 15);
    auto                 ret  = fibonacci(0);
    long                 nfib = details::random_entry(v);
    for(int64_t i = 0; i < niter; ++i)
    {
        nfib = details::random_entry(v);
        ret += details::fibonacci(nfib);
    }
    printf("fibonacci(%li) * %li = %li\n", (long) nfib, (long) niter, ret);

    curr.stop();
    peak.stop();
}

template <typename _Tp>
std::string
get_info(const _Tp& obj)
{
    std::stringstream ss;
    auto              _unit = static_cast<double>(_Tp::get_unit());
    ss << "value = " << obj.get_value() / _unit << " " << _Tp::get_display_unit()
       << ", accum = " << obj.get_accum() / _unit << " " << _Tp::get_display_unit()
       << std::endl;
    return ss.str();
}
}  // namespace details

//--------------------------------------------------------------------------------------//

class rusage_tests : public ::testing::Test
{
};

//--------------------------------------------------------------------------------------//

TEST_F(rusage_tests, peak_rss)
{
    CHECK_AVAILABLE(peak_rss);
    std::cout << "[" << __FUNCTION__ << "]> peak:     " << peak << std::endl;
    std::cout << "[" << __FUNCTION__ << "]> expected: " << tot_size << " "
              << memory_unit.second << std::endl;
    std::cout << "[" << __FUNCTION__ << "]> info:     " << details::get_info(peak);
    ASSERT_NEAR(tot_size, peak.get(), peak_tolerance);
}

//--------------------------------------------------------------------------------------//

TEST_F(rusage_tests, current_rss)
{
    CHECK_AVAILABLE(current_rss);
    std::cout << "[" << __FUNCTION__ << "]> current:  " << curr << std::endl;
    std::cout << "[" << __FUNCTION__ << "]> expected: " << tot_size << " "
              << memory_unit.second << std::endl;
    std::cout << "[" << __FUNCTION__ << "]> info:     " << details::get_info(curr);
    ASSERT_NEAR(tot_size, curr.get(), curr_tolerance);
}

//--------------------------------------------------------------------------------------//

int
main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    // preform allocation only once here
    tim::settings::precision()    = 9;
    tim::settings::memory_units() = memory_unit.second;
    tim::settings::process();
    details::allocate();

    return RUN_ALL_TESTS();
}

//--------------------------------------------------------------------------------------//
