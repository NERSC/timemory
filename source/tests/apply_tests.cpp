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

#include "test_macros.hpp"

TIMEMORY_TEST_DEFAULT_MAIN

#include "gtest/gtest.h"

#include "timemory/mpl/apply.hpp"
#include "timemory/timemory.hpp"

#include <array>
#include <chrono>
#include <iostream>
#include <random>
#include <thread>
#include <vector>

static const double epsilon = 1.1 * std::numeric_limits<double>::epsilon();

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
}  // namespace details

//--------------------------------------------------------------------------------------//

class apply_tests : public ::testing::Test
{
protected:
    TIMEMORY_TEST_DEFAULT_SUITE_BODY
};

//--------------------------------------------------------------------------------------//

TEST_F(apply_tests, set_value)
{
    const double           value = 10.0;
    std::array<double, 12> arr;
    tim::apply<void>::set_value(arr, value);
    for(auto& itr : arr)
    {
        auto diff = itr - value;
        printf("itr value = %16.12f (expected = %16.12f); diff = %16.12e\n", itr, value,
               diff);
        ASSERT_NEAR(itr, value, epsilon);
    }
}

//--------------------------------------------------------------------------------------//
// declare a component and set it to always off
TIMEMORY_DECLARE_COMPONENT(always_off)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::always_off, false_type)
//
TEST_F(apply_tests, traits)
{
    using namespace tim::component;
    tim::trait::apply<tim::trait::runtime_enabled>::set<
        wall_clock, user_clock, system_clock, cpu_clock, cpu_util, peak_rss>(false);
    tim::trait::apply<tim::trait::runtime_enabled>::set<user_clock, peak_rss, always_off>(
        true);

    enum
    {
        WallClockIdx = 0,
        UserClockIdx,
        SystemClockIdx,
        CpuClockIdx,
        CpuUtilIdx,
        PeakRssIdx,
        AlwaysOffIdx
    };

    auto check = tim::trait::apply<tim::trait::runtime_enabled>::get<
        wall_clock, user_clock, system_clock, cpu_clock, cpu_util, peak_rss,
        always_off>();

    EXPECT_FALSE(std::get<WallClockIdx>(check));
    EXPECT_TRUE(std::get<UserClockIdx>(check));
    EXPECT_FALSE(std::get<SystemClockIdx>(check));
    EXPECT_FALSE(std::get<CpuClockIdx>(check));
    EXPECT_FALSE(std::get<CpuUtilIdx>(check));
    EXPECT_TRUE(std::get<PeakRssIdx>(check));
    EXPECT_FALSE(std::get<AlwaysOffIdx>(check));
}

//--------------------------------------------------------------------------------------//

class chained_tests : public ::testing::Test
{};

using namespace tim::component;
static long count_val = 0;

namespace details
{
//
template <typename BundleT = void>
long
fibonacci(long);
//
template <typename BundleT>
long
fibonacci(long n)
{
    return BundleT{ "fibonacci" }
        .start()
        .execute([n]() {
            return (n < 2) ? n : (fibonacci<BundleT>(n - 1) + fibonacci<BundleT>(n - 2));
        })
        .stop()
        .return_result();
}
//
template <>
long
fibonacci<void>(long n)
{
    return (n < 2) ? n : (fibonacci<void>(n - 1) + fibonacci<void>(n - 2));
}
//
template <>
long
fibonacci<long>(long n)
{
    ++count_val;
    return (n < 2) ? n : (fibonacci<long>(n - 1) + fibonacci<long>(n - 2));
}
//
}  // namespace details
//--------------------------------------------------------------------------------------//

TEST_F(chained_tests, fibonacci)
{
    using bundle_t     = tim::component_bundle<TIMEMORY_API, trip_count>;
    using timer_t      = tim::lightweight_tuple<wall_clock, tim::quirk::auto_start>;
    using auto_timer_t = tim::auto_bundle<TIMEMORY_API, wall_clock>;

    tim::settings::precision() = 6;
    long nfib                  = 30;

    std::pair<timer_t, long> count_tmp =
        timer_t{ "count" }.start().execute(details::fibonacci<long>, nfib).stop();

    auto real_ret = timer_t{ "real" }
                        .start()
                        .execute(details::fibonacci<void>, nfib)
                        .stop()
                        .get_bundle_and_result();

    auto test_ret = auto_timer_t{ "test" }
                        .execute(details::fibonacci<bundle_t>, nfib)
                        .stop()
                        .get_bundle_and_result();

    auto real_timer = real_ret.first;
    auto test_timer = test_ret.first;
    auto real_val   = real_ret.second;
    auto test_val   = test_ret.second;

    EXPECT_EQ(real_val, count_tmp.second);
    EXPECT_EQ(real_val, test_val);

    auto tc_data  = tim::storage<trip_count>::instance()->get();
    long meas_val = 0;
    long laps_val = 0;
    for(auto& itr : tc_data)
    {
        meas_val += itr.data().get();
        laps_val += itr.data().get_laps();
    }

    EXPECT_EQ(count_val, meas_val);
    EXPECT_EQ(count_val, laps_val);

    std::cout << real_timer << std::endl;
    std::cout << test_timer << std::endl;

    auto tc_storage = tim::storage<trip_count>::instance()->get();

    EXPECT_EQ(tc_storage.size(), nfib);
}

//--------------------------------------------------------------------------------------//
