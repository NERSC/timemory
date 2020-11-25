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

#include <array>
#include <chrono>
#include <iostream>
#include <random>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include "timemory/timemory.hpp"

using namespace tim;
using namespace tim::stl;
using namespace tim::stl::ostream;

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

}  // namespace details

//--------------------------------------------------------------------------------------//

class stl_overload_tests : public ::testing::Test
{
protected:
    TIMEMORY_TEST_DEFAULT_SUITE_BODY
};

//--------------------------------------------------------------------------------------//

#define GET(VAR, N) std::get<N>(VAR)

constexpr size_t N         = 2;
const double     tolerance = static_cast<double>(std::numeric_limits<float>::epsilon());

using int_type    = int;
using real_type   = double;
using vector_type = std::vector<real_type>;
using array_type  = std::array<int_type, N>;
using pair_type   = std::pair<vector_type, array_type>;
using tuple_type  = std::tuple<int_type, real_type, pair_type>;

//--------------------------------------------------------------------------------------//

TEST_F(stl_overload_tests, divide)
{
    int_type    int_v    = 4;
    real_type   real_v   = 4.0;
    vector_type vector_v = { real_v, 1.5 * real_v };
    array_type  array_v  = { { int_v, 3 * int_v } };
    pair_type   pair_v   = { vector_v, array_v };

    int_type    int_op    = 2;
    real_type   real_op   = 2.0;
    vector_type vector_op = { real_op, real_op };
    array_type  array_op  = { { int_op, int_op } };
    pair_type   pair_op   = { vector_op, array_op };

    tuple_type init_v{ int_v, real_v, pair_v };
    tuple_type init_op{ int_op, real_op, pair_op };

    tuple_type same_ret = init_v / init_op;
    tuple_type fund_ret = init_v / 2;

    std::stringstream ss;
    ss.precision(2);
    ss.width(4);
    ss.setf(std::ios_base::fixed | std::ios_base::dec | std::ios_base::showpoint);
    ss << '\n';
    ss << "INIT VAL : " << init_v << '\n';
    ss << "INIT OP  : " << init_op << '\n';
    ss << "SAME RET : " << same_ret << '\n';
    ss << "FUND RET : " << fund_ret << '\n';
    ss << '\n';
    std::cout << ss.str();

    // check int
    ASSERT_EQ(GET(same_ret, 0), 2);
    // check double
    ASSERT_NEAR(GET(same_ret, 1), 2.0, tolerance);
    // check vector
    ASSERT_NEAR(GET(same_ret, 2).first[0], 2.0, tolerance);
    ASSERT_NEAR(GET(same_ret, 2).first[1], 3.0, tolerance);
    // check array
    ASSERT_EQ(GET(same_ret, 2).second[0], 2);
    ASSERT_EQ(GET(same_ret, 2).second[1], 6);

    // check int
    ASSERT_EQ(GET(fund_ret, 0), 2);
    // check double
    ASSERT_NEAR(GET(fund_ret, 1), 2.0, tolerance);
    // check vector
    ASSERT_NEAR(GET(fund_ret, 2).first[0], 2.0, tolerance);
    ASSERT_NEAR(GET(fund_ret, 2).first[1], 3.0, tolerance);
    // check array
    ASSERT_EQ(GET(fund_ret, 2).second[0], 2);
    ASSERT_EQ(GET(fund_ret, 2).second[1], 6);
}

//--------------------------------------------------------------------------------------//

TEST_F(stl_overload_tests, multiply)
{
    int_type    int_v    = 4;
    real_type   real_v   = 4.0;
    vector_type vector_v = { real_v, 1.5 * real_v };
    array_type  array_v  = { { int_v, 3 * int_v } };
    pair_type   pair_v   = { vector_v, array_v };

    int_type    int_op    = 2;
    real_type   real_op   = 2.0;
    vector_type vector_op = { real_op, real_op };
    array_type  array_op  = { { int_op, int_op } };
    pair_type   pair_op   = { vector_op, array_op };

    tuple_type init_v{ int_v, real_v, pair_v };
    tuple_type init_op{ int_op, real_op, pair_op };

    tuple_type same_ret = init_v * init_op;
    tuple_type fund_ret = init_v * 2;

    std::stringstream ss;
    ss.precision(2);
    ss.width(4);
    ss.setf(std::ios_base::fixed | std::ios_base::dec | std::ios_base::showpoint);
    ss << '\n';
    ss << "INIT VAL : " << init_v << '\n';
    ss << "INIT OP  : " << init_op << '\n';
    ss << "SAME RET : " << same_ret << '\n';
    ss << "FUND RET : " << fund_ret << '\n';
    ss << '\n';
    std::cout << ss.str();

    // check int
    ASSERT_EQ(GET(same_ret, 0), 8);
    // check double
    ASSERT_NEAR(GET(same_ret, 1), 8.0, tolerance);
    // check vector
    ASSERT_NEAR(GET(same_ret, 2).first[0], 8.0, tolerance);
    ASSERT_NEAR(GET(same_ret, 2).first[1], 12.0, tolerance);
    // check array
    ASSERT_EQ(GET(same_ret, 2).second[0], 8);
    ASSERT_EQ(GET(same_ret, 2).second[1], 24);

    // check int
    ASSERT_EQ(GET(fund_ret, 0), 8);
    // check double
    ASSERT_NEAR(GET(fund_ret, 1), 8.0, tolerance);
    // check vector
    ASSERT_NEAR(GET(fund_ret, 2).first[0], 8.0, tolerance);
    ASSERT_NEAR(GET(fund_ret, 2).first[1], 12.0, tolerance);
    // check array
    ASSERT_EQ(GET(fund_ret, 2).second[0], 8);
    ASSERT_EQ(GET(fund_ret, 2).second[1], 24);
}

//--------------------------------------------------------------------------------------//

TEST_F(stl_overload_tests, statistics)
{
    int_type    int_v    = 4;
    real_type   real_v   = 4.0;
    vector_type vector_v = { real_v, 1.5 * real_v };
    array_type  array_v  = { { int_v, 3 * int_v } };
    pair_type   pair_v   = { vector_v, array_v };

    int_type    int_op    = 2;
    real_type   real_op   = 2.0;
    vector_type vector_op = { real_op, real_op };
    array_type  array_op  = { { int_op, int_op } };
    pair_type   pair_op   = { vector_op, array_op };

    auto apply = [](statistics<tuple_type>& stat_v, const tuple_type& init_v) {
        std::stringstream ss;
        ss.precision(2);
        ss.width(4);
        ss.setf(std::ios_base::fixed | std::ios_base::dec | std::ios_base::showpoint);

        stat_v += init_v;

        tuple_type sum_ret  = stat_v.get_sum();
        tuple_type sqr_ret  = stat_v.get_sqr();
        tuple_type mean_ret = stat_v.get_mean();
        tuple_type min_ret  = stat_v.get_min();
        tuple_type max_ret  = stat_v.get_max();
        tuple_type var_ret  = stat_v.get_variance();
        tuple_type std_ret  = stat_v.get_stddev();

        ss << '\n';
        ss << "ANSWER   :\n";
        ss << "    INIT : " << init_v << '\n';
        ss << "     SUM : " << sum_ret << '\n';
        ss << "     SQR : " << sqr_ret << '\n';
        ss << "    MEAN : " << mean_ret << '\n';
        ss << "     MIN : " << min_ret << '\n';
        ss << "     MAX : " << max_ret << '\n';
        ss << "     VAR : " << var_ret << '\n';
        ss << "  STDDEV : " << std_ret << '\n';
        ss << "    DATA : " << stat_v << '\n';

        std::cout << ss.str();
    };

    tuple_type init_v{ int_v, real_v, pair_v };
    tuple_type init_op{ int_op, real_op, pair_op };

    statistics<tuple_type> stat_v;

    stat_v += init_v;
    stat_v -= init_v;

    tuple_type sum_ret  = stat_v.get_sum();
    tuple_type sqr_ret  = stat_v.get_sqr();
    tuple_type mean_ret = stat_v.get_mean();
    tuple_type min_ret  = stat_v.get_min();
    tuple_type max_ret  = stat_v.get_max();
    tuple_type var_ret  = stat_v.get_variance();
    tuple_type std_ret  = stat_v.get_stddev();

    std::stringstream ss;
    ss.precision(2);
    ss.width(4);
    ss.setf(std::ios_base::fixed | std::ios_base::dec | std::ios_base::showpoint);
    ss << '\n';
    ss << "INITIAL  :\n";
    ss << "     SUM : " << sum_ret << '\n';
    ss << "     SQR : " << sqr_ret << '\n';
    ss << "    MEAN : " << mean_ret << '\n';
    ss << "     MIN : " << min_ret << '\n';
    ss << "     MAX : " << max_ret << '\n';
    ss << "     VAR : " << var_ret << '\n';
    ss << "  STDDEV : " << std_ret << '\n';
    ss << "    DATA : " << stat_v << '\n';
    std::cout << ss.str();

    apply(stat_v, init_op);
    for(int i = 0; i < 10; ++i)
        apply(stat_v, init_v);

    std::cout << '\n';
}

//--------------------------------------------------------------------------------------//
