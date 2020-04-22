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
{}  // namespace details

//--------------------------------------------------------------------------------------//

class apply_tests : public ::testing::Test
{};

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

int
main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    tim::settings::verbose()     = 0;
    tim::settings::debug()       = false;
    tim::settings::json_output() = true;
    tim::timemory_init(&argc, &argv);
    tim::settings::dart_output() = true;
    tim::settings::dart_count()  = 1;
    tim::settings::banner()      = false;

    tim::settings::dart_type() = "peak_rss";
    // TIMEMORY_VARIADIC_BLANK_AUTO_TUPLE("PEAK_RSS", ::tim::component::peak_rss);
    auto ret = RUN_ALL_TESTS();

    tim::timemory_finalize();
    tim::dmp::finalize();
    return ret;
}

//--------------------------------------------------------------------------------------//
