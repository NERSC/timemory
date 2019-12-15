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

#include <cassert>
#include <chrono>
#include <cmath>
#include <fstream>
#include <future>
#include <iostream>
#include <iterator>
#include <random>
#include <thread>
#include <unordered_map>
#include <vector>

#define TIMEMORY_STRICT_VARIADIC_CONCAT

#include <timemory/timemory.hpp>
#include <timemory/utility/signals.hpp>

using namespace tim::component;

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

//--------------------------------------------------------------------------------------//
// fibonacci calculation
int64_t
fibonacci(int32_t n)
{
    return (n < 2) ? n : fibonacci(n - 1) + fibonacci(n - 2);
}
}  // namespace details

//--------------------------------------------------------------------------------------//

class variadic_tests : public ::testing::Test
{};

//--------------------------------------------------------------------------------------//

TEST_F(variadic_tests, variadic)
{
    static constexpr size_t nz = 7;
    std::array<size_t, nz>  sizes;
    size_t                  n = 0;
    std::generate(sizes.begin(), sizes.end(), [&]() { return n++; });

    {
        using tuple_t  = tim::auto_tuple<wall_clock, peak_rss>;
        using list_t   = tim::auto_list<cpu_roofline_dp_flops, gpu_roofline_flops>;
        using hybrid_t = tim::auto_hybrid<tuple_t, list_t>;
        sizes[0]       = hybrid_t::size();
    }

    {
        using tuple_t  = tim::component_tuple<wall_clock, peak_rss>;
        using list_t   = tim::auto_list<cpu_roofline_dp_flops, gpu_roofline_flops>;
        using hybrid_t = tim::auto_hybrid<tuple_t, list_t>;
        sizes[1]       = hybrid_t::size();
    }

    {
        using tuple_t  = tim::component_tuple<wall_clock, peak_rss>;
        using list_t   = tim::component_list<cpu_roofline_dp_flops, gpu_roofline_flops>;
        using hybrid_t = tim::auto_hybrid<tuple_t, list_t>;
        sizes[2]       = hybrid_t::size();
    }

    {
        using tuple_t  = tim::component_tuple<wall_clock, peak_rss>;
        using list_t   = tim::component_list<cpu_roofline_dp_flops, gpu_roofline_flops,
                                           cpu_roofline_sp_flops>;
        using hybrid_t = tim::auto_hybrid<tuple_t, list_t>;
        sizes[3]       = hybrid_t::size();
    }

    {
        using tuple_t  = tim::component_tuple<wall_clock, peak_rss>;
        using list_t   = tim::component_list<gpu_roofline_dp_flops, cpu_roofline_flops,
                                           gpu_roofline_sp_flops>;
        using hybrid_t = tim::auto_hybrid<tuple_t, list_t>;
        sizes[4]       = hybrid_t::size();
    }

    {
        using tuple_t  = tim::component_tuple<wall_clock, peak_rss>;
        using list_t   = tim::component_list<gpu_roofline_dp_flops, gpu_roofline_flops,
                                           gpu_roofline_sp_flops>;
        using hybrid_t = tim::auto_hybrid<tuple_t, list_t>;
        sizes[5]       = hybrid_t::size();
    }

    {
        using tuple_t  = tim::component_tuple<wall_clock, peak_rss>;
        using list_t   = tim::component_list<cpu_roofline_dp_flops, cpu_roofline_flops,
                                           cpu_roofline_sp_flops>;
        using hybrid_t = tim::auto_hybrid<tuple_t, list_t>;
        sizes[6]       = hybrid_t::size();
    }

    std::cout << "\n";
    for(size_t i = 0; i < nz; ++i)
        std::cout << "size[" << i << "] = " << sizes[i] << std::endl;
    std::cout << "\n";

    EXPECT_EQ(sizes[0], sizes[1]);
    EXPECT_EQ(sizes[0], sizes[2]);
    EXPECT_EQ(sizes[1], sizes[2]);

    for(size_t i = 3; i < nz; ++i)
    {
        ASSERT_LE(sizes[0], sizes[i]);
        ASSERT_LE(sizes[1], sizes[i]);
        ASSERT_LE(sizes[2], sizes[i]);
    }

    using tup_t     = tim::component_tuple<wall_clock>;
    using tup_add_t = tim::component_tuple<cpu_clock>;
    using lst_t     = tim::component_list<peak_rss>;
    using lst_add_t = tim::component_list<page_rss>;
    using hybrid_t  = tim::concat<tim::auto_hybrid<tup_t, lst_t>, tup_add_t, lst_add_t>;

    auto hsize = hybrid_t::size();
    auto tsize = hybrid_t::tuple_type::size();
    auto lsize = hybrid_t::list_type::size();

    std::cout << "\nhybrid        : " << tim::demangle<hybrid_t>() << "\n";
    std::cout << "\nhybrid tuple  : " << tim::demangle<typename hybrid_t::tuple_type>();
    std::cout << "\nhybrid list   : " << tim::demangle<typename hybrid_t::list_type>();
    std::cout << "\n";
    {
        hybrid_t hybrid(details::get_test_name());
        auto     ret = details::fibonacci(41);
        printf("\nfibonacci(41) = %li\n\n", (long int) ret);
    }

    EXPECT_EQ(hsize, 4);
    EXPECT_EQ(tsize, 2);
    EXPECT_EQ(lsize, 2);
}

//--------------------------------------------------------------------------------------//

TEST_F(variadic_tests, concat)
{
    using lhs_t = tim::component_tuple<real_clock, system_clock>;
    using rhs_t = tim::component_tuple<real_clock, cpu_clock>;

    using comp_t0 = tim::remove_duplicates<
        typename tim::component_tuple<lhs_t, rhs_t>::component_type>;
    using comp_t1 = tim::remove_duplicates<
        typename tim::auto_tuple<lhs_t, rhs_t, user_clock>::component_type>;
    using comp_t2 = tim::remove_duplicates<typename tim::auto_tuple<
        lhs_t, tim::component_list<rhs_t, user_clock>>::component_type>;

    using data_t0 =
        tim::remove_duplicates<typename tim::component_list<lhs_t, rhs_t>::data_type>;
    using data_t1 = tim::remove_duplicates<
        typename tim::auto_list<lhs_t, rhs_t, user_clock>::data_type>;
    using data_t2 = tim::remove_duplicates<
        typename tim::auto_list<lhs_t, tim::auto_tuple<rhs_t, user_clock>>::data_type>;

    std::cout << "\n" << std::flush;
    std::cout << "comp_t0 = " << tim::demangle<comp_t0>() << "\n";
    std::cout << "comp_t1 = " << tim::demangle<comp_t1>() << "\n";
    std::cout << "comp_t2 = " << tim::demangle<comp_t2>() << "\n";
    std::cout << "\n" << std::flush;

    std::cout << "data_t0 = " << tim::demangle<data_t0>() << "\n";
    std::cout << "data_t1 = " << tim::demangle<data_t1>() << "\n";
    std::cout << "data_t2 = " << tim::demangle<data_t2>() << "\n";
    std::cout << "\n" << std::flush;

    EXPECT_EQ(comp_t0::size(), 3);
    EXPECT_EQ(comp_t1::size(), 4);
    EXPECT_EQ(comp_t2::size(), 4);

    EXPECT_EQ(std::tuple_size<data_t0>::value, 3);
    EXPECT_EQ(std::tuple_size<data_t1>::value, 4);
    EXPECT_EQ(std::tuple_size<data_t2>::value, 4);
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

    tim::dmp::finalize();
    return ret;
}

//--------------------------------------------------------------------------------------//
