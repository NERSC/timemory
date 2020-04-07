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

#include "timemory/timemory.hpp"
#include <chrono>
#include <iostream>
#include <random>
#include <thread>
#include <vector>

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

inline void
demangle(const std::string& _inst)
{
    std::cout << "\n"
              << "original:\n\t " << _inst << "\n"
              << "demangle:\n\t " << tim::demangle(_inst.c_str()) << "\n"
              << std::endl;
}

template <typename Tp>
inline void
demangle()
{
    std::cout << "\n"
              << "original:\n\t " << typeid(Tp).name() << "\n"
              << "demangle:\n\t " << tim::demangle<Tp>() << "\n"
              << std::endl;
}

}  // namespace details

//--------------------------------------------------------------------------------------//

class mangle_tests : public ::testing::Test
{};

//--------------------------------------------------------------------------------------//

TEST_F(mangle_tests, sample)
{
    std::string function_name =
        "_ZN5amrex13launch_globalIZNS_3ForIlZ27doEsirkepovDepositionShapeNILi3EEvPKdS4_"
        "S4_S4_S4_S4_S4_PKiRKNS_6Array4IdEESA_SA_ldRKSt5arrayIdLm3EESC_NS_4Dim3EdlEUllE_"
        "vEEvT_OT0_mEUlvE_EEvSH_";

    using namespace tim::component;
    std::cout << "\n" << tim::demangle(function_name) << "\n";
    std::cout << "\n" << tim::demangle<tim::auto_tuple<wall_clock, cuda_event>>() << "\n";
    std::cout << std::endl;
}

//--------------------------------------------------------------------------------------//

TEST_F(mangle_tests, cuda_kernel)
{
    std::string function_name =
        "_ZN5amrex13launch_globalIZNS_3ForIlZ27doEsirkepovDepositionShapeNILi3EEvPKdS4_"
        "S4_S4_S4_S4_S4_PKiRKNS_6Array4IdEESA_SA_ldRKSt5arrayIdLm3EESC_NS_4Dim3EdlEUllE_"
        "vEEvT_OT0_mEUlvE_EEvSH_";

    details::demangle(function_name);
    using namespace tim::component;
    std::cout << tim::demangle(function_name) << "\n";
    std::cout << tim::demangle<tim::auto_tuple<wall_clock, cuda_event>>() << "\n";
}

//--------------------------------------------------------------------------------------//

TEST_F(mangle_tests, data_type) { details::demangle<double>(); }

//--------------------------------------------------------------------------------------//

TEST_F(mangle_tests, struct_type)
{
    using namespace tim::component;
    details::demangle<wall_clock>();
}

//--------------------------------------------------------------------------------------//

TEST_F(mangle_tests, variadic_struct_type) { details::demangle<tim::auto_tuple<>>(); }

//--------------------------------------------------------------------------------------//

TEST_F(mangle_tests, function)
{
    using Type = decltype(details::get_test_name);
    details::demangle<Type>();
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
