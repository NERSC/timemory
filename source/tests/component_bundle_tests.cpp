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

#define TIMEMORY_STRICT_VARIADIC_CONCAT

#include "timemory/timemory.hpp"
#include "timemory/utility/signals.hpp"
#include "timemory/variadic/functional.hpp"

#include "gtest/gtest.h"

#include <cassert>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <fstream>
#include <future>
#include <iostream>
#include <iterator>
#include <mutex>
#include <random>
#include <thread>
#include <unordered_map>
#include <vector>

using namespace tim::component;
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
    return std::string(::testing::UnitTest::GetInstance()->current_test_suite()->name()) +
           "." + ::testing::UnitTest::GetInstance()->current_test_info()->name();
}

//--------------------------------------------------------------------------------------//
// fibonacci calculation
int64_t
fibonacci(int32_t n)
{
    return (n < 2) ? n : fibonacci(n - 1) + fibonacci(n - 2);
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
}  // namespace details

//--------------------------------------------------------------------------------------//

static int    _argc = 0;
static char** _argv = nullptr;

//--------------------------------------------------------------------------------------//

class component_bundle_tests : public ::testing::Test
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
            tim::dmp::initialize(_argc, _argv);
            tim::timemory_init(_argc, _argv);
            tim::settings::dart_output() = true;
            tim::settings::dart_count()  = 1;
            tim::settings::banner()      = false;
        }
    }
};

//--------------------------------------------------------------------------------------//

TEST_F(component_bundle_tests, variadic)
{
    static constexpr size_t nz = 7;
    std::array<size_t, nz>  sizes;
    size_t                  n = 0;
    std::generate(sizes.begin(), sizes.end(), [&]() { return n++; });

    {
        using bundle_t = tim::auto_bundle<TIMEMORY_API, wall_clock, peak_rss>;
        sizes[0]       = bundle_t::size();
    }

    {
        using bundle_t = tim::component_bundle<TIMEMORY_API, wall_clock, peak_rss>;
        sizes[1]       = bundle_t::size();
    }

    {
        using bundle_t =
            tim::component_bundle_t<TIMEMORY_API, wall_clock, peak_rss,
                                    cpu_roofline_dp_flops*, gpu_roofline_flops*>;
        sizes[2] = bundle_t::size();
    }

    {
        using bundle_t =
            tim::auto_bundle_t<TIMEMORY_API, wall_clock, peak_rss, cpu_roofline_dp_flops*,
                               gpu_roofline_flops*, cpu_roofline_sp_flops*>;
        sizes[3] = bundle_t::size();
    }

    {
        using bundle_t =
            tim::component_bundle_t<TIMEMORY_API, wall_clock, peak_rss,
                                    gpu_roofline_dp_flops*, cpu_roofline_flops,
                                    gpu_roofline_sp_flops*>;
        sizes[4] = bundle_t::size();
    }

    {
        using bundle_t =
            tim::component_bundle_t<TIMEMORY_API, wall_clock, peak_rss,
                                    gpu_roofline_dp_flops*, gpu_roofline_flops*,
                                    gpu_roofline_sp_flops*>;
        sizes[5] = bundle_t::size();
    }

    {
        using bundle_t =
            tim::auto_bundle_t<TIMEMORY_API, wall_clock, peak_rss, cpu_roofline_dp_flops*,
                               cpu_roofline_flops, cpu_roofline_sp_flops*>;
        sizes[6] = bundle_t::size();
    }

    std::cout << "\n";
    for(size_t i = 0; i < nz; ++i)
        std::cout << "size[" << i << "] = " << sizes[i] << std::endl;
    std::cout << "\n";

    EXPECT_EQ(sizes[0], sizes[1]);

    for(size_t i = 2; i < nz; ++i)
    {
        EXPECT_EQ(sizes[0], sizes[i]);
        EXPECT_EQ(sizes[1], sizes[i]);
    }

    using bundle_t = tim::auto_bundle<TIMEMORY_API, wall_clock, cpu_clock, peak_rss*>;

    auto hsize = bundle_t::size();
    EXPECT_EQ(hsize, 3);

    auto wsize = tim::storage<wall_clock>::instance()->size();
    auto csize = tim::storage<cpu_clock>::instance()->size();
    auto psize = tim::storage<peak_rss>::instance()->size();

    std::cout << "\nbundle        : " << tim::demangle<bundle_t>() << "\n";
    std::cout << "\n";
    long nfib = 30;
    long ival = 200;
    long nitr = 1000;
    for(long i = 0; i < nitr; ++i)
    {
        bundle_t bundle(details::get_test_name());
        auto     ret = details::fibonacci(nfib);
        if(i % ival == (ival - 1))
            printf("\nfibonacci(%li) = %li\n\n", (long int) nfib, (long int) ret);
    }

    wsize = tim::storage<wall_clock>::instance()->size() - wsize;
    csize = tim::storage<cpu_clock>::instance()->size() - csize;
    psize = tim::storage<peak_rss>::instance()->size() - psize;

    EXPECT_EQ(wsize, 1);
    EXPECT_EQ(csize, 1);
    EXPECT_EQ(psize, 0);
}

//--------------------------------------------------------------------------------------//

TEST_F(component_bundle_tests, get)
{
    tim::trait::runtime_enabled<cpu_roofline<float>>::set(false);
    tim::trait::runtime_enabled<cpu_roofline<double>>::set(false);

    using lhs_t =
        tim::component_bundle_t<TIMEMORY_API, wall_clock, user_clock*, system_clock,
                                cpu_roofline<double>*, gpu_roofline<float>>;
    using rhs_t = tim::auto_bundle_t<TIMEMORY_API, wall_clock*, cpu_clock*,
                                     cpu_roofline<float>*, gpu_roofline<double>>;

    using lhs_data_t = typename lhs_t::data_type;
    using rhs_data_t = typename rhs_t::data_type;
    using rhs_comp_t = typename rhs_t::component_type;

    lhs_t::get_initializer() = [](auto& cl) {
        cl.template initialize<wall_clock, user_clock>();
    };
    rhs_t::get_initializer() = [](auto& cl) {
        cl.template initialize<wall_clock, cpu_clock>();
    };

    auto lhs = lhs_t(TIMEMORY_JOIN("/", details::get_test_name(), "lhs"));
    auto rhs = rhs_t(TIMEMORY_JOIN("/", details::get_test_name(), "rhs"));

    tim::invoke::disjoint::start(std::forward_as_tuple(lhs, rhs));
    tim::invoke::disjoint::mark_begin(std::forward_as_tuple(lhs, rhs));

    std::this_thread::sleep_for(std::chrono::seconds(1));
    details::consume(1000);

    tim::invoke::disjoint::mark_end(std::forward_as_tuple(lhs, rhs));
    tim::invoke::disjoint::stop(std::forward_as_tuple(lhs, rhs));

    auto cb = lhs.get();
    auto ab = rhs.get();

    std::cout << "\n" << std::flush;

    std::cout << "rhs_t      = " << tim::demangle<rhs_t>() << "\n";
    std::cout << "lhs_t      = " << tim::demangle<lhs_t>() << "\n";
    std::cout << "rhs_comp_t = " << tim::demangle<rhs_comp_t>() << "\n";
    std::cout << "\n" << std::flush;

    std::cout << "lhs_data_t = " << tim::demangle<lhs_data_t>() << "\n";
    std::cout << "rhs_data_t = " << tim::demangle<rhs_data_t>() << "\n";
    std::cout << "\n" << std::flush;

    std::cout << "cb         = " << tim::demangle<decltype(cb)>() << "\n";
    std::cout << "ab         = " << tim::demangle<decltype(ab)>() << "\n";
    std::cout << "\n" << std::flush;

    tim::invoke::print(std::cout, lhs, rhs);

    EXPECT_NEAR(std::get<0>(cb), 2.0, 0.1);
    EXPECT_NEAR(std::get<0>(ab), 2.0, 0.1);
    EXPECT_NEAR(std::get<1>(cb) + std::get<2>(cb), 1.0, 0.1);
    EXPECT_NEAR(std::get<1>(ab), 1.0, 0.1);
}

//--------------------------------------------------------------------------------------//
//  these types are available on every OS
//
template <typename ApiT>
struct test
{
    using direct_auto_t =
        tim::auto_bundle<ApiT, wall_clock, cpu_clock, peak_rss, page_rss*, user_clock*>;
    using direct_comp_t = tim::component_bundle<ApiT, wall_clock, cpu_clock, peak_rss,
                                                page_rss*, user_clock*>;
    using derive_auto_t = typename direct_comp_t::auto_type;
    using derive_comp_t = typename direct_auto_t::component_type;

    using direct_auto_data_t = typename direct_auto_t::data_type;
    using direct_comp_data_t = typename direct_comp_t::data_type;
    using derive_auto_data_t = typename derive_auto_t::data_type;
    using derive_comp_data_t = typename derive_comp_t::data_type;
};

template <typename Tp>
constexpr auto
fixed_count(int) -> decltype(Tp::fixed_count(), uint64_t())
{
    return Tp::fixed_count();
}

template <typename Tp>
constexpr auto
fixed_count(long)
{
    return (tim::mpl::get_tuple_size<Tp>::value -
            tim::mpl::get_tuple_size<
                typename tim::get_true_types<std::is_pointer, Tp>::type>::value);
}

template <typename Tp>
constexpr auto
optional_count(int) -> decltype(Tp::optional_count(), uint64_t())
{
    return Tp::optional_count();
}

template <typename Tp>
constexpr auto
optional_count(long)
{
    return (tim::mpl::get_tuple_size<
            typename tim::get_true_types<std::is_pointer, Tp>::type>::value);
}

//--------------------------------------------------------------------------------------//

TEST_F(component_bundle_tests, native_type_fixed_size_check)
{
    using api_t = TIMEMORY_API;
    EXPECT_EQ(fixed_count<test<api_t>::direct_auto_t>(0), 3);
    EXPECT_EQ(fixed_count<test<api_t>::direct_comp_t>(0), 3);
    EXPECT_EQ(fixed_count<test<api_t>::derive_auto_t>(0), 3);
    EXPECT_EQ(fixed_count<test<api_t>::derive_comp_t>(0), 3);
}

//--------------------------------------------------------------------------------------//

TEST_F(component_bundle_tests, native_data_fixed_size_check)
{
    using api_t = TIMEMORY_API;
    EXPECT_EQ(fixed_count<test<api_t>::direct_auto_data_t>(0), 3);
    EXPECT_EQ(fixed_count<test<api_t>::direct_comp_data_t>(0), 3);
    EXPECT_EQ(fixed_count<test<api_t>::derive_auto_data_t>(0), 3);
    EXPECT_EQ(fixed_count<test<api_t>::derive_comp_data_t>(0), 3);
}

//--------------------------------------------------------------------------------------//

TEST_F(component_bundle_tests, native_type_optional_size_check)
{
    using api_t = TIMEMORY_API;
    EXPECT_EQ(optional_count<test<api_t>::direct_auto_t>(0), 2);
    EXPECT_EQ(optional_count<test<api_t>::direct_comp_t>(0), 2);
    EXPECT_EQ(optional_count<test<api_t>::derive_auto_t>(0), 2);
    EXPECT_EQ(optional_count<test<api_t>::derive_comp_t>(0), 2);
}

//--------------------------------------------------------------------------------------//

TEST_F(component_bundle_tests, native_data_optional_size_check)
{
    using api_t = TIMEMORY_API;
    EXPECT_EQ(optional_count<test<api_t>::direct_auto_data_t>(0), 2);
    EXPECT_EQ(optional_count<test<api_t>::direct_comp_data_t>(0), 2);
    EXPECT_EQ(optional_count<test<api_t>::derive_auto_data_t>(0), 2);
    EXPECT_EQ(optional_count<test<api_t>::derive_comp_data_t>(0), 2);
}

//--------------------------------------------------------------------------------------//

TEST_F(component_bundle_tests, native_type_count_wo_init)
{
    using api_t = TIMEMORY_API;
    auto A      = test<api_t>::direct_auto_t(
        TIMEMORY_JOIN("/", details::get_test_name(), "test<TIMEMORY_API>::direct_auto"));
    auto B = test<api_t>::direct_comp_t(
        TIMEMORY_JOIN("/", details::get_test_name(), "test<TIMEMORY_API>::direct_comp"));
    auto C = test<api_t>::derive_auto_t(
        TIMEMORY_JOIN("/", details::get_test_name(), "test<TIMEMORY_API>::derive_auto"));
    auto D = test<api_t>::derive_comp_t(
        TIMEMORY_JOIN("/", details::get_test_name(), "test<TIMEMORY_API>::derive_comp"));

    EXPECT_EQ(A.count(), 3);
    EXPECT_EQ(B.count(), 3);
    EXPECT_EQ(C.count(), 3);
    EXPECT_EQ(D.count(), 3);
}

//--------------------------------------------------------------------------------------//

TEST_F(component_bundle_tests, native_type_count_w_init)
{
    using api_t                                   = TIMEMORY_API;
    test<api_t>::direct_auto_t::get_initializer() = [](auto& cb) {
        // initialize one of two pointers and a type which does not belong
        cb.template initialize<page_rss, system_clock>();
    };

    auto A = test<api_t>::direct_auto_t(
        TIMEMORY_JOIN("/", details::get_test_name(), "test<TIMEMORY_API>::direct_auto"));
    auto C = test<api_t>::derive_auto_t(
        TIMEMORY_JOIN("/", details::get_test_name(), "test<TIMEMORY_API>::derive_auto"));

    test<api_t>::derive_comp_t::get_initializer() = [](auto& cb) {
        // initialize both pointers
        cb.template initialize<page_rss, user_clock>();
    };

    auto B = test<api_t>::direct_comp_t(
        TIMEMORY_JOIN("/", details::get_test_name(), "test<TIMEMORY_API>::direct_comp"));
    auto D = test<api_t>::derive_comp_t(
        TIMEMORY_JOIN("/", details::get_test_name(), "test<TIMEMORY_API>::derive_comp"));

    EXPECT_EQ(A.count(), 4);
    EXPECT_EQ(B.count(), 5);
    EXPECT_EQ(C.count(), 4);
    EXPECT_EQ(D.count(), 5);

    test<api_t>::derive_auto_t::get_initializer() = [](auto&) {};
    test<api_t>::direct_comp_t::get_initializer() = [](auto&) {};
}

//--------------------------------------------------------------------------------------//

// declare a new API
TIMEMORY_DEFINE_API(custom_tag)
// make the API unavailable
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, api::custom_tag, false_type)
// macro for API
#define CUSTOM_API tim::api::custom_tag

//--------------------------------------------------------------------------------------//

TEST_F(component_bundle_tests, custom_type_fixed_size_check)
{
    using api_t = CUSTOM_API;
    EXPECT_EQ(fixed_count<test<api_t>::direct_auto_t>(0), 0);
    EXPECT_EQ(fixed_count<test<api_t>::direct_comp_t>(0), 0);
    EXPECT_EQ(fixed_count<test<api_t>::derive_auto_t>(0), 0);
    EXPECT_EQ(fixed_count<test<api_t>::derive_comp_t>(0), 0);
}

//--------------------------------------------------------------------------------------//

TEST_F(component_bundle_tests, custom_data_fixed_size_check)
{
    using api_t = CUSTOM_API;
    EXPECT_EQ(fixed_count<test<api_t>::direct_auto_data_t>(0), 0);
    EXPECT_EQ(fixed_count<test<api_t>::direct_comp_data_t>(0), 0);
    EXPECT_EQ(fixed_count<test<api_t>::derive_auto_data_t>(0), 0);
    EXPECT_EQ(fixed_count<test<api_t>::derive_comp_data_t>(0), 0);
}

//--------------------------------------------------------------------------------------//

TEST_F(component_bundle_tests, custom_type_optional_size_check)
{
    using api_t = CUSTOM_API;
    EXPECT_EQ(optional_count<test<api_t>::direct_auto_t>(0), 0);
    EXPECT_EQ(optional_count<test<api_t>::direct_comp_t>(0), 0);
    EXPECT_EQ(optional_count<test<api_t>::derive_auto_t>(0), 0);
    EXPECT_EQ(optional_count<test<api_t>::derive_comp_t>(0), 0);
}

//--------------------------------------------------------------------------------------//

TEST_F(component_bundle_tests, custom_data_optional_size_check)
{
    using api_t = CUSTOM_API;
    EXPECT_EQ(optional_count<test<api_t>::direct_auto_data_t>(0), 0);
    EXPECT_EQ(optional_count<test<api_t>::direct_comp_data_t>(0), 0);
    EXPECT_EQ(optional_count<test<api_t>::derive_auto_data_t>(0), 0);
    EXPECT_EQ(optional_count<test<api_t>::derive_comp_data_t>(0), 0);
}

//--------------------------------------------------------------------------------------//

TEST_F(component_bundle_tests, custom_type_count_wo_init)
{
    using api_t = CUSTOM_API;
    auto A      = test<api_t>::direct_auto_t(
        TIMEMORY_JOIN("/", details::get_test_name(), "test<CUSTOM_API>::direct_auto"));
    auto B = test<api_t>::direct_comp_t(
        TIMEMORY_JOIN("/", details::get_test_name(), "test<CUSTOM_API>::direct_comp"));
    auto C = test<api_t>::derive_auto_t(
        TIMEMORY_JOIN("/", details::get_test_name(), "test<CUSTOM_API>::derive_auto"));
    auto D = test<api_t>::derive_comp_t(
        TIMEMORY_JOIN("/", details::get_test_name(), "test<CUSTOM_API>::derive_comp"));

    EXPECT_EQ(A.count(), 0);
    EXPECT_EQ(B.count(), 0);
    EXPECT_EQ(C.count(), 0);
    EXPECT_EQ(D.count(), 0);
}

//--------------------------------------------------------------------------------------//

TEST_F(component_bundle_tests, custom_type_count_w_init)
{
    using api_t                                   = CUSTOM_API;
    test<api_t>::direct_auto_t::get_initializer() = [](auto& cb) {
        // initialize one of two pointers and a type which does not belong
        cb.template initialize<page_rss, system_clock>();
    };

    auto A = test<api_t>::direct_auto_t(
        TIMEMORY_JOIN("/", details::get_test_name(), "test<api_t>::direct_auto"));
    auto C = test<api_t>::derive_auto_t(
        TIMEMORY_JOIN("/", details::get_test_name(), "test<CUSTOM_API>::derive_auto"));

    test<api_t>::derive_comp_t::get_initializer() = [](auto& cb) {
        // initialize both pointers
        cb.template initialize<page_rss, user_clock>();
    };

    auto B = test<api_t>::direct_comp_t(
        TIMEMORY_JOIN("/", details::get_test_name(), "test<CUSTOM_API>::direct_comp"));
    auto D = test<api_t>::derive_comp_t(
        TIMEMORY_JOIN("/", details::get_test_name(), "test<CUSTOM_API>::derive_comp"));

    EXPECT_EQ(A.count(), 0);
    EXPECT_EQ(B.count(), 0);
    EXPECT_EQ(C.count(), 0);
    EXPECT_EQ(D.count(), 0);

    test<api_t>::derive_auto_t::get_initializer() = [](auto&) {};
    test<api_t>::direct_comp_t::get_initializer() = [](auto&) {};
}

//--------------------------------------------------------------------------------------//

int
main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    _argc    = argc;
    _argv    = argv;
    auto ret = RUN_ALL_TESTS();
    tim::timemory_finalize();
    tim::dmp::finalize();
    return ret;
}

//--------------------------------------------------------------------------------------//
