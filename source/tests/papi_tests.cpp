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
#include <x86intrin.h>

using namespace tim::component;
using mutex_t   = std::mutex;
using lock_t    = std::unique_lock<mutex_t>;
using condvar_t = std::condition_variable;

static constexpr uint64_t FLOPS  = 16;
static constexpr uint64_t TRIALS = 2;
static constexpr uint64_t SIZE   = 1024 * 1024 * 64;
// tolerance of 5.0e-5
static const int64_t ops_tolerance = (TRIALS * SIZE * FLOPS) / 2000;
static const int64_t lst_tolerance = (TRIALS * SIZE * FLOPS) / 2000;

#define CHECK_WORKING()                                                                  \
    if(!tim::papi::working())                                                            \
    {                                                                                    \
        printf("Skipping test because PAPI is not working\n");                           \
        return;                                                                          \
    }

#define CHECK_AVAILABLE(type)                                                            \
    CHECK_WORKING();                                                                     \
    if(!tim::trait::is_available<type>::value)                                           \
    {                                                                                    \
        printf("Skipping test because either component is not available\n");             \
        return;                                                                          \
    }

template <typename _Tp>
using ptr_t = std::shared_ptr<_Tp>;
template <typename _Tp>
using return_type = std::tuple<ptr_t<_Tp>, int64_t, int64_t>;

namespace details
{
//--------------------------------------------------------------------------------------//
template <typename _Tp, int64_t _Unroll, typename _Component, typename... _Args>
return_type<_Component>
run_cpu_ops_kernel(int64_t ntrials, int64_t nsize, _Args&&... _args)
{
    // auto op_func = [](_Tp& a, const _Tp& b, const _Tp& c) { a = b + c; };
    auto op_func           = [](_Tp& a, const _Tp& b, const _Tp& c) { a = a * b + c; };
    auto store_func        = [](_Tp& a, const _Tp& b) { a = b; };
    auto bytes_per_elem = sizeof(_Tp);
    auto vec_size = TIMEMORY_VEC / 16;
    auto mem_access_per_elem = 2;

    int64_t nops         = _Unroll;
    int64_t working_size = nsize * ntrials;
    int64_t total_bytes  = working_size * bytes_per_elem * mem_access_per_elem / vec_size;
    int64_t total_ops    = working_size * nops;

    std::cout << "              Nops = " << nops << "\n";
    std::cout << "           ntrials = " << ntrials << "\n";
    std::cout << " bytes per element = " << bytes_per_elem << "\n";
    std::cout << " accesses per elem = " << mem_access_per_elem << "\n";
    std::cout << "      working size = " << working_size << "\n";
    std::cout << "       total bytes = " << total_bytes << "\n";
    std::cout << "  total operations = " << total_ops << "\n";

    //_Tp* array = new _Tp[nsize];
    std::vector<_Tp, tim::ert::aligned_allocator<_Tp, 64>> array(nsize);
    std::memset(array.data(), 0, nsize * sizeof(_Tp));

    _Component::invoke_thread_init();

    using pointer = ptr_t<_Component>;
    pointer obj   = pointer(new _Component(std::forward<_Args>(_args)...));

    obj->start();
    tim::ert::cpu_ops_kernel<_Unroll>(ntrials, op_func, store_func, nsize, array.data());
    obj->stop();

    _Component::invoke_thread_finalize();

    // return zeros if not working
    if(!tim::papi::working())
    {
        std::cout << "\n\nPAPI is not working so returning zeros instead of total_ops = "
                  << total_ops << " and total_bytes = " << total_bytes << "\n\n"
                  << std::endl;
        return return_type<_Component>(obj, 0, 0);
    }
    return return_type<_Component>(obj, total_ops, total_bytes);
}
//--------------------------------------------------------------------------------------//
inline std::string
get_test_name()
{
    return ::testing::UnitTest::GetInstance()->current_test_info()->name();
}
//--------------------------------------------------------------------------------------//
template <typename _Tp>
void
report(const _Tp& measured_count, const _Tp& explicit_count, const _Tp& tolerance)
{
    _Tp    diff = measured_count - explicit_count;
    double err  = (diff / static_cast<double>(explicit_count)) * 100.0;
    double ratio = static_cast<double>(explicit_count) / measured_count;
    std::cout << get_test_name() << std::endl;
    std::cout << "    Measured:  " << measured_count << std::endl;
    std::cout << "    Expected:  " << explicit_count << std::endl;
    std::cout << "   Tolerance:  " << tolerance << std::endl;
    std::cout << "  Difference:  " << diff << std::endl;
    std::cout << "       Ratio:  " << ratio << std::endl;
    std::cout << "   Abs Error:  " << err << " %" << std::endl;
}
//--------------------------------------------------------------------------------------//
}  // namespace details

//--------------------------------------------------------------------------------------//

class papi_tests : public ::testing::Test
{
};

//--------------------------------------------------------------------------------------//
/*
TEST_F(papi_tests, working)
{
    using test_type = papi_tuple<PAPI_TOT_CYC>;
    CHECK_AVAILABLE(test_type);

    // simple test to see if PAPI is enabled but doesn't have sufficient permissions.
    // Let this always succeed...
    test_type::invoke_thread_init();
    test_type::invoke_thread_finalize();
    SUCCEED();
}
*/
//--------------------------------------------------------------------------------------//

TEST_F(papi_tests, tuple_single_precision_ops)
{
    using test_type = papi_tuple<PAPI_SP_OPS>;
    CHECK_AVAILABLE(test_type);

    auto ret = details::run_cpu_ops_kernel<float, FLOPS, test_type>(TRIALS, SIZE);

    auto obj            = std::get<0>(ret);
    auto total_expected = std::get<1>(ret);
    auto total_measured = obj->get<int64_t>()[0];

    int idx = 0;
    for(auto itr : test_type::get_overhead())
        std::cout << "Overhead for counter " << idx++ << " is " << itr << std::endl;

    details::report(total_measured, total_expected, ops_tolerance);
    if(std::abs<int64_t>(total_measured - total_expected) < ops_tolerance)
        SUCCEED();
    else
        FAIL();
}

//--------------------------------------------------------------------------------------//

TEST_F(papi_tests, array_single_precision_ops)
{
    using test_type = papi_array_t;
    CHECK_AVAILABLE(test_type);

    papi_array_t::get_events_func() = []() { return std::vector<int>({ PAPI_SP_OPS }); };
    auto ret = details::run_cpu_ops_kernel<float, FLOPS, test_type>(TRIALS, SIZE);

    auto obj            = std::get<0>(ret);
    auto total_expected = std::get<1>(ret);
    auto total_measured = obj->get<int64_t>()[0];

    details::report(total_measured, total_expected, ops_tolerance);
    if(std::abs<int64_t>(total_measured - total_expected) < ops_tolerance)
        SUCCEED();
    else
        FAIL();
}

//--------------------------------------------------------------------------------------//

TEST_F(papi_tests, tuple_double_precision_ops)
{
    using test_type = papi_tuple<PAPI_DP_OPS>;
    CHECK_AVAILABLE(test_type);

    auto ret = details::run_cpu_ops_kernel<double, FLOPS, test_type>(TRIALS, SIZE);

    auto obj            = std::get<0>(ret);
    auto total_expected = std::get<1>(ret);
    auto total_measured = obj->get<int64_t>()[0];

    int idx = 0;
    for(auto itr : test_type::get_overhead())
        std::cout << "Overhead for counter " << idx++ << " is " << itr << std::endl;

    details::report(total_measured, total_expected, ops_tolerance);
    if(std::abs<int64_t>(total_measured - total_expected) < ops_tolerance)
        SUCCEED();
    else
        FAIL();
}

//--------------------------------------------------------------------------------------//

TEST_F(papi_tests, array_double_precision_ops)
{
    using test_type = papi_array_t;
    CHECK_AVAILABLE(test_type);

    papi_array_t::get_events_func() = []() { return std::vector<int>({ PAPI_DP_OPS }); };
    auto ret = details::run_cpu_ops_kernel<double, FLOPS, test_type>(TRIALS, SIZE);

    auto obj            = std::get<0>(ret);
    auto total_expected = std::get<1>(ret);
    auto total_measured = obj->get<int64_t>()[0];

    details::report(total_measured, total_expected, ops_tolerance);
    if(std::abs<int64_t>(total_measured - total_expected) < ops_tolerance)
        SUCCEED();
    else
        FAIL();
}

//--------------------------------------------------------------------------------------//

TEST_F(papi_tests, tuple_load_store_ins)
{
    using test_type = papi_tuple<PAPI_LST_INS>;
    CHECK_AVAILABLE(test_type);

    auto ret = details::run_cpu_ops_kernel<double, FLOPS, test_type>(TRIALS, SIZE);

    auto obj            = std::get<0>(ret);
    auto total_expected = std::get<2>(ret);
    auto total_measured = obj->get<int64_t>()[0];

    int idx = 0;
    for(auto itr : test_type::get_overhead())
        std::cout << "Overhead for counter " << idx++ << " is " << itr << std::endl;

    details::report(total_measured, total_expected, ops_tolerance);
    if(std::abs<int64_t>(total_measured - total_expected) < ops_tolerance)
        SUCCEED();
    else
        FAIL();
}

//--------------------------------------------------------------------------------------//

TEST_F(papi_tests, array_load_store_ins)
{
    using test_type = papi_array_t;
    CHECK_AVAILABLE(test_type);

    papi_array_t::get_events_func() = []() { return std::vector<int>({ PAPI_LST_INS }); };
    auto ret = details::run_cpu_ops_kernel<float, FLOPS, test_type>(TRIALS, SIZE);

    auto obj            = std::get<0>(ret);
    auto total_expected = std::get<2>(ret);
    auto total_measured = obj->get<int64_t>()[0];

    details::report(total_measured, total_expected, lst_tolerance);
    if(std::abs<int64_t>(total_measured - total_expected) < lst_tolerance)
        SUCCEED();
    else
        FAIL();
}

//--------------------------------------------------------------------------------------//

int
main(int argc, char** argv)
{
    tim::papi::init();
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

//--------------------------------------------------------------------------------------//
