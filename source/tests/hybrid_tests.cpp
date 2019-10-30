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

#include <timemory/timemory.hpp>

#include <chrono>
#include <condition_variable>
#include <functional>
#include <iostream>
#include <limits>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using namespace tim::component;
using mutex_t        = std::mutex;
using lock_t         = std::unique_lock<mutex_t>;
using string_t       = std::string;
using stringstream_t = std::stringstream;

using tuple_t = tim::component_tuple<real_clock, cpu_clock, cpu_util, peak_rss>;
using list_t  = tim::component_list<real_clock, cpu_clock, cpu_util, peak_rss, page_rss,
                                   papi_array_t, cuda_event, cupti_counters, caliper>;
using auto_hybrid_t = tim::auto_hybrid<tuple_t, list_t>;
using hybrid_t      = typename auto_hybrid_t::component_type;

static const int64_t niter       = 20;
static const int64_t nelements   = 0.95 * (tim::units::get_page_size() * 500);
static const auto    memory_unit = std::pair<int64_t, string_t>(tim::units::KiB, "KiB");
// static auto          tot_size    = nelements * sizeof(int64_t) / memory_unit.first;

// acceptable absolute error
static const double util_tolerance  = 2.5;
static const double timer_tolerance = 0.02125;
// static const double peak_tolerance  = 5 * tim::units::MiB;
// acceptable relative error
static const double util_epsilon  = 0.5;
static const double timer_epsilon = 0.02;
// acceptable compose error
static const double compose_tolerance = 1.0e-9;

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
inline void
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

// this function ensures an allocation cannot be optimized
template <typename _Tp>
inline size_t
random_entry(const std::vector<_Tp>& v)
{
    std::mt19937 rng;
    rng.seed(std::random_device()());
    std::uniform_int_distribution<std::mt19937::result_type> dist(0, v.size() - 1);
    return v.at(dist(rng));
}

inline void
allocate()
{
    std::vector<int64_t> v(nelements, 15);
    auto                 ret  = fibonacci(0);
    long                 nfib = details::random_entry(v);
    for(int64_t i = 0; i < niter; ++i)
    {
        nfib = details::random_entry(v);
        ret += details::fibonacci(nfib);
    }
    printf("fibonacci(%li) * %li = %li\n", (long) nfib, (long) niter, ret);
}

template <typename _Tp, typename _Up, typename _Vp = typename _Tp::value_type,
          typename _Func = std::function<_Vp(_Vp)>>
inline void
print_info(const _Tp& obj, const _Up& expected, const string_t& unit,
           _Func _func = [](const _Vp& _obj) { return _obj; })
{
    std::cout << std::endl;
    std::cout << "[" << get_test_name() << "]>  measured : " << obj << std::endl;
    std::cout << "[" << get_test_name() << "]>  expected : " << expected << " " << unit
              << std::endl;
    std::cout << "[" << get_test_name() << "]>     value : " << _func(obj.get_value())
              << std::endl;
    std::cout << "[" << get_test_name() << "]>     accum : " << _func(obj.get_accum())
              << std::endl;
}

}  // namespace details

//--------------------------------------------------------------------------------------//

class hybrid_tests : public ::testing::Test
{
protected:
    void SetUp() override
    {
#if defined(TIMEMORY_USE_PAPI)
        papi_array_t::get_initializer() = []() {
            return std::vector<int>({ PAPI_TOT_CYC, PAPI_LST_INS });
        };
#endif
        list_t::get_initializer() = [](list_t& l) {
            l.initialize<real_clock, cpu_clock, cpu_util, peak_rss, page_rss,
                         papi_array_t, caliper>();
        };
    }
};

//--------------------------------------------------------------------------------------//

TEST_F(hybrid_tests, type_check)
{
    using list_type = typename hybrid_t::list_type;
    printf("\n");
    std::cout << "list_t    = " << tim::demangle<list_t>() << std::endl;
    std::cout << "list_type = " << tim::demangle<list_type>() << std::endl;
    printf("\n");
    ASSERT_TRUE((std::is_same<list_type, list_t>::value));
}

//--------------------------------------------------------------------------------------//

TEST_F(hybrid_tests, hybrid)
{
    hybrid_t obj(details::get_test_name());

    obj.start();
    std::thread t(details::consume, 500);
    details::do_sleep(250);
    details::consume(750);
    t.join();
    obj.stop();
    std::cout << "\n" << obj << std::endl;

    auto clock_convert    = [](const int64_t& _obj) { return _obj; };
    auto cpu_util_convert = [](const std::pair<int64_t, int64_t>& val) {
        return static_cast<double>(val.first) / val.second * 100.0;
    };

    auto& t_rc   = obj.get_tuple().get<real_clock>();
    auto& t_cpu  = obj.get_tuple().get<cpu_clock>();
    auto& t_util = obj.get_tuple().get<cpu_util>();

    details::print_info(t_rc, 1.0, "sec", clock_convert);
    details::print_info(t_cpu, 1.25, "sec", clock_convert);
    details::print_info(t_util, 125.0, "%", cpu_util_convert);

    ASSERT_NEAR(1.0, t_rc.get(), timer_tolerance);
    ASSERT_NEAR(1.25, t_cpu.get(), timer_tolerance);
    ASSERT_NEAR(125.0, t_util.get(), util_tolerance);

    auto* l_rc   = obj.get_list().get<real_clock>();
    auto* l_cpu  = obj.get_list().get<cpu_clock>();
    auto* l_util = obj.get_list().get<cpu_util>();

    ASSERT_TRUE(l_rc != nullptr);
    ASSERT_TRUE(l_cpu != nullptr);
    ASSERT_TRUE(l_util != nullptr);

    details::print_info(*l_rc, 1.0, "sec", clock_convert);
    details::print_info(*l_cpu, 1.25, "sec", clock_convert);
    details::print_info(*l_util, 125.0, "%", cpu_util_convert);

    ASSERT_NEAR(t_rc.get(), l_rc->get(), timer_epsilon);
    ASSERT_NEAR(t_cpu.get(), l_cpu->get(), timer_epsilon);
    ASSERT_NEAR(t_util.get(), l_util->get(), util_epsilon);
}

//--------------------------------------------------------------------------------------//

TEST_F(hybrid_tests, auto_timer)
{
    tim::auto_timer obj(details::get_test_name());
    std::thread     t(details::consume, 500);
    details::do_sleep(250);
    details::consume(750);
    t.join();
    obj.stop();
    std::cout << "\n" << obj << std::endl;

    auto clock_convert    = [](const int64_t& _obj) { return _obj; };
    auto cpu_util_convert = [](const std::pair<int64_t, int64_t>& val) {
        return static_cast<double>(val.first) / val.second * 100.0;
    };

    auto  _cpu  = obj.get_lhs().get<user_clock>() + obj.get_lhs().get<system_clock>();
    auto& _rc   = obj.get_lhs().get<real_clock>();
    auto& _util = obj.get_lhs().get<cpu_util>();

    details::print_info(_rc, 1.0, "sec", clock_convert);
    details::print_info(_cpu, 1.25, "sec", clock_convert);
    details::print_info(_util, 125.0, "%", cpu_util_convert);

    ASSERT_NEAR(1.0, _rc.get(), timer_tolerance);
    ASSERT_NEAR(1.25, _cpu.get(), timer_tolerance);
    ASSERT_NEAR(125.0, _util.get(), util_tolerance);

    auto _cpu2 = obj.get<user_clock>() + obj.get<system_clock>();
    ASSERT_NEAR(1.0e-9, _cpu.get(), _cpu2.get());

    cpu_clock _cpu_obj = obj.get<user_clock>() + obj.get<system_clock>();
    double    _cpu_val = obj.get<user_clock>().get() + obj.get<system_clock>().get();
    ASSERT_NEAR(_cpu_obj.get(), _cpu_val, compose_tolerance);
    details::print_info(_cpu_obj, _cpu_val, "sec");

    auto _obj  = tim::get(obj);
    auto _cpu3 = std::get<1>(_obj) + std::get<2>(_obj);

    ASSERT_NEAR(1.0e-9, _cpu.get(), _cpu3);
}

//--------------------------------------------------------------------------------------//

TEST_F(hybrid_tests, compose)
{
    using bundle_t = tim::component_tuple<user_clock, system_clock>;
    using result_t = std::tuple<double, double>;

    bundle_t obj(details::get_test_name());
    obj.start();
    details::do_sleep(250);  // in millseconds
    details::consume(750);   // in millseconds
    obj.stop();
    std::cout << "\n" << obj << std::endl;

    result_t  _cpu_ret = obj.get();
    cpu_clock _cpu_obj = obj.get<user_clock>() + obj.get<system_clock>();
    double    _cpu_val = obj.get<user_clock>().get() + obj.get<system_clock>().get();

    details::print_info(_cpu_obj, 0.75, "sec");

    ASSERT_NEAR(0.75, _cpu_val, timer_tolerance);
    ASSERT_NEAR(_cpu_obj.get(), _cpu_val, compose_tolerance);
    ASSERT_NEAR(_cpu_val, std::get<0>(_cpu_ret) + std::get<1>(_cpu_ret),
                compose_tolerance);

    printf("\n");
}

//--------------------------------------------------------------------------------------//

int
main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    tim::settings::timing_units() = "sec";
    tim::settings::memory_units() = "KiB";
    tim::settings::precision()    = 6;
    tim::timemory_init(argc, argv);
    tim::settings::verbose() += 1;
    tim::settings::debug()       = true;
    tim::settings::dart_output() = true;
    tim::settings::dart_count()  = 1;
    tim::settings::banner()      = false;

    return RUN_ALL_TESTS();
}

//--------------------------------------------------------------------------------------//
