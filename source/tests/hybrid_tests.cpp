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
using condvar_t      = std::condition_variable;
using string_t       = std::string;
using stringstream_t = std::stringstream;

using tuple_t = tim::component_tuple<real_clock, cpu_clock, cpu_util, peak_rss>;
using list_t = tim::component_list<real_clock, cpu_clock, cpu_util, peak_rss, current_rss,
                                   papi_array_t, cuda_event, cupti_event, caliper>;
using auto_hybrid_t = tim::auto_hybrid<tuple_t, list_t>;
using hybrid_t      = auto_hybrid_t::component_type;

static const int64_t niter       = 20;
static const int64_t nelements   = 0.95 * (tim::units::get_page_size() * 500);
static const auto    memory_unit = std::pair<int64_t, string_t>(tim::units::KiB, "KiB");
static auto          tot_size    = nelements * sizeof(int64_t) / memory_unit.first;

static const float util_epsilon    = 1.0e-1;
static const float util_tolerance  = 2.5;
static const float timer_epsilon   = 2.5e-3;
static const float timer_tolerance = 0.01;
static const float peak_tolerance  = 5 * tim::units::MiB;

#define CHECK_AVAILABLE(type)                                                            \
    if(!tim::trait::is_available<type>::value)                                           \
        return;

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
void
do_sleep(long n)
{
    std::this_thread::sleep_for(std::chrono::milliseconds(n));
}

// this function consumes an unknown number of cpu resources
long
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
    auto now = std::chrono::system_clock::now();
    // get elapsed
    auto until = now + std::chrono::milliseconds(n);
    // try until time point
    while(std::chrono::system_clock::now() < until)
        try_lk.try_lock();
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

template <typename _Tp, typename _Func>
string_t
get_info(const _Tp& obj, _Func&& _func)
{
    stringstream_t ss;
    auto           _unit = static_cast<double>(_Tp::get_unit());
    ss << "value = " << _func(obj.get_value()) / _unit << " " << _Tp::get_display_unit()
       << ", accum = " << _func(obj.get_accum()) / _unit << " " << _Tp::get_display_unit()
       << std::endl;
    return ss.str();
}

template <typename _Tp, typename _Up, typename _Vp = typename _Tp::value_type,
          typename _Func = std::function<_Vp(_Vp)>>
void
print_info(
    const _Tp& obj, const _Up& expected, string_t unit,
    _Func _func = [](const _Vp& obj) { return obj; })
{
    std::cout << std::endl;
    std::cout << "[" << get_test_name() << "]>  measured : " << obj << std::endl;
    std::cout << "[" << get_test_name() << "]>  expected : " << expected << " " << unit
              << std::endl;
    std::cout << "[" << get_test_name() << "]> data info : " << get_info(obj, _func)
              << std::endl;
}

}  // namespace details

//--------------------------------------------------------------------------------------//

class hybrid_tests : public ::testing::Test
{
};

//--------------------------------------------------------------------------------------//

TEST_F(hybrid_tests, hybrid)
{
    hybrid_t obj(details::get_test_name());
    obj.start();
    std::thread t(details::consume, 1000);
    details::consume(500);
    details::do_sleep(500);
    t.join();
    obj.stop();
    std::cout << "\n" << obj << std::endl;

    auto clock_convert    = [](const int64_t& obj) { return obj; };
    auto cpu_util_convert = [](const std::pair<int64_t, int64_t>& val) {
        return static_cast<double>(val.first) / val.second * 100.0;
    };

    details::print_info(obj.get_tuple().get<real_clock>(), 1.0, "sec", clock_convert);
    details::print_info(obj.get_tuple().get<cpu_clock>(), 1.5, "sec", clock_convert);
    details::print_info(obj.get_tuple().get<cpu_util>(), 150.0, "%", cpu_util_convert);

    ASSERT_NEAR(1.0, obj.get_tuple().get<real_clock>().get(), timer_tolerance);
    ASSERT_NEAR(1.5, obj.get_tuple().get<cpu_clock>().get(), timer_tolerance);
    ASSERT_NEAR(150.0, obj.get_tuple().get<cpu_util>().get(), util_tolerance);

    ASSERT_NEAR(obj.get_tuple().get<real_clock>().get(),
                obj.get_list().get<real_clock>()->get(), timer_epsilon);
    ASSERT_NEAR(obj.get_tuple().get<cpu_clock>().get(),
                obj.get_list().get<cpu_clock>()->get(), timer_epsilon);
    ASSERT_NEAR(obj.get_tuple().get<cpu_util>().get(),
                obj.get_list().get<cpu_util>()->get(), util_epsilon);

    obj.start();
    details::allocate();
    obj.stop();
    std::cout << obj << std::endl;

    details::print_info(obj.get_tuple().get<peak_rss>(), tot_size, "KiB");
    ASSERT_NEAR(tot_size, obj.get_tuple().get<peak_rss>().get(), peak_tolerance);
}

//--------------------------------------------------------------------------------------//

TEST_F(hybrid_tests, auto_timer)
{
    tim::auto_timer obj(details::get_test_name());
    std::thread     t(details::consume, 1000);
    details::consume(500);
    details::do_sleep(500);
    t.join();
    obj.stop();
    std::cout << "\n" << obj << std::endl;

    auto clock_convert    = [](const int64_t& obj) { return obj; };
    auto cpu_util_convert = [](const std::pair<int64_t, int64_t>& val) {
        return static_cast<double>(val.first) / val.second * 100.0;
    };

    details::print_info(obj.get_lhs().get<real_clock>(), 1.0, "sec", clock_convert);
    details::print_info(obj.get_lhs().get<cpu_clock>(), 1.5, "sec", clock_convert);
    details::print_info(obj.get_lhs().get<cpu_util>(), 150.0, "%", cpu_util_convert);

    ASSERT_NEAR(1.0, obj.get_lhs().get<real_clock>().get(), timer_tolerance);
    ASSERT_NEAR(1.5, obj.get_lhs().get<cpu_clock>().get(), timer_tolerance);
    ASSERT_NEAR(150.0, obj.get_lhs().get<cpu_util>().get(), util_tolerance);

    obj.start();
    details::allocate();
    obj.stop();
    std::cout << obj << std::endl;

    details::print_info(obj.get_lhs().get<peak_rss>(), tot_size, "KiB");
    // ASSERT_NEAR(tot_size, obj.get<peak_rss>().get(), peak_tolerance);
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
    tim::settings::file_output() = false;
    tim::settings::verbose() += 1;

    list_t::get_initializer() = [](list_t& l) {
        l.initialize<real_clock, cpu_clock, cpu_util, peak_rss, current_rss, papi_array_t,
                     cuda_event, cupti_event, caliper>();
    };

    return RUN_ALL_TESTS();
}

//--------------------------------------------------------------------------------------//
