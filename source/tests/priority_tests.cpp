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
#include <mutex>
#include <random>
#include <thread>
#include <vector>

#include <timemory/timemory.hpp>

using mutex_t = std::mutex;
using lock_t  = std::unique_lock<mutex_t>;

using namespace tim::component;

// make different types to access and change traits individually

template <size_t Idx>
struct test_clock : public base<test_clock<Idx>>
{
    using ratio_t    = std::nano;
    using value_type = int64_t;
    using this_type  = test_clock<Idx>;
    using base_type  = base<test_clock<Idx>, value_type>;
    using string_t   = std::string;

    // since this is a template class, need these statements
    using base_type::accum;
    using base_type::is_transient;
    using base_type::set_started;
    using base_type::set_stopped;
    using base_type::value;

    static const short                   precision    = wall_clock::precision;
    static const short                   width        = wall_clock::width;
    static const std::ios_base::fmtflags format_flags = wall_clock::format_flags;

    static int64_t    unit() { return wall_clock::unit(); }
    static string_t   label() { return string_t("test_clock_") + std::to_string(Idx); }
    static string_t   description() { return "wall time"; }
    static string_t   display_unit() { return wall_clock::display_unit(); }
    static value_type record() { return wall_clock::record(); }

    double get_display() const
    {
        auto val = (is_transient) ? accum : value;
        return static_cast<double>(val / static_cast<double>(ratio_t::den) *
                                   wall_clock::get_unit());
    }

    double get() const { return get_display(); }

    void start()
    {
        set_started();
        value = record();
    }

    void stop()
    {
        auto tmp = record();
        accum += (tmp - value);
        value = std::move(tmp);
        set_stopped();
    }
};

using priority_start_wc = test_clock<0>;
using priority_stop_wc  = test_clock<1>;

//--------------------------------------------------------------------------------------//

namespace tim
{
namespace trait
{
template <>
struct is_timing_category<priority_start_wc> : public std::true_type
{};
template <>
struct is_timing_category<priority_stop_wc> : public std::true_type
{};
template <>
struct uses_timing_units<priority_start_wc> : public std::true_type
{};
template <>
struct uses_timing_units<priority_stop_wc> : public std::true_type
{};
template <>
struct start_priority<priority_start_wc> : public std::true_type
{};
template <>
struct stop_priority<priority_stop_wc> : public std::true_type
{};
}  // namespace trait
}  // namespace tim

//--------------------------------------------------------------------------------------//

using tuple_t =
    tim::component_tuple<wall_clock, cpu_clock, priority_start_wc, priority_stop_wc>;

using prior_start_t = tuple_t::prior_start_t;
using prior_stop_t  = tuple_t::prior_stop_t;
using stand_start_t = tuple_t::stand_start_t;
using stand_stop_t  = tuple_t::stand_stop_t;

using apply_v = tim::apply<void>;

//--------------------------------------------------------------------------------------//
// this function consumes approximately "n" milliseconds of wall time
//
#define do_sleep(N) std::this_thread::sleep_for(std::chrono::milliseconds(N))

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
}  // namespace details

//--------------------------------------------------------------------------------------//

class priority_tests : public ::testing::Test
{
protected:
    void SetUp() override
    {
        priority_start_wc::get_label() = "priority-start-clock";
        priority_stop_wc::get_label()  = "priority-stop-clock";
    }
};

//--------------------------------------------------------------------------------------//

TEST_F(priority_tests, simple_check)
{
    tuple_t t(details::get_test_name(), true);

    // start/stop all to check laps
    t.start();
    t.stop();

    t.start();

    do_sleep(250);  // TOTAL TIME: 0.25 seconds

    // priority_start(t);

    do_sleep(250);  // TOTAL TIME: 0.50 seconds

    // standard_start(t);

    do_sleep(500);  // TOTAL TIME: 1.00 seconds

    // priority_stop(t);

    do_sleep(125);  // TOTAL TIME: 1.125 seconds

    // standard_stop(t);

    do_sleep(125);  // TOTAL TIME: 1.25 seconds

    t.stop();

    // t.start();
    // details::consume(500);
    // t.stop();

    auto& native_wc = t.get<wall_clock>();
    auto& pstart_wc = t.get<priority_start_wc>();
    auto& pstop_wc  = t.get<priority_stop_wc>();

    printf("\n");
    std::cout << native_wc << std::endl;
    std::cout << pstart_wc << std::endl;
    std::cout << pstop_wc << std::endl;
    printf("\n");
    std::cout << t << std::endl;
    printf("\n");

    // each start()/stop() on tuple_t resets internal lap count to zero
    ASSERT_EQ(native_wc.nlaps(), 1);
    ASSERT_EQ(pstart_wc.nlaps(), 1);
    ASSERT_EQ(pstop_wc.nlaps(), 1);
    // tuple_t will have total number of start()/stop()
    ASSERT_EQ(t.laps(), 2);
}

//--------------------------------------------------------------------------------------//

TEST_F(priority_tests, start_stop)
{
    // lambdas to ensure inline
    auto priority_start = [](tuple_t& t) { apply_v::access<prior_start_t>(t.data()); };
    auto priority_stop  = [](tuple_t& t) { apply_v::access<prior_stop_t>(t.data()); };
    auto standard_start = [](tuple_t& t) { apply_v::access<stand_start_t>(t.data()); };
    auto standard_stop  = [](tuple_t& t) { apply_v::access<stand_stop_t>(t.data()); };

    tuple_t t(details::get_test_name(), true);

    // start/stop all to check laps
    t.start();
    t.stop();

    t.get<wall_clock>().start();

    do_sleep(250);  // TOTAL TIME: 0.25 seconds

    priority_start(t);

    do_sleep(250);  // TOTAL TIME: 0.50 seconds

    standard_start(t);

    do_sleep(500);  // TOTAL TIME: 1.00 seconds

    priority_stop(t);

    do_sleep(125);  // TOTAL TIME: 1.125 seconds

    standard_stop(t);

    do_sleep(125);  // TOTAL TIME: 1.25 seconds

    t.get<wall_clock>().stop();

    // t.start();
    // details::consume(500);
    // t.stop();

    auto& native_wc = t.get<wall_clock>();
    auto& pstart_wc = t.get<priority_start_wc>();
    auto& pstop_wc  = t.get<priority_stop_wc>();

    printf("\n");
    std::cout << native_wc << std::endl;
    std::cout << pstart_wc << std::endl;
    std::cout << pstop_wc << std::endl;
    printf("\n");
    std::cout << t << std::endl;
    printf("\n");

    double native_exp = 1.25;
    double pstart_exp = 0.875;
    double pstop_exp  = 0.5;

    ASSERT_NEAR(pstart_exp, pstart_wc.get(), 0.125);
    ASSERT_NEAR(pstop_exp, pstop_wc.get(), 0.125);
    ASSERT_NEAR(native_exp, native_wc.get(), 0.125);
}

//--------------------------------------------------------------------------------------//

int
main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    tim::timemory_init(argc, argv);
    tim::settings::precision()   = 6;
    tim::settings::width()       = 15;
    tim::settings::dart_output() = true;
    tim::settings::dart_count()  = 1;
    tim::settings::banner()      = false;

    return RUN_ALL_TESTS();
}

//--------------------------------------------------------------------------------------//
