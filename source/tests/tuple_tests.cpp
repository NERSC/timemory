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

#include <timemory/timemory.hpp>
#include <timemory/utility/signals.hpp>

using namespace tim::stl_overload;
using namespace tim::component;

using papi_tuple_t = papi_tuple<PAPI_TOT_CYC, PAPI_TOT_INS, PAPI_LST_INS>;

using auto_tuple_t =
    tim::auto_tuple<real_clock, thread_cpu_clock, thread_cpu_util, process_cpu_clock,
                    process_cpu_util, peak_rss, page_rss>;

using full_measurement_t =
    tim::component_tuple<peak_rss, page_rss, stack_rss, data_rss, num_swap, num_io_in,
                         num_io_out, num_minor_page_faults, num_major_page_faults,
                         num_msg_sent, num_msg_recv, num_signals,
                         voluntary_context_switch, priority_context_switch, papi_tuple_t>;

using measurement_t =
    tim::component_tuple<real_clock, system_clock, user_clock, cpu_clock, cpu_util,
                         thread_cpu_clock, thread_cpu_util, process_cpu_clock,
                         process_cpu_util, monotonic_clock, monotonic_raw_clock,
                         papi_tuple_t>;

using printed_t = tim::component_tuple<real_clock, system_clock, user_clock, cpu_clock,
                                       thread_cpu_clock, process_cpu_clock>;

// measure wall-clock, thread cpu-clock + process cpu-utilization
using small_set_t = tim::auto_tuple<real_clock, thread_cpu_clock, process_cpu_util,
                                    caliper, papi_tuple_t>;

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
template <typename _Tp>
size_t
random_entry(const std::vector<_Tp>& v)
{
    std::mt19937 rng;
    rng.seed(std::random_device()());
    std::uniform_int_distribution<std::mt19937::result_type> dist(0, v.size() - 1);
    return v.at(dist(rng));
}
template <typename _Tp>
void
serialize(const std::string& fname, const std::string& title, const _Tp& obj)
{
    static constexpr auto spacing = cereal::JSONOutputArchive::Options::IndentChar::space;
    std::stringstream     ss;
    {
        // ensure json write final block during destruction before the file is closed
        //                                  args: precision, spacing, indent size
        cereal::JSONOutputArchive::Options opts(12, spacing, 4);
        cereal::JSONOutputArchive          oa(ss, opts);
        oa(cereal::make_nvp(title, obj));
    }
    std::ofstream ofs(fname.c_str());
    ofs << ss.str() << std::endl;
}
//--------------------------------------------------------------------------------------//
// fibonacci calculation
int64_t
fibonacci(int32_t n)
{
    return (n < 2) ? n : fibonacci(n - 1) + fibonacci(n - 2);
}
//--------------------------------------------------------------------------------------//
// fibonacci calculation
int64_t
fibonacci(int32_t n, int64_t cutoff)
{
    if(n > cutoff)
    {
        TIMEMORY_BASIC_MARKER(auto_tuple_t, n);
        return (n < 2) ? n : fibonacci(n - 1, cutoff) + fibonacci(n - 2, cutoff);
    }
    return (n < 2) ? n : fibonacci(n - 1) + fibonacci(n - 2);
}
//--------------------------------------------------------------------------------------//
// time fibonacci with return type and arguments
// e.g. std::function < int32_t ( int32_t ) >
int64_t
time_fibonacci(int32_t n)
{
    TIMEMORY_MARKER(auto_tuple_t, "");
    return fibonacci(n);
}
//--------------------------------------------------------------------------------------//
// time fibonacci with return type and arguments
// e.g. std::function < int32_t ( int32_t ) >
int64_t
time_fibonacci(int32_t n, int32_t cutoff)
{
    TIMEMORY_MARKER(auto_tuple_t, "");
    return fibonacci(n, cutoff);
}

}  // namespace details

//--------------------------------------------------------------------------------------//

class tuple_tests : public ::testing::Test
{};

//--------------------------------------------------------------------------------------//

TEST_F(tuple_tests, usage)
{
    auto test_1_marker = TIMEMORY_HANDLE(auto_tuple_t, "");

    full_measurement_t _use_beg("test_1_usage_begin");
    full_measurement_t _use_delta("test_1_usage_delta");
    full_measurement_t _use_end("test_1_usage_end");

    auto n = 5000000;
    _use_beg.record();
    _use_delta.start();
    std::vector<int64_t> v(n, 30);
    long                 nfib = details::random_entry(v);
    details::time_fibonacci(nfib);
    _use_delta.stop();
    _use_end.record();

    std::cout << "usage (begin): " << _use_beg << std::endl;
    std::cout << "usage (delta): " << _use_delta << std::endl;
    std::cout << "usage (end):   " << _use_end << std::endl;

    std::vector<std::pair<std::string, full_measurement_t>> measurements = {
        { "begin", _use_beg }, { "delta", _use_delta }, { "end", _use_end }
    };
    // serialize("rusage.json", "usage", measurements);
    test_1_marker.stop();
}

//--------------------------------------------------------------------------------------//

TEST_F(tuple_tests, all_threads)
{
    tim::settings::collapse_threads() = false;
    auto manager                      = tim::manager::instance();
    tim::manager::get_storage<auto_tuple_t>::clear(manager);
    auto starting_storage_size = tim::manager::get_storage<auto_tuple_t>::size(manager);
    auto data_size             = auto_tuple_t::size();

    using pair_t  = std::pair<std::string, measurement_t>;
    using mutex_t = std::mutex;
    using lock_t  = std::unique_lock<mutex_t>;

    mutex_t              mtx;
    std::vector<pair_t>  measurements;
    measurement_t        runtime("", false);
    printed_t            runtime_printed("", false);
    std::atomic<int64_t> ret;
    std::stringstream    lambda_ss;

    {
        TIMEMORY_MARKER(auto_tuple_t, "");

        auto run_fib = [&](long n, std::promise<void>* p) {
            std::stringstream ss;
            ss << "fibonacci(" << n << ")";

            TIMEMORY_BLANK_MARKER(auto_tuple_t, "run_fib");
            if(p != nullptr)
                p->set_value();
            measurement_t _tm("thread " + ss.str(), false);
            _tm.start();
            ret += details::time_fibonacci(n, n - 2);
            _tm.stop();

            lock_t lk(mtx);
            measurements.push_back(pair_t(ss.str(), _tm));
            lambda_ss << _tm << std::endl;
        };

        runtime_printed.start();
        runtime.start();
        {
            std::promise<void> _p1, _p2;
            std::future<void>  _f1 = _p1.get_future();
            std::future<void>  _f2 = _p2.get_future();
            std::thread        _t1(run_fib, 42, &_p1);
            std::thread        _t2(run_fib, 42, &_p2);

            _f1.wait();
            _f2.wait();

            run_fib(40, nullptr);

            _t1.join();
            _t2.join();
        }
        runtime.stop();
        runtime_printed.stop();
    }

    std::cout << "\n" << lambda_ss.str() << std::endl;
    std::cout << "total runtime: " << runtime << std::endl;
    std::cout << "std::get: " << std::get<0>(runtime) << std::endl;
    std::cout << "fibonacci total: " << ret.load() << "\n" << std::endl;
    std::cout << "runtime process cpu time: " << runtime.get<process_cpu_clock>() << "\n";
    std::cout << "measured data: " << runtime_printed.get() << std::endl;

    measurements.insert(measurements.begin(), pair_t("run", runtime));
    // serialize("timing.json", "runtime", measurements);

    // auto _test = std::tuple<int, double, std::string>{ 0, 0.2, "test" };
    // std::cout << "\nVARIADIC TUPLE PRINT: " << _test << "\n" << std::endl;

    auto rc_storage = tim::storage<wall_clock>::instance()->get();
    {
        printf("\n");
        size_t w = 0;
        for(const auto& itr : rc_storage)
            w = std::max<size_t>(w, std::get<2>(itr).length());
        for(const auto& itr : rc_storage)
        {
            std::cout << std::setw(w) << std::left << std::get<2>(itr) << " : "
                      << std::get<1>(itr);
            auto _hierarchy = std::get<5>(itr);
            for(size_t i = 0; i < _hierarchy.size(); ++i)
            {
                if(i == 0)
                    std::cout << " :: ";
                std::cout << _hierarchy[i];
                if(i + 1 < _hierarchy.size())
                    std::cout << "/";
            }
            std::cout << std::endl;
        }
        printf("\n");
    }

    auto final_storage_size = tim::manager::get_storage<auto_tuple_t>::size(manager);
    auto expected           = (final_storage_size - starting_storage_size);

    EXPECT_EQ(expected, 13 * data_size);

    const size_t store_size = 13;

    if(tim::trait::is_available<wall_clock>::value)
        EXPECT_EQ(tim::storage<wall_clock>::instance()->get().size(), store_size);

    if(tim::trait::is_available<thread_cpu_clock>::value)
        EXPECT_EQ(tim::storage<thread_cpu_clock>::instance()->get().size(), store_size);

    if(tim::trait::is_available<thread_cpu_util>::value)
        EXPECT_EQ(tim::storage<thread_cpu_util>::instance()->get().size(), store_size);

    if(tim::trait::is_available<process_cpu_clock>::value)
        EXPECT_EQ(tim::storage<process_cpu_clock>::instance()->get().size(), store_size);

    if(tim::trait::is_available<process_cpu_util>::value)
        EXPECT_EQ(tim::storage<process_cpu_util>::instance()->get().size(), store_size);

    if(tim::trait::is_available<peak_rss>::value)
        EXPECT_EQ(tim::storage<peak_rss>::instance()->get().size(), store_size);

    if(tim::trait::is_available<page_rss>::value)
        EXPECT_EQ(tim::storage<page_rss>::instance()->get().size(), store_size);
}

//--------------------------------------------------------------------------------------//

TEST_F(tuple_tests, collapsed_threads)
{
    tim::settings::collapse_threads() = true;
    auto manager                      = tim::manager::instance();
    tim::manager::get_storage<auto_tuple_t>::clear(manager);
    auto starting_storage_size = tim::manager::get_storage<auto_tuple_t>::size(manager);
    auto data_size             = auto_tuple_t::size();

    std::atomic<int64_t> ret;
    // accumulate metrics on full run
    TIMEMORY_BASIC_CALIPER(tot, auto_tuple_t, "[total]");

    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    // run a fibonacci calculation and accumulate metric
    auto run_fibonacci = [&](long n) {
        TIMEMORY_BLANK_MARKER(auto_tuple_t, "run_fibonacci");
        ret += details::time_fibonacci(n, n - 2);
    };

    {
        // run longer fibonacci calculations on two threads
        TIMEMORY_BASIC_CALIPER(master_thread_a, auto_tuple_t, "[master_thread]/0");
        {
            std::thread t1(run_fibonacci, 40);
            t1.join();
            std::thread t2(run_fibonacci, 41);
            t2.join();
        }
        TIMEMORY_CALIPER_APPLY(master_thread_a, stop);
    }

    {
        // run longer fibonacci calculations on two threads
        TIMEMORY_BASIC_CALIPER(master_thread_a, auto_tuple_t, "[master_thread]/1");

        std::thread t1(run_fibonacci, 41);
        std::thread t2(run_fibonacci, 40);
        // run_fibonacci(42);

        t1.join();
        t2.join();

        TIMEMORY_CALIPER_APPLY(master_thread_a, stop);
    }

    TIMEMORY_CALIPER_APPLY(tot, stop);

    std::cout << "\nfibonacci total: " << ret.load() << "\n" << std::endl;

    auto rc_storage = tim::storage<wall_clock>::instance()->get();
    {
        printf("\n");
        size_t w = 0;
        for(const auto& itr : rc_storage)
            w = std::max<size_t>(w, std::get<2>(itr).length());
        for(const auto& itr : rc_storage)
        {
            std::cout << std::setw(w) << std::left << std::get<2>(itr) << " : "
                      << std::get<1>(itr);
            auto _hierarchy = std::get<5>(itr);
            for(size_t i = 0; i < _hierarchy.size(); ++i)
            {
                if(i == 0)
                    std::cout << " :: ";
                std::cout << _hierarchy[i];
                if(i + 1 < _hierarchy.size())
                    std::cout << "/";
            }
            std::cout << std::endl;
        }
        printf("\n");
    }

    auto final_storage_size = tim::manager::get_storage<auto_tuple_t>::size(manager);
    auto expected           = (final_storage_size - starting_storage_size);

    EXPECT_EQ(expected, 19 * data_size);

    const size_t store_size = 15;

    if(tim::trait::is_available<wall_clock>::value)
        EXPECT_EQ(tim::storage<wall_clock>::instance()->get().size(), store_size);

    if(tim::trait::is_available<thread_cpu_clock>::value)
        EXPECT_EQ(tim::storage<thread_cpu_clock>::instance()->get().size(), store_size);

    if(tim::trait::is_available<thread_cpu_util>::value)
        EXPECT_EQ(tim::storage<thread_cpu_util>::instance()->get().size(), store_size);

    if(tim::trait::is_available<process_cpu_clock>::value)
        EXPECT_EQ(tim::storage<process_cpu_clock>::instance()->get().size(), store_size);

    if(tim::trait::is_available<process_cpu_util>::value)
        EXPECT_EQ(tim::storage<process_cpu_util>::instance()->get().size(), store_size);

    if(tim::trait::is_available<peak_rss>::value)
        EXPECT_EQ(tim::storage<peak_rss>::instance()->get().size(), store_size);

    if(tim::trait::is_available<page_rss>::value)
        EXPECT_EQ(tim::storage<page_rss>::instance()->get().size(), store_size);
}

//--------------------------------------------------------------------------------------//

TEST_F(tuple_tests, measure)
{
    tim::component_tuple<page_rss, peak_rss> prss(TIMEMORY_LABEL(""));
    {
        TIMEMORY_VARIADIC_BASIC_AUTO_TUPLE("[init]", page_rss, peak_rss);
        // just record the peak rss
        prss.measure();
        std::cout << "  Current rss: " << prss << std::endl;
    }

    {
        TIMEMORY_VARIADIC_AUTO_TUPLE("[delta]", page_rss, peak_rss);
        // do something, where you want delta peak rss
        auto                 n = 10000000;
        std::vector<int64_t> v(n, 10);
        long                 nfib = details::random_entry(v);
        details::fibonacci(nfib);
        prss.stop();
        std::cout << "Change in rss: " << prss << std::endl;
    }

    // prss.reset();
    prss.measure();
    std::cout << "  Current rss: " << prss << std::endl;
}

//--------------------------------------------------------------------------------------//

int
main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    tim::settings::verbose()     = 0;
    tim::settings::debug()       = false;
    tim::settings::json_output() = true;
    tim::timemory_init(argc, argv);  // parses environment, sets output paths
    tim::settings::dart_output() = true;
    tim::settings::dart_count()  = 1;
    tim::settings::banner()      = false;

    return RUN_ALL_TESTS();
}

//--------------------------------------------------------------------------------------//
