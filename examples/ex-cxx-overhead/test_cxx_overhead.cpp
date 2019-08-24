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
//

#include <cstdint>

#include <timemory/timemory.hpp>
#include <timemory/utility/signals.hpp>
#include <timemory/utility/testing.hpp>

using namespace tim::component;

// using auto_tuple_t = tim::auto_tuple<real_clock>;
using namespace tim::component;

using auto_tuple_t  = tim::auto_tuple<real_clock, system_clock, user_clock, trip_count>;
using timer_tuple_t = typename tim::auto_tuple<real_clock>::component_type;

using papi_tuple_t = papi_array<8>;
using global_tuple_t =
    tim::auto_tuple<real_clock, user_clock, system_clock, cpu_clock, cpu_util, peak_rss,
                    current_rss, priority_context_switch, voluntary_context_switch,
                    caliper, papi_tuple_t>;

static int64_t nmeasure = 0;
using result_type       = std::tuple<timer_tuple_t, int64_t, int64_t>;

namespace mode
{
struct basic
{
};
struct blank
{
};
struct none
{
};
}

//======================================================================================//

void
print_result(const std::string& prefix, int64_t result, int64_t num_meas,
             int64_t num_uniq)
{
    std::cout << std::setw(20) << prefix << " answer : " << result
              << " (# measurments: " << num_meas << ", # unique: " << num_uniq << ")"
              << std::endl;
}

//======================================================================================//

int64_t
fibonacci(int64_t n)
{
    return (n < 2) ? n : (fibonacci(n - 1) + fibonacci(n - 2));
}

//======================================================================================//

template <typename _Tp, tim::enable_if_t<std::is_same<_Tp, mode::none>::value, int> = 0>
int64_t
fibonacci(int64_t n, int64_t cutoff)
{
    tim::consume_parameters(cutoff);
    return fibonacci(n);
}

//======================================================================================//

template <typename _Tp, tim::enable_if_t<std::is_same<_Tp, mode::blank>::value, int> = 0>
int64_t
fibonacci(int64_t n, int64_t cutoff)
{
    if(n > cutoff)
    {
        nmeasure += auto_tuple_t::size();
        TIMEMORY_BLANK_OBJECT(auto_tuple_t, __FUNCTION__);
        return (n < 2) ? n
                       : (fibonacci<_Tp>(n - 1, cutoff) + fibonacci<_Tp>(n - 2, cutoff));
    }
    return fibonacci(n);
}

//======================================================================================//

template <typename _Tp, tim::enable_if_t<std::is_same<_Tp, mode::basic>::value, int> = 0>
int64_t
fibonacci(int64_t n, int64_t cutoff)
{
    if(n > cutoff)
    {
        nmeasure += auto_tuple_t::size();
        TIMEMORY_BASIC_OBJECT(auto_tuple_t, "[", n, "]");
        return (n < 2) ? n
                       : (fibonacci<_Tp>(n - 1, cutoff) + fibonacci<_Tp>(n - 2, cutoff));
    }
    return fibonacci(n);
}

//======================================================================================//

template <typename _Tp>
result_type
run(int64_t n, int64_t cutoff)
{
    bool is_none  = std::is_same<_Tp, mode::none>::value;
    bool is_blank = std::is_same<_Tp, mode::blank>::value;
    bool is_basic = std::is_same<_Tp, mode::basic>::value;

    bool        with_timing = !(is_none);
    std::string space       = (with_timing) ? " " : "";
    auto        signature   = TIMEMORY_LABEL(" [with timing = ", space, with_timing, "]");

    nmeasure = 0;

    timer_tuple_t timer(signature, false);
    timer.start();
    int64_t result = fibonacci<_Tp>(n, cutoff);
    timer.stop();

    int64_t nuniq =
        (is_blank) ? ((n - cutoff) * auto_tuple_t::size()) : (is_basic) ? nmeasure : 0;

    print_result(signature, result, nmeasure, nuniq);
    return result_type(timer, nmeasure, nuniq);
}

//======================================================================================//

int
main(int argc, char** argv)
{
    tim::settings::timing_scientific() = true;
#if !defined(TIMEMORY_USE_GPERF)
    // heap-profiler will take a long timer if enabled
    tim::settings::json_output() = true;
#endif
    tim::timemory_init(argc, argv);
    tim::settings::cout_output() = false;
    tim::print_env();

    // default calc: fibonacci(43)
    int nfib = 43;
    if(argc > 1)
        nfib = atoi(argv[1]);

    // only record auto_timers when n > cutoff
    int cutoff = nfib - 20;
    if(argc > 2)
        cutoff = atoi(argv[2]);

    int nitr = 1;
    if(argc > 3)
        nitr = atoi(argv[3]);

    std::vector<timer_tuple_t> timer_list;

    std::cout << "\nRunning fibonacci(n = " << nfib << ", cutoff = " << cutoff << ")...\n"
              << std::endl;

    tim::auto_tuple<> empty_test("test");
    tim::consume_parameters(empty_test);

    auto warmup = run<mode::none>(nfib, nfib);
    std::cout << "[warmup]" << std::get<0>(warmup) << std::endl;

    int64_t ex_measure = 0;
    int64_t ex_unique  = 0;

    TIMEMORY_CALIPER(global, global_tuple_t, "[", argv[0], "]");

    std::cout << std::endl;

    //----------------------------------------------------------------------------------//
    //      run without timing
    //----------------------------------------------------------------------------------//
    for(int i = 0; i < nitr; ++i)
    {
        auto&& ret = run<mode::none>(nfib, nfib);
        if(i == 0)
        {
            timer_list.push_back(std::get<0>(ret));
            ex_measure += std::get<1>(ret);
            ex_unique += std::get<2>(ret);
        }
        else
        {
            timer_list.at(0) += std::get<0>(ret);
        }
    }

    auto        nmeas = 0;
    auto        nuniq = 0;
    std::string prefix;

    //----------------------------------------------------------------------------------//
    //      run with "blank" signature
    //----------------------------------------------------------------------------------//
    for(int i = 0; i < nitr; ++i)
    {
        auto&& ret = run<mode::blank>(nfib, cutoff);
        if(i == 0)
        {
            timer_list.push_back(std::get<0>(ret));
            nmeas = std::get<1>(ret);
            nuniq = std::get<2>(ret);
            ex_measure += std::get<1>(ret);
            ex_unique += std::get<2>(ret);
        }
        else
        {
            timer_list.back() += std::get<0>(ret);
        }
    }

    prefix = std::to_string(nuniq) + " unique measurements and " + std::to_string(nmeas) +
             " total measurements";
    timer_list.push_back((timer_list.back() / nitr) - (timer_list.at(0) / nitr));
    timer_list.push_back(timer_list.back() / nmeas);
    timer_list.at(timer_list.size() - 2).rekey("difference vs. " + prefix);
    timer_list.at(timer_list.size() - 1).rekey("average overhead of " + prefix);

    //----------------------------------------------------------------------------------//
    //      run with "basic" signature
    //----------------------------------------------------------------------------------//
    for(int i = 0; i < nitr; ++i)
    {
        auto&& ret = run<mode::basic>(nfib, cutoff);
        if(i == 0)
        {
            timer_list.push_back(std::get<0>(ret));
            nmeas = std::get<1>(ret);
            nuniq = std::get<2>(ret);
            ex_measure += std::get<1>(ret);
            ex_unique += std::get<2>(ret);
        }
        else
        {
            timer_list.back() += std::get<0>(ret);
        }
    }

    prefix = std::to_string(nuniq) + " unique measurements and " + std::to_string(nmeas) +
             " total measurements";
    timer_list.push_back((timer_list.back() / nitr) - (timer_list.at(0) / nitr));
    timer_list.push_back(timer_list.back() / nmeas);
    timer_list.at(timer_list.size() - 2).rekey("difference vs. " + prefix);
    timer_list.at(timer_list.size() - 1).rekey("average overhead of " + prefix);

    TIMEMORY_CALIPER_APPLY(global, stop);

    std::cout << std::endl;

    std::cout << "\nReport from " << ex_measure << " total measurements and " << ex_unique
              << " unique measurements: " << std::endl;

    for(auto& itr : timer_list)
        std::cout << "    " << itr << std::endl;

    auto l1_size  = tim::ert::cache_size::get<1>();
    auto l2_size  = tim::ert::cache_size::get<2>();
    auto l3_size  = tim::ert::cache_size::get<3>();
    auto max_size = tim::ert::cache_size::get_max();
    std::cout << "\n[INFO]> L1 cache size: " << (l1_size / tim::units::kilobyte)
              << " KB, L2 cache size: " << (l2_size / tim::units::kilobyte)
              << " KB, L3 cache size: " << (l3_size / tim::units::kilobyte)
              << " KB, max cache size: " << (max_size / tim::units::kilobyte) << " KB\n"
              << std::endl;

    int64_t rc_unique =
        (tim::storage<real_clock>::instance()->size() - 1) * auto_tuple_t::size();
    printf("Expected size: %li, actual size: %li\n", (long) ex_unique, (long) rc_unique);
    return (rc_unique == ex_unique) ? EXIT_SUCCESS : EXIT_FAILURE;
}
