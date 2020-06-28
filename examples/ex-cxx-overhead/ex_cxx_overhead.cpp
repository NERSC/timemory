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
//

#include <cstdint>

#include "timemory/runtime/configure.hpp"
#include "timemory/runtime/invoker.hpp"
#include "timemory/timemory.hpp"
#include "timemory/utility/signals.hpp"
#include "timemory/utility/testing.hpp"

using namespace tim::component;

using auto_tuple_t  = tim::auto_tuple_t<wall_clock, user_global_bundle>;
using timer_tuple_t = tim::component_tuple_t<wall_clock, cpu_clock, peak_rss>;

using papi_tuple_t = papi_array<8>;
using global_tuple_t =
    tim::auto_tuple_t<wall_clock, user_clock, system_clock, cpu_clock, cpu_util, peak_rss,
                      page_rss, priority_context_switch, voluntary_context_switch,
                      caliper, tau_marker, papi_tuple_t, trip_count>;

static int64_t nmeasure     = 0;
static int64_t toolkit_size = 2;
using result_type           = std::tuple<timer_tuple_t, int64_t, int64_t>;

namespace mode
{
struct basic
{};
struct blank
{};
struct none
{};
struct basic_pointer
{};
struct blank_pointer
{};
struct measure
{};
}  // namespace mode

//======================================================================================//

static bool&
do_print_result()
{
    static bool _instance = true;
    return _instance;
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

template <typename Tp, tim::enable_if_t<std::is_same<Tp, mode::none>::value, int> = 0>
int64_t
fibonacci(int64_t n, int64_t)
{
    return fibonacci(n);
}

//======================================================================================//

template <typename Tp, tim::enable_if_t<std::is_same<Tp, mode::measure>::value, int> = 0>
int64_t
fibonacci(int64_t n, int64_t cutoff)
{
    if(n > cutoff)
    {
        nmeasure += toolkit_size;
        return (n < 2) ? n
                       : (fibonacci<Tp>(n - 1, cutoff) + fibonacci<Tp>(n - 2, cutoff));
    }
    return fibonacci(n);
}

//======================================================================================//

template <typename Tp, tim::enable_if_t<std::is_same<Tp, mode::blank>::value, int> = 0>
int64_t
fibonacci(int64_t n, int64_t cutoff)
{
    if(n > cutoff)
    {
        TIMEMORY_BLANK_MARKER(auto_tuple_t, __FUNCTION__);
        return (n < 2) ? n
                       : (fibonacci<Tp>(n - 1, cutoff) + fibonacci<Tp>(n - 2, cutoff));
    }
    return fibonacci(n);
}

//======================================================================================//

template <typename Tp, tim::enable_if_t<std::is_same<Tp, mode::basic>::value, int> = 0>
int64_t
fibonacci(int64_t n, int64_t cutoff)
{
    if(n > cutoff)
    {
        // TIMEMORY_BASIC_MARKER(auto_tuple_t, "[", n, "]");
        auto labeler = [](int _n) { return TIMEMORY_JOIN("", "fibonacci[", _n, "]"); };
        auto fib     = [](int _n, int _cutoff) {
            return (_n < 2) ? _n
                            : (fibonacci<Tp>(_n - 1, _cutoff) +
                               fibonacci<Tp>(_n - 2, _cutoff));
        };
        return tim::runtime::invoke<auto_tuple_t>(labeler(n), fib, n, cutoff);
    }
    return fibonacci(n);
}

//======================================================================================//

template <typename Tp,
          tim::enable_if_t<std::is_same<Tp, mode::blank_pointer>::value, int> = 0>
int64_t
fibonacci(int64_t n, int64_t cutoff)
{
    if(n > cutoff)
    {
        TIMEMORY_BLANK_POINTER(auto_tuple_t, __FUNCTION__);
        return (n < 2) ? n
                       : (fibonacci<Tp>(n - 1, cutoff) + fibonacci<Tp>(n - 2, cutoff));
    }
    return fibonacci(n);
}

//======================================================================================//

template <typename Tp,
          tim::enable_if_t<std::is_same<Tp, mode::basic_pointer>::value, int> = 0>
int64_t
fibonacci(int64_t n, int64_t cutoff)
{
    if(n > cutoff)
    {
        TIMEMORY_BASIC_POINTER(auto_tuple_t, "[", n, "]");
        return (n < 2) ? n
                       : (fibonacci<Tp>(n - 1, cutoff) + fibonacci<Tp>(n - 2, cutoff));
    }
    return fibonacci(n);
}

//======================================================================================//

template <typename Tp>
result_type
run(int64_t n, int64_t cutoff, bool store = true)
{
    // bool is_none  = std::is_same<Tp, mode::none>::value;
    bool is_blank = std::is_same<Tp, mode::blank>::value ||
                    std::is_same<Tp, mode::blank_pointer>::value;
    bool is_basic = std::is_same<Tp, mode::basic>::value ||
                    std::is_same<Tp, mode::basic_pointer>::value;

    // bool        with_timing = !(is_none);
    // std::string space       = (with_timing) ? " " : "";
    auto signature =
        TIMEMORY_LABEL(" [with timing = ", tim::demangle(typeid(Tp).name()), "]");

    nmeasure = 0;
    fibonacci<mode::measure>(n, cutoff);

    timer_tuple_t timer(signature, store);
    timer.start();
    int64_t result = fibonacci<Tp>(n, cutoff);
    timer.stop();

    int64_t nuniq =
        (is_blank) ? ((n - cutoff) * toolkit_size) : (is_basic) ? nmeasure : 0;

    auto _alt = timer;
    if(do_print_result())
        print_result(signature, result, nmeasure, nuniq);
    return result_type(_alt, nmeasure, nuniq);
}

//======================================================================================//

template <typename Tp>
void
launch(const int nitr, const int nfib, const int cutoff, int64_t& ex_measure,
       int64_t& ex_unique, std::vector<timer_tuple_t>& timer_list)
{
    int64_t nmeas = 0;
    int64_t nuniq = 0;

    do_print_result() = true;
    for(int i = 0; i < nitr; ++i)
    {
        auto&& ret = run<Tp>(nfib, cutoff);
        if(i == 0)
        {
            timer_list.push_back(std::get<0>(ret));
            nmeas = std::get<1>(ret);
            nuniq = std::get<2>(ret);
            ex_measure += std::get<1>(ret);
            ex_unique += std::get<2>(ret);
            do_print_result() = false;
        }
        else
        {
            timer_list.back() += std::get<0>(ret);
        }
    }

    std::string prefix = std::to_string(nuniq) + " unique measurements and " +
                         std::to_string(nmeas) + " total measurements (" +
                         tim::demangle(typeid(Tp).name()) + ")";
    timer_list.push_back((timer_list.back() / nitr) - (timer_list.at(0) / nitr));
    timer_list.push_back(timer_list.back() / nmeas);
    timer_list.at(timer_list.size() - 2).rekey("difference vs. " + prefix);
    timer_list.at(timer_list.size() - 1).rekey("average overhead of " + prefix);
}
//======================================================================================//

int
main(int argc, char** argv)
{
    tim::settings::timing_scientific() = true;
    tim::settings::auto_output()       = true;
    tim::settings::json_output()       = false;
    tim::settings::text_output()       = true;
    tim::settings::memory_units()      = "kB";
    tim::settings::memory_precision()  = 3;
    tim::settings::width()             = 12;
    tim::settings::timing_precision()  = 6;
    tim::timemory_init(&argc, &argv);
    tim::settings::cout_output() = false;
    tim::print_env();

    // default calc: fibonacci(43)
    int nfib = 43;
    if(argc > 1)
        nfib = atoi(argv[1]);

    // only record auto_timers when n > cutoff
    int cutoff = nfib - 15;
    if(argc > 2)
        cutoff = atoi(argv[2]);

    int nitr = 1;
    if(argc > 3)
        nitr = atoi(argv[3]);

    auto env_tool = tim::get_env<std::string>("EX_CXX_OVERHEAD_COMPONENTS", "");
    auto env_enum = tim::enumerate_components(tim::delimit(env_tool));
    env_enum.erase(std::remove_if(env_enum.begin(), env_enum.end(),
                                  [](int c) { return c == WALL_CLOCK; }),
                   env_enum.end());
    toolkit_size = env_enum.size() + 1;
    tim::configure<user_global_bundle>(env_enum);

    std::vector<timer_tuple_t> timer_list;

    std::cout << "\nRunning fibonacci(n = " << nfib << ", cutoff = " << cutoff << ")...\n"
              << std::endl;

    tim::auto_tuple<> empty_test("test");
    tim::consume_parameters(empty_test);

    auto warmup = run<mode::none>(nfib, nfib, false);
    std::cout << "[warmup]" << std::get<0>(warmup) << std::endl;

    int64_t ex_measure = 0;
    int64_t ex_unique  = 0;

    TIMEMORY_CALIPER(global, global_tuple_t, "[", argv[0], "]");

    std::cout << std::endl;

    //----------------------------------------------------------------------------------//
    //      run without timing
    //----------------------------------------------------------------------------------//
    do_print_result() = true;
    for(int i = 0; i < nitr; ++i)
    {
        auto&& ret = run<mode::none>(nfib, nfib);
        if(i == 0)
        {
            timer_list.push_back(std::get<0>(ret));
            ex_measure += std::get<1>(ret);
            ex_unique += std::get<2>(ret);
            do_print_result() = false;
        }
        else
        {
            timer_list.at(0) += std::get<0>(ret);
        }
    }

    //----------------------------------------------------------------------------------//
    //      run various modes
    //----------------------------------------------------------------------------------//
    launch<mode::blank>(nitr, nfib, cutoff, ex_measure, ex_unique, timer_list);
    launch<mode::blank_pointer>(nitr, nfib, cutoff, ex_measure, ex_unique, timer_list);
    launch<mode::basic>(nitr, nfib, cutoff, ex_measure, ex_unique, timer_list);
    launch<mode::basic_pointer>(nitr, nfib, cutoff, ex_measure, ex_unique, timer_list);

    TIMEMORY_CALIPER_APPLY(global, stop);

    std::cout << std::endl;

    std::cout << "\nReport from " << ex_measure << " total measurements and " << ex_unique
              << " unique measurements: " << std::endl;

    int nc = -1;
    for(auto& itr : timer_list)
    {
        std::cout << "    " << itr << std::endl;
        auto _nc = nc++;
        if(_nc % 3 == 2 || _nc < 0)
            std::cout << "\n";
    }

    auto l1_size  = tim::ert::cache_size::get<1>();
    auto l2_size  = tim::ert::cache_size::get<2>();
    auto l3_size  = tim::ert::cache_size::get<3>();
    auto max_size = tim::ert::cache_size::get_max();
    std::cout << "\n[INFO]> L1 cache size: " << (l1_size / tim::units::kilobyte)
              << " KB, L2 cache size: " << (l2_size / tim::units::kilobyte)
              << " KB, L3 cache size: " << (l3_size / tim::units::kilobyte)
              << " KB, max cache size: " << (max_size / tim::units::kilobyte) << " KB\n"
              << std::endl;

    // int ret = 0;

    if(!tim::settings::enabled())
    {
        printf("timemory was disabled.\n");
        // ret = EXIT_SUCCESS;
    }
    else if(tim::settings::flat_profile())
    {
        ex_unique = ((nfib - cutoff) + 1) * toolkit_size;
        int64_t rc_unique =
            (tim::storage<wall_clock>::instance()->size() - 6) * toolkit_size;
        printf("Expected size: %li, actual size: %li\n", (long) ex_unique,
               (long) rc_unique);
        // ret = (rc_unique == ex_unique) ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    else
    {
        int64_t rc_unique =
            (tim::storage<wall_clock>::instance()->size() - 5) * toolkit_size - 4;
        printf("Expected size: %li, actual size: %li\n", (long) ex_unique,
               (long) rc_unique);
        // ret = (rc_unique == ex_unique) ? EXIT_SUCCESS : EXIT_FAILURE;
    }

    tim::timemory_finalize();

    return 0;
}
