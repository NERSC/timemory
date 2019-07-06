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

#include <timemory/testing.hpp>
#include <timemory/timemory.hpp>

using namespace tim::component;
static int64_t nlaps = 0;

// using auto_tuple_t = tim::auto_tuple<real_clock>;

using auto_tuple_t =
    tim::auto_tuple<tim::component::real_clock, tim::component::system_clock,
                    tim::component::user_clock>;
using timer_tuple_t = typename auto_tuple_t::component_type;

using papi_tuple_t = papi_tuple<0, PAPI_TOT_CYC, PAPI_TOT_INS>;
using global_tuple_t =
    tim::auto_tuple<real_clock, user_clock, system_clock, cpu_clock, cpu_util, peak_rss,
                    current_rss, priority_context_switch, voluntary_context_switch,
                    papi_tuple_t>;

//======================================================================================//

void
print_result(const std::string& prefix, int64_t result)
{
    std::cout << std::setw(20) << prefix << " answer : " << result << std::endl;
}

//======================================================================================//

int64_t
fibonacci(int64_t n)
{
    return (n < 2) ? n : (fibonacci(n - 1) + fibonacci(n - 2));
}

//======================================================================================//

int64_t
fibonacci(int64_t n, int64_t cutoff)
{
    if(n > cutoff)
    {
        nlaps += auto_tuple_t::size();
        // TIMEMORY_BASIC_AUTO_TUPLE(auto_tuple_t, "");
        TIMEMORY_BASIC_AUTO_TUPLE(auto_tuple_t, "[", n, "]");
        // TIMEMORY_BLANK_AUTO_TUPLE(auto_tuple_t, __FUNCTION__);
        return (n < 2) ? n : (fibonacci(n - 1, cutoff) + fibonacci(n - 2, cutoff));
    }
    return fibonacci(n);
}

//======================================================================================//

timer_tuple_t
run(int64_t n, bool with_timing, int64_t cutoff)
{
    auto signature = TIMEMORY_AUTO_SIGN(" [with timing = ", ((with_timing) ? " " : ""),
                                        with_timing, "]");
    timer_tuple_t timer(signature);
    int64_t       result = 0;
    {
        auto auto_timer = timer_tuple_t::auto_type(timer, __LINE__);
        result          = (with_timing) ? fibonacci(n, cutoff) : fibonacci(n);
    }
    print_result(signature, result);
    return timer;
}

//======================================================================================//

int
main(int argc, char** argv)
{
    tim::settings::timing_scientific() = true;
    tim::settings::cout_output()       = false;
    tim::timemory_init(argc, argv);
    tim::settings::cout_output() = false;
#if !defined(TIMEMORY_USE_GPERF)
    // heap-profiler will take a long timer if enabled
    tim::settings::json_output() = true;
#endif

    // default calc: fibonacci(43)
    int nfib = 43;
    if(argc > 1)
        nfib = atoi(argv[1]);

    // only record auto_timers when n > cutoff
    int cutoff = nfib - 27;
    if(argc > 2)
        cutoff = atoi(argv[2]);

    std::cout << "Running fibonacci(n = " << nfib << ", cutoff = " << cutoff << ")..."
              << std::endl;
    tim::consume_parameters(tim::manager::instance());
    tim::auto_tuple<>          empty_test("test");
    std::vector<timer_tuple_t> timer_list;

    std::cout << std::endl;
    {
        nlaps = 0;
        bool enable_auto_timers;
        TIMEMORY_AUTO_TUPLE(global_tuple_t, "[", argv[0], "]");
        // run without timing first
        timer_list.push_back(run(nfib, enable_auto_timers = false, nfib));
        timer_list.push_back(run(nfib, enable_auto_timers = true, cutoff));
    }
    std::cout << std::endl;

    timer_list.push_back(timer_list.at(1) - timer_list.at(0));
    timer_list.push_back(timer_list.back() / nlaps);
    timer_list.at(timer_list.size() - 2).key() = "timing difference";
    timer_list.at(timer_list.size() - 1).key() = "average overhead per timer";

    std::cout << "\nReports from " << nlaps << " total laps: " << std::endl;

    for(auto& itr : timer_list)
        std::cout << "\t" << itr << std::endl;

    tim::print_env();
    return 0;
}
