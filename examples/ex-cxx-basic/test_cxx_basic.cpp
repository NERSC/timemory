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

#include <chrono>
#include <thread>
#include <timemory/timemory.hpp>

using namespace tim::component;

using papi_tuple_t = papi_tuple<PAPI_TOT_CYC, PAPI_TOT_INS, PAPI_LST_INS>;
using real_tuple_t = tim::auto_tuple<real_clock, papi_tuple_t, caliper>;
using auto_tuple_t =
    tim::auto_tuple<real_clock, cpu_clock, cpu_util, peak_rss, papi_tuple_t, caliper>;
using comp_tuple_t = typename auto_tuple_t::component_type;
using auto_list_t  = tim::auto_list<real_clock, cpu_clock, cpu_util, peak_rss, caliper>;

void
some_func();
void
another_func();
intmax_t
fibonacci(intmax_t n);

//======================================================================================//

int
main(int argc, char** argv)
{
    // runtime customization of auto_list_t initialization
    auto_list_t::get_initializer() = [](auto_list_t& al) {
        const std::string default_env = "real_clock,cpu_clock,cpu_util,caliper";
        tim::env::initialize(al, "TIMEMORY_COMPONENTS", default_env);
        al.report_at_exit(true);
    };

    tim::settings::timing_units()      = "sec";
    tim::settings::timing_width()      = 12;
    tim::settings::timing_precision()  = 6;
    tim::settings::timing_scientific() = false;
    tim::settings::memory_units()      = "KB";
    tim::settings::memory_width()      = 12;
    tim::settings::memory_precision()  = 3;
    tim::settings::memory_scientific() = false;
    tim::timemory_init(argc, argv);

    // create a component tuple (does not auto-start)
    comp_tuple_t main("overall timer", true);
    main.start();
    for(auto n : { 15, 20, 25 })
    {
        // create a caliper handle to an auto_tuple_t and have it report when destroyed
        TIMEMORY_BLANK_AUTO_TUPLE_CALIPER(fib, auto_tuple_t, "fibonacci(", n, ")");
        TIMEMORY_CALIPER_APPLY(fib, report_at_exit, true);
        // run calculation
        auto ret = fibonacci(n);
        // manually stop the auto_tuple_t
        TIMEMORY_CALIPER_APPLY(fib, stop);
        printf("\nfibonacci(%i) = %li\n", n, (long int) ret);
    }
    // stop and print
    main.stop();
    std::cout << "\n" << main << std::endl;

    some_func();
    another_func();
}

//======================================================================================//

intmax_t
fibonacci(intmax_t n)
{
    TIMEMORY_BASIC_AUTO_TUPLE(real_tuple_t, "");
    return (n < 2) ? n : fibonacci(n - 1) + fibonacci(n - 2);
}

void
some_func()
{
    auto_tuple_t at("some_func", __LINE__);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
}

void
another_func()
{
    auto_list_t al("another_func", __LINE__);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
}
