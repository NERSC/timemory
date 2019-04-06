// MIT License
//
// Copyright (c) 2018, The Regents of the University of California,
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

#include <timemory/auto_tuple.hpp>
#include <timemory/component_tuple.hpp>
#include <timemory/environment.hpp>
#include <timemory/manager.hpp>
#include <timemory/mpi.hpp>
#include <timemory/papi.hpp>
#include <timemory/rusage.hpp>
#include <timemory/signal_detection.hpp>
#include <timemory/testing.hpp>

using namespace tim::component;
using auto_tuple_t  = tim::auto_tuple<real_clock>;
using timer_tuple_t = tim::component_tuple<real_clock, system_clock, process_cpu_clock>;

static intmax_t nlaps = 0;
//======================================================================================//

intmax_t
fibonacci(intmax_t n, intmax_t cutoff)
{
    if(n > cutoff)
    {
        ++nlaps;
        TIMEMORY_AUTO_TUPLE(auto_tuple_t, "");
        return (n < 2) ? 1L : (fibonacci(n - 2, cutoff) + fibonacci(n - 1, cutoff));
    }
    else
    {
        return (n < 2) ? 1L : (fibonacci(n - 2, cutoff) + fibonacci(n - 1, cutoff));
    }
}

//======================================================================================//

void
print_result(std::string prefix, intmax_t result)
{
    std::cout << std::setw(20) << prefix << " answer : " << result << std::endl;
}

//======================================================================================//

timer_tuple_t
run(intmax_t n, bool with_timing, intmax_t cutoff)
{
    std::stringstream ss, ss_b;
    ss_b << std::boolalpha << with_timing;
    ss << __FUNCTION__ << " [with timing = " << std::boolalpha << std::setw(5)
       << ss_b.str() << "]";

    timer_tuple_t timer(ss.str());
    timer.start();
    auto result = (with_timing) ? fibonacci(n, cutoff) : fibonacci(n, n);
    timer.stop();
    print_result(ss.str(), result);
    return timer;
}

//======================================================================================//

int
main(int argc, char** argv)
{
    setenv("TIMEMORY_TIMING_SCIENTIFIC", "1", 1);

    // default calc: fibonacci(40)
    int nfib = 43;
    if(argc > 1)
        nfib = atoi(argv[1]);

    // only record auto_timers when n > cutoff
    int cutoff = nfib - 25;
    if(argc > 2)
        cutoff = atoi(argv[2]);

    tim::consume_parameters(tim::manager::instance());

    std::vector<timer_tuple_t> timer_list;
    std::cout << std::endl;
    // run without timing first so overhead is not started yet
    timer_list.push_back(run(nfib, false, nfib));  // without timing
    nlaps = 0;
    timer_list.push_back(run(nfib, true, cutoff));  // with timing
    std::cout << std::endl;
    timer_list.push_back(timer_list.at(1) - timer_list.at(0));
    timer_list.push_back(timer_list.back() / nlaps);

    std::cout << "\nReports from " << nlaps << " total laps: " << std::endl;
    for(auto& itr : timer_list)
    {
        std::cout << "\t" << itr << std::endl;
    }

    std::cout << std::endl;

    return 0;
}
