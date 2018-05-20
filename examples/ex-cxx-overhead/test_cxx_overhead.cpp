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

#include "timemory/macros.hpp"
#include "timemory/manager.hpp"
#include "timemory/auto_timer.hpp"
#include "timemory/timer.hpp"

//===========================================================================//
int64_t fibonacci(int64_t n, int64_t cutoff)
{
    if(n > cutoff)
    {
        //std::stringstream ss; ss << "[" << n << "]";
        //TIMEMORY_AUTO_TIMER(ss.str());
        TIMEMORY_AUTO_TIMER();
        return (n < 2) ? 1L : (fibonacci(n-2, cutoff) +
                               fibonacci(n-1, cutoff));
    }
    else
    {
        return (n < 2) ? 1L : (fibonacci(n-2, cutoff) +
                               fibonacci(n-1, cutoff));
    }
}

//===========================================================================//
void print_result(std::string prefix, int64_t result)
{
    std::cout << std::setw(20) << prefix << " answer : " << result << std::endl;
}

//===========================================================================//
tim::timer run(int64_t n, bool with_timing, int64_t cutoff)
{
    std::stringstream ss;
    ss << __FUNCTION__ << " [with timing = "
       << std::boolalpha << with_timing << "]";

    tim::timer timer(ss.str());
    timer.start();
    auto result = (with_timing) ? fibonacci(n, cutoff)
                                : fibonacci(n, n);
    timer.stop();
    print_result(ss.str(), result);
    return timer;
}

//===========================================================================//
int main(int argc, char** argv)
{
    // default calc: fibonacci(40)
    int nfib = 40;
    if(argc > 1)
        nfib = atoi(argv[1]);

    // only record auto_timers when n > cutoff
    int cutoff = nfib - 25;
    if(argc > 2)
        cutoff = atoi(argv[2]);

    tim::format::timer::default_unit(tim::units::msec);
    tim::format::timer::default_scientific(false);

    tim::manager* manager = tim::manager::instance();

    std::vector<tim::timer> timer_list;
    std::cout << std::endl;
    // run without timing first so overhead is not started yet
    timer_list.push_back(run(nfib, false, nfib)); // without timing
    timer_list.push_back(run(nfib, true, cutoff)); // with timing
    std::cout << std::endl;
    timer_list.push_back(timer_list.at(1) - timer_list.at(0));
    timer_list.back().format()->prefix("Timer difference");
    manager->missing_timer()->stop();
    timer_list.push_back(tim::timer(timer_list.back()));
    timer_list.back().accum() /= manager->total_laps();
    timer_list.back().format()->prefix("TiMemory avg. overhead");
    timer_list.back().record_memory(false);
    timer_list.back().format()->format(": %w %T (wall), %u %T (user), %s %T (sys), %t %T (cpu)");
    timer_list.back().format()->unit(tim::units::usec);

    manager->write_report("test_output/cxx_timing_overhead.out");
    manager->write_json("test_output/cxx_timing_overhead.json");
    manager->report(true);

    std::cout << "\nReports: " << std::endl;
    for(auto& itr : timer_list)
    {
        itr.format()->default_width(40);
        itr.format()->align_width(true);
        std::cout << "\t" << itr.as_string() << std::endl;
    }

    std::cout << std::endl;
    manager->write_missing(std::cout);

    return 0;
}
