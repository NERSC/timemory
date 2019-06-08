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

using graph_t          = tim::graph<std::string>;
using graph_iterator_t = typename graph_t::iterator;
using namespace tim::component;
using auto_tuple_t  = tim::auto_tuple<real_clock>;
using timer_tuple_t = tim::component_tuple<real_clock, system_clock, process_cpu_clock>;
using papi_tuple_t  = papi_tuple<0, PAPI_TOT_CYC, PAPI_TOT_INS, PAPI_BR_MSP, PAPI_BR_PRC>;
using global_tuple_t =
    tim::auto_tuple<real_clock, system_clock, thread_cpu_clock, thread_cpu_util,
                    process_cpu_clock, process_cpu_util, peak_rss, current_rss,
                    papi_tuple_t>;

//======================================================================================//

int64_t
fibonacci(int64_t n, int64_t cutoff, graph_t& g, graph_iterator_t itr)
{
    if(n > cutoff)
    {
        auto str = TIMEMORY_BASIC_AUTO_SIGN("[" + std::to_string(n) + "]");
        itr      = g.append_child(itr, str);
    }
    return (n < 2)
               ? n
               : (fibonacci(n - 2, cutoff, g, itr) + fibonacci(n - 1, cutoff, g, itr));
}

//======================================================================================//

void
print_result(const std::string& prefix, int64_t result)
{
    std::cout << std::setw(20) << prefix << " answer : " << result << std::endl;
}

//======================================================================================//

int
main(int argc, char** argv)
{
    tim::timemory_init(argc, argv);
    // default calc: fibonacci(40)
    int nfib = 40;
    if(argc > 1)
        nfib = atoi(argv[1]);

    // only record auto_timers when n > cutoff
    int cutoff = nfib - 5;
    if(argc > 2)
        cutoff = atoi(argv[2]);

    graph_t          mgraph;
    graph_iterator_t itr  = mgraph.set_head("top");
    auto             itr1 = mgraph.append_child(itr, "one");
    auto             itr2 = mgraph.append_child(itr, "two");
    mgraph.append_child(itr1, "three");
    mgraph.append_child(itr2, "four");
    mgraph.append_child(itr2, "five");

    std::vector<timer_tuple_t> timer_list;
    std::cout << std::endl;
    fibonacci(nfib, cutoff, mgraph, itr1);
    std::cout << std::endl;

    for(const auto& gitr : mgraph)
    {
        std::cout << gitr << std::endl;
    }

    std::cout << std::endl;

    auto format = [](const std::string& g) {
        std::stringstream ss;
        ss << "[test] " << g;
        return ss.str();
    };

    tim::print_graph_hierarchy(mgraph, format);

    std::cout << std::endl;

    return 0;
}

//======================================================================================//
