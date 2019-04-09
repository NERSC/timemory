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

#include "timemory/auto_timer.hpp"
#include "timemory/graph.hpp"
#include "timemory/macros.hpp"
#include "timemory/manager.hpp"
#include "timemory/timer.hpp"

typedef tim::graph<std::string>    graph_t;
typedef typename graph_t::iterator graph_iterator_t;

//======================================================================================//

intmax_t
fibonacci(intmax_t n, intmax_t cutoff, graph_t& g, graph_iterator_t itr)
{
    if(n > cutoff)
    {
        auto str = TIMEMORY_BASIC_AUTO_SIGN("[" + std::to_string(n) + "]");
        itr      = g.append_child(itr, str);
    }
    return (n < 2)
               ? 1L
               : (fibonacci(n - 2, cutoff, g, itr) + fibonacci(n - 1, cutoff, g, itr));
}

//======================================================================================//

void
print_result(const std::string& prefix, intmax_t result)
{
    std::cout << std::setw(20) << prefix << " answer : " << result << std::endl;
}

//======================================================================================//

int
main(int argc, char** argv)
{
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
    graph_iterator_t top  = itr;
    auto             itr1 = mgraph.append_child(itr, "one");
    auto             itr2 = mgraph.append_child(itr, "two");
    mgraph.append_child(itr1, "three");
    mgraph.append_child(itr2, "four");
    mgraph.append_child(itr2, "five");

    // tim::manager* manager = tim::manager::instance();

    std::vector<tim::timer> timer_list;
    std::cout << std::endl;
    fibonacci(nfib, cutoff, mgraph, itr1);
    std::cout << std::endl;

    for(auto gitr : mgraph)
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
