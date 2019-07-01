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
#include <iostream>
#include <thread>
#include <timemory/testing.hpp>
#include <timemory/timemory.hpp>

using namespace tim::component;  // RELEVANT
using roofline_t   = cpu_roofline<PAPI_DP_OPS>;
using auto_tuple_t = tim::auto_tuple<real_clock, cpu_clock, roofline_t>;
using auto_list_t  = tim::auto_list<real_clock, cpu_clock, roofline_t>;

//--------------------------------------------------------------------------------------//

double
fib(double n);
void
check(auto_list_t& l);
void
check_const(const auto_list_t& l);

//--------------------------------------------------------------------------------------//

int
main(int argc, char** argv)
{
    tim::timemory_init(argc, argv);  // RELEVANT

    // RELEVANT
    using comp_tuple_t = typename auto_tuple_t::component_type;
    comp_tuple_t _main("overall timer", true);

    _main.start();  // RELEVANT
    for(auto n : { 35, 38, 45 })
    {
        // RELEVANT
        auto label = tim::str::join("", "fibonacci(", n, ")");
        TIMEMORY_BLANK_AUTO_TUPLE(auto_tuple_t, label);
        auto ret = fib(n);
        printf("fib(%i) = %f\n", n, ret);
    }
    _main.stop();  // RELEVANT

    std::cout << "\n" << _main << "\n" << std::endl;  // RELEVANT
    auto_list_t l(__FUNCTION__, false);
    check(l);
    check_const(l);
}

//--------------------------------------------------------------------------------------//

double
fib(double n)
{
    return (n < 2.0) ? n : 1.0 * (fib(n - 1) + fib(n - 2));
}

//--------------------------------------------------------------------------------------//

void
check_const(const auto_list_t& l)
{
    const real_clock* rc = l.get<real_clock>();
    std::cout << "type: " << tim::demangle(typeid(rc).name()) << std::endl;
}

//--------------------------------------------------------------------------------------//

void
check(auto_list_t& l)
{
    real_clock* rc = l.get<real_clock>();
    std::cout << "type: " << tim::demangle(typeid(rc).name()) << std::endl;
}

//--------------------------------------------------------------------------------------//
