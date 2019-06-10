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
#include <timemory/timemory.hpp>

using namespace tim::component;

using roofline_t   = cpu_roofline<PAPI_DP_OPS>;
using auto_tuple_t = tim::auto_tuple<real_clock, cpu_clock, cpu_util, roofline_t>;
using comp_tuple_t = typename auto_tuple_t::component_type;

template <typename _Tp>
_Tp
fibonacci(const _Tp& n)
{
    // fibonacci using floating point
    return (n < _Tp(2)) ? n
                        : (1.0 * fibonacci(n - _Tp(1))) + (1.0 * fibonacci(n - _Tp(2)));
}

int
main(int argc, char** argv)
{
    // STEP 4: configure output and parse env  (optional)
    tim::settings::precision() = 6;
    tim::timemory_init(argc, argv);
    tim::print_env();
    std::cout << std::endl;

    comp_tuple_t main("overall timer", true);
    main.start();
    for(auto n : { 35, 38, 45 })
    {
        auto_tuple_t t(tim::str::join("", "fibonacci(", n, ")"));
        auto         ret = fibonacci<double>(n);
        printf("fibonacci(%i) = %li\n", n, static_cast<long int>(ret));
    }

    // ERT Kernel goes here
    main.stop();
    std::cout << main << "\n" << std::endl;
}
