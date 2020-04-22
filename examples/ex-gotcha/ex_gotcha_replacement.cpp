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

#if defined(DEBUG) && !defined(VERBOSE)
#    define VERBOSE
#endif

#include "ex_gotcha_lib.hpp"
#include "timemory/timemory.hpp"

#include <array>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <set>
#include <string>
#include <vector>

using namespace tim;
using namespace tim::component;
using std::cout;
using std::vector;

using wc_t  = component_tuple<wall_clock>;
using got_t = gotcha<2, wc_t>;

extern "C" double
exp(double);

extern double
sum_exp(const vector<double>&);

int
main(int argc, char** argv)
{
    tim::timemory_init(argc, argv);

    int n = 100000;
    if(argc > 1) n = atoi(argv[1]);

    got_t::get_initializer() = [=]() {
        TIMEMORY_C_GOTCHA(got_t, 0, exp);
        TIMEMORY_CXX_GOTCHA(got_t, 1, sum_exp);
    };

    double ret = 0.0;
    for(int i = 0; i < n; ++i)
    {
        auto_tuple<got_t> obj("example");
        ret += sum_exp({ i + 1.0, 2.0 * (i + 1.0) });
    }

    tim::timemory_finalize();
}
