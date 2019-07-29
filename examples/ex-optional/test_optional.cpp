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

#include "test_optional.hpp"  // file that includes optional usage
#include <chrono>
#include <cstdio>
#include <thread>
#include <vector>

//--------------------------------------------------------------------------------------//
//
//      Declarations
//
//--------------------------------------------------------------------------------------//

namespace impl
{
long fibonacci(long n);
}
long fibonacci(long n);
void status();

//--------------------------------------------------------------------------------------//
//
//      Construct a main that uses macro tricks to avoid:
//
//      #ifdef USE_TIMEMORY
//          ....
//      #endif
//
//--------------------------------------------------------------------------------------//

int main(int argc, char** argv)
{
    status();

    //
    //  Dummy functions when USE_TIMEMORY not defined
    //
    tim::timemory_init(argc, argv);

    //
    //  Provide some work
    //
    std::vector<long> fibvalues;
    for(int i = 1; i < argc; ++i) fibvalues.push_back(atol(argv[i]));
    if(fibvalues.empty()) fibvalues.push_back(43);

    //
    // create an auto tuple accessible via a caliper integer or expand to nothing
    //
    TIMEMORY_AUTO_TUPLE_CALIPER(main, auto_tuple_t, "");
    TIMEMORY_AUTO_TUPLE_CALIPER(0, auto_tuple_t, "[]");

    //
    // call <auto_tuple_t>.report_at_exit(true) or expand to nothing
    //
    TIMEMORY_CALIPER_APPLY(main, report_at_exit, true);
    TIMEMORY_CALIPER_APPLY(0, report_at_exit, true);

    //
    //  Execute the work
    //
    for(const auto& itr : fibvalues)
    {
        auto ret = fibonacci(itr);
        printf("fibonacci(%li) = %li\n", itr, ret);
    }

    //
    // call <auto_tuple_t>.stop() or expand to nothing
    //
    TIMEMORY_CALIPER_APPLY(main, stop);

    //
    // sleep for 1 second so difference between two calipers
    //
    std::this_thread::sleep_for(std::chrono::seconds(1));

    //
    // call <auto_tuple_t>.stop() or expand to nothing
    //
    TIMEMORY_CALIPER_APPLY(0, stop);

    status();
}

//--------------------------------------------------------------------------------------//
//
//      Implementations
//
//--------------------------------------------------------------------------------------//

long impl::fibonacci(long n)
{
    return (n < 2) ? n : (impl::fibonacci(n - 1) + impl::fibonacci(n - 2));
}

long fibonacci(long n)
{
    TIMEMORY_BASIC_AUTO_TUPLE(auto_tuple_t, "(", n, ")");
    return impl::fibonacci(n);
}

void status()
{
#if defined(USE_TIMEMORY)
    printf("\n#----------------- TIMEMORY is enabled  ----------------#\n\n");
#else
    printf("\n#----------------- TIMEMORY is disabled ----------------#\n\n");
#endif
}
