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

#include "timemory/library.h"
#include "timemory/variadic/macros.hpp"
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define LABEL(...) TIMEMORY_LABEL(__VA_ARGS__)

long
fib(long n)
{
    return (n < 2) ? n : (fib(n - 1) + fib(n - 2));
}

int
main(int argc, char** argv)
{
    long nfib = (argc > 1) ? atol(argv[1]) : 43;
    int  nitr = (argc > 2) ? atoi(argv[2]) : 2;

    timemory_init_library(argc, argv);

    uint64_t id0 = timemory_get_begin_record("main/total");
    long     ans = fib(nfib);

    for(int i = 0; i < nitr; ++i)
    {
        uint64_t id1 = timemory_get_begin_record(TIMEMORY_JOIN("/", "nested", i).c_str());
        ans += fib(nfib + 1);
        timemory_end_record(id1);
    }

    timemory_end_record(id0);

    printf("Answer = %li\n", ans);
    timemory_finalize_library();
    return EXIT_SUCCESS;
}
