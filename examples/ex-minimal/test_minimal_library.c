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

#include <stdio.h>
#include <timemory/ctimemory.h>

extern void timemory_init_library(int, char**);
extern void timemory_finalize_library();
extern void timemory_begin_record(const char*, uint64_t*);
extern void timemory_end_record(uint64_t);

long fib(long n) { return (n < 2) ? n : (fib(n - 1) + fib(n - 2)); }

int main(int argc, char** argv)
{
    long     nfib = (argc > 1) ? atol(argv[1]) : 43;
    uint64_t id0, id1;

    timemory_init_library(argc, argv);

    timemory_begin_record(argv[0], &id0);
    long ans = fib(nfib);

    timemory_begin_record(argv[0], &id1);
    ans += fib(nfib + 1);

    timemory_end_record(id1);
    timemory_end_record(id0);

    printf("Answer = %li\n", ans);
    timemory_finalize_library();
    return EXIT_SUCCESS;
}
