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

#include <cstdio>
#include <cstdlib>
#include <random>

//--------------------------------------------------------------------------------------//
// include these headers for declaration w/o instantiation
//
#include "timemory/mpl/types.hpp"

//
// all types use compact json
//
TIMEMORY_DEFINE_CONCRETE_TRAIT(pretty_json, void, std::false_type)

//--------------------------------------------------------------------------------------//
// include rest of headers
//
#include "timemory/timemory.hpp"

//
// configure these two types to always record statistics
//
TIMEMORY_DEFINE_CONCRETE_TRAIT(record_statistics, component::wall_clock, std::true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(record_statistics, component::written_bytes,
                               std::true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(record_statistics, component::read_bytes, std::true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(record_statistics, component::cpu_clock, std::false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(flat_storage, component::monotonic_clock, std::true_type)

//
// shorthand
//
using namespace tim::component;
using namespace tim;

//
// bundle of tools
//
using tuple_t = auto_tuple<wall_clock, monotonic_clock, cpu_clock, read_bytes,
                           written_bytes, peak_rss, current_peak_rss>;

//--------------------------------------------------------------------------------------//

long
fib(long n)
{
    return (n < 2) ? n : (fib(n - 1) + fib(n - 2));
}

//--------------------------------------------------------------------------------------//

int
main(int argc, char** argv)
{
    settings::destructor_report() = true;
    tim::timemory_init(argc, argv);
    tim::enable_signal_detection();

    long nfib = (argc > 1) ? atol(argv[1]) : 40;
    int  nitr = (argc > 2) ? atoi(argv[2]) : 10;

    std::random_device              rd;
    std::mt19937                    gen(rd());
    std::uniform_int_distribution<> size_dist(50000, 1000000);
    std::uniform_int_distribution<> value_dist(0, 1000000);

    for(int i = 0; i < nitr; ++i)
    {
        TIMEMORY_MARKER(tuple_t, "total");
        long ans = fib(nfib);

        TIMEMORY_BLANK_MARKER(tuple_t, "nested/", i % 5);
        ans += fib(nfib + (i % 3));

        // allocate some memory
        std::vector<int32_t> v(size_dist(gen), value_dist(gen));
        // use the memory
        ans *= v.at(value_dist(gen) % v.size());

        TIMEMORY_CONDITIONAL_BASIC_MARKER(i % 3 == 2, tuple_t, "occasional/", i % 2);
        ans += fib(nfib - 1);

        printf("Answer = %li\n", ans);
        if(i > 0)
            tim::settings::destructor_report() = false;
    }

    tim::timemory_finalize();
    return EXIT_SUCCESS;
}

//--------------------------------------------------------------------------------------//
