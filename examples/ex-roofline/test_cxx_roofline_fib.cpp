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
#include <timemory/signal_detection.hpp>
#include <timemory/testing.hpp>
#include <timemory/timemory.hpp>

using namespace tim::component;
using float_type   = double;
using fib_list_t   = std::vector<int64_t>;
using roofline_t   = cpu_roofline<float_type, PAPI_DP_OPS>;
using auto_tuple_t = tim::auto_tuple<real_clock, cpu_clock, cpu_util, roofline_t>;
using auto_list_t  = tim::auto_list<real_clock, cpu_clock, cpu_util, roofline_t>;

// unless specified number of threads, use the number of available cores
#if !defined(NUM_THREADS)
#    define NUM_THREADS std::thread::hardware_concurrency()
#endif

//--------------------------------------------------------------------------------------//

float_type
fibonacci_a(float_type n);
float_type
fibonacci_b(float_type n);
void
check(auto_list_t& l);
void
check_const(const auto_list_t& l);

//--------------------------------------------------------------------------------------//

int
main(int argc, char** argv)
{
    tim::settings::json_output() = true;
    tim::timemory_init(argc, argv);
    tim::print_env();

    int64_t    num_threads   = NUM_THREADS;         // default number of threads
    int64_t    working_size  = 16;                  // default working set size
    int64_t    memory_factor = 8;                   // default multiple of max cache size
    fib_list_t fib_values    = { { 35, 38, 43 } };  // default values for fibonacci calcs

    if(argc > 1)
        num_threads = atol(argv[1]);
    if(argc > 2)
        working_size = atol(argv[2]);
    if(argc > 3)
        memory_factor = atol(argv[3]);
    if(argc > 4)
    {
        fib_values.clear();
        for(int i = 4; i < argc; ++i)
            fib_values.push_back(atol(argv[i]));
    }

    // overload the finalization function that runs ERT calculations
    roofline_t::get_finalize_function() = [=]() {
        using _Tp = float_type;
        // these are the kernel functions we want to calculate the peaks with
        auto store_func = [](_Tp& a, const _Tp& b) { a = b; };
        auto add_func   = [](_Tp& a, const _Tp& b, const _Tp& c) { a = b + c; };
        auto fma_func   = [](_Tp& a, const _Tp& b, const _Tp& c) { a = a * b + c; };
        // test getting the cache info
        auto l1_size = tim::ert::cache_size::get<1>();
        auto l2_size = tim::ert::cache_size::get<2>();
        auto l3_size = tim::ert::cache_size::get<3>();
        auto lm_size = tim::ert::cache_size::get_max();
        // log the cache info
        std::cout << "[INFO]> L1 cache size: " << (l1_size / tim::units::kilobyte)
                  << " KB, L2 cache size: " << (l2_size / tim::units::kilobyte)
                  << " KB, L3 cache size: " << (l3_size / tim::units::kilobyte)
                  << " KB, max cache size: " << (lm_size / tim::units::kilobyte)
                  << " KB\n"
                  << std::endl;
        // log how many threads were used
        printf("[INFO]> Running ERT with %li threads...\n\n",
               static_cast<long>(num_threads));
        // create the execution parameters
        tim::ert::exec_params params(working_size, memory_factor * lm_size, num_threads);
        // create the operation counter
        auto op_counter = new tim::ert::cpu::operation_counter<_Tp>(params, 64);
        // set bytes per element
        op_counter->bytes_per_element = sizeof(_Tp);
        // set number of memory accesses per element from two functions
        op_counter->memory_accesses_per_element = 2;
        // run the operation counter kernels
        tim::ert::cpu_ops_main<1>(*op_counter, add_func, store_func);
        tim::ert::cpu_ops_main<4>(*op_counter, fma_func, store_func);
        // return this data for processing
        return op_counter;
    };

    {
        //auto_tuple_t _main("overall_timer", __LINE__, tim::language::cxx(), true);

        for(const auto& n : fib_values)
        {
            auto label = tim::str::join("", "fibonacci_a(", n, ")");
            TIMEMORY_BLANK_AUTO_TUPLE(auto_tuple_t, label);
            auto ret = fibonacci_a(n);
            printf("fibonacci_a(%li) = %.1f\n", static_cast<long>(n), ret);
        }

        for(const auto& n : fib_values)
        {
            auto label = tim::str::join("", "fibonacci_b(", n, ")");
            TIMEMORY_BLANK_AUTO_TUPLE(auto_tuple_t, label);
            auto ret = fibonacci_b(n);
            printf("fibonacci_b(%li) = %.1f\n", static_cast<long>(n), ret);
        }
    }

    auto_list_t l(__FUNCTION__, false);
    check(l);
    check_const(l);
    std::cout << std::endl;
}

//--------------------------------------------------------------------------------------//

#define ftwo static_cast<float_type>(2.0)
#define fone static_cast<float_type>(1.0)

//--------------------------------------------------------------------------------------//

float_type
fibonacci_a(float_type n)
{
    return (n < ftwo) ? n : fone * (fibonacci_a(n - 1) + fibonacci_a(n - 2));
}

//--------------------------------------------------------------------------------------//

float_type
fibonacci_b(float_type n)
{
    return (n < ftwo) ? n : ((fone/0.5 * fibonacci_b(n - 1)) + fibonacci_b(n - 2));
}

//--------------------------------------------------------------------------------------//

void
check_const(const auto_list_t& l)
{
    const real_clock* rc = l.get<real_clock>();
    std::cout << "[demangle-test]> type: " << tim::demangle(typeid(rc).name())
              << std::endl;
}

//--------------------------------------------------------------------------------------//

void
check(auto_list_t& l)
{
    real_clock* rc = l.get<real_clock>();
    std::cout << "[demangle-test]> type: " << tim::demangle(typeid(rc).name())
              << std::endl;
}

//--------------------------------------------------------------------------------------//
