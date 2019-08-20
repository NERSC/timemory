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
#include <random>
#include <thread>
#include <timemory/timemory.hpp>
#include <timemory/utility/signals.hpp>
#include <timemory/utility/testing.hpp>

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
fibonacci(float_type n);
float_type
random_fibonacci(float_type n);
void
check(auto_list_t& l);
void
     check_const(const auto_list_t& l);
void customize_roofline(int64_t, int64_t, int64_t);

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

    //
    // override method for determining how many threads to run
    //
    roofline_t::get_finalize_threads_function() = [=]() { return num_threads; };

    //
    // allow for customizing the roofline
    //
    if(tim::get_env("CUSTOMIZE_ROOFLINE", true))
        customize_roofline(num_threads, working_size, memory_factor);

    //
    // initialize the storage for components that only are recorded in worker threads
    //
    // tim::manager::instance()->initialize_storage<thread_cpu_clock, thread_cpu_util>();

    //
    // execute fibonacci in a thread
    //
    auto exec_fibonacci = [&](int64_t n) {
        TIMEMORY_BLANK_AUTO_TUPLE_CALIPER(0, auto_tuple_t, "fibonacci(", n, ")");
        auto ret = fibonacci(n);
        TIMEMORY_CALIPER_APPLY(0, stop);
        printf("fibonacci(%li) = %.1f\n", static_cast<long>(n), ret);
    };

    //
    // execute random_fibonacci in a thread
    //
    auto exec_random_fibonacci = [&](int64_t n) {
        TIMEMORY_BLANK_AUTO_TUPLE_CALIPER(1, auto_tuple_t, "random_fibonacci(", n, ")");
        auto ret = random_fibonacci(n);
        TIMEMORY_CALIPER_APPLY(1, stop);
        printf("random_fibonacci(%li) = %.1f\n", static_cast<long>(n), ret);
    };

    //
    // overall timing
    //
    auto _main = TIMEMORY_BLANK_AUTO_TUPLE_INSTANCE(auto_tuple_t, "overall_timer");
    _main.report_at_exit(true);
    real_clock total;
    total.start();

    //
    // run fibonacci calculations
    //
    for(const auto& n : fib_values)
    {
        std::vector<std::thread> threads;
        for(int64_t i = 0; i < num_threads; ++i)
            threads.push_back(std::thread(exec_fibonacci, n));
        for(auto& itr : threads)
            itr.join();
    }

    //
    // run the random fibonacci calculations
    //
    for(const auto& n : fib_values)
    {
        std::vector<std::thread> threads;
        for(int64_t i = 0; i < num_threads; ++i)
            threads.push_back(std::thread(exec_random_fibonacci, n));
        for(auto& itr : threads)
            itr.join();
    }

    //
    // stop the overall timing
    //
    _main.stop();

    //
    // overall timing
    //
    std::this_thread::sleep_for(std::chrono::seconds(1));
    total.stop();

    std::cout << "Total time: " << total << std::endl;

    auto_list_t l(__FUNCTION__, __LINE__);
    check(l);
    check_const(l);
    std::cout << std::endl;
}

//--------------------------------------------------------------------------------------//

#define ftwo static_cast<float_type>(2.0)
#define fone static_cast<float_type>(1.0)

//--------------------------------------------------------------------------------------//

float_type
fibonacci(float_type n)
{
    return (n < ftwo) ? n : fone * (fibonacci(n - 1) + fibonacci(n - 2));
}

//--------------------------------------------------------------------------------------//

float_type
random_fibonacci(float_type n)
{
    // this is intentionally different between runs so that we scatter
    // the arithmetic intensity vs. the compute rate between runs to simulate
    // different algorithms
    static std::atomic<int>                tid;
    static thread_local std::random_device rd;
    static thread_local std::mt19937       gen(rd() + tid++);
    auto                                   get_random = [&]() {
        return 1.75 * std::generate_canonical<float_type, 16>(gen) + 0.9;
    };
    auto m1 = get_random();
    auto m2 = get_random();
    return (n < ftwo) ? n : (random_fibonacci(n - m1) + random_fibonacci(n - m2));
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

void
customize_roofline(int64_t num_threads, int64_t working_size, int64_t memory_factor)
{
    // overload the finalization function that runs ERT calculations
    roofline_t::get_finalizer() = [=]() {
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
        tim::ert::cpu_ops_main<4, 5, 6, 7, 8>(*op_counter, fma_func, store_func);
        // return this data for processing
        return op_counter;
    };
}
//--------------------------------------------------------------------------------------//
