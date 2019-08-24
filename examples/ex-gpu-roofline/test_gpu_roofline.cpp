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

#if !defined(ROOFLINE_FP_BYTES)
#    define ROOFLINE_FP_BYTES 8
#endif

#if ROOFLINE_FP_BYTES == 8
using float_type = double;
#elif ROOFLINE_FP_BYTES == 4
using float_type = float;
#else
#    error "ROOFLINE_FP_BYTES must be either 4 or 8"
#endif

using roofline_t     = gpu_roofline<float_type>;
using fib_list_t     = std::vector<int64_t>;
using auto_tuple_t   = tim::auto_tuple<real_clock, cpu_clock, cpu_util, roofline_t>;
using auto_list_t    = tim::auto_list<real_clock, cpu_clock, cpu_util, roofline_t>;
using device_t       = tim::device::gpu;
using params_t       = tim::device::params<device_t>;
using stream_t       = tim::cuda::stream_t;
using default_device = device_t;

// unless specified number of threads, use the number of available cores
#if !defined(NUM_THREADS)
#    define NUM_THREADS 1
#endif

#if !defined(NUM_STREAMS)
#    define NUM_STREAMS 8
#endif

//--------------------------------------------------------------------------------------//
//
template <typename _Tp>
DEVICE_CALLABLE inline void
add_func(_Tp& a, const _Tp& b, const _Tp& c)
{
    a = b + c;
}
//--------------------------------------------------------------------------------------//
//
template <typename _Tp>
DEVICE_CALLABLE inline void
fma_func(_Tp& a, const _Tp& b, const _Tp& c)
{
    a = a * b + c;
}

//--------------------------------------------------------------------------------------//
// saxpy calculation
//
template <typename _Tp>
GLOBAL_CALLABLE void
saxpy(int64_t n, _Tp a, _Tp* x, _Tp* y)
{
    auto range = tim::device::grid_strided_range<default_device, 0>(n);
    for(int i = range.begin(); i < range.end(); i += range.stride())
        fma_func<_Tp>(y[i], a, x[i]);
}

void customize_roofline(int64_t, int64_t, int64_t);

//--------------------------------------------------------------------------------------//

int
main(int argc, char** argv)
{
    tim::settings::json_output() = true;
    tim::timemory_init(argc, argv);
    cuInit(0);

    int64_t    num_threads   = NUM_THREADS;  // default number of threads
    int64_t    num_streams   = NUM_STREAMS;  // default number of streams
    int64_t    working_size  = 16;           // default working set size
    int64_t    memory_factor = 8;            // default multiple of max cache size
    int64_t    iterations    = 100;
    int64_t    block_size    = 512;
    int64_t    grid_size     = 0;
    int64_t    data_size     = 1000;
    float_type factor        = 2.0;

    auto usage = [=]() {
        using lli = long long int;
        printf(
            "%s [%s = %lli] [%s = %lli] [%s = %lli] [%s = %lli] [%s = %lli] [%s = %lli] "
            "[%s = %lli] [%s = %lli] [%s = %f]\n",
            argv[0], "num_threads", (lli) num_threads, "num_streams", (lli) num_streams,
            "working_size", (lli) working_size, "memory_factor", (lli) memory_factor,
            "iterations", (lli) iterations, "block_size", (lli) block_size, "grid_size",
            (lli) grid_size, "data_size", (lli) data_size, "factor", factor);
    };

    for(auto i = 0; i < argc; ++i)
        if(strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0)
        {
            usage();
            exit(EXIT_SUCCESS);
        }

    int curr_arg = 1;
    if(argc > curr_arg)
        num_threads = atol(argv[curr_arg++]);
    if(argc > curr_arg)
        num_streams = atol(argv[curr_arg++]);
    if(argc > curr_arg)
        working_size = atol(argv[curr_arg++]);
    if(argc > curr_arg)
        memory_factor = atol(argv[curr_arg++]);
    if(argc > curr_arg)
        iterations = atol(argv[curr_arg++]);
    if(argc > curr_arg)
        block_size = atol(argv[curr_arg++]);
    if(argc > curr_arg)
        grid_size = atol(argv[curr_arg++]);
    if(argc > curr_arg)
        data_size = atol(argv[curr_arg++]);
    if(argc > curr_arg)
        factor = (float_type) atol(argv[curr_arg++]);

    //
    // override method for determining how many threads and streams to run
    //
    roofline_t::get_num_threads_finalizer() = [=]() { return num_threads; };
    roofline_t::get_num_streams_finalizer() = [=]() { return num_streams; };

    //
    // allow for customizing the roofline
    //
    if(tim::get_env("CUSTOMIZE_ROOFLINE", false))
        customize_roofline(num_threads, working_size, memory_factor);

    params_t params(grid_size, block_size, 0, 0);

    std::vector<stream_t> streams;
    if(num_streams > 1)
    {
        streams.resize(num_streams);
        for(auto& itr : streams)
            tim::cuda::stream_create(itr);
    }

    int64_t chunk_size   = data_size / num_streams;
    int64_t chunk_modulo = data_size % num_streams;

    float_type* y = tim::cuda::malloc<float_type>(data_size);
    float_type* x = tim::cuda::malloc<float_type>(data_size);

    printf("y = %p, x = %p\n", y, x);
    // tim::device::launch(data_size, params,
    //                    tim::ert::initialize_buffer<device_t, float_type, int64_t>, y,
    //                    0.0, data_size);
    tim::cuda::memset(y, 0, data_size);
    tim::cuda::device_sync();
    printf("y = %p, x = %p\n", y, x);

    // tim::device::launch(data_size, params,
    //                    tim::ert::initialize_buffer<device_t, float_type, int64_t>, x,
    //                    1.0, data_size);
    tim::cuda::memset(x, 1, data_size);
    tim::cuda::device_sync();
    printf("y = %p, x = %p\n", y, x);

    //
    // execute fibonacci in a thread
    //
    auto exec_saxpy = [&](int64_t n) {
        TIMEMORY_BLANK_CALIPER(0, auto_tuple_t, "saxpy(", n, ")");

        int64_t offset = n * chunk_size;
        int64_t nsize  = chunk_size;
        if(n + 1 == streams.size())
            nsize += chunk_modulo;

        auto*    _x      = x + offset;
        auto*    _y      = y + offset;
        stream_t _stream = 0;

        if(!streams.empty())
            _stream = streams.at(n % streams.size());

        for(int64_t i = 0; i < iterations; ++i)
            tim::device::launch(nsize, _stream, params, saxpy<float_type>, nsize, factor,
                                _x, _y);

        TIMEMORY_CALIPER_APPLY(0, stop);
    };

    //
    // overall timing
    //
    auto _main = TIMEMORY_BLANK_INSTANCE(auto_tuple_t, "overall_timer");
    _main.report_at_exit(true);
    real_clock total;
    total.start();

    //
    // run saxpy calculations
    //
    if(num_threads > 1)
    {
        std::vector<std::thread> threads;
        for(int64_t i = 0; i < num_threads; ++i)
            threads.push_back(std::thread(exec_saxpy, i));
        for(auto& itr : threads)
            itr.join();
    }
    else
    {
        exec_saxpy(0);
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
}

//--------------------------------------------------------------------------------------//

void
customize_roofline(int64_t num_threads, int64_t working_size, int64_t memory_factor)
{
    // overload the finalization function that runs ERT calculations
    roofline_t::get_finalizer() = [=]() {
        using _Tp = float_type;
        // these are the kernel functions we want to calculate the peaks with
        auto store_func = [] TIMEMORY_LAMBDA(_Tp & a, const _Tp& b) { a = b; };
        auto add_func   = [] TIMEMORY_LAMBDA(_Tp & a, const _Tp& b, const _Tp& c) {
            a = b + c;
        };
        auto fma_func = [] TIMEMORY_LAMBDA(_Tp & a, const _Tp& b, const _Tp& c) {
            a = a * b + c;
        };
        // test getting the cache info
        auto lm_size = tim::ert::cache_size::get_max();
        // log the cache info
        std::cout << "[INFO]> max cache size: " << (lm_size / tim::units::kilobyte)
                  << " KB\n"
                  << std::endl;
        // log how many threads were used
        printf("[INFO]> Running ERT with %li threads...\n\n",
               static_cast<long>(num_threads));
        // create the execution parameters
        tim::ert::exec_params params(working_size, memory_factor * lm_size, num_threads);
        // create the operation counter
        auto op_counter = new tim::ert::operation_counter<device_t, _Tp>(params, 64);
        // set bytes per element
        op_counter->bytes_per_element = sizeof(_Tp);
        // set number of memory accesses per element from two functions
        op_counter->memory_accesses_per_element = 2;
        // run the operation counter kernels
        tim::ert::ops_main<1>(*op_counter, add_func, store_func);
        tim::ert::ops_main<4, 5, 6, 7, 8>(*op_counter, fma_func, store_func);
        // return this data for processing
        return op_counter;
    };
}
//--------------------------------------------------------------------------------------//
