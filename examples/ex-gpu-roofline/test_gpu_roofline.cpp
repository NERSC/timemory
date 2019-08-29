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

#if ROOFLINE_FP_BYTES == 2
#    define TIMEMORY_CUDA_FP16
#endif

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

using float_type     = double;
using cpu_float_type = double;

#elif ROOFLINE_FP_BYTES == 4

using float_type     = float;
using cpu_float_type = float;

#elif ROOFLINE_FP_BYTES == 2

using float_type     = tim::device::gpu::fp16_t;
using cpu_float_type = tim::device::cpu::fp16_t;

#else

#    error "ROOFLINE_FP_BYTES must be either 2, 4, or 8"

#endif

using cpu_roofline_t = cpu_roofline<cpu_float_type>;
using gpu_roofline_t = gpu_roofline<float_type>;
using fib_list_t     = std::vector<int64_t>;
using auto_tuple_t =
    tim::auto_tuple<real_clock, cpu_clock, cpu_util, gpu_roofline_t, cpu_roofline_t>;
using device_t       = tim::device::gpu;
using params_t       = tim::device::params<device_t>;
using stream_t       = tim::cuda::stream_t;
using default_device = device_t;
using fp16_t         = tim::cuda::fp16_t;

// unless specified number of threads, use the number of available cores
#if !defined(NUM_THREADS)
#    define NUM_THREADS 1
#endif

#if !defined(NUM_STREAMS)
#    define NUM_STREAMS 1
#endif

//--------------------------------------------------------------------------------------//
// amypx calculation
//
template <typename _Tp>
GLOBAL_CALLABLE void
amypx(int64_t n, _Tp* x, _Tp* y, int64_t nitr)
{
    auto range = tim::device::grid_strided_range<default_device, 0, int32_t>(n);
    for(int64_t j = 0; j < nitr; ++j)
    {
        for(int i = range.begin(); i < range.end(); i += range.stride())
            y[i] = static_cast<_Tp>(2.0) * y[i] + x[i];
    }
}

//--------------------------------------------------------------------------------------//
// amypx calculation
//
template <>
GLOBAL_CALLABLE void
amypx(int64_t n, fp16_t* x, fp16_t* y, int64_t nitr)
{
    auto range = tim::device::grid_strided_range<default_device, 0, int32_t>(n);
    for(int64_t j = 0; j < nitr; ++j)
    {
        for(int i = range.begin(); i < range.end(); i += range.stride())
        {
            fp16_t a = { 2.0, 2.0 };
            y[i]     = a * y[i] + x[i];
        }
    }
}

//--------------------------------------------------------------------------------------//

template <typename _Tp>
void customize_gpu_roofline(int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);

//--------------------------------------------------------------------------------------//

template <typename _Tp>
void
exec_amypx(int64_t data_size, int64_t nitr, params_t params)
{
    auto label = TIMEMORY_JOIN("_", data_size, nitr, tim::demangle(typeid(_Tp).name()));

    _Tp* y = tim::cuda::malloc<_Tp>(data_size);
    _Tp* x = tim::cuda::malloc<_Tp>(data_size);
    tim::device::launch(data_size, params, amypx<_Tp>, data_size, x, y, 1);
    tim::cuda::device_sync();
    tim::device::launch(data_size, params,
                        tim::ert::initialize_buffer<device_t, _Tp, int64_t>, y, 0.0,
                        data_size);
    tim::device::launch(data_size, params,
                        tim::ert::initialize_buffer<device_t, _Tp, int64_t>, x, 1.0,
                        data_size);
    tim::cuda::device_sync();

    TIMEMORY_BLANK_CALIPER(0, auto_tuple_t, "amypx_", label);
    tim::device::launch(data_size, params, amypx<_Tp>, data_size, x, y, nitr);
    tim::cuda::device_sync();
    TIMEMORY_CALIPER_APPLY(0, stop);

    tim::cuda::free(x);
    tim::cuda::free(y);
}

//--------------------------------------------------------------------------------------//

template <>
void
exec_amypx<fp16_t>(int64_t data_size, int64_t nitr, params_t params)
{
    using _Tp  = fp16_t;
    auto label = TIMEMORY_JOIN("_", data_size, nitr, tim::demangle(typeid(_Tp).name()));

    _Tp* y = tim::cuda::malloc<_Tp>(data_size);
    _Tp* x = tim::cuda::malloc<_Tp>(data_size);
    tim::device::launch(data_size, params, amypx<_Tp>, data_size, x, y, 1);
    tim::cuda::device_sync();
    tim::device::launch(data_size, params,
                        tim::ert::initialize_buffer<device_t, _Tp, int64_t>, y,
                        _Tp({ 0.0, 0.0 }), data_size);
    tim::device::launch(data_size, params,
                        tim::ert::initialize_buffer<device_t, _Tp, int64_t>, x,
                        _Tp({ 1.0, 1.0 }), data_size);
    tim::cuda::device_sync();

    TIMEMORY_BLANK_CALIPER(0, auto_tuple_t, "amypx_", label);
    tim::device::launch(data_size, params, amypx<_Tp>, data_size, x, y, nitr);
    tim::cuda::device_sync();
    TIMEMORY_CALIPER_APPLY(0, stop);

    tim::cuda::free(x);
    tim::cuda::free(y);
}

//--------------------------------------------------------------------------------------//

int
main(int argc, char** argv)
{
    tim::settings::json_output() = true;
    tim::timemory_init(argc, argv);
    // TIMEMORY_CONFIGURE(gpu_roofline_t);
    tim::cuda::device_query();
    tim::cuda::set_device(0);

    int64_t num_threads   = NUM_THREADS;               // default number of threads
    int64_t num_streams   = NUM_STREAMS;               // default number of streams
    int64_t working_size  = 1 * tim::units::megabyte;  // default working set size
    int64_t memory_factor = 10;                        // default multiple of 500 MB
    int64_t iterations    = 1000;
    int64_t block_size    = 1024;
    int64_t grid_size     = 0;
    int64_t data_size     = 10000000;

    auto usage = [&]() {
        using lli = long long int;
        printf(
            "\n%s [%s = %lli] [%s = %lli] [%s = %lli] [%s = %lli] [%s = %lli] [%s = "
            "%lli] "
            "[%s = %lli] [%s = %lli]\n\n",
            argv[0], "num_threads", (lli) num_threads, "num_streams", (lli) num_streams,
            "working_size", (lli) working_size, "memory_factor", (lli) memory_factor,
            "iterations", (lli) iterations, "block_size", (lli) block_size, "grid_size",
            (lli) grid_size, "data_size", (lli) data_size);
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

    usage();

    customize_gpu_roofline<float_type>(num_threads, working_size, memory_factor,
                                       num_streams, grid_size, block_size);

    // ensure cpu version also runs the same number of threads
    using cpu_ert_config_t = typename cpu_roofline_t::ert_config_type<cpu_float_type>;
    cpu_ert_config_t::get_num_threads() = [=]() { return num_threads; };

    params_t params(grid_size, block_size, 0, 0);

    real_clock total;
    total.start();

    std::vector<stream_t> streams;
    if(num_streams > 1)
    {
        streams.resize(num_streams);
        for(auto& itr : streams)
            tim::cuda::stream_create(itr);
    }

    //
    // overall timing
    //
    auto _main = TIMEMORY_BLANK_INSTANCE(auto_tuple_t, argv[0]);
    _main.report_at_exit(true);

    //
    // run amypx calculations
    //
    using fp16_t = tim::device::gpu::fp16_t;

    exec_amypx<fp16_t>(data_size / 2, iterations * 2, params);
    exec_amypx<fp16_t>(data_size, iterations, params);
    exec_amypx<fp16_t>(data_size * 2, iterations / 2, params);

    exec_amypx<float>(data_size / 2, iterations * 2, params);
    exec_amypx<float>(data_size, iterations, params);
    exec_amypx<float>(data_size * 2, iterations / 2, params);

    exec_amypx<double>(data_size / 2, iterations * 2, params);
    exec_amypx<double>(data_size, iterations, params);
    exec_amypx<double>(data_size * 2, iterations / 2, params);

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

template <typename _Tp>
void
customize_gpu_roofline(int64_t num_threads, int64_t working_size, int64_t memory_factor,
                       int64_t num_streams, int64_t grid_size, int64_t block_size)
{
    using ert_params_t   = typename gpu_roofline_t::ert_params_t;
    using ert_data_t     = typename gpu_roofline_t::ert_data_t;
    using ert_data_ptr_t = typename gpu_roofline_t::ert_data_ptr_t;
    using ert_counter_t  = typename gpu_roofline_t::ert_counter_type<_Tp>;
    using ert_config_t   = typename gpu_roofline_t::ert_config_type<_Tp>;
    using ert_executor_t = typename gpu_roofline_t::ert_executor_type<_Tp>;

    //
    // simple modifications to override method number of threads, number of streams,
    // block size, and grid size
    //
    ert_config_t::get_num_threads() = [=]() { return num_threads; };
    ert_config_t::get_num_streams() = [=]() { return num_streams; };
    ert_config_t::get_block_size()  = [=]() { return block_size; };
    ert_config_t::get_grid_size()   = [=]() { return grid_size; };

    //
    // fully customize the roofline
    //
    if(tim::get_env<bool>("CUSTOMIZE_ROOFLINE", false))
    {
        // overload the finalization function that runs ERT calculations
        ert_config_t::get_executor() = [=](ert_data_ptr_t data) {
            // test getting the cache info
            auto lm_size = 500 * tim::units::megabyte;
            // log the cache info
            std::cout << "[INFO]> max cache size: " << (lm_size / tim::units::kilobyte)
                      << " KB\n"
                      << std::endl;
            // log how many threads were used
            printf("[INFO]> Running ERT with %li threads...\n\n",
                   static_cast<long>(num_threads));
            // create the execution parameters
            ert_params_t params(working_size, memory_factor * lm_size, num_threads,
                                num_streams, grid_size, block_size);
            // create the operation _counter
            ert_counter_t _counter(params, data, 64);
            return _counter;
        };

        // does the execution of ERT
        auto callback = [=](ert_counter_t& _counter) {
            // these are the kernel functions we want to calculate the peaks with
            auto store_func = [] TIMEMORY_LAMBDA(_Tp & a, const _Tp& b) { a = b; };
            auto add_func   = [] TIMEMORY_LAMBDA(_Tp & a, const _Tp& b, const _Tp& c) {
                a = b + c;
            };
            auto fma_func = [] TIMEMORY_LAMBDA(_Tp & a, const _Tp& b, const _Tp& c) {
                a = a * b + c;
            };

            // set bytes per element
            _counter.bytes_per_element = sizeof(_Tp);
            // set number of memory accesses per element from two functions
            _counter.memory_accesses_per_element = 2;

            // set the label
            _counter.label = "scalar_add";
            // run the operation _counter kernels
            tim::ert::ops_main<1>(_counter, add_func, store_func);

            // set the label
            _counter.label = "vector_add";
            // run the operation _counter kernels
            tim::ert::ops_main<4, 8, 16>(_counter, add_func, store_func);

            // set the label
            _counter.label = "vector_fma";
            tim::ert::ops_main<2, 4, 6, 8, 10, 12, 16, 32, 64, 96, 128, 192, 256, 512>(
                _counter, fma_func, store_func);
        };

        // set the callback
        gpu_roofline_t::set_executor_callback<_Tp>(callback);
    }
}

//--------------------------------------------------------------------------------------//
