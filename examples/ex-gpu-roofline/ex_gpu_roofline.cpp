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

using device_t       = tim::device::gpu;
using params_t       = tim::device::params<device_t>;
using stream_t       = tim::cuda::stream_t;
using default_device = device_t;

#if !defined(TIMEMORY_DISABLE_CUDA_HALF2)
using fp16_t = tim::cuda::fp16_t;
#endif

//--------------------------------------------------------------------------------------//

#if ROOFLINE_FP_BYTES == 8

using gpu_roofline_t = gpu_roofline_dp_flops;
using cpu_roofline_t = cpu_roofline_dp_flops;

#elif ROOFLINE_FP_BYTES == 4

using gpu_roofline_t = gpu_roofline_sp_flops;
using cpu_roofline_t = cpu_roofline_sp_flops;

#elif(ROOFLINE_FP_BYTES == 2) && !defined(TIMEMORY_DISABLE_CUDA_HALF2)

using gpu_roofline_t = gpu_roofline_hp_flops;
using cpu_roofline_t = cpu_roofline_sp_flops;

#else

using gpu_roofline_t = gpu_roofline_flops;
using cpu_roofline_t = cpu_roofline_flops;

#endif

//--------------------------------------------------------------------------------------//

using auto_tuple_t =
    tim::auto_tuple<real_clock, cpu_clock, cpu_util, gpu_roofline_t, cpu_roofline_t>;

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
#if !defined(TIMEMORY_DISABLE_CUDA_HALF2)
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
#endif

//--------------------------------------------------------------------------------------//

template <typename _Tp>
void customize_roofline(int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);

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
#if !defined(TIMEMORY_DISABLE_CUDA_HALF2)
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
#endif
//--------------------------------------------------------------------------------------//

int
main(int argc, char** argv)
{
    tim::settings::json_output() = true;
    tim::timemory_init(argc, argv);
    // TIMEMORY_CONFIGURE(gpu_roofline_t);
    tim::cuda::device_query();
    tim::cuda::set_device(0);

    int64_t num_threads   = 1;                         // default number of threads
    int64_t num_streams   = 1;                         // default number of streams
    int64_t working_size  = 1 * tim::units::megabyte;  // default working set size
    int64_t memory_factor = 10;                        // default multiple of 500 MB
    int64_t iterations    = 1000;
    int64_t block_size    = 1024;
    int64_t grid_size     = 0;
    int64_t data_size     = 10000000;

    auto usage = [&]() {
        using lli = long long int;
        printf("\n%s [%s = %lli] [%s = %lli] [%s = %lli] [%s = %lli] [%s = %lli] [%s = "
               "%lli] "
               "[%s = %lli] [%s = %lli]\n\n",
               argv[0], "num_threads", (lli) num_threads, "num_streams",
               (lli) num_streams, "working_size", (lli) working_size, "memory_factor",
               (lli) memory_factor, "iterations", (lli) iterations, "block_size",
               (lli) block_size, "grid_size", (lli) grid_size, "data_size",
               (lli) data_size);
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

#if !defined(TIMEMORY_DISABLE_CUDA_HALF2)
    customize_roofline<fp16_t>(num_threads, working_size, memory_factor, num_streams,
                               grid_size, block_size);
#endif

    customize_roofline<float>(num_threads, working_size, memory_factor, num_streams,
                              grid_size, block_size);
    customize_roofline<double>(num_threads, working_size, memory_factor, num_streams,
                               grid_size, block_size);

    // ensure cpu version also runs the same number of threads
    tim::settings::ert_num_threads_cpu() = num_threads;

    params_t params(grid_size, block_size, 0, 0);

    real_clock total;
    total.start();

    //
    // overall timing
    //
    auto _main = TIMEMORY_BLANK_HANDLE(auto_tuple_t, argv[0]);
    _main.report_at_exit(true);

    //
    // run amypx calculations
    //

#if !defined(TIMEMORY_DISABLE_CUDA_HALF2)
    exec_amypx<fp16_t>(data_size / 2, iterations * 2, params);
    exec_amypx<fp16_t>(data_size, iterations, params);
    exec_amypx<fp16_t>(data_size * 2, iterations / 2, params);
#endif

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
customize_roofline(int64_t num_threads, int64_t working_size, int64_t memory_factor,
                   int64_t num_streams, int64_t grid_size, int64_t block_size)
{
    using namespace tim;
    using counter_t         = component::real_clock;
    using ert_data_t        = ert::exec_data;
    using ert_params_t      = ert::exec_params;
    using ert_data_ptr_t    = std::shared_ptr<ert_data_t>;
    using ert_executor_type = ert::executor<device_t, _Tp, ert_data_t, counter_t>;
    using ert_config_type   = typename ert_executor_type::configuration_type;
    using ert_counter_type  = typename ert_executor_type::counter_type;

    //
    // simple modifications to override method number of threads, number of streams,
    // block size, and grid size
    //
    ert_config_type::configure(num_threads, 64, num_streams, block_size, grid_size);

    //
    // fully customize the roofline
    //
    if(tim::get_env<bool>("CUSTOMIZE_ROOFLINE", false))
    {
        // overload the finalization function that runs ERT calculations
        ert_config_type::get_executor() = [=](ert_data_ptr_t data) {
            auto lm_size = 100 * units::megabyte;
            // create the execution parameters
            ert_params_t params(working_size, memory_factor * lm_size, num_threads,
                                num_streams, grid_size, block_size);
            // create the operation _counter
            ert_counter_type _counter(params, data, 64);
            std::cout << _counter << std::endl;
            return _counter;
        };

        // does the execution of ERT
        auto callback = [=](ert_counter_type& _counter) {
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
            _counter.label = "vector_fma";
            tim::ert::ops_main<2, 4, 6, 8, 10, 12, 16, 32, 64, 96, 128, 192, 256, 512>(
                _counter, fma_func, store_func);
        };

        // set the callback
        gpu_roofline_t::set_executor_callback<_Tp>(callback);
    }
}

//--------------------------------------------------------------------------------------//
