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

#include "gtest/gtest.h"

#include <timemory/backends/device.hpp>
#include <timemory/components/cupti_event.hpp>
#include <timemory/timemory.hpp>

#include <chrono>
#include <iostream>
#include <random>
#include <thread>
#include <vector>

using namespace tim::component;

//--------------------------------------------------------------------------------------//

using default_device = tim::device::default_device;
using params_t       = tim::device::params<default_device>;

static auto max_size = tim::get_env("MAX_SIZE", 64);
static auto num_data = tim::get_env("NUM_SIZE", 100);
static auto num_iter = tim::get_env("NUM_ITER", 10);

//--------------------------------------------------------------------------------------//

namespace details
{
//--------------------------------------------------------------------------------------//
//  Get the current tests name
//
inline std::string
get_test_name()
{
    return ::testing::UnitTest::GetInstance()->current_test_info()->name();
}
//--------------------------------------------------------------------------------------//
// saxpy calculation
//
GLOBAL_CALLABLE void
saxpy(int64_t n, float a, float* x, float* y)
{
    auto range = tim::device::grid_strided_range<default_device, 0>(n);
    for(int i = range.begin(); i < range.end(); i += range.stride())
    {
        y[i] = a * x[i] + y[i];
    }
}
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
//  print an array to a string
//
template <typename _Tp>
std::string
array_to_string(const _Tp& arr, const std::string& delimiter = ", ",
                const int& _width = 16, const int& _break = 8,
                const std::string& _break_delim = "\t")
{
    auto size      = std::distance(arr.begin(), arr.end());
    using int_type = decltype(size);
    std::stringstream ss;
    for(int_type i = 0; i < size; ++i)
    {
        ss << std::setw(_width) << arr.at(i);
        if(i + 1 < size)
            ss << delimiter;
        if((i + 1) % _break == 0 && (i + 1) < size)
            ss << "\n" << _break_delim;
    }
    return ss.str();
}
//======================================================================================//
namespace impl
{
template <typename T>
GLOBAL_CALLABLE void
KERNEL_A(T* begin, int n)
{
    auto range = tim::device::grid_strided_range<default_device, 0>(n);
    for(int i = range.begin(); i < range.end(); i += range.stride())
    {
        if(i < n)
            *(begin + i) += 2.0f * n;
    }
}

//--------------------------------------------------------------------------------------//

template <typename T>
GLOBAL_CALLABLE void
KERNEL_B(T* begin, int n)
{
    auto range = tim::device::grid_strided_range<default_device, 0>(n);
    for(int i = range.begin(); i < range.end(); i += range.stride())
    {
        if(i < n / 2)
            *(begin + i) *= 2.0f;
        else if(i >= n / 2 && i < n)
            *(begin + i) += 3.0f;
    }
}
}  // namespace impl
//--------------------------------------------------------------------------------------//

template <typename T>
void
KERNEL_A(T* arg, int size, tim::cuda::stream_t stream = 0)
{
    params_t params(64, 2, 0, stream);
    tim::device::launch(params, impl::KERNEL_A<T>, arg, size);
}

//--------------------------------------------------------------------------------------//
template <typename T>
void
KERNEL_B(T* arg, int size, tim::cuda::stream_t stream = 0)
{
    params_t params(2, 64, 0, stream);
    tim::device::launch(params, impl::KERNEL_B<T>, arg, size / 2);
}

}  // namespace details

//--------------------------------------------------------------------------------------//

class cupti_tests : public ::testing::Test
{
};

//--------------------------------------------------------------------------------------//

TEST_F(cupti_tests, ert)
{
    /*
    CUdevice device;
    CUDA_DRIVER_API_CALL(cuInit(0));
    CUDA_DRIVER_API_CALL(cuDeviceGet(&device, 0));

    auto event_names  = tim::cupti::available_events(device);
    auto metric_names = tim::cupti::available_metrics(device);
    std::sort(event_names.begin(), event_names.end());
    std::sort(metric_names.begin(), metric_names.end());

    using size_type = decltype(event_names.size());
    size_type wevt  = 10;
    size_type wmet  = 10;
    for(const auto& itr : event_names)
        wevt = std::max(itr.size(), wevt);
    for(const auto& itr : metric_names)
        wmet = std::max(itr.size(), wmet);

    std::cout << "Event names: \n\t"
              << details::array_to_string(event_names, ", ", wevt, 200 / wevt)
              << std::endl;
    std::cout << "Metric names: \n\t"
              << details::array_to_string(metric_names, ", ", wmet, 200 / wmet)
              << std::endl;
    */
    cupti_event::get_device_setter() = []() { return std::vector<int>({ 0 }); };
    cupti_event::get_event_setter()  = []() {
        return std::vector<std::string>({ "active_warps", "active_cycles", "global_load",
                                          "global_store", "gld_inst_32bit",
                                          "gst_inst_32bit" });
    };
    cupti_event::get_metric_setter() = []() {
        return std::vector<std::string>({ "inst_per_warp", "branch_efficiency",
                                          "warp_execution_efficiency", "flop_count_sp",
                                          "flop_count_sp_add", "flop_count_sp_fma",
                                          "flop_count_sp_mul", "flop_sp_efficiency",
                                          "gld_efficiency", "gst_efficiency" });
    };

    using _Tp                 = double;
    using operation_counter_t = tim::ert::gpu::operation_counter<_Tp>;

    tim::ert::exec_params params(16, 64 * 64);
    auto                  op_counter = new operation_counter_t(params, 64);

    std::vector<float> cpu_data(num_data, 0);
    float*             data = tim::device::gpu::alloc<float>(num_data);
    tim::cuda::memcpy(data, cpu_data.data(), num_data, tim::cuda::host_to_device_v, 0);

    using tuple_t = tim::auto_tuple<real_clock, cupti_event>::component_type;
    tuple_t timer(details::get_test_name());

    timer.start();
    // tim::cupti::profiler profiler(cupti_event::get_event_setter()(),
    //                              cupti_event::get_metric_setter()());
    // profiler.start();
    for(int i = 0; i < num_iter; ++i)
    {
        printf("\n[%s]> iteration %i...\n", __FUNCTION__, i);
        // tim::ert::gpu_ops_main<1>(*op_counter, details::add_func<_Tp>);
        // tim::ert::gpu_ops_main<2, 4, 8>(*op_counter, details::fma_func<_Tp>);
        details::KERNEL_A(data, num_data);
        details::KERNEL_B(data, num_data);
    }
    tim::cuda::device_sync();
    timer.stop();
    // profiler.stop();

    // std::cout << *op_counter << std::endl;
    std::cout << timer << std::endl;
    // printf("Event Trace\n");
    // profiler.print_event_values(std::cout);
    // printf("Metric Trace\n");
    // profiler.print_metric_values(std::cout);

    tim::device::gpu::free(data);
    printf("\n");
    tim::cuda::device_reset();
}

//--------------------------------------------------------------------------------------//

int
main(int argc, char** argv)
{
    auto manager = tim::manager::instance();
    tim::consume_parameters(manager);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

//--------------------------------------------------------------------------------------//
