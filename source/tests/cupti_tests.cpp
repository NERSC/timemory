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

static const auto num_data = 96;
static const auto num_iter = 10;
static const auto num_blck = 64;
static const auto num_grid = 2;
static const auto epsilon  = std::numeric_limits<float>::epsilon();

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
KERNEL_A(T* arr, int size, tim::cuda::stream_t stream = 0)
{
    // this kernel is designed for size / (64.0 * 2.0) operations
    // of 128
    params_t params(num_grid, num_blck, 0, stream);
    tim::device::launch(params, impl::KERNEL_A<T>, arr, size);
}

//--------------------------------------------------------------------------------------//
template <typename T>
void
KERNEL_B(T* arr, int size, tim::cuda::stream_t stream = 0)
{
    // this kernel is designed for (64.0 / 2.0) * (size / (64.0 * 2.0)) operations
    params_t params(num_blck, num_grid, 0, stream);
    tim::device::launch(params, impl::KERNEL_B<T>, arr, size / 2);
}

}  // namespace details

//--------------------------------------------------------------------------------------//

class cupti_tests : public ::testing::Test
{
};

//--------------------------------------------------------------------------------------//

TEST_F(cupti_tests, available)
{
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
}

//--------------------------------------------------------------------------------------//

TEST_F(cupti_tests, kernels)
{
    cupti_event::get_device_setter() = []() { return std::vector<int>({ 0 }); };
    cupti_event::get_event_setter()  = []() {
        return std::vector<std::string>({ "active_warps", "active_cycles", "global_load",
                                          "global_store", "gld_inst_32bit",
                                          "gst_inst_32bit" });
    };
    cupti_event::get_metric_setter() = []() {
        return std::vector<std::string>({ "inst_per_warp", "branch_efficiency",
                                          "warp_execution_efficiency", "flop_count_sp",
                                          "flop_sp_efficiency", "gld_efficiency",
                                          "gst_efficiency" });
    };

    std::vector<float> cpu_data(num_data, 0);
    float*             data = tim::device::gpu::alloc<float>(num_data);
    tim::cuda::memcpy(data, cpu_data.data(), num_data, tim::cuda::host_to_device_v, 0);

    using tuple_t = tim::auto_tuple<real_clock, cupti_event>::component_type;
    tuple_t timer(details::get_test_name());

    timer.start();
    for(int i = 0; i < num_iter; ++i)
    {
        printf("\n[%s]> iteration %i...\n", __FUNCTION__, i);
        details::KERNEL_A(data, num_data);
        details::KERNEL_B(data, num_data);
    }
    timer.stop();
    std::cout << timer << std::endl;
    tim::device::gpu::free(data);
    tim::cuda::device_sync();
    tim::cuda::device_reset();
    printf("\n");

    auto cupti_data   = timer.get<cupti_event>().get();
    auto cupti_labels = timer.get<cupti_event>().label_array();
    std::cout << "CUPTI: data size = " << cupti_data.size()
              << ", label size = " << cupti_labels.size() << "\n"
              << std::endl;

    double num_active = num_blck * num_grid;
    double ratio      = num_data / static_cast<double>(num_active);

    auto A_glob_load     = (2.0 * num_grid) * ratio;
    auto B_glob_load     = (1.0 * num_blck / num_grid) * ratio;
    auto A_glob_store    = (2.0 * num_grid) * ratio;
    auto B_glob_store    = (1.0 * num_blck / num_grid) * ratio;
    auto A_warp_eff      = (num_blck % 32 == 0) ? 1.0 : ((num_blck % 32) / 32.0);
    auto B_warp_eff      = (num_grid % 32 == 0) ? 1.0 : ((num_grid % 32) / 32.0);
    auto A_gld_inst_32b  = 1.0 * num_data;
    auto B_gld_inst_32b  = 0.5 * num_data;
    auto A_gst_inst_32b  = 1.0 * num_data;
    auto B_gst_inst_32b  = 0.5 * num_data;
    auto A_flop_count_sp = 2.00 * num_data;
    auto B_flop_count_sp = 0.75 * num_data;
    // these are inherent to kernel design
    auto A_glob_load_eff  = 100.0;
    auto B_glob_load_eff  = 25.0;
    auto A_glob_store_eff = 100.0;
    auto B_glob_store_eff = 25.0;

    std::map<std::string, double> cupti_map;
    for(uint64_t i = 0; i < cupti_data.size(); ++i)
    {
        std::cout << "    " << std::setw(28) << cupti_labels[i] << " : " << std::setw(12)
                  << std::setprecision(6) << cupti_data[i] << std::endl;
        cupti_map[cupti_labels[i]] = cupti_data[i];
    }

    auto global_load      = num_iter * (A_glob_load + B_glob_load);
    auto global_store     = num_iter * (A_glob_store + B_glob_store);
    auto warp_eff         = 0.5 * (A_warp_eff + B_warp_eff) * 100.0;
    auto global_load_eff  = 0.5 * (A_glob_load_eff + B_glob_load_eff);
    auto global_store_eff = 0.5 * (A_glob_store_eff + B_glob_store_eff);
    auto gld_inst_32b     = num_iter * (A_gld_inst_32b + B_gld_inst_32b);
    auto gst_inst_32b     = num_iter * (A_gst_inst_32b + B_gst_inst_32b);
    auto flop_count_sp    = num_iter * (A_flop_count_sp + B_flop_count_sp);

    printf("A flop = %f\n", A_flop_count_sp);
    printf("B flop = %f\n", B_flop_count_sp);

    ASSERT_NEAR(cupti_map["global_load"], global_load, epsilon);
    ASSERT_NEAR(cupti_map["global_store"], global_store, epsilon);
    ASSERT_NEAR(cupti_map["warp_execution_efficiency"], warp_eff, epsilon);
    ASSERT_NEAR(cupti_map["gld_efficiency"], global_load_eff, epsilon);
    ASSERT_NEAR(cupti_map["gst_efficiency"], global_store_eff, epsilon);
    ASSERT_NEAR(cupti_map["gld_inst_32bit"], gld_inst_32b, epsilon);
    ASSERT_NEAR(cupti_map["gst_inst_32bit"], gst_inst_32b, epsilon);
    ASSERT_NEAR(cupti_map["flop_count_sp"], flop_count_sp, epsilon);
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
