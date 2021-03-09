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

#include <cassert>
#include <chrono>
#include <cmath>
#include <fstream>
#include <future>
#include <iomanip>
#include <iterator>
#include <random>
#include <thread>
#include <unordered_map>
#include <vector>

#include "timemory/ert/kernels.hpp"
#include "timemory/timemory.hpp"
#include "timemory/utility/signals.hpp"
#include "timemory/utility/testing.hpp"

#if defined(TIMEMORY_USE_CUPTI)
#    include "timemory/components/cupti/cupti_counters.hpp"
#endif

using namespace tim::component;

// using papi_tuple_t = papi_tuple<PAPI_TOT_CYC, PAPI_TOT_INS, PAPI_BR_MSP, PAPI_BR_PRC>;
using auto_tuple_t = tim::auto_tuple_t<wall_clock, system_clock, cpu_clock, cpu_util,
                                       nvtx_marker, papi_array_t>;
using comp_tuple_t = typename auto_tuple_t::component_type;
using cuda_tuple_t = tim::auto_tuple_t<cuda_event, nvtx_marker>;
using counter_t    = ert_timer;
using ert_data_t   = tim::ert::exec_data<counter_t>;

//======================================================================================//

#define CUDA_CHECK_LAST_ERROR()                                                          \
    {                                                                                    \
        tim::cuda::stream_sync(0);                                                       \
        cudaError err = cudaGetLastError();                                              \
        if(cudaSuccess != err)                                                           \
        {                                                                                \
            fprintf(stderr, "cudaCheckError() failed at %s@'%s':%i : %s\n",              \
                    __FUNCTION__, __FILE__, __LINE__, cudaGetErrorString(err));          \
            std::stringstream ss;                                                        \
            ss << "cudaCheckError() failed at " << __FUNCTION__ << "@'" << __FILE__      \
               << "':" << __LINE__ << " : " << cudaGetErrorString(err);                  \
            throw std::runtime_error(ss.str());                                          \
        }                                                                                \
    }

//======================================================================================//

template <typename Tp>
std::string
array_to_string(const Tp& arr, const std::string& delimiter = ", ",
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

//--------------------------------------------------------------------------------------//

static const int nitr = 10;
static int64_t   N    = 50 * (1 << 23);
static auto      Nsub = N / nitr;

//--------------------------------------------------------------------------------------//
// saxpy calculation
TIMEMORY_GLOBAL_FUNCTION void
warmup(int64_t n)
{
    int i   = blockIdx.x * blockDim.x + threadIdx.x;
    int tmp = 0;
    if(i < n)
        tmp += i;
}

//--------------------------------------------------------------------------------------//
// saxpy calculation
TIMEMORY_GLOBAL_FUNCTION void
saxpy(int64_t n, float a, float* x, float* y)
{
    auto itr = tim::device::grid_strided_range<tim::device::default_device, 0>(n);
    for(int i = itr.begin(); i < itr.end(); i += itr.stride())
        y[i] = a * x[i] + y[i];
}
//--------------------------------------------------------------------------------------//

void
warmup()
{
    int     block = 128;
    int     ngrid = 128;
    int64_t val   = 256;
    warmup<<<ngrid, block>>>(val);
    // CUDA_CHECK_LAST_ERROR();
}

//======================================================================================//

void
print_info(const std::string&);
void
print_string(const std::string& str);
void
test_1_saxpy();
void
test_2_saxpy_async();
void
test_3_saxpy_pinned();
void
test_4_saxpy_async_pinned();
void
test_5_mt_saxpy_async();
void
test_6_mt_saxpy_async_pinned();
void
test_7_cupti_available();
void
test_8_cupti_subset();
void
test_9_cupti_counters();
void
test_10_cupti_metric();

//======================================================================================//

int
main(int argc, char** argv)
{
    if(N % nitr != 0)
    {
        throw std::runtime_error("Error N is not a multiple of nitr");
    }

    tim::settings::timing_scientific() = true;
    tim::timemory_init(argc, argv);
    tim::settings::json_output() = true;
    tim::enable_signal_detection();

    int ndevices = tim::cuda::device_count();
    warmup();

    auto* timing = new tim::component_tuple<wall_clock, system_clock, cpu_clock, cpu_util,
                                            nvtx_marker>("Tests runtime", true);

    timing->start();

    CONFIGURE_TEST_SELECTOR(10);

    int num_fail = 0;
    int num_test = 0;

    if(ndevices == 0)
    {
        for(auto i : { 3, 4, 6 })
        {
            if(tests.count(i) > 0)
                tests.erase(tests.find(i));
        }
    }

    std::cout << "# tests: " << tests.size() << std::endl;
    try
    {
        RUN_TEST(1, test_1_saxpy, num_test, num_fail);
        RUN_TEST(2, test_2_saxpy_async, num_test, num_fail);
        RUN_TEST(3, test_3_saxpy_pinned, num_test, num_fail);
        RUN_TEST(4, test_4_saxpy_async_pinned, num_test, num_fail);
        RUN_TEST(5, test_5_mt_saxpy_async, num_test, num_fail);
        RUN_TEST(6, test_6_mt_saxpy_async_pinned, num_test, num_fail);
        RUN_TEST(7, test_7_cupti_available, num_test, num_fail);
        RUN_TEST(8, test_8_cupti_subset, num_test, num_fail);
        RUN_TEST(9, test_9_cupti_counters, num_test, num_fail);
        RUN_TEST(10, test_10_cupti_metric, num_test, num_fail);
    } catch(std::exception& e)
    {
        std::cerr << e.what() << std::endl;
    }

    timing->stop();
    std::cout << "\n" << *timing << std::endl;

    TEST_SUMMARY(argv[0], num_test, num_fail);
    delete timing;

    tim::timemory_finalize();
    exit(num_fail);
}

//======================================================================================//

void
print_info(const std::string& func)
{
    if(tim::dmp::rank() == 0)
    {
        std::cout << "\n[" << tim::dmp::rank() << "]\e[1;33m TESTING \e[0m["
                  << "\e[1;36m" << func << "\e[0m"
                  << "]...\n"
                  << std::endl;
    }
}

//======================================================================================//

void
print_string(const std::string& str)
{
    std::stringstream _ss;
    _ss << "[" << tim::dmp::rank() << "] " << str << std::endl;
    std::cout << _ss.str();
}

//======================================================================================//

void
test_1_saxpy()
{
    print_info(__FUNCTION__);
    warmup();
    TIMEMORY_BASIC_MARKER(auto_tuple_t, "");

    comp_tuple_t _clock("Runtime");
    _clock.start();

    float*      x;
    float*      y;
    float*      d_x;
    float*      d_y;
    int         block    = 512;
    int         ngrid    = (N + block - 1) / block;
    float       nseconds = 0.0f;
    float       maxError = 0.0f;
    float       sumError = 0.0f;
    cuda_event* evt      = nullptr;

    {
        TIMEMORY_BASIC_MARKER(auto_tuple_t, "[cpu_malloc]");
        x = tim::device::cpu::alloc<float>(N);
        y = tim::device::cpu::alloc<float>(N);
    }

    {
        TIMEMORY_BASIC_MARKER(auto_tuple_t, "[gpu_malloc]");
        d_x = tim::cuda::malloc<float>(N);
        d_y = tim::cuda::malloc<float>(N);
        CUDA_CHECK_LAST_ERROR();
    }

    {
        TIMEMORY_BASIC_MARKER(auto_tuple_t, "[assign]");
        for(int i = 0; i < N; i++)
        {
            x[i] = 1.0f;
            y[i] = 2.0f;
        }
    }

    {
        TIMEMORY_BASIC_MARKER(auto_tuple_t, "[create_event]");
        evt = new cuda_event();
    }

    {
        TIMEMORY_BASIC_MARKER(auto_tuple_t, "[H2D]");
        evt->mark_begin();
        tim::cuda::memcpy(d_x, x, N, tim::cuda::host_to_device_v);
        tim::cuda::memcpy(d_y, y, N, tim::cuda::host_to_device_v);
        evt->mark_end();
        CUDA_CHECK_LAST_ERROR();
    }

    for(int i = 0; i < 1; ++i)
    {
        TIMEMORY_BASIC_MARKER(auto_tuple_t, "[", i, "]");
        // Perform SAXPY on 1M elements
        evt->mark_begin();
        saxpy<<<ngrid, block>>>(N, 1.0f, d_x, d_y);
        evt->mark_end();
    }

    {
        TIMEMORY_BASIC_MARKER(auto_tuple_t, "[D2H]");
        evt->mark_begin();
        cudaMemcpy(y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);
        evt->mark_end();
    }

    tim::cuda::device_sync();

    {
        TIMEMORY_BASIC_MARKER(auto_tuple_t, "[check]");
        for(int64_t i = 0; i < N; i++)
        {
            maxError = std::max(maxError, std::abs(y[i] - 2.0f));
            sumError += std::abs(y[i] - 2.0f);
        }
    }

    evt->sync();
    nseconds += evt->get();
    _clock.stop();
    {
        TIMEMORY_BASIC_MARKER(auto_tuple_t, "[output]");
        std::cout << "Event: " << *evt << std::endl;
        std::cout << _clock << std::endl;
        printf("Max error: %8.4e\n", maxError);
        printf("Sum error: %8.4e\n", sumError);
        printf("Effective Bandwidth (GB/s): %f\n",
               N * 4 * 3 / nseconds / tim::units::gigabyte);
        printf("Kernel Runtime (sec): %16.12e\n", nseconds);
    }

    delete evt;

    tim::device::gpu::free(d_x);
    tim::device::gpu::free(d_y);
    tim::device::cpu::free(x);
    tim::device::cpu::free(y);

    tim::cuda::device_sync();
    // tim::cuda::device_reset();
}

//======================================================================================//

void
test_2_saxpy_async()
{
    print_info(__FUNCTION__);
    warmup();
    TIMEMORY_BASIC_MARKER(auto_tuple_t, "");

    comp_tuple_t _clock("Runtime");
    _clock.start();

    float*        x;
    float*        y;
    float*        d_x;
    float*        d_y;
    int           block    = 512;
    int           ngrid    = (Nsub + block - 1) / block;
    float         nseconds = 0.0f;
    float         maxError = 0.0f;
    float         sumError = 0.0f;
    cuda_event**  evt      = new cuda_event*[nitr];
    cudaStream_t* stream   = new cudaStream_t[nitr];

    auto _sync = [&]() {
        for(int i = 0; i < nitr; i++)
            tim::cuda::stream_sync(stream[i]);
    };

    {
        TIMEMORY_BASIC_MARKER(auto_tuple_t, "[cpu_malloc]");
        x = tim::device::cpu::alloc<float>(N);
        y = tim::device::cpu::alloc<float>(N);
    }

    {
        TIMEMORY_BASIC_MARKER(auto_tuple_t, "[gpu_malloc]");
        d_x = tim::cuda::malloc<float>(N);
        d_y = tim::cuda::malloc<float>(N);
    }

    {
        TIMEMORY_BASIC_MARKER(auto_tuple_t, "[assign]");
        for(int i = 0; i < N; i++)
        {
            x[i] = 1.0f;
            y[i] = 2.0f;
        }
    }

    {
        TIMEMORY_BASIC_MARKER(auto_tuple_t, "[create]");
        for(int i = 0; i < nitr; ++i)
        {
            tim::cuda::stream_create(stream[i]);
            evt[i] = new cuda_event(stream[i]);
            evt[i]->start();
        }
    }

    {
        TIMEMORY_BASIC_MARKER(auto_tuple_t, "[H2D]");
        for(int i = 0; i < nitr; ++i)
        {
            auto   offset = Nsub * i;
            float* _x     = x + offset;
            float* _y     = y + offset;
            float* _dx    = d_x + offset;
            float* _dy    = d_y + offset;
            evt[i]->mark_begin();
            tim::cuda::memcpy(_dx, _x, Nsub, tim::cuda::host_to_device_v, stream[i]);
            tim::cuda::memcpy(_dy, _y, Nsub, tim::cuda::host_to_device_v, stream[i]);
            evt[i]->mark_end();
        }
    }

    for(int i = 0; i < nitr; ++i)
    {
        TIMEMORY_BASIC_MARKER(auto_tuple_t, "[", i, "]");

        auto   offset = Nsub * i;
        float* _dx    = d_x + offset;
        float* _dy    = d_y + offset;

        evt[i]->mark_begin();
        // Perform SAXPY on 1M elements
        saxpy<<<ngrid, block, 0, stream[i]>>>(Nsub, 1.0f, _dx, _dy);
        evt[i]->mark_end();
    }

    {
        TIMEMORY_BASIC_MARKER(auto_tuple_t, "[D2H]");
        for(int i = 0; i < nitr; ++i)
        {
            auto   offset = Nsub * i;
            float* _y     = y + offset;
            float* _dy    = d_y + offset;
            evt[i]->mark_begin();
            tim::cuda::memcpy(_y, _dy, Nsub, tim::cuda::device_to_host_v, stream[i]);
            evt[i]->mark_end();
        }
    }

    _sync();

    {
        TIMEMORY_BASIC_MARKER(auto_tuple_t, "[check]");
        for(int64_t i = 0; i < N; i++)
        {
            maxError = std::max(maxError, std::abs(y[i] - 2.0f));
            sumError += std::abs(y[i] - 2.0f);
        }
    }

    _clock.stop();
    {
        TIMEMORY_BASIC_MARKER(auto_tuple_t, "[output]");
        cuda_event _evt = **evt;
        for(int i = 1; i < nitr; ++i)
        {
            evt[i]->stop();
            nseconds += evt[i]->get();
            _evt += *(evt[i]);
        }
        std::cout << "Event: " << _evt << std::endl;
        std::cout << _clock << std::endl;
        printf("Max error: %8.4e\n", maxError);
        printf("Sum error: %8.4e\n", sumError);
        printf("Effective Bandwidth (GB/s): %f\n",
               N * 4 * 3 / nseconds / tim::units::gigabyte);
        printf("Kernel Runtime (sec): %16.12e\n", nseconds);
    }

    for(int i = 0; i < nitr; ++i)
        delete evt[i];
    delete[] evt;

    tim::device::gpu::free(d_x);
    tim::device::gpu::free(d_y);
    tim::device::cpu::free(x);
    tim::device::cpu::free(y);

    tim::cuda::device_sync();
}

//======================================================================================//

void
test_3_saxpy_pinned()
{
    print_info(__FUNCTION__);
    warmup();
    TIMEMORY_BASIC_MARKER(auto_tuple_t, "");

    comp_tuple_t _clock("Runtime");
    _clock.start();

    float*      x;
    float*      y;
    float*      d_x;
    float*      d_y;
    int         block    = 512;
    int         ngrid    = (N + block - 1) / block;
    float       nseconds = 0.0f;
    float       maxError = 0.0f;
    float       sumError = 0.0f;
    cuda_event* evt      = nullptr;

    {
        TIMEMORY_BASIC_MARKER(auto_tuple_t, "[cpu_malloc]");
        x = tim::cuda::malloc_host<float>(N);
        y = tim::cuda::malloc_host<float>(N);
    }

    {
        TIMEMORY_BASIC_MARKER(auto_tuple_t, "[gpu_malloc]");
        d_x = tim::cuda::malloc<float>(N);
        d_y = tim::cuda::malloc<float>(N);
    }

    {
        TIMEMORY_BASIC_MARKER(auto_tuple_t, "[assign]");
        for(int i = 0; i < N; i++)
        {
            x[i] = 1.0f;
            y[i] = 2.0f;
        }
    }

    {
        TIMEMORY_BASIC_MARKER(auto_tuple_t, "[create_event]");
        evt = new cuda_event();
        evt->start();
    }

    {
        TIMEMORY_BASIC_MARKER(auto_tuple_t, "[H2D]");
        evt->mark_begin();
        tim::cuda::memcpy(d_x, x, N, tim::cuda::host_to_device_v);
        tim::cuda::memcpy(d_y, y, N, tim::cuda::host_to_device_v);
        evt->mark_end();
    }

    for(int i = 0; i < 1; ++i)
    {
        TIMEMORY_BASIC_MARKER(auto_tuple_t, "[", i, "]");
        evt->mark_begin();
        // Perform SAXPY on 1M elements
        saxpy<<<ngrid, block>>>(N, 1.0f, d_x, d_y);
        evt->mark_end();
    }

    {
        TIMEMORY_BASIC_MARKER(auto_tuple_t, "[D2H]");
        evt->mark_begin();
        tim::cuda::memcpy(y, d_y, N, tim::cuda::device_to_host_v);
        evt->mark_end();
    }

    tim::cuda::device_sync();

    {
        TIMEMORY_BASIC_MARKER(auto_tuple_t, "[check]");
        for(int64_t i = 0; i < N; i++)
        {
            maxError = std::max(maxError, std::abs(y[i] - 2.0f));
            sumError += std::abs(y[i] - 2.0f);
        }
    }

    _clock.stop();
    evt->stop();
    nseconds += evt->get();
    {
        TIMEMORY_BASIC_MARKER(auto_tuple_t, "[output]");
        std::cout << "Event: " << *evt << std::endl;
        std::cout << _clock << std::endl;
        printf("Max error: %8.4e\n", maxError);
        printf("Sum error: %8.4e\n", sumError);
        printf("Effective Bandwidth (GB/s): %f\n",
               N * 4 * 3 / nseconds / tim::units::gigabyte);
        printf("Kernel Runtime (sec): %16.12e\n", nseconds);
    }

    delete evt;

    tim::cuda::free_host(x);
    tim::cuda::free_host(y);
    tim::cuda::free(d_x);
    tim::cuda::free(d_y);

    tim::cuda::device_sync();
    // tim::cuda::device_reset();
}

//======================================================================================//

void
test_4_saxpy_async_pinned()
{
    print_info(__FUNCTION__);
    warmup();
    TIMEMORY_BASIC_MARKER(auto_tuple_t, "");

    comp_tuple_t _clock("Runtime");
    _clock.start();

    float*        x;
    float*        y;
    float*        d_x;
    float*        d_y;
    int           block    = 512;
    int           ngrid    = (Nsub + block - 1) / block;
    float         nseconds = 0.0f;
    float         maxError = 0.0f;
    float         sumError = 0.0f;
    cuda_event**  evt      = new cuda_event*[nitr];
    cudaStream_t* stream   = new cudaStream_t[nitr];

    auto _sync = [&]() {
        for(int i = 0; i < nitr; i++)
            tim::cuda::stream_sync(stream[i]);
    };

    {
        TIMEMORY_BASIC_MARKER(auto_tuple_t, "[cpu_malloc]");
        x = tim::cuda::malloc_host<float>(N);
        y = tim::cuda::malloc_host<float>(N);
    }

    {
        TIMEMORY_BASIC_MARKER(auto_tuple_t, "[gpu_malloc]");
        d_x = tim::cuda::malloc<float>(N);
        d_y = tim::cuda::malloc<float>(N);
    }

    {
        TIMEMORY_BASIC_MARKER(auto_tuple_t, "[assign]");
        for(int i = 0; i < N; i++)
        {
            x[i] = 1.0f;
            y[i] = 2.0f;
        }
    }

    {
        TIMEMORY_BASIC_MARKER(auto_tuple_t, "[create]");
        for(int i = 0; i < nitr; ++i)
        {
            tim::cuda::stream_create(stream[i]);
            evt[i] = new cuda_event(stream[i]);
            evt[i]->start();
        }
    }

    {
        TIMEMORY_BASIC_MARKER(auto_tuple_t, "[H2D]");
        for(int i = 0; i < nitr; ++i)
        {
            auto   offset = Nsub * i;
            float* _x     = x + offset;
            float* _y     = y + offset;
            float* _dx    = d_x + offset;
            float* _dy    = d_y + offset;
            evt[i]->mark_begin();
            tim::cuda::memcpy(_dx, _x, Nsub, tim::cuda::host_to_device_v, stream[i]);
            tim::cuda::memcpy(_dy, _y, Nsub, tim::cuda::host_to_device_v, stream[i]);
            evt[i]->mark_end();
        }
    }

    for(int i = 0; i < nitr; ++i)
    {
        TIMEMORY_BASIC_MARKER(auto_tuple_t, "[", i, "]");

        auto   offset = Nsub * i;
        float* _dx    = d_x + offset;
        float* _dy    = d_y + offset;

        evt[i]->mark_begin();
        // Perform SAXPY on 1M elements
        saxpy<<<ngrid, block, 0, stream[i]>>>(Nsub, 1.0f, _dx, _dy);
        evt[i]->mark_end();
    }

    {
        TIMEMORY_BASIC_MARKER(auto_tuple_t, "[D2H]");
        for(int i = 0; i < nitr; ++i)
        {
            auto   offset = Nsub * i;
            float* _y     = y + offset;
            float* _dy    = d_y + offset;
            evt[i]->mark_begin();
            tim::cuda::memcpy(_y, _dy, Nsub, tim::cuda::device_to_host_v, stream[i]);
            evt[i]->mark_end();
        }
    }

    _sync();

    {
        TIMEMORY_BASIC_MARKER(auto_tuple_t, "[check]");
        for(int64_t i = 0; i < N; i++)
        {
            maxError = std::max(maxError, std::abs(y[i] - 2.0f));
            sumError += std::abs(y[i] - 2.0f);
        }
    }

    _clock.stop();
    {
        TIMEMORY_BASIC_MARKER(auto_tuple_t, "[output]");
        cuda_event _evt = **evt;
        for(int i = 1; i < nitr; ++i)
        {
            evt[i]->stop();
            nseconds += evt[i]->get();
            _evt += *(evt[i]);
        }
        std::cout << "Event: " << _evt << std::endl;
        std::cout << _clock << std::endl;
        printf("Max error: %8.4e\n", maxError);
        printf("Sum error: %8.4e\n", sumError);
        printf("Effective Bandwidth (GB/s): %f\n",
               N * 4 * 3 / nseconds / tim::units::gigabyte);
        printf("Kernel Runtime (sec): %16.12e\n", nseconds);
    }

    for(int i = 0; i < nitr; ++i)
    {
        delete evt[i];
        tim::cuda::stream_destroy(stream[i]);
    }
    delete[] evt;

    tim::cuda::free(d_x);
    tim::cuda::free(d_y);
    tim::cuda::free_host(x);
    tim::cuda::free_host(y);

    tim::cuda::device_sync();
    // tim::cuda::device_reset();
}

//======================================================================================//

void
test_5_mt_saxpy_async()
{
    print_info(__FUNCTION__);
    warmup();
    TIMEMORY_BASIC_MARKER(auto_tuple_t, "");
    auto lambda_op = tim::string::join("", "::", _TIM_FUNC);

    comp_tuple_t _clock("Runtime");
    _clock.start();

    using data_t        = std::tuple<cuda_event, float, float, float>;
    using data_vector_t = std::vector<data_t>;

    data_vector_t data_vector(nitr);

    auto run_thread = [&](int i) {
        float*              x;
        float*              y;
        float*              d_x;
        float*              d_y;
        int                 block    = 512;
        int                 ngrid    = (Nsub + block - 1) / block;
        float               nseconds = 0.0f;
        float               maxError = 0.0f;
        float               sumError = 0.0f;
        tim::cuda::stream_t stream;
        tim::cuda::stream_create(stream);
        cuda_event evt(stream);
        evt.start();

        TIMEMORY_BASIC_MARKER(auto_tuple_t, lambda_op, "[run_thread]");

        {
            TIMEMORY_BASIC_MARKER(auto_tuple_t, lambda_op, "[cpu_malloc]");
            x = tim::device::cpu::alloc<float>(Nsub);
            y = tim::device::cpu::alloc<float>(Nsub);
        }

        {
            TIMEMORY_BASIC_MARKER(auto_tuple_t, lambda_op, "[gpu_malloc]");
            d_x = tim::cuda::malloc<float>(Nsub);
            d_y = tim::cuda::malloc<float>(Nsub);
        }

        {
            TIMEMORY_BASIC_MARKER(auto_tuple_t, lambda_op, "[assign]");
            for(int i = 0; i < Nsub; i++)
            {
                x[i] = 1.0f;
                y[i] = 2.0f;
            }
        }

        {
            TIMEMORY_BASIC_MARKER(auto_tuple_t, lambda_op, "[H2D]");
            evt.mark_begin();
            tim::cuda::memcpy(d_x, x, Nsub, tim::cuda::host_to_device_v, stream);
            tim::cuda::memcpy(d_y, y, Nsub, tim::cuda::host_to_device_v, stream);
            evt.mark_end();
        }

        {
            TIMEMORY_BASIC_MARKER(auto_tuple_t, lambda_op, "[", i, "]");
            // Perform SAXPY on 1M elements
            evt.mark_begin();
            saxpy<<<ngrid, block, 0, stream>>>(Nsub, 1.0f, d_x, d_y);
            evt.mark_end();
        }

        {
            TIMEMORY_BASIC_MARKER(auto_tuple_t, lambda_op, "[D2H]");
            evt.mark_begin();
            tim::cuda::memcpy(y, d_y, Nsub, tim::cuda::device_to_host_v);
            evt.mark_end();
        }

        tim::cuda::stream_sync(stream);
        tim::cuda::stream_destroy(stream);

        {
            TIMEMORY_BASIC_MARKER(auto_tuple_t, lambda_op, "[check]");
            for(int64_t i = 0; i < Nsub; i++)
            {
                maxError = std::max(maxError, std::abs(y[i] - 2.0f));
                sumError += std::abs(y[i] - 2.0f);
            }
        }

        evt.stop();
        nseconds += evt.get();
        data_vector[i] = std::make_tuple(evt, nseconds, maxError, sumError);

        tim::device::gpu::free(d_x);
        tim::device::gpu::free(d_y);
        tim::device::cpu::free(x);
        tim::device::cpu::free(y);
    };

    std::vector<std::thread> threads;
    for(int i = 0; i < nitr; i++)
        threads.emplace_back(run_thread, i);

    for(int i = 0; i < nitr; i++)
        threads[i].join();

    cuda_event evt      = std::get<0>(data_vector[0]);
    float      nseconds = std::get<1>(data_vector[0]);
    float      maxError = std::get<2>(data_vector[0]);
    float      sumError = std::get<3>(data_vector[0]);

    for(int i = 1; i < nitr; i++)
    {
        evt += std::get<0>(data_vector[i]);
        nseconds += std::get<1>(data_vector[i]);
        maxError = std::max(maxError, std::get<2>(data_vector[i]));
        sumError += std::get<3>(data_vector[i]);
    }

    _clock.stop();
    {
        TIMEMORY_BASIC_MARKER(auto_tuple_t, "[output]");
        std::cout << "Event: " << evt << std::endl;
        std::cout << _clock << std::endl;
        printf("Max error: %8.4e\n", maxError);
        printf("Sum error: %8.4e\n", sumError);
        printf("Effective Bandwidth (GB/s): %f\n",
               N * 4 * 3 / nseconds / tim::units::gigabyte);
        printf("Kernel Runtime (sec): %16.12e\n", nseconds);
    }

    tim::cuda::device_sync();
    // tim::cuda::device_reset();
}

//======================================================================================//

void
test_6_mt_saxpy_async_pinned()
{
    print_info(__FUNCTION__);
    warmup();
    TIMEMORY_BASIC_MARKER(auto_tuple_t, "");
    auto lambda_op = tim::string::join("", "::", _TIM_FUNC);

    comp_tuple_t _clock("Runtime");
    _clock.start();

    using data_t        = std::tuple<cuda_event, float, float, float>;
    using data_vector_t = std::vector<data_t>;

    data_vector_t data_vector(nitr);

    auto run_thread = [&](int i) {
        float*              x;
        float*              y;
        float*              d_x;
        float*              d_y;
        int                 block    = 512;
        int                 ngrid    = (Nsub + block - 1) / block;
        float               nseconds = 0.0f;
        float               maxError = 0.0f;
        float               sumError = 0.0f;
        tim::cuda::stream_t stream;
        tim::cuda::stream_create(stream);
        cuda_event* evt = new cuda_event(stream);
        evt->start();

        TIMEMORY_BASIC_MARKER(auto_tuple_t, lambda_op, "[run_thread]");

        {
            TIMEMORY_BASIC_MARKER(auto_tuple_t, lambda_op, "[cpu_malloc]");
            x = tim::cuda::malloc_host<float>(Nsub);
            y = tim::cuda::malloc_host<float>(Nsub);
        }

        {
            TIMEMORY_BASIC_MARKER(auto_tuple_t, lambda_op, "[gpu_malloc]");
            d_x = tim::cuda::malloc_host<float>(Nsub);
            d_y = tim::cuda::malloc_host<float>(Nsub);
        }

        {
            TIMEMORY_BASIC_MARKER(auto_tuple_t, lambda_op, "[assign]");
            for(int i = 0; i < Nsub; i++)
            {
                x[i] = 1.0f;
                y[i] = 2.0f;
            }
        }

        {
            TIMEMORY_BASIC_MARKER(auto_tuple_t, lambda_op, "[H2D]");
            evt->mark_begin();
            tim::cuda::memcpy(d_x, x, Nsub, tim::cuda::host_to_device_v, stream);
            tim::cuda::memcpy(d_y, y, Nsub, tim::cuda::host_to_device_v, stream);
            evt->mark_end();
        }

        {
            TIMEMORY_BASIC_MARKER(auto_tuple_t, lambda_op, "[", i, "]");

            // Perform SAXPY on 1M elements
            evt->mark_begin();
            saxpy<<<ngrid, block, 0, stream>>>(Nsub, 1.0f, d_x, d_y);
            evt->mark_end();
        }

        {
            TIMEMORY_BASIC_MARKER(auto_tuple_t, lambda_op, "[D2H]");
            evt->mark_begin();
            tim::cuda::memcpy(y, d_y, Nsub, tim::cuda::device_to_host_v, stream);
            evt->mark_end();
        }

        tim::cuda::stream_sync(stream);
        tim::cuda::stream_destroy(stream);

        {
            TIMEMORY_BASIC_MARKER(auto_tuple_t, lambda_op, "[check]");
            for(int64_t i = 0; i < Nsub; i++)
            {
                maxError = std::max(maxError, std::abs(y[i] - 2.0f));
                sumError += std::abs(y[i] - 2.0f);
            }
        }

        evt->stop();
        nseconds += evt->get();

        data_vector[i] = std::make_tuple(*evt, nseconds, maxError, sumError);

        tim::cuda::free_host(x);
        tim::cuda::free_host(y);
        tim::cuda::free(d_x);
        tim::cuda::free(d_y);
        delete evt;
    };

    std::vector<std::thread> threads;
    for(int i = 0; i < nitr; i++)
        threads.emplace_back(run_thread, i);

    for(int i = 0; i < nitr; i++)
        threads[i].join();

    cuda_event evt      = std::get<0>(data_vector[0]);
    float      nseconds = std::get<1>(data_vector[0]);
    float      maxError = std::get<2>(data_vector[0]);
    float      sumError = std::get<3>(data_vector[0]);

    for(int i = 1; i < nitr; i++)
    {
        evt += std::get<0>(data_vector[i]);
        nseconds += std::get<1>(data_vector[i]);
        maxError = std::max(maxError, std::get<2>(data_vector[i]));
        sumError += std::get<3>(data_vector[i]);
    }

    _clock.stop();
    {
        TIMEMORY_BASIC_MARKER(auto_tuple_t, "[output]");
        std::cout << "Event: " << evt << std::endl;
        std::cout << _clock << std::endl;
        printf("Max error: %8.4e\n", maxError);
        printf("Sum error: %8.4e\n", sumError);
        printf("Effective Bandwidth (GB/s): %f\n",
               N * 4 * 3 / nseconds / tim::units::gigabyte);
        printf("Kernel Runtime (sec): %16.12e\n", nseconds);
    }

    tim::cuda::device_sync();
    // tim::cuda::device_reset();
}

//======================================================================================//
namespace impl
{
template <typename T>
TIMEMORY_GLOBAL_FUNCTION void
KERNEL_A(T* begin, int n)
{
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        if(i < n)
            *(begin + i) += 2.0f * n;
    }
}

//--------------------------------------------------------------------------------------//

template <typename T>
TIMEMORY_GLOBAL_FUNCTION void
KERNEL_B(T* begin, int n)
{
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
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
    impl::KERNEL_A<<<2, 64, 0, stream>>>(arg, size);
}

//--------------------------------------------------------------------------------------//
template <typename T>
void
KERNEL_B(T* arg, int size, tim::cuda::stream_t stream = 0)
{
    impl::KERNEL_B<<<64, 2, 0, stream>>>(arg, size / 2);
}

//======================================================================================//
#if defined(TIMEMORY_USE_CUPTI)

static auto max_size = tim::get_env("MAX_SIZE", 64);
static auto num_data = tim::get_env("NUM_SIZE", 100);
static auto num_iter = tim::get_env("NUM_ITER", 10);

void
test_7_cupti_available()
{
    print_info(__FUNCTION__);

    CUdevice device;
    TIMEMORY_CUDA_DRIVER_API_CALL(cuInit(0));
    TIMEMORY_CUDA_DRIVER_API_CALL(cuDeviceGet(&device, 0));

    auto reduce_size = [](std::vector<std::string>& arr) {
        std::sort(arr.begin(), arr.end());
        if(arr.size() > (size_t) max_size)
            arr.resize(max_size);
    };

    auto event_names  = tim::cupti::available_events(device);
    auto metric_names = tim::cupti::available_metrics(device);
    reduce_size(event_names);
    reduce_size(metric_names);

    using size_type = decltype(event_names.size());
    size_type wevt  = 30;
    size_type wmet  = 30;
    for(const auto& itr : event_names)
        wevt = std::max(itr.size(), wevt);
    for(const auto& itr : metric_names)
        wmet = std::max(itr.size(), wmet);

    std::cout << "Event names: \n\t"
              << array_to_string(event_names, ", ", wevt, 180 / wevt) << std::endl;
    std::cout << "Metric names: \n\t"
              << array_to_string(metric_names, ", ", wmet, 180 / wmet) << std::endl;

    std::vector<float> cpu_data(num_data, 0);
    float*             data;
    TIMEMORY_CUDA_RUNTIME_API_CALL(cudaMalloc(&data, num_data * sizeof(float)));
    TIMEMORY_CUDA_RUNTIME_API_CALL(cudaMemcpy(
        data, cpu_data.data(), num_data * sizeof(float), cudaMemcpyHostToDevice));

    tim::cupti::profiler profiler(event_names, metric_names);
    const int            passes = profiler.passes();
    printf("Passes: %d\n", passes);

    profiler.start();
    for(int i = 0; i < num_iter; ++i)
    {
        printf("\n[%s]> iteration %i...\n", __FUNCTION__, i);
        KERNEL_A(data, num_data);
        KERNEL_B(data, num_data);
    }
    tim::cuda::device_sync();
    profiler.stop();

    printf("Event Trace\n");
    profiler.print_event_values(std::cout);
    printf("Metric Trace\n");
    profiler.print_metric_values(std::cout);

    auto names = profiler.get_kernel_names();
    std::cout << "Kernel names: \n\t" << array_to_string(names, "\n\t", 16, names.size())
              << std::endl;

    TIMEMORY_CUDA_RUNTIME_API_CALL(cudaMemcpy(
        cpu_data.data(), data, num_data * sizeof(float), cudaMemcpyDeviceToHost));
    TIMEMORY_CUDA_RUNTIME_API_CALL(cudaFree(data));

    printf("\n");
    std::cout << "Data values: \n\t" << array_to_string(cpu_data, ", ", 8, 10)
              << std::endl;
    printf("\n");
}

//======================================================================================//

void
test_8_cupti_subset()
{
    print_info(__FUNCTION__);
    // tim::cuda::device_reset();

    TIMEMORY_CUDA_DRIVER_API_CALL(cuInit(0));
    std::vector<std::string> event_names{ "active_warps",   "active_cycles",
                                          "global_load",    "global_store",
                                          "gld_inst_32bit", "gst_inst_32bit" };
    std::vector<std::string> metric_names{
        "inst_per_warp",     "branch_efficiency",  "warp_execution_efficiency",
        "flop_count_sp",     "flop_count_sp_add",  "flop_count_sp_fma",
        "flop_count_sp_mul", "flop_sp_efficiency", "gld_efficiency",
        "gst_efficiency"
    };

    std::vector<float> cpu_data(num_data, 0);
    float*             data;
    TIMEMORY_CUDA_RUNTIME_API_CALL(cudaMalloc(&data, num_data * sizeof(float)));
    TIMEMORY_CUDA_RUNTIME_API_CALL(cudaMemcpy(
        data, cpu_data.data(), num_data * sizeof(float), cudaMemcpyHostToDevice));

    tim::cupti::profiler profiler(event_names, metric_names);
    // Get #passes required to compute all metrics and events
    const int passes = profiler.passes();
    printf("Passes: %d\n", passes);

    profiler.start();
    for(int i = 0; i < num_iter; ++i)
    {
        printf("\n[%s]> iteration %i...\n", __FUNCTION__, i);
        KERNEL_A(data, num_data);
        KERNEL_B(data, num_data);
    }
    tim::cuda::device_sync();
    profiler.stop();

    printf("Event Trace\n");
    profiler.print_event_values(std::cout);
    printf("Metric Trace\n");
    profiler.print_metric_values(std::cout);

    auto names = profiler.get_kernel_names();
    std::cout << "Kernel names: \n\t" << array_to_string(names, "\n\t", 16, names.size())
              << std::endl;

    TIMEMORY_CUDA_RUNTIME_API_CALL(cudaMemcpy(
        cpu_data.data(), data, num_data * sizeof(float), cudaMemcpyDeviceToHost));
    TIMEMORY_CUDA_RUNTIME_API_CALL(cudaFree(data));

    printf("\n");
    std::cout << "Data values: \n\t" << array_to_string(cpu_data, ", ", 8, 10)
              << std::endl;
    printf("\n");
    // tim::cuda::device_reset();
}

//======================================================================================//

void
test_9_cupti_counters()
{
    print_info(__FUNCTION__);
    // tim::cuda::device_reset();

    CUdevice device;
    TIMEMORY_CUDA_DRIVER_API_CALL(cuInit(0));
    TIMEMORY_CUDA_DRIVER_API_CALL(cuDeviceGet(&device, 0));

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
              << array_to_string(event_names, ", ", wevt, 200 / wevt) << std::endl;
    std::cout << "Metric names: \n\t"
              << array_to_string(metric_names, ", ", wmet, 200 / wmet) << std::endl;

    cupti_counters::get_device_initializer() = []() { return 0; };
    cupti_counters::get_event_initializer()  = []() {
        return std::vector<std::string>({ "active_warps", "active_cycles", "global_load",
                                          "global_store", "gld_inst_32bit",
                                          "gst_inst_32bit" });
    };
    cupti_counters::get_metric_initializer() = []() {
        return std::vector<std::string>({ "inst_per_warp", "branch_efficiency",
                                          "warp_execution_efficiency", "flop_count_sp",
                                          "flop_count_sp_add", "flop_count_sp_fma",
                                          "flop_count_sp_mul", "flop_sp_efficiency",
                                          "gld_efficiency", "gst_efficiency" });
    };

    using Tp        = double;
    using counter_t = tim::ert::counter<tim::device::gpu, Tp>;

    auto store_func = [] TIMEMORY_LAMBDA(Tp & a, const Tp& b) { a = b; };
    auto add_func   = [] TIMEMORY_LAMBDA(Tp & a, const Tp& b, const Tp& c) { a = b + c; };
    auto fma_func   = [] TIMEMORY_LAMBDA(Tp & a, const Tp& b, const Tp& c) {
        a = a * b + c;
    };

    tim::ert::exec_params       params(16, 64 * 64);
    std::shared_ptr<ert_data_t> exec_data(new ert_data_t);
    auto                        _counter = new counter_t(params, exec_data, 64);

    std::vector<float> cpu_data(num_data, 0);
    float*             data;
    TIMEMORY_CUDA_RUNTIME_API_CALL(cudaMalloc(&data, num_data * sizeof(float)));
    TIMEMORY_CUDA_RUNTIME_API_CALL(cudaMemcpy(
        data, cpu_data.data(), num_data * sizeof(float), cudaMemcpyHostToDevice));

    cupti_counters profiler;
    profiler.start();
    for(int i = 0; i < num_iter; ++i)
    {
        printf("\n[%s]> iteration %i...\n", __FUNCTION__, i);
        tim::ert::ops_main<1>(*_counter, add_func, store_func);
        tim::ert::ops_main<2, 4, 8>(*_counter, fma_func, store_func);
        std::cout << *_counter << std::endl;
        KERNEL_A(data, num_data);
        KERNEL_B(data, num_data);
    }
    tim::cuda::device_sync();
    profiler.stop();
    std::cout << __FUNCTION__ << " : " << profiler << std::endl;

    TIMEMORY_CUDA_RUNTIME_API_CALL(cudaMemcpy(
        cpu_data.data(), data, num_data * sizeof(float), cudaMemcpyDeviceToHost));
    TIMEMORY_CUDA_RUNTIME_API_CALL(cudaFree(data));

    printf("\n");
    // tim::cuda::device_reset();
}

//======================================================================================//

void
test_10_cupti_metric()
{
    print_info(__FUNCTION__);

    constexpr int      num_data = 100;
    std::vector<float> cpu_data(num_data, 0);
    float*             data;
    TIMEMORY_CUDA_RUNTIME_API_CALL(cudaMalloc(&data, num_data * sizeof(float)));
    TIMEMORY_CUDA_RUNTIME_API_CALL(cudaMemcpy(
        data, cpu_data.data(), num_data * sizeof(float), cudaMemcpyHostToDevice));

    tim::cupti::profiler prof({ "gst_inst_32bit" }, { "flop_count_sp" });

    prof.start();
    for(int i = 0; i < 10; ++i)
    {
        KERNEL_A(data, num_data);
        KERNEL_B(data, num_data);
    }
    _LOG("calling sync...\n");
    tim::cuda::device_sync();
    prof.stop();

    TIMEMORY_CUDA_RUNTIME_API_CALL(cudaMemcpy(
        cpu_data.data(), data, num_data * sizeof(float), cudaMemcpyDeviceToHost));
    TIMEMORY_CUDA_RUNTIME_API_CALL(cudaFree(data));

    printf("\n");
    std::cout << "Data values: \n\t" << array_to_string(cpu_data, ", ", 8, 10)
              << std::endl;
    printf("\n");
}

//======================================================================================//

#else  // defined(TIMEMORY_USE_CUPTI)

//======================================================================================//

void
test_7_cupti_available()
{
    print_info(__FUNCTION__);
    printf("CUPTI is not available...\n");
}

//======================================================================================//

void
test_8_cupti_subset()
{
    print_info(__FUNCTION__);
    printf("CUPTI is not available...\n");
}

//======================================================================================//

void
test_9_cupti_counters()
{
    print_info(__FUNCTION__);
    printf("CUPTI is not available...\n");
}

//======================================================================================//

void
test_10_cupti_metric()
{
    print_info(__FUNCTION__);
    printf("CUPTI is not available...\n");
}

//======================================================================================//

#endif  // defined(TIMEMORY_USE_CUPTI)
