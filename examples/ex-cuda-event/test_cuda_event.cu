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
#include <iterator>
#include <random>
#include <thread>
#include <unordered_map>
#include <vector>

#include <timemory/timemory.hpp>

using namespace tim::component;

using papi_tuple_t = papi_event<0, PAPI_RES_STL, PAPI_TOT_CYC, PAPI_BR_MSP, PAPI_BR_PRC>;
using auto_tuple_t =
    tim::auto_tuple<real_clock, system_clock, cpu_clock, cpu_util, papi_tuple_t>;
using comp_tuple_t = typename auto_tuple_t::component_type;
using cuda_tuple_t = tim::auto_tuple<cuda_event>;

//======================================================================================//

static const int nitr = 4;
static int64_t   N    = 50 * (1 << 23);
static auto      Nsub = N / nitr;

//--------------------------------------------------------------------------------------//
// saxpy calculation
__global__ void
warmup(int64_t n)
{
    int i   = blockIdx.x * blockDim.x + threadIdx.x;
    int tmp = 0;
    if(i < n)
        tmp += i;
}

//--------------------------------------------------------------------------------------//
// saxpy calculation
__global__ void
saxpy(int64_t n, float a, float* x, float* y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < n)
    {
        atomicAdd(&y[i], y[i] - (a * x[i]));
    }
    // if(i < 8)
    //    printf("i = %li, y = %8.4e, x = %8.4e, y = %8.4e\n", i, y[i], x[i], y[i]);
}
//--------------------------------------------------------------------------------------//

void
warmup()
{
    int     block = 16 * 512;
    int     ngrid = 512;
    int64_t val   = 256;
    warmup<<<ngrid, block>>>(val);
    cudaDeviceSynchronize();
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

//======================================================================================//

int
main(int argc, char** argv)
{
    if(N % nitr != 0)
    {
        throw std::runtime_error("Error N is not a multiple of nitr");
    }

    cuda_event::get_format_flags() = std::ios_base::scientific;
    tim::timemory_init(argc, argv);
    tim::env::json_output() = true;

    int ndevices = 0;
    cudaGetDeviceCount(&ndevices);
    warmup();

    auto* timing =
        new tim::component_tuple<real_clock, system_clock, cpu_clock, cpu_util>(
            "Tests runtime", true);

    timing->start();

    CONFIGURE_TEST_SELECTOR(6);

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
    }
    catch(std::exception& e)
    {
        std::cerr << e.what() << std::endl;
    }

    timing->stop();
    std::cout << "\n" << *timing << std::endl;

    TEST_SUMMARY(argv[0], num_test, num_fail);
    delete timing;

    exit(num_fail);
}

//======================================================================================//

void
print_info(const std::string& func)
{
    if(tim::mpi_rank() == 0)
    {
        std::cout << "\n[" << tim::mpi_rank() << "]\e[1;33m TESTING \e[0m["
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
    _ss << "[" << tim::mpi_rank() << "] " << str << std::endl;
    std::cout << _ss.str();
}

//======================================================================================//

void
test_1_saxpy()
{
    print_info(__FUNCTION__);
    warmup();
    TIMEMORY_BASIC_AUTO_TUPLE(auto_tuple_t, "");

    comp_tuple_t _clock("Runtime");
    _clock.start();

    float*      x;
    float*      y;
    float*      d_x;
    float*      d_y;
    int         block        = 512;
    int         ngrid        = (N + block - 1) / block;
    float       milliseconds = 0.0f;
    float       maxError     = 0.0f;
    float       sumError     = 0.0f;
    cuda_event* evt          = nullptr;

    {
        TIMEMORY_BASIC_AUTO_TUPLE(auto_tuple_t, "[malloc]");
        x = (float*) malloc(N * sizeof(float));
        y = (float*) malloc(N * sizeof(float));
    }

    {
        TIMEMORY_BASIC_AUTO_TUPLE(auto_tuple_t, "[cudaMalloc]");
        cudaMalloc(&d_x, N * sizeof(float));
        cudaMalloc(&d_y, N * sizeof(float));
    }

    {
        TIMEMORY_BASIC_AUTO_TUPLE(auto_tuple_t, "[assign]");
        for(int i = 0; i < N; i++)
        {
            x[i] = 1.0f;
            y[i] = 2.0f;
        }
    }

    {
        TIMEMORY_BASIC_AUTO_TUPLE(auto_tuple_t, "[create_event]");
        evt = new cuda_event();
    }

    {
        TIMEMORY_BASIC_AUTO_TUPLE(auto_tuple_t, "[H2D]");
        cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice);
    }

    for(int i = 0; i < nitr; ++i)
    {
        TIMEMORY_BASIC_AUTO_TUPLE(auto_tuple_t, "[", i, "]");
        evt->start();

        // Perform SAXPY on 1M elements
        saxpy<<<ngrid, block>>>(N, 1.0f, d_x, d_y);

        evt->stop();
        milliseconds += evt->value;
    }

    {
        TIMEMORY_BASIC_AUTO_TUPLE(auto_tuple_t, "[D2H]");
        cudaMemcpy(y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);
    }

    {
        TIMEMORY_BASIC_AUTO_TUPLE(auto_tuple_t, "[check]");
        for(int64_t i = 0; i < N; i++)
        {
            maxError = std::max(maxError, std::abs(y[i] - 2.0f));
            sumError += std::abs(y[i] - 2.0f);
        }
    }

    _clock.stop();
    {
        TIMEMORY_BASIC_AUTO_TUPLE(auto_tuple_t, "[output]");
        std::cout << "Event: " << *evt << std::endl;
        std::cout << _clock << std::endl;
        printf("Max error: %8.4e\n", maxError);
        printf("Sum error: %8.4e\n", sumError);
        printf("Effective Bandwidth (GB/s): %f\n", N * 4 * 3 / milliseconds / 1e6);
        printf("Kernel Runtime (sec): %16.12e\n", milliseconds / 1e6);
    }

    delete evt;
    cudaDeviceSynchronize();
    cudaDeviceReset();
}

//======================================================================================//

void
test_2_saxpy_async()
{
    print_info(__FUNCTION__);
    warmup();
    TIMEMORY_BASIC_AUTO_TUPLE(auto_tuple_t, "");

    comp_tuple_t _clock("Runtime");
    _clock.start();

    float*        x;
    float*        y;
    float*        d_x;
    float*        d_y;
    int           block        = 512;
    int           ngrid        = (Nsub + block - 1) / block;
    float         milliseconds = 0.0f;
    float         maxError     = 0.0f;
    float         sumError     = 0.0f;
    cuda_event**  evt          = new cuda_event*[nitr];
    cudaStream_t* stream       = new cudaStream_t[nitr];

    auto _sync = [&]() {
        for(int i = 0; i < nitr; i++)
            cudaStreamSynchronize(stream[i]);
    };

    {
        TIMEMORY_BASIC_AUTO_TUPLE(auto_tuple_t, "[malloc]");
        x = (float*) malloc(N * sizeof(float));
        y = (float*) malloc(N * sizeof(float));
    }

    {
        TIMEMORY_BASIC_AUTO_TUPLE(auto_tuple_t, "[cudaMalloc]");
        cudaMalloc(&d_x, N * sizeof(float));
        cudaMalloc(&d_y, N * sizeof(float));
    }

    {
        TIMEMORY_BASIC_AUTO_TUPLE(auto_tuple_t, "[assign]");
        for(int i = 0; i < N; i++)
        {
            x[i] = 1.0f;
            y[i] = 2.0f;
        }
    }

    {
        TIMEMORY_BASIC_AUTO_TUPLE(auto_tuple_t, "[create]");
        for(int i = 0; i < nitr; ++i)
        {
            cudaStreamCreate(&stream[i]);
            evt[i] = new cuda_event(stream[i]);
        }
    }

    {
        TIMEMORY_BASIC_AUTO_TUPLE(auto_tuple_t, "[H2D]");
        for(int i = 0; i < nitr; ++i)
        {
            auto   offset = Nsub * i;
            float* _x     = x + offset;
            float* _dx    = d_x + offset;
            float* _y     = y + offset;
            float* _dy    = d_y + offset;
            cudaMemcpyAsync(_dx, _x, Nsub * sizeof(float), cudaMemcpyHostToDevice,
                            stream[i]);
            cudaMemcpyAsync(_dy, _y, Nsub * sizeof(float), cudaMemcpyHostToDevice,
                            stream[i]);
        }
    }

    for(int i = 0; i < nitr; ++i)
    {
        TIMEMORY_BASIC_AUTO_TUPLE(auto_tuple_t, "[", i, "]");

        auto   offset = Nsub * i;
        float* _dx    = d_x + offset;
        float* _dy    = d_y + offset;

        evt[i]->start();

        // Perform SAXPY on 1M elements
        saxpy<<<ngrid, block, 0, stream[i]>>>(N, 1.0f, _dx, _dy);

        evt[i]->stop();
        milliseconds += evt[i]->value;
    }

    _sync();
    {
        TIMEMORY_BASIC_AUTO_TUPLE(auto_tuple_t, "[D2H]");
        for(int i = 0; i < nitr; ++i)
        {
            auto   offset = Nsub * i;
            float* _y     = y + offset;
            float* _dy    = d_y + offset;
            cudaMemcpyAsync(_y, _dy, Nsub * sizeof(float), cudaMemcpyDeviceToHost,
                            stream[i]);
        }
    }

    _sync();
    {
        TIMEMORY_BASIC_AUTO_TUPLE(auto_tuple_t, "[check]");
        for(int64_t i = 0; i < N; i++)
        {
            maxError = std::max(maxError, std::abs(y[i] - 2.0f));
            sumError += std::abs(y[i] - 2.0f);
        }
    }

    _clock.stop();
    {
        TIMEMORY_BASIC_AUTO_TUPLE(auto_tuple_t, "[output]");
        cuda_event _evt = **evt;
        for(int i = 1; i < nitr; ++i)
            _evt += *(evt[i]);
        std::cout << "Event: " << _evt << std::endl;
        std::cout << _clock << std::endl;
        printf("Max error: %8.4e\n", maxError);
        printf("Sum error: %8.4e\n", sumError);
        printf("Effective Bandwidth (GB/s): %f\n", N * 4 * 3 / milliseconds / 1e6);
        printf("Kernel Runtime (sec): %16.12e\n", milliseconds / 1e6);
    }

    for(int i = 0; i < nitr; ++i)
        delete evt[i];
    delete[] evt;
    cudaDeviceSynchronize();
    cudaDeviceReset();
}

//======================================================================================//

void
test_3_saxpy_pinned()
{
    print_info(__FUNCTION__);
    warmup();
    TIMEMORY_BASIC_AUTO_TUPLE(auto_tuple_t, "");

    comp_tuple_t _clock("Runtime");
    _clock.start();

    float*      x;
    float*      y;
    float*      d_x;
    float*      d_y;
    int         block        = 512;
    int         ngrid        = (N + block - 1) / block;
    float       milliseconds = 0.0f;
    float       maxError     = 0.0f;
    float       sumError     = 0.0f;
    cuda_event* evt          = nullptr;

    {
        TIMEMORY_BASIC_AUTO_TUPLE(auto_tuple_t, "[malloc]");
        cudaMallocHost(&x, N * sizeof(float));
        cudaMallocHost(&y, N * sizeof(float));
    }

    {
        TIMEMORY_BASIC_AUTO_TUPLE(auto_tuple_t, "[cudaMalloc]");
        cudaMalloc(&d_x, N * sizeof(float));
        cudaMalloc(&d_y, N * sizeof(float));
    }

    {
        TIMEMORY_BASIC_AUTO_TUPLE(auto_tuple_t, "[assign]");
        for(int i = 0; i < N; i++)
        {
            x[i] = 1.0f;
            y[i] = 2.0f;
        }
    }

    {
        TIMEMORY_BASIC_AUTO_TUPLE(auto_tuple_t, "[create_event]");
        evt = new cuda_event();
    }

    {
        TIMEMORY_BASIC_AUTO_TUPLE(auto_tuple_t, "[H2D]");
        cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice);
    }

    for(int i = 0; i < nitr; ++i)
    {
        TIMEMORY_BASIC_AUTO_TUPLE(auto_tuple_t, "[", i, "]");
        evt->start();

        // Perform SAXPY on 1M elements
        saxpy<<<ngrid, block>>>(N, 1.0f, d_x, d_y);

        evt->stop();
        milliseconds += evt->value;
    }

    {
        TIMEMORY_BASIC_AUTO_TUPLE(auto_tuple_t, "[D2H]");
        cudaMemcpy(y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);
    }

    {
        TIMEMORY_BASIC_AUTO_TUPLE(auto_tuple_t, "[check]");
        for(int64_t i = 0; i < N; i++)
        {
            maxError = std::max(maxError, std::abs(y[i] - 2.0f));
            sumError += std::abs(y[i] - 2.0f);
        }
    }

    _clock.stop();
    {
        TIMEMORY_BASIC_AUTO_TUPLE(auto_tuple_t, "[output]");
        std::cout << "Event: " << *evt << std::endl;
        std::cout << _clock << std::endl;
        printf("Max error: %8.4e\n", maxError);
        printf("Sum error: %8.4e\n", sumError);
        printf("Effective Bandwidth (GB/s): %f\n", N * 4 * 3 / milliseconds / 1e6);
        printf("Kernel Runtime (sec): %16.12e\n", milliseconds / 1e6);
    }

    delete evt;
    cudaDeviceSynchronize();
    cudaDeviceReset();
}

//======================================================================================//

void
test_4_saxpy_async_pinned()
{
    print_info(__FUNCTION__);
    warmup();
    TIMEMORY_BASIC_AUTO_TUPLE(auto_tuple_t, "");

    comp_tuple_t _clock("Runtime");
    _clock.start();

    float*        x;
    float*        y;
    float*        d_x;
    float*        d_y;
    int           block        = 512;
    int           ngrid        = (Nsub + block - 1) / block;
    float         milliseconds = 0.0f;
    float         maxError     = 0.0f;
    float         sumError     = 0.0f;
    cuda_event**  evt          = new cuda_event*[nitr];
    cudaStream_t* stream       = new cudaStream_t[nitr];

    auto _sync = [&]() {
        for(int i = 0; i < nitr; i++)
            cudaStreamSynchronize(stream[i]);
    };

    {
        TIMEMORY_BASIC_AUTO_TUPLE(auto_tuple_t, "[malloc]");
        cudaMallocHost(&x, N * sizeof(float));
        cudaMallocHost(&y, N * sizeof(float));
    }

    {
        TIMEMORY_BASIC_AUTO_TUPLE(auto_tuple_t, "[cudaMalloc]");
        cudaMalloc(&d_x, N * sizeof(float));
        cudaMalloc(&d_y, N * sizeof(float));
    }

    {
        TIMEMORY_BASIC_AUTO_TUPLE(auto_tuple_t, "[assign]");
        for(int i = 0; i < N; i++)
        {
            x[i] = 1.0f;
            y[i] = 2.0f;
        }
    }

    {
        TIMEMORY_BASIC_AUTO_TUPLE(auto_tuple_t, "[create]");
        for(int i = 0; i < nitr; ++i)
        {
            cudaStreamCreate(&stream[i]);
            evt[i] = new cuda_event(stream[i]);
        }
    }

    {
        TIMEMORY_BASIC_AUTO_TUPLE(auto_tuple_t, "[H2D]");
        for(int i = 0; i < nitr; ++i)
        {
            auto offset = Nsub * i;
            cudaMemcpyAsync(d_x + offset, x + offset, Nsub * sizeof(float),
                            cudaMemcpyHostToDevice, stream[i]);
            cudaMemcpyAsync(d_y + offset, y + offset, Nsub * sizeof(float),
                            cudaMemcpyHostToDevice, stream[i]);
        }
    }

    for(int i = 0; i < nitr; ++i)
    {
        TIMEMORY_BASIC_AUTO_TUPLE(auto_tuple_t, "[", i, "]");

        auto offset = Nsub * i;

        evt[i]->start();

        // Perform SAXPY on 1M elements
        saxpy<<<ngrid, block, 0, stream[i]>>>(N, 1.0f, d_x + offset, d_y + offset);

        evt[i]->stop();
        milliseconds += evt[i]->value;
    }

    {
        TIMEMORY_BASIC_AUTO_TUPLE(auto_tuple_t, "[D2H]");
        cudaMemcpy(y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);
        for(int i = 0; i < nitr; ++i)
        {
            auto offset = Nsub * i;
            cudaMemcpyAsync(y + offset, d_y + offset, Nsub * sizeof(float),
                            cudaMemcpyDeviceToHost, stream[i]);
        }
    }

    {
        TIMEMORY_BASIC_AUTO_TUPLE(auto_tuple_t, "[check]");
        _sync();
        for(int64_t i = 0; i < N; i++)
        {
            maxError = std::max(maxError, std::abs(y[i] - 2.0f));
            sumError += std::abs(y[i] - 2.0f);
        }
    }

    _clock.stop();
    {
        TIMEMORY_BASIC_AUTO_TUPLE(auto_tuple_t, "[output]");
        cuda_event _evt = **evt;
        for(int i = 1; i < nitr; ++i)
            _evt += *(evt[i]);
        std::cout << "Event: " << _evt << std::endl;
        std::cout << _clock << std::endl;
        printf("Max error: %8.4e\n", maxError);
        printf("Sum error: %8.4e\n", sumError);
        printf("Effective Bandwidth (GB/s): %f\n", N * 4 * 3 / milliseconds / 1e6);
        printf("Kernel Runtime (sec): %16.12e\n", milliseconds / 1e6);
    }

    for(int i = 0; i < nitr; ++i)
        delete evt[i];
    delete[] evt;
    cudaDeviceSynchronize();
    cudaDeviceReset();
}

//======================================================================================//

void
test_5_mt_saxpy_async()
{
    print_info(__FUNCTION__);
    warmup();
    TIMEMORY_BASIC_AUTO_TUPLE(auto_tuple_t, "");
    auto lambda_op = tim::str::join("", "::", __TIMEMORY_FUNCTION__);

    comp_tuple_t _clock("Runtime");
    _clock.start();

    using data_t        = std::tuple<cuda_event, float, float, float>;
    using data_vector_t = std::vector<data_t>;

    data_vector_t data_vector(nitr);

    auto run_thread = [&](int i) {
        float*     x;
        float*     y;
        float*     d_x;
        float*     d_y;
        int        block        = 512;
        int        ngrid        = (Nsub + block - 1) / block;
        float      milliseconds = 0.0f;
        float      maxError     = 0.0f;
        float      sumError     = 0.0f;
        cuda_event evt;

        {
            TIMEMORY_BASIC_AUTO_TUPLE(auto_tuple_t, lambda_op, "[malloc]");
            x = (float*) malloc(Nsub * sizeof(float));
            y = (float*) malloc(Nsub * sizeof(float));
        }

        {
            TIMEMORY_BASIC_AUTO_TUPLE(auto_tuple_t, lambda_op, "[cudaMalloc]");
            cudaMalloc(&d_x, Nsub * sizeof(float));
            cudaMalloc(&d_y, Nsub * sizeof(float));
        }

        {
            TIMEMORY_BASIC_AUTO_TUPLE(auto_tuple_t, lambda_op, "[assign]");
            for(int i = 0; i < Nsub; i++)
            {
                x[i] = 1.0f;
                y[i] = 2.0f;
            }
        }

        {
            TIMEMORY_BASIC_AUTO_TUPLE(auto_tuple_t, lambda_op, "[H2D]");
            cudaMemcpy(d_x, x, Nsub * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_y, y, Nsub * sizeof(float), cudaMemcpyHostToDevice);
        }

        {
            TIMEMORY_BASIC_AUTO_TUPLE(auto_tuple_t, lambda_op, "[", i, "]");

            evt.start();

            // Perform SAXPY on 1M elements
            saxpy<<<ngrid, block>>>(Nsub, 1.0f, d_x, d_y);

            evt.stop();
            milliseconds += evt.value;
        }

        {
            TIMEMORY_BASIC_AUTO_TUPLE(auto_tuple_t, lambda_op, "[D2H]");
            cudaMemcpy(y, d_y, Nsub * sizeof(float), cudaMemcpyDeviceToHost);
        }

        {
            TIMEMORY_BASIC_AUTO_TUPLE(auto_tuple_t, lambda_op, "[check]");
            for(int64_t i = 0; i < Nsub; i++)
            {
                maxError = std::max(maxError, std::abs(y[i] - 2.0f));
                sumError += std::abs(y[i] - 2.0f);
            }
        }

        data_vector[i] =
            std::move(std::make_tuple(evt, milliseconds, maxError, sumError));
    };

    std::vector<std::thread> threads;
    for(int i = 0; i < nitr; i++)
        threads.push_back(std::move(std::thread(run_thread, i)));

    for(int i = 0; i < nitr; i++)
        threads[i].join();

    cuda_event evt          = std::get<0>(data_vector[0]);
    float      milliseconds = std::get<1>(data_vector[0]);
    float      maxError     = std::get<2>(data_vector[0]);
    float      sumError     = std::get<3>(data_vector[0]);

    for(int i = 1; i < nitr; i++)
    {
        evt += std::get<0>(data_vector[i]);
        milliseconds += std::get<1>(data_vector[i]);
        maxError = std::max(maxError, std::get<2>(data_vector[i]));
        sumError += std::get<3>(data_vector[i]);
    }

    _clock.stop();
    {
        TIMEMORY_BASIC_AUTO_TUPLE(auto_tuple_t, "[output]");
        std::cout << "Event: " << evt << std::endl;
        std::cout << _clock << std::endl;
        printf("Max error: %8.4e\n", maxError);
        printf("Sum error: %8.4e\n", sumError);
        printf("Effective Bandwidth (GB/s): %f\n", N * 4 * 3 / milliseconds / 1e6);
        printf("Kernel Runtime (sec): %16.12e\n", milliseconds / 1e6);
    }

    cudaDeviceSynchronize();
    cudaDeviceReset();
}

//======================================================================================//

void
test_6_mt_saxpy_async_pinned()
{
    print_info(__FUNCTION__);
    warmup();
    TIMEMORY_BASIC_AUTO_TUPLE(auto_tuple_t, "");
    auto lambda_op = tim::str::join("", "::", __TIMEMORY_FUNCTION__);

    comp_tuple_t _clock("Runtime");
    _clock.start();

    using data_t        = std::tuple<cuda_event, float, float, float>;
    using data_vector_t = std::vector<data_t>;

    data_vector_t data_vector(nitr);

    auto run_thread = [&](int i) {
        float*      x;
        float*      y;
        float*      d_x;
        float*      d_y;
        int         block        = 512;
        int         ngrid        = (Nsub + block - 1) / block;
        float       milliseconds = 0.0f;
        float       maxError     = 0.0f;
        float       sumError     = 0.0f;
        cuda_event* evt          = new cuda_event();

        {
            TIMEMORY_BASIC_AUTO_TUPLE(auto_tuple_t, lambda_op, "[malloc]");
            cudaMallocHost(&x, Nsub * sizeof(float));
            cudaMallocHost(&y, Nsub * sizeof(float));
        }

        {
            TIMEMORY_BASIC_AUTO_TUPLE(auto_tuple_t, lambda_op, "[cudaMalloc]");
            cudaMalloc(&d_x, Nsub * sizeof(float));
            cudaMalloc(&d_y, Nsub * sizeof(float));
        }

        {
            TIMEMORY_BASIC_AUTO_TUPLE(auto_tuple_t, lambda_op, "[assign]");
            for(int i = 0; i < Nsub; i++)
            {
                x[i] = 1.0f;
                y[i] = 2.0f;
            }
        }

        {
            TIMEMORY_BASIC_AUTO_TUPLE(auto_tuple_t, lambda_op, "[H2D]");
            cudaMemcpy(d_x, x, Nsub * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_y, y, Nsub * sizeof(float), cudaMemcpyHostToDevice);
        }

        {
            TIMEMORY_BASIC_AUTO_TUPLE(auto_tuple_t, lambda_op, "[", i, "]");
            evt->start();

            // Perform SAXPY on 1M elements
            saxpy<<<ngrid, block>>>(Nsub, 1.0f, d_x, d_y);

            evt->stop();
            milliseconds += evt->value;
        }

        {
            TIMEMORY_BASIC_AUTO_TUPLE(auto_tuple_t, lambda_op, "[D2H]");
            cudaMemcpy(y, d_y, Nsub * sizeof(float), cudaMemcpyDeviceToHost);
        }

        {
            TIMEMORY_BASIC_AUTO_TUPLE(auto_tuple_t, lambda_op, "[check]");
            for(int64_t i = 0; i < Nsub; i++)
            {
                maxError = std::max(maxError, std::abs(y[i] - 2.0f));
                sumError += std::abs(y[i] - 2.0f);
            }
        }

        data_vector[i] =
            std::move(std::make_tuple(*evt, milliseconds, maxError, sumError));
    };

    std::vector<std::thread> threads;
    for(int i = 0; i < nitr; i++)
        threads.push_back(std::move(std::thread(run_thread, i)));

    for(int i = 0; i < nitr; i++)
        threads[i].join();

    cuda_event evt          = std::move(std::get<0>(data_vector[0]));
    float      milliseconds = std::get<1>(data_vector[0]);
    float      maxError     = std::get<2>(data_vector[0]);
    float      sumError     = std::get<3>(data_vector[0]);

    for(int i = 1; i < nitr; i++)
    {
        evt += std::get<0>(data_vector[i]);
        milliseconds += std::get<1>(data_vector[i]);
        maxError = std::max(maxError, std::get<2>(data_vector[i]));
        sumError += std::get<3>(data_vector[i]);
    }

    _clock.stop();
    {
        TIMEMORY_BASIC_AUTO_TUPLE(auto_tuple_t, "[output]");
        std::cout << "Event: " << evt << std::endl;
        std::cout << _clock << std::endl;
        printf("Max error: %8.4e\n", maxError);
        printf("Sum error: %8.4e\n", sumError);
        printf("Effective Bandwidth (GB/s): %f\n", N * 4 * 3 / milliseconds / 1e6);
        printf("Kernel Runtime (sec): %16.12e\n", milliseconds / 1e6);
    }

    cudaDeviceSynchronize();
    cudaDeviceReset();
}

//======================================================================================//
