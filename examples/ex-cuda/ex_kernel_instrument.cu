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

#include "timemory/backends/device.hpp"
#include "timemory/components/macros.hpp"
#include "timemory/macros.hpp"
#include "timemory/timemory.hpp"

#include <chrono>
#include <iostream>
#include <random>
#include <thread>
#include <vector>

namespace device = tim::device;
namespace mpl    = tim::mpl;
namespace comp   = tim::component;
namespace cuda   = tim::cuda;
// static constexpr int     nitr     = 10;
// static constexpr int     nstreams = 4;
// static constexpr int64_t nthreads = 512;
// static constexpr int64_t N        = 50 * (1 << 20);
// static constexpr int64_t N        = 4096;
using default_device = device::default_device;
using tim::component_tuple;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 600)
__device__ long long int
atomicAdd(long long int* address, long long int val);
#endif

#if !defined(TIMEMORY_USE_CUDA)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::cuda_device_timer, false_type)
#endif

TIMEMORY_DECLARE_COMPONENT(cuda_device_timer)
TIMEMORY_SET_COMPONENT_API(component::cuda_device_timer, tpls::nvidia, device::gpu,
                           category::external, os::agnostic)
TIMEMORY_STATISTICS_TYPE(component::cuda_device_timer, std::vector<float>)
TIMEMORY_DEFINE_CONCRETE_TRAIT(uses_timing_units, component::cuda_device_timer, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_timing_category, component::cuda_device_timer,
                               true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(base_has_accum, component::cuda_device_timer, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(echo_enabled, component::cuda_device_timer, false_type)

namespace tim
{
namespace component
{
struct cuda_device_timer : base<cuda_device_timer, std::vector<long long int>>
{
    using value_type = std::vector<long long int>;
    using base_type  = base<cuda_device_timer, value_type>;

    TIMEMORY_HOST_DEVICE_FUNCTION cuda_device_timer() = default;
    TIMEMORY_HOST_DEVICE_FUNCTION ~cuda_device_timer();
    TIMEMORY_HOST_DEVICE_FUNCTION cuda_device_timer(cuda_device_timer&&) = default;
    TIMEMORY_HOST_DEVICE_FUNCTION cuda_device_timer& operator=(cuda_device_timer&&) =
        default;

    cuda_device_timer(const cuda_device_timer& rhs);
    cuda_device_timer& operator=(const cuda_device_timer& rhs);

    static std::string              label();
    static std::string              description();
    static std::vector<std::string> label_array();
    static std::vector<int64_t>     unit_array();
    static std::vector<std::string> display_unit_array();

    std::vector<float> get() const;
    std::string        get_display() const;

    void allocate(device::gpu = {}, size_t nthreads = 2048);
    void deallocate();
    void start(device::gpu, size_t nthreads = 2048);

#if defined(__CUDA_ARCH__)
    TIMEMORY_DEVICE_FUNCTION void device_init();
    TIMEMORY_DEVICE_FUNCTION void start();
    TIMEMORY_DEVICE_FUNCTION void stop();
#else
    TIMEMORY_HOST_FUNCTION void device_init();
    TIMEMORY_HOST_FUNCTION void start();
    TIMEMORY_HOST_FUNCTION void stop();
#endif

private:
    static size_t& max_threads();

    bool           m_copy       = false;
    int            m_device_num = cuda::get_device();
    int            m_idx        = 0;
    size_t         m_threads    = 0;
    long long int* m_buff       = nullptr;
    long long int* m_data       = nullptr;
    long long int* m_host       = nullptr;
};
}  // namespace component
}  // namespace tim

//--------------------------------------------------------------------------------------//
// saxpy calculation
//
TIMEMORY_GLOBAL_FUNCTION void
saxpy_inst(int64_t n, float a, float* x, float* y, comp::cuda_device_timer _timer)
{
    _timer.device_init();
    _timer.start();
    auto range = device::grid_strided_range<default_device, 0>(n);
    for(int i = range.begin(); i < range.end(); i += range.stride())
    {
        y[i] = a * x[i] + y[i];
    }
    _timer.stop();
}

//--------------------------------------------------------------------------------------//

void
run_saxpy(int nitr, int nstreams, int64_t block_size, int64_t N)
{
    using params_t = device::params<default_device>;
    using stream_t = default_device::stream_t;
    using tuple_t  = component_tuple<comp::wall_clock, comp::cpu_clock, comp::cpu_util,
                                    comp::cuda_event, comp::nvtx_marker>;

    mpl::append_type_t<comp::cuda_device_timer, tuple_t> tot{ __FUNCTION__ };
    tot.start(device::gpu{}, block_size);

    float*                x         = device::cpu::alloc<float>(N);
    float*                y         = device::cpu::alloc<float>(N);
    float    data_size = (3.0 * N * sizeof(float)) / tim::units::gigabyte;
    std::vector<stream_t> streams(std::max<int>(nstreams, 1));
    for(auto& itr : streams)
        cuda::stream_create(itr);
    stream_t stream = streams.at(0);
    params_t params(params_t::compute(N, block_size), block_size);

    auto sync_streams = [&streams]() {
        for(auto& itr : streams)
            cuda::stream_sync(itr);
    };

    std::cout << "\n"
              << __FUNCTION__ << " launching on " << default_device::name()
              << " with parameters: " << params << "\n"
              << std::endl;

    for(int i = 0; i < N; ++i)
    {
        x[i] = 1.0;
        y[i] = 2.0;
    }

    float* d_x = device::gpu::alloc<float>(N);
    float* d_y = device::gpu::alloc<float>(N);
    cuda::memcpy(d_x, x, N, cuda::host_to_device_v, stream);
    cuda::memcpy(d_y, y, N, cuda::host_to_device_v, stream);

    sync_streams();
    for(int i = 0; i < nitr; ++i)
    {
        params.stream = streams.at(i % streams.size());
        device::launch(params, saxpy_inst, N, 1.0, d_x, d_y,
                       *tot.get<comp::cuda_device_timer>());
    }
    sync_streams();

    cuda::memcpy(y, d_y, N, cuda::device_to_host_v, stream);
    cuda::device_sync();
    tot.stop();
    cuda::free(d_x);
    cuda::free(d_y);

    float maxError = 0.0;
    float sumError = 0.0;
    for(int64_t i = 0; i < N; i++)
    {
        maxError = std::max<float>(maxError, std::abs(y[i] - 2.0f));
        sumError += (y[i] > 2.0f) ? (y[i] - 2.0f) : (2.0f - y[i]);
    }

    device::cpu::free(x);
    device::cpu::free(y);

    auto ce = tot.get<comp::cuda_event>();
    auto rc = tot.get<comp::wall_clock>();

    printf("Max error: %8.4e\n", (double) maxError);
    printf("Sum error: %8.4e\n", (double) sumError);
    printf("Total amount of data (GB): %f\n", (double) data_size);
    if(ce)
    {
        printf("Effective Bandwidth (GB/s): %f\n", (double) (data_size / ce->get()));
        printf("Kernel Runtime (sec): %16.12e\n", (double) ce->get());
    }
    if(rc)
        printf("Wall-clock time (sec): %16.12e\n", (double) rc->get());
    if(ce)
        std::cout << __FUNCTION__ << " cuda event: " << *ce << std::endl;
    if(rc)
        std::cout << __FUNCTION__ << " real clock: " << *rc << std::endl;
    std::cout << tot << std::endl;
    std::cout << std::endl;

    for(auto& itr : streams)
        cuda::stream_destroy(itr);
}

//--------------------------------------------------------------------------------------//

int
main(int argc, char** argv)
{
    tim::timemory_init(argc, argv);
    tim::timemory_argparse(&argc, &argv);

    for(auto blocks : { 32, 64, 128, 256, 512 })
        run_saxpy(10, 4, blocks, 50 * (1 << 20));

    tim::timemory_finalize();
    return EXIT_SUCCESS;
}

//--------------------------------------------------------------------------------------//

tim::component::cuda_device_timer::cuda_device_timer(const cuda_device_timer& rhs)
: base_type{ rhs }
, m_copy{ true }
, m_device_num{ rhs.m_device_num }
, m_idx{ rhs.m_idx }
, m_threads{ rhs.m_threads }
, m_buff{ rhs.m_buff }
, m_data{ rhs.m_data }
, m_host{ rhs.m_host }
{}

tim::component::cuda_device_timer&
tim::component::cuda_device_timer::operator=(const cuda_device_timer& rhs)
{
    if(this != &rhs)
    {
        base_type::operator=(rhs);
        m_copy             = true;
        m_device_num       = rhs.m_device_num;
        m_idx              = rhs.m_idx;
        m_threads          = rhs.m_threads;
        m_buff             = rhs.m_buff;
        m_data             = rhs.m_data;
        m_host             = rhs.m_host;
    }
    return *this;
}

TIMEMORY_HOST_DEVICE_FUNCTION
tim::component::cuda_device_timer::~cuda_device_timer()
{
#if !defined(__CUDA_ARCH__)
    deallocate();
#endif
}

std::string
tim::component::cuda_device_timer::label()
{
    return "cuda_device_timer";
}
std::string
tim::component::cuda_device_timer::description()
{
    return "instruments clock64() within a CUDA kernel";
}

std::vector<std::string>
tim::component::cuda_device_timer::label_array()
{
    std::vector<std::string> _labels{};
    for(size_t i = 0; i < max_threads(); ++i)
        _labels.emplace_back(std::string{ "thr" } + std::to_string(i));
    return _labels;
}

std::vector<int64_t>
tim::component::cuda_device_timer::unit_array()
{
    return std::vector<int64_t>(max_threads(), unit());
}

std::vector<std::string>
tim::component::cuda_device_timer::display_unit_array()
{
    return std::vector<std::string>(max_threads(), display_unit());
}

std::vector<float>
tim::component::cuda_device_timer::get() const
{
    std::vector<float> _value{};
    auto               _factor     = static_cast<float>(units::msec) * unit();
    auto               _clock_rate = cuda::get_device_clock_rate(m_device_num);
    for(const auto& itr : get_value())
        _value.emplace_back(itr / static_cast<float>(_clock_rate) / _factor);
    return _value;
}

std::string
tim::component::cuda_device_timer::get_display() const
{
    auto               _data = get();
    std::ostringstream _oss{};
    _oss.setf(get_format_flags());
    for(size_t i = 0; i < _data.size(); ++i)
    {
        if(_data.at(i) > 0.0)
            _oss << ", " << std::setprecision(get_precision()) << std::setw(get_width())
                 << _data.at(i) << " thr" << i;
    }
    auto _str = _oss.str();
    if(_str.length() > 2)
    {
        _oss << " " << get_display_unit();
        return _str.substr(2);
    }
    return _str;
}

void
tim::component::cuda_device_timer::allocate(device::gpu, size_t nthreads)
{
    if(nthreads > m_threads)
    {
        deallocate();
        m_copy    = false;
        m_threads = nthreads;
        m_host    = cuda::malloc_host<long long int>(m_threads);
        m_buff    = cuda::malloc<long long int>(m_threads);
        m_data    = cuda::malloc<long long int>(m_threads);
        cuda::memset(m_buff, 0, m_threads);
        cuda::memset(m_data, 0, m_threads);
        max_threads() = std::max<size_t>(max_threads(), nthreads);
        cuda::check(cuda::get_last_error());
    }
}

void
tim::component::cuda_device_timer::deallocate()
{
    if(m_copy)
        return;
    m_threads = 0;
    if(m_buff)
        cuda::free(m_buff);
    if(m_data)
        cuda::free(m_data);
    if(m_host)
        cuda::free_host(m_host);
}

void
tim::component::cuda_device_timer::start(device::gpu, size_t nthreads)
{
    set_is_invalid(false);
    m_device_num = cuda::get_device();
    allocate(device::gpu{}, nthreads);
}

#if defined(__CUDA_ARCH__)

TIMEMORY_DEVICE_FUNCTION
void
tim::component::cuda_device_timer::device_init()
{
    m_idx =
        (blockDim.x * blockDim.y * blockIdx.z) + (blockDim.x * blockIdx.y) + threadIdx.x;
    assert(m_idx < m_threads);
}

TIMEMORY_DEVICE_FUNCTION
void
tim::component::cuda_device_timer::start()
{
    if(m_buff)
        m_buff[m_idx] = clock64();
}

TIMEMORY_DEVICE_FUNCTION
void
tim::component::cuda_device_timer::stop()
{
    if(m_buff && m_data)
        atomicAdd(&(m_data[m_idx]), (clock64() - m_buff[m_idx]));
}

#else

TIMEMORY_HOST_FUNCTION
void
tim::component::cuda_device_timer::device_init()
{}

TIMEMORY_HOST_FUNCTION
void
tim::component::cuda_device_timer::start()
{
    set_is_invalid(m_data == nullptr);
}

TIMEMORY_HOST_FUNCTION
void
tim::component::cuda_device_timer::stop()
{
    if(m_data && m_host)
    {
        TIMEMORY_CUDA_RUNTIME_API_CALL(cudaStreamSynchronize(0));
        TIMEMORY_CUDA_RUNTIME_API_CALL(
            cuda::memcpy(m_host, m_data, m_threads, cuda::device_to_host_v));
        auto _value = get_value();
        _value.resize(m_threads, 0);
        for(size_t i = 0; i < m_threads; ++i)
            _value[i] += m_host[i];
        set_value(std::move(_value));
    }
}

#endif

size_t&
tim::component::cuda_device_timer::max_threads()
{
    static size_t _value = 0;
    return _value;
}

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 600)
__device__ long long int
atomicAdd(long long int* address, long long int val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*) address;
    unsigned long long int  old            = *address_as_ull;
    unsigned long long int  assumed;
    do
    {
        assumed = old;
        old     = atomicCAS(address_as_ull, assumed, val + assumed);
        // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while(assumed != old);
    return old;
}
#endif
