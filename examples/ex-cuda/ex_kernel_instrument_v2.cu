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

struct cuda_data_tag
{};
using cuda_device_timer_data = comp::data_tracker<float, cuda_data_tag>;

TIMEMORY_STATISTICS_TYPE(cuda_device_timer_data, float)
TIMEMORY_DEFINE_CONCRETE_TRAIT(uses_timing_units, cuda_device_timer_data, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_timing_category, cuda_device_timer_data, true_type)

namespace tim
{
namespace component
{
struct cuda_device_timer : base<cuda_device_timer, void>
{
    using value_type   = void;
    using base_type  = base<cuda_device_timer, value_type>;
    using tracker_type = tim::component_bundle<TIMEMORY_API, cuda_device_timer_data>;

    TIMEMORY_HOST_DEVICE_FUNCTION cuda_device_timer() = default;
    TIMEMORY_HOST_DEVICE_FUNCTION ~cuda_device_timer();
    TIMEMORY_HOST_DEVICE_FUNCTION cuda_device_timer(cuda_device_timer&&) = default;
    TIMEMORY_HOST_DEVICE_FUNCTION cuda_device_timer& operator=(cuda_device_timer&&) =
        default;

    cuda_device_timer(const cuda_device_timer& rhs);
    cuda_device_timer& operator=(const cuda_device_timer& rhs);

    static void                     preinit();
    static std::string              label();
    static std::string              description();

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

    void set_prefix(const char* _prefix) { m_prefix = _prefix; }

private:
    static size_t& max_threads();

    bool           m_copy       = false;
    int            m_device_num = cuda::get_device();
    int            m_idx        = 0;
    size_t         m_threads    = 0;
    long long int* m_buff       = nullptr;
    long long int* m_data       = nullptr;
    long long int* m_host       = nullptr;
    const char*    m_prefix     = nullptr;
    tracker_type   m_tracker    = {};
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

void
tim::component::cuda_device_timer::preinit()
{
    cuda_device_timer_data::label()       = label();
    cuda_device_timer_data::description() = description();
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
    if(m_prefix)
    {
        m_tracker.rekey(m_prefix);
        m_tracker.start();
    }
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
    if(m_data != nullptr && m_prefix != nullptr)
    {
        m_tracker.rekey(m_prefix);
        m_tracker.start();
    }
}

TIMEMORY_HOST_FUNCTION
void
tim::component::cuda_device_timer::stop()
{
    if(m_data && m_host)
    {
        auto _factor =
            static_cast<float>(units::msec) * cuda_device_timer_data::get_unit();
        auto _clock_rate = cuda::get_device_clock_rate(m_device_num);
        TIMEMORY_CUDA_RUNTIME_API_CALL(cudaStreamSynchronize(0));
        TIMEMORY_CUDA_RUNTIME_API_CALL(
            cuda::memcpy(m_host, m_data, m_threads, cuda::device_to_host_v));
        for(size_t i = 0; i < m_threads; ++i)
        {
            if(m_host[i] <= 0)
                continue;
            float _value = m_host[i] / static_cast<float>(_clock_rate) / _factor;
            m_tracker.store(
                [](float lhs, float rhs) { return std::max<float>(lhs, rhs); }, _value);
            // m_tracker.add_secondary(TIMEMORY_JOIN("_", "thread", i),
            // std::move(_value));
            auto _scope = tim::scope::config{} + tim::scope::tree{};
            tracker_type _child{ TIMEMORY_JOIN("_", "thread", i), _scope };
            _child.start();
            _child.store(std::plus<float>{}, _value);
            _child.stop();
        }
        m_tracker.stop();
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

TIMEMORY_INITIALIZE_STORAGE(comp::cuda_device_timer, cuda_device_timer_data)
