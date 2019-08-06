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

#pragma once

#include "timemory/backends/cuda.hpp"
#include "timemory/backends/mpi.hpp"
#include "timemory/components/timing.hpp"
#include "timemory/utility/macros.hpp"

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#if defined(__INTEL_COMPILER)
#    define ASSUME_ALIGNED_ARRAY(ARRAY, WIDTH) __assume_aligned(ARRAY, WIDTH)
#elif defined(__xlC__)
#    define ASSUME_ALIGNED_ARRAY(ARRAY, WIDTH) __alignx(WIDTH, ARRAY)
#else
#    define ASSUME_ALIGNED_ARRAY(ARRAY, WIDTH)
#endif

// #define ERT_FLOP 256
// #define ERT_TRIALS_MIN 100
// #define ERT_WORKING_SET_MIN 100
// #define ERT_MEMORY_MAX 64 * 64 * 64 * 64

#if !defined(_WINDOWS)
#    define RESTRICT(TYPE) TYPE __restrict__
#else
#    define RESTRICT(TYPE) TYPE
#endif

#include <cstddef>

#if defined(_MACOS)
#    include <sys/sysctl.h>
#elif defined(_WINDOWS)
#    include <cstdlib>
#    include <windows.h>
#elif defined(_LINUX)
#    include <cstdio>
#endif

namespace tim
{
namespace ert
{
using std::size_t;

namespace impl
{
#if defined(_MACOS)
inline size_t
cache_size(const int& level)
{
    // configure sysctl query
    //      L1  ->  hw.l1dcachesize
    //      L2  ->  hw.l2cachesize
    //      L3  ->  hw.l3cachesize
    //
    std::stringstream query;
    query << "hw.l" << level;
    if(level == 1)
        query << "d";
    query << "cachesize";

    size_t line_size        = 0;
    size_t sizeof_line_size = sizeof(line_size);
    sysctlbyname(query.str().c_str(), &line_size, &sizeof_line_size, 0, 0);
    return line_size;
}

#elif defined(_WINDOWS)

inline size_t
cache_size(const int& level)
{
    size_t                                line_size   = 0;
    DWORD                                 buffer_size = 0;
    DWORD                                 i           = 0;
    SYSTEM_LOGICAL_PROCESSOR_INFORMATION* buffer      = 0;

    GetLogicalProcessorInformation(0, &buffer_size);
    buffer = (SYSTEM_LOGICAL_PROCESSOR_INFORMATION*) malloc(buffer_size);
    GetLogicalProcessorInformation(&buffer[0], &buffer_size);

    for(i = 0; i != buffer_size / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION); ++i)
    {
        if(buffer[i].Relationship == RelationCache && buffer[i].Cache.Level == level)
        {
            line_size = buffer[i].Cache.Size;
            break;
        }
    }

    free(buffer);
    return line_size;
}

#elif defined(_LINUX)

inline size_t
cache_size(const int& _level)
{
    // L1 has a data and instruction cache, index0 should be data
    auto level = (_level == 1) ? 0 : (_level);
    // location of files
    std::stringstream fpath;
    fpath << "/sys/devices/system/cpu/cpu0/cache/index" << level << "/";

    // files to read
    static thread_local std::array<std::string, 3> files(
        { { "number_of_sets", "ways_of_associativity", "coherency_line_size" } });

    uint64_t product = 1;
    for(unsigned i = 0; i < files.size(); ++i)
    {
        std::string   fname = fpath.str() + files[i];
        std::ifstream ifs(fname.c_str());
        if(ifs)
        {
            uint64_t val;
            ifs >> val;
            product *= val;
        }
        else
        {
            throw std::runtime_error("Unable to open file: " + fname);
        }
        ifs.close();
    }
    return (product > 1) ? product : 0;
}

#else

#    warning Unrecognized platform
inline size_t
cache_size()
{
    return 0;
}

#endif

}  // namespace impl

//--------------------------------------------------------------------------------------//
//  get the size of the L1 (data), L2, or L3 cache
//
namespace cache_size
{
template <size_t _Level>
inline size_t
get()
{
    // only enable queries 1, 2, 3
    static_assert(_Level > 0 && _Level < 4,
                  "Request for cache level that is not supported");

    // avoid multiple queries
    static size_t _value = impl::cache_size(_Level);
    return _value;
}

inline size_t
get(const int& _level)
{
    // only enable queries 1, 2, 3
    if(_level < 1 || _level > 3)
    {
        std::stringstream ss;
        ss << "tim::ert::cache_size::get(" << _level << ") :: "
           << "Requesting invalid cache level";
        throw std::runtime_error(ss.str());
    }
    // avoid multiple queries
    static std::vector<size_t> _values(
        { { impl::cache_size(1), impl::cache_size(2), impl::cache_size(3) } });
    return _values.at(_level - 1);
}

inline size_t
get_max()
{
    // this is useful for system like KNL that do not have L3 cache
    for(auto level : { 3, 2, 1 })
    {
        try
        {
            auto sz = impl::cache_size(level);
            // if this succeeded, we can return the value
            return sz;
        }
        catch(...)
        {
            continue;
        }
    }
    return 0;
}
};

//--------------------------------------------------------------------------------------//
//  aligned allocation
//
template <typename _Tp>
_Tp*
allocate_aligned(size_t size, size_t alignment)
{
#if defined(__INTEL_COMPILER)
    return static_cast<_Tp*>(_mm_malloc(size * sizeof(_Tp), alignment));
#elif defined(_UNIX)
    void* ptr = nullptr;
    auto  ret = posix_memalign(&ptr, alignment, size * sizeof(_Tp));
    consume_parameters(ret);
    return static_cast<_Tp*>(ptr);
#elif defined(_WINDOWS)
    return static_cast<_Tp*>(_aligned_malloc(size * sizeof(_Tp), alignment));
#endif
}

//--------------------------------------------------------------------------------------//
//  free aligned array
//
template <typename _Tp>
void
free_aligned(_Tp* ptr)
{
#if defined(__INTEL_COMPILER)
    _mm_free(static_cast<void*>(ptr));
#elif defined(_UNIX) || defined(_WINDOWS)
    free(static_cast<void*>(ptr));
#endif
}

//--------------------------------------------------------------------------------------//
//  execution params
//
struct exec_params
{
    exec_params(uint64_t _work_set = 16, uint64_t mem_max = 8 * cache_size::get_max(),
                uint64_t _nthread = 1)
    : working_set_min(_work_set)
    , memory_max(mem_max)
    , nthreads(_nthread)
    {
    }

    uint64_t working_set_min = 16;
    uint64_t memory_max      = 8 * cache_size::get_max();  // default is 8 * L3 cache size
    const uint64_t nthreads  = 1;
    const uint64_t nrank     = tim::mpi_rank();
    const uint64_t nproc     = tim::mpi_size();

    template <typename Archive>
    void serialize(Archive& ar, const unsigned int)
    {
        ar(serializer::make_nvp("working_set_min", working_set_min),
           serializer::make_nvp("memory_max", memory_max),
           serializer::make_nvp("nthreads", nthreads),
           serializer::make_nvp("nrank", nrank), serializer::make_nvp("nproc", nproc));
    }
};

//--------------------------------------------------------------------------------------//
//  creates a multithreading barrier
//
class thread_barrier
{
public:
    using size_type = int64_t;
    using mutex_t   = std::mutex;
    using condvar_t = std::condition_variable;
    using atomic_t  = std::atomic<size_type>;
    using lock_t    = std::unique_lock<mutex_t>;

public:
    explicit thread_barrier(const size_t& nthreads)
    : m_num_threads(nthreads)
    {
    }

    thread_barrier(const thread_barrier&) = delete;
    thread_barrier(thread_barrier&&)      = delete;

    thread_barrier& operator=(const thread_barrier&) = delete;
    thread_barrier& operator=(thread_barrier&&) = delete;

    size_type size() const { return m_num_threads; }

    // call from worker thread -- spin wait (fast)
    void spin_wait()
    {
        if(is_master())
            throw std::runtime_error("master thread calling worker wait function");

        {
            lock_t lk(m_mutex);
            ++m_counter;
            ++m_waiting;
        }

        while(m_counter < m_num_threads)
        {
            while(spin_lock.test_and_set(std::memory_order_acquire))  // acquire lock
                ;                                                     // spin
            spin_lock.clear(std::memory_order_release);
        }

        {
            lock_t lk(m_mutex);
            --m_waiting;
            if(m_waiting == 0)
                m_counter = 0;  // reset barrier
        }
    }

    // call from worker thread -- condition variable wait (slower)
    void cv_wait()
    {
        if(is_master())
            throw std::runtime_error("master thread calling worker wait function");

        lock_t lk(m_mutex);
        ++m_counter;
        ++m_waiting;
        m_cv.wait(lk, [&] { return m_counter >= m_num_threads; });
        m_cv.notify_one();
        --m_waiting;
        if(m_waiting == 0)
            m_counter = 0;  // reset barrier
    }

    // check if this is the thread the created barrier
    bool is_master() const { return std::this_thread::get_id() == m_master; }

private:
    // the constructing thread will be set to master
    std::thread::id  m_master      = std::this_thread::get_id();
    size_type        m_num_threads = 0;  // number of threads that will wait on barrier
    size_type        m_waiting     = 0;  // number of threads waiting on lock
    size_type        m_counter     = 0;  // number of threads that have entered wait func
    std::atomic_flag spin_lock     = ATOMIC_FLAG_INIT;  // for spin lock
    mutex_t          m_mutex;
    condvar_t        m_cv;
};

//--------------------------------------------------------------------------------------//
//  measure CPU floating-point or integer operations
//
namespace cpu
{
template <typename _Tp, typename _Counter = tim::component::real_clock>
class operation_counter
{
public:
    using string_t            = std::string;
    using counter_type        = _Counter;
    using counter_create_func = std::function<counter_type()>;
    using units_type          = decltype(counter_type::unit());
    using labels_type         = std::array<string_t, 8>;
    using mutex_t             = std::mutex;
    using lock_t              = std::unique_lock<mutex_t>;
    using result_type =
        std::tuple<uint64_t, uint64_t, float, uint64_t, uint64_t, float, float, float>;
    using result_array = std::vector<result_type>;

public:
    operation_counter() = default;

    explicit operation_counter(const exec_params& _params, uint64_t _align = sizeof(_Tp))
    : params(_params)
    , align(_align)
    {
        nsize = params.memory_max / params.nproc / params.nthreads;
        nsize = nsize & (~(align - 1));
        nsize = nsize / sizeof(_Tp);
        nsize = std::max<uint64_t>(nsize, 1);
    }

    // overload how to create the counter
    operation_counter(const exec_params& _params, const counter_create_func& _func,
                      uint64_t _align = sizeof(_Tp))
    : params(_params)
    , align(_align)
    , create_counter(_func)
    {
    }

    ~operation_counter() { delete pmutex; }

public:
    _Tp* get_buffer()
    {
        _Tp* buffer = allocate_aligned<_Tp>(nsize, align);
        for(uint64_t i = 0; i < nsize; ++i)
            buffer[i] = _Tp(1);
        return buffer;
    }

    void destroy_buffer(_Tp* buffer) { free_aligned(buffer); }

    counter_type get_counter() const { return create_counter(); }

    inline void record(counter_type& _counter, int n, int t, uint64_t nops)
    {
        uint64_t working_set_size = n * params.nthreads * params.nproc;
        uint64_t total_bytes =
            t * working_set_size * bytes_per_element * memory_accesses_per_element;
        uint64_t total_ops = t * working_set_size * nops;
        auto     seconds   = _counter.get() * counter_units;
        // lock before storing result
        lock_t lk(*pmutex);
        data.push_back(result_type(working_set_size * bytes_per_element, t, seconds,
                                   total_bytes, total_ops, total_bytes / seconds,
                                   total_ops / seconds, nops));
    }

    void        set_labels(const labels_type& _labels) { labels = _labels; }
    labels_type get_labels() const { return labels; }
    void set_create_counter(counter_create_func& f) { create_counter = std::bind(f); }

public:
    exec_params  params                      = exec_params();
    int          bytes_per_element           = 0;
    int          memory_accesses_per_element = 0;
    uint64_t     align                       = sizeof(_Tp);
    uint64_t     nsize                       = 0;
    units_type   counter_units               = tim::units::sec;
    result_array data;

    template <typename Archive>
    void serialize(Archive& ar, const unsigned int)
    {
        ar(serializer::make_nvp("params", params), serializer::make_nvp("labels", labels),
           serializer::make_nvp("data", data));
    }

protected:
    std::mutex* pmutex = new std::mutex;
    labels_type labels =
        labels_type({ { "working-set", "trials", "seconds", "total-bytes", "total-ops",
                        "bytes-per-sec", "ops-per-sec", "ops-per-set" } });
    counter_create_func create_counter = []() -> counter_type { return counter_type(); };

public:
    friend std::ostream& operator<<(std::ostream& os, const operation_counter& obj)
    {
        for(const auto& itr : obj.data)
        {
            obj.write<0>(os, itr, ", ");
            obj.write<1>(os, itr, ", ");
            obj.write<2>(os, itr, ", ");
            obj.write<3>(os, itr, ", ");
            obj.write<4>(os, itr, ", ");
            obj.write<5>(os, itr, ", ");
            obj.write<6>(os, itr, ", ");
            obj.write<7>(os, itr, "\n");
        }
        return os;
    }

private:
    template <size_t _N>
    void write(std::ostream& os, const result_type& ret, const string_t& _trailing) const
    {
        os << std::setw(12) << std::get<_N>(labels) << " = " << std::setw(12)
           << std::get<_N>(ret) << _trailing;
    }
};
}  // namespace cpu

//--------------------------------------------------------------------------------------//
//  measure GPU floating-point or integer operations
//
namespace gpu
{
template <typename _Tp>
class operation_counter
{
public:
    using string_t = std::string;
    using timer_t  = tim::component::real_clock;
    using result_type =
        std::tuple<uint64_t, uint64_t, float, uint64_t, uint64_t, float, float, float>;
    using result_array = std::vector<result_type>;
    using labels_type  = std::array<string_t, 8>;

public:
    operation_counter() = default;
    explicit operation_counter(const exec_params& _params, size_t _align = sizeof(_Tp))
    : params(_params)
    , align(_align)
    {
    }
    ~operation_counter()
    {
        delete rc;
        cuda::free(buffer);
    }

public:
    _Tp* initialize()
    {
        nsize  = params.memory_max / params.nproc / params.nthreads;
        nsize  = nsize & (~(align - 1));
        nsize  = nsize / sizeof(_Tp);
        buffer = cuda::malloc<_Tp>(nsize);
        cuda::memset<_Tp>(buffer, 0, nsize);
        return buffer;
    }

    inline void start()
    {
        rc = new timer_t();
        rc->start();
        // return rc;
    }

    inline void stop(int n, int t, size_t nops)
    {
        rc->stop();
        uint64_t working_set_size = n * params.nthreads * params.nproc;
        uint64_t total_bytes =
            t * working_set_size * bytes_per_element * memory_accesses_per_element;
        uint64_t total_ops = t * working_set_size * nops;
        auto     seconds   = rc->get() * tim::units::sec;
        data.push_back(result_type(working_set_size * bytes_per_element, t, seconds,
                                   total_bytes, total_ops, total_bytes / seconds,
                                   total_ops / seconds,
                                   total_ops / static_cast<float>(total_bytes)));
        delete rc;
        rc = nullptr;
    }

public:
    exec_params    params                      = exec_params();
    int            grid_size                   = 32;
    int            block_size                  = 32;
    int            shmem                       = 0;
    cuda::stream_t stream                      = 0;
    int            bytes_per_element           = 0;
    int            memory_accesses_per_element = 0;
    size_t         align                       = 32;
    uint64_t       nsize                       = 0;
    timer_t*       rc                          = nullptr;
    result_array   data;
    labels_type    labels =
        labels_type({ { "working-set", "trials", "seconds", "total-bytes", "total-ops",
                        "bytes-per-sec", "ops-per-sec", "intensity" } });
    _Tp* buffer = nullptr;

public:
    friend std::ostream& operator<<(std::ostream& os, const operation_counter& obj)
    {
        for(const auto& itr : obj.data)
        {
            obj.write<0>(os, itr, ", ");
            obj.write<1>(os, itr, ", ");
            obj.write<2>(os, itr, ", ");
            obj.write<3>(os, itr, ", ");
            obj.write<4>(os, itr, ", ");
            obj.write<5>(os, itr, ", ");
            obj.write<6>(os, itr, ", ");
            obj.write<7>(os, itr, "\n");
        }
        return os;
    }

private:
    template <size_t _N>
    void write(std::ostream& os, const result_type& ret, const string_t& _trailing) const
    {
        os << std::setw(12) << std::get<_N>(labels) << " = " << std::setw(12)
           << std::get<_N>(ret) << _trailing;
    }
};
}  // namespace gpu

}  // namespace ert
}  // namespace tim
