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
#include "timemory/backends/device.hpp"
#include "timemory/backends/mpi.hpp"
#include "timemory/components/timing.hpp"
#include "timemory/details/settings.hpp"
#include "timemory/ert/aligned_allocator.hpp"
#include "timemory/utility/macros.hpp"

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

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

}  // namespace cache_size

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
//  execution params
//
struct exec_params
{
    explicit exec_params(uint64_t _work_set = 16,
                         uint64_t mem_max   = 8 * cache_size::get_max(),
                         uint64_t _nthread = 1, uint64_t _nstream = 1,
                         uint64_t _grid_size = 0, uint64_t _block_size = 32)
    : working_set_min(_work_set)
    , memory_max(mem_max)
    , nthreads(_nthread)
    , nstreams(_nstream)
    , grid_size(_grid_size)
    , block_size(_block_size)
    {
    }

    ~exec_params()                  = default;
    exec_params(const exec_params&) = default;
    exec_params(exec_params&&)      = default;
    exec_params& operator=(const exec_params&) = default;
    exec_params& operator=(exec_params&&) = default;

    uint64_t working_set_min = 16;
    uint64_t memory_max      = 8 * cache_size::get_max();  // default is 8 * L3 cache size
    uint64_t nthreads        = 1;
    uint64_t nrank           = tim::mpi::rank();
    uint64_t nproc           = tim::mpi::size();
    uint64_t nstreams        = 1;
    uint64_t grid_size       = 0;
    uint64_t block_size      = 32;
    uint64_t shmem_size      = 0;

    template <typename Archive>
    void serialize(Archive& ar, const unsigned int)
    {
        ar(serializer::make_nvp("working_set_min", working_set_min),
           serializer::make_nvp("memory_max", memory_max),
           serializer::make_nvp("nthreads", nthreads),
           serializer::make_nvp("nrank", nrank), serializer::make_nvp("nproc", nproc),
           serializer::make_nvp("nstreams", nstreams),
           serializer::make_nvp("grid_size", grid_size),
           serializer::make_nvp("block_size", block_size),
           serializer::make_nvp("shmem_size", shmem_size));
    }
};

//--------------------------------------------------------------------------------------//
//  execution data -- reuse this for multiple types
//
class exec_data
{
public:
    using value_type     = std::tuple<std::string, uint64_t, uint64_t, double, uint64_t,
                                  uint64_t, double, double, double>;
    using labels_type    = std::array<string_t, std::tuple_size<value_type>::value>;
    using value_array    = std::vector<value_type>;
    using iterator       = typename value_array::iterator;
    using const_iterator = typename value_array::const_iterator;

    static constexpr std::size_t size() { return std::tuple_size<value_type>::value; }

    exec_data()                 = default;
    ~exec_data()                = default;
    exec_data(const exec_data&) = delete;
    exec_data(exec_data&&)      = default;
    exec_data& operator=(const exec_data&) = delete;
    exec_data& operator=(exec_data&&) = default;

    exec_data& operator+=(const value_type& entry)
    {
        {
            std::unique_lock<std::mutex> lk(pmutex, std::defer_lock);
            if(!lk.owns_lock())
                lk.lock();
            // std::cout << "Adding " << std::get<0>(entry) << "..." << std::endl;
            m_values.push_back(entry);
        }
        return *this;
    }
    exec_data& operator+=(const exec_data& rhs)
    {
        {
            std::unique_lock<std::mutex> lk(pmutex);
            if(!lk.owns_lock())
                lk.lock();
            for(const auto& itr : rhs.m_values)
                m_values.push_back(itr);
        }
        return *this;
    }

    void        set_labels(const labels_type& _labels) { m_labels = _labels; }
    labels_type get_labels() const { return m_labels; }

    iterator       begin() { return m_values.begin(); }
    const_iterator begin() const { return m_values.begin(); }
    iterator       end() { return m_values.end(); }
    const_iterator end() const { return m_values.end(); }

protected:
    labels_type m_labels = {
        "label",     "working-set",   "trials",      "seconds",     "total-bytes",
        "total-ops", "bytes-per-sec", "ops-per-sec", "ops-per-set",
    };
    value_array m_values;
    std::mutex  pmutex;

public:
    friend std::ostream& operator<<(std::ostream& os, const exec_data& obj)
    {
        for(const auto& itr : obj.m_values)
        {
            os << std::setw(24) << std::get<0>(itr) << ": ";
            obj.write<1>(os, itr, ", ", 10);
            obj.write<2>(os, itr, ", ", 6);
            obj.write<3>(os, itr, ", ", 12);
            obj.write<4>(os, itr, ", ", 12);
            obj.write<5>(os, itr, ", ", 12);
            obj.write<6>(os, itr, ", ", 12);
            obj.write<7>(os, itr, ", ", 12);
            obj.write<8>(os, itr, "\n", 4);
        }
        return os;
    }

    template <typename Archive>
    void serialize(Archive& ar, const unsigned int)
    {
        // ar(serializer::make_nvp("labels", m_labels));
        // using size_type = typename value_array::size_type;
        // ar(cereal::make_size_tag(
        //    static_cast<size_type>(m_values.size())));  // number of elements
        constexpr auto sz = std::tuple_size<value_type>::value;
        // for(size_type i = 0; i < m_values.size(); ++i)
        ar.setNextName("ert");
        ar.startNode();
        ar.makeArray();
        for(auto& itr : m_values)
        {
            ar.startNode();
            _serialize(ar, itr, make_index_sequence<sz>{});
            ar.finishNode();
        }
        ar.finishNode();
    }

private:
    template <size_t _N>
    void write(std::ostream& os, const value_type& ret, const string_t& _trailing,
               const int32_t& _width) const
    {
        os << std::setw(10) << std::get<_N>(m_labels) << " = " << std::setw(_width)
           << std::get<_N>(ret) << _trailing;
    }

    template <typename _Archive, size_t... _Idx>
    void _serialize(_Archive& ar, value_type& _tuple, index_sequence<_Idx...>)
    {
        ar(serializer::make_nvp(std::get<_Idx>(m_labels), std::get<_Idx>(_tuple))...);
    }
};

//--------------------------------------------------------------------------------------//
//
//      CPU -- initialize buffer
//
//--------------------------------------------------------------------------------------//

template <typename _Device, typename _Tp, typename _Intp = int32_t,
          device::enable_if_cpu_t<_Device> = 0>
void
initialize_buffer(_Tp* A, const _Tp& value, const _Intp& nsize)
{
    auto range = device::grid_strided_range<_Device, 0, _Intp>(nsize);
    for(auto i = range.begin(); i < range.end(); i += range.stride())
    {
        if(i < nsize)
            A[i] = value;
        else
            printf("[%s]> Warning! Request for A[%lli] which is > %lli...\n",
                   __FUNCTION__, (long long) i, (long long) nsize);
    }
}

//--------------------------------------------------------------------------------------//
//
//      GPU -- initialize buffer
//
//--------------------------------------------------------------------------------------//

template <typename _Device, typename _Tp, typename _Intp = int32_t,
          device::enable_if_gpu_t<_Device> = 0>
GLOBAL_CALLABLE void
initialize_buffer(_Tp* A, _Tp value, _Intp nsize)
{
    auto range = device::grid_strided_range<_Device, 0, _Intp>(nsize);
    for(auto i = range.begin(); i < range.end(); i += range.stride())
        A[i] = value;
}

//--------------------------------------------------------------------------------------//
//  measure floating-point or integer operations
//
template <typename _Device, typename _Tp, typename _ExecData = exec_data,
          typename _Counter = component::real_clock>
class counter
{
public:
    using string_t            = std::string;
    using mutex_t             = std::mutex;
    using lock_t              = std::unique_lock<mutex_t>;
    using counter_type        = _Counter;
    using counter_create_func = std::function<counter_type()>;
    using exec_data_t         = _ExecData;
    using units_type          = decltype(counter_type::unit());
    using data_type           = typename exec_data_t::value_type;
    using data_ptr_t          = std::shared_ptr<exec_data_t>;
    using ull                 = unsigned long long;

public:
    counter() = default;

    explicit counter(const exec_params& _params, uint64_t _align = 8 * sizeof(_Tp),
                     data_ptr_t _exec_data = nullptr)
    : params(_params)
    , align(_align)
    , data(_exec_data)
    {
        compute_internal();
    }

    // overload how to create the counter
    counter(const exec_params& _params, const counter_create_func& _func,
            uint64_t _align = 8 * sizeof(_Tp), data_ptr_t _exec_data = nullptr)
    : params(_params)
    , align(_align)
    , create_counter(_func)
    , data(_exec_data)
    {
        compute_internal();
    }

    explicit counter(const exec_params& _params, data_ptr_t _exec_data,
                     uint64_t _align = 8 * sizeof(_Tp))
    : params(_params)
    , align(_align)
    , data(_exec_data)
    {
        compute_internal();
    }

    ~counter() {}

private:
    void compute_internal()
    {
        if(device::is_cpu<_Device>::value)
            params.nstreams = 1;
        nsize = params.memory_max / params.nproc / params.nthreads;
        nsize = nsize & (~(align - 1));
        nsize = nsize / sizeof(_Tp);
        nsize = std::max<uint64_t>(nsize, 1);
    }

public:
    _Tp* get_buffer()
    {
        // check alignment and
        align = std::max<uint64_t>(align, 8 * sizeof(_Tp));
        compute_internal();

        if(settings::debug())
            printf("[%s]> nsize = %llu\n", __FUNCTION__, (ull) nsize);
        _Tp* buffer = allocate_aligned<_Tp, _Device>(nsize, align);
        if(settings::debug())
            printf("[%s]> buffer = %p\n", __FUNCTION__, buffer);
        device::params<_Device> params(0, 512, 0, 0);
        device::launch(nsize, params, initialize_buffer<_Device, _Tp, uint64_t>, buffer,
                       _Tp(1), nsize);
        return buffer;
    }

    void destroy_buffer(_Tp* buffer) { free_aligned<_Tp, _Device>(buffer); }

    counter_type get_counter() const { return create_counter(); }

    inline void record(counter_type& _counter, int n, int t, uint64_t nops)
    {
        // std::cout << "Recording " << label << "..." << std::endl;
        uint64_t working_set_size = n * params.nthreads * params.nproc;
        uint64_t total_bytes =
            t * working_set_size * bytes_per_element * memory_accesses_per_element;
        uint64_t          total_ops = t * working_set_size * nops;
        auto              seconds   = _counter.get() * counter_units;
        std::stringstream ss;
        if(label.length() > 0)
            ss << label << "_" << nops;
        else
        {
            if(nops > 1)
                ss << "vector_" << nops;
            else
                ss << "scalar_" << nops;
        }
        data->operator+=(data_type(ss.str(), working_set_size * bytes_per_element, t,
                                   seconds, total_bytes, total_ops, total_bytes / seconds,
                                   total_ops / seconds, nops));
    }

    void set_create_counter(counter_create_func& f) { create_counter = std::bind(f); }

public:
    exec_params params                      = exec_params();
    int         bytes_per_element           = 0;
    int         memory_accesses_per_element = 0;
    uint64_t    align                       = sizeof(_Tp);
    uint64_t    nsize                       = 0;
    units_type  counter_units               = tim::units::sec;
    data_ptr_t  data                        = nullptr;
    std::string label                       = "";

    template <typename Archive>
    void serialize(Archive& ar, const unsigned int)
    {
        if(!data.get())  // for input
            data = data_ptr_t(new _ExecData());
        ar(serializer::make_nvp("params", params), serializer::make_nvp("data", *data));
    }

protected:
    counter_create_func create_counter = []() -> counter_type { return counter_type(); };

public:
    friend std::ostream& operator<<(std::ostream& os, const counter& obj)
    {
        if(obj.data)
            os << (*obj.data);
        return os;
    }
};

}  // namespace ert
}  // namespace tim
