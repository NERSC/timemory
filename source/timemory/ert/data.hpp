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
#include "timemory/bits/settings.hpp"
#include "timemory/components/timing.hpp"
#include "timemory/ert/aligned_allocator.hpp"
#include "timemory/ert/barrier.hpp"
#include "timemory/ert/cache_size.hpp"
#include "timemory/ert/types.hpp"
#include "timemory/utility/macros.hpp"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace tim
{
namespace ert
{
using std::size_t;
using namespace std::placeholders;

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
    {}

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

    friend std::ostream& operator<<(std::ostream& os, const exec_params& obj)
    {
        std::stringstream ss;
        ss << "working_set_min = " << obj.working_set_min << ", "
           << "memory_max = " << obj.memory_max << ", "
           << "nthreads = " << obj.nthreads << ", "
           << "nrank = " << obj.nrank << ", "
           << "nproc = " << obj.nproc << ", "
           << "nstreams = " << obj.nstreams << ", "
           << "grid_size = " << obj.grid_size << ", "
           << "block_size = " << obj.block_size << ", "
           << "shmem_size = " << obj.shmem_size;
        os << ss.str();
        return os;
    }
};

//--------------------------------------------------------------------------------------//
//  execution data -- reuse this for multiple types
//
class exec_data
{
public:
    using value_type =
        std::tuple<std::string, uint64_t, uint64_t, double, uint64_t, uint64_t, double,
                   double, double, std::string, std::string, exec_params>;
    using labels_type    = std::array<string_t, std::tuple_size<value_type>::value>;
    using value_array    = std::vector<value_type>;
    using size_type      = typename value_array::size_type;
    using iterator       = typename value_array::iterator;
    using const_iterator = typename value_array::const_iterator;

    //----------------------------------------------------------------------------------//
    //
    exec_data() = default;
    ~exec_data() { delete pmutex; }
    exec_data(const exec_data&) = delete;
    exec_data(exec_data&&)      = default;
    exec_data& operator=(const exec_data&) = delete;
    exec_data& operator=(exec_data&&) = default;

public:
    //----------------------------------------------------------------------------------//
    //
    void           set_labels(const labels_type& _labels) { m_labels = _labels; }
    labels_type    get_labels() const { return m_labels; }
    size_type      size() { return m_values.size(); }
    iterator       begin() { return m_values.begin(); }
    const_iterator begin() const { return m_values.begin(); }
    iterator       end() { return m_values.end(); }
    const_iterator end() const { return m_values.end(); }

public:
    //----------------------------------------------------------------------------------//
    //
    exec_data& operator+=(const value_type& entry)
    {
        {
            std::unique_lock<std::mutex> lk(*pmutex, std::defer_lock);
            if(!lk.owns_lock())
                lk.lock();
            // std::cout << "Adding " << std::get<0>(entry) << "..." << std::endl;
            m_values.push_back(entry);
        }
        return *this;
    }

    //----------------------------------------------------------------------------------//
    //
    exec_data& operator+=(const exec_data& rhs)
    {
        {
            std::unique_lock<std::mutex> lk(*pmutex);
            if(!lk.owns_lock())
                lk.lock();
            for(const auto& itr : rhs.m_values)
                m_values.push_back(itr);
        }
        return *this;
    }

protected:
    labels_type m_labels = { { "label", "working-set", "trials", "seconds", "total-bytes",
                               "total-ops", "bytes-per-sec", "ops-per-sec", "ops-per-set",
                               "device", "dtype", "exec-params" } };
    value_array m_values;
    std::mutex* pmutex = new std::mutex;

public:
    //----------------------------------------------------------------------------------//
    //
    friend std::ostream& operator<<(std::ostream& os, const exec_data& obj)
    {
        std::stringstream ss;
        for(const auto& itr : obj.m_values)
        {
            ss << std::setw(24) << std::get<0>(itr) << " (device: " << std::get<9>(itr)
               << ", dtype = " << std::get<10>(itr) << "): ";
            obj.write<1>(ss, itr, ", ", 10);
            obj.write<2>(ss, itr, ", ", 6);
            obj.write<3>(ss, itr, ", ", 12);
            obj.write<4>(ss, itr, ", ", 12);
            obj.write<5>(ss, itr, ", ", 12);
            obj.write<6>(ss, itr, ", ", 12);
            obj.write<7>(ss, itr, ", ", 12);
            obj.write<8>(ss, itr, "\n", 4);
        }
        os << ss.str();
        return os;
    }

    //----------------------------------------------------------------------------------//
    //
    template <typename Archive>
    void serialize(Archive& ar, const unsigned int)
    {
        constexpr auto sz = std::tuple_size<value_type>::value;
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
    //----------------------------------------------------------------------------------//
    //
    template <size_t _N>
    void write(std::ostream& os, const value_type& ret, const string_t& _trailing,
               const int32_t& _width) const
    {
        os << std::setw(10) << std::get<_N>(m_labels) << " = " << std::setw(_width)
           << std::get<_N>(ret) << _trailing;
    }

    //----------------------------------------------------------------------------------//
    //
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
        A[i] = value;
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
template <typename _Device, typename _Tp, typename _ExecData, typename _Counter>
class counter
{
public:
    using string_t      = std::string;
    using mutex_t       = std::recursive_mutex;
    using lock_t        = std::unique_lock<mutex_t>;
    using counter_type  = _Counter;
    using exec_data_t   = _ExecData;
    using this_type     = counter<_Device, _Tp, _ExecData, _Counter>;
    using callback_type = std::function<void(uint64_t, this_type&)>;
    using units_type    = decltype(counter_type::unit());
    using data_type     = typename exec_data_t::value_type;
    using data_ptr_t    = std::shared_ptr<exec_data_t>;
    using ull           = unsigned long long;

public:
    //----------------------------------------------------------------------------------//
    //  default construction
    //
    counter()               = default;
    ~counter()              = default;
    counter(const counter&) = default;
    counter(counter&&)      = default;
    counter& operator=(const counter&) = default;
    counter& operator=(counter&&) = default;

    //----------------------------------------------------------------------------------//
    // standard creation
    //
    explicit counter(const exec_params& _params, data_ptr_t _exec_data,
                     uint64_t _align = 8 * sizeof(_Tp))
    : params(_params)
    , align(_align)
    , data(_exec_data)
    {
        compute_internal();
    }

    //----------------------------------------------------------------------------------//
    // overload how to create the counter with a callback function
    //
    counter(const exec_params& _params, const callback_type& _func, data_ptr_t _exec_data,
            uint64_t _align = 8 * sizeof(_Tp))
    : params(_params)
    , align(_align)
    , configure_callback(_func)
    , data(_exec_data)
    {
        compute_internal();
    }

public:
    //----------------------------------------------------------------------------------//
    ///  allocate a buffer for the ERT calculation
    ///     uses this function if device is CPU or device is GPU and type is not half2
    ///
    template <typename _Up = _Tp, typename Dev = _Device,
              typename std::enable_if<(std::is_same<Dev, device::cpu>::value ||
                                       (std::is_same<Dev, device::gpu>::value &&
                                        !std::is_same<_Up, cuda::fp16_t>::value)),
                                      int>::type = 0>
    _Up* get_buffer()
    {
        // check alignment and
        align = std::max<uint64_t>(align, 8 * sizeof(_Up));
        compute_internal();

        if(settings::debug())
            printf("[%s]> nsize = %llu\n", __FUNCTION__, (ull) nsize);
        _Up* buffer = allocate_aligned<_Up, _Device>(nsize, align);
        if(settings::debug())
            printf("[%s]> buffer = %p\n", __FUNCTION__, buffer);
        device::params<_Device> _params(0, 512, 0, 0);
        device::launch(nsize, _params, initialize_buffer<_Device, _Up, uint64_t>, buffer,
                       _Up(1), nsize);
        return buffer;
    }

    //----------------------------------------------------------------------------------//
    ///  allocate a buffer for the ERT calculation
    ///     uses this function if device is GPU and type is half2
    ///
    template <typename _Up = _Tp, typename Dev = _Device,
              typename std::enable_if<(std::is_same<_Up, cuda::fp16_t>::value &&
                                       std::is_same<Dev, device::gpu>::value),
                                      int>::type = 0>
    _Up* get_buffer()
    {
        // check alignment and
        align = std::max<uint64_t>(align, 8 * sizeof(_Up));
        compute_internal();

        if(settings::debug())
            printf("[%s]> nsize = %llu\n", __FUNCTION__, (ull) nsize);
        _Up* buffer = allocate_aligned<_Up, _Device>(nsize, align);
        if(settings::debug())
            printf("[%s]> buffer = %p\n", __FUNCTION__, buffer);
        device::params<_Device> _params(0, 512, 0, 0);
        device::launch(nsize, _params, initialize_buffer<_Device, _Up, uint32_t>, buffer,
                       _Up{ 1, 1 }, nsize);
        return buffer;
    }

    //----------------------------------------------------------------------------------//
    //  destroy associated buffer
    //
    void destroy_buffer(_Tp* buffer) { free_aligned<_Tp, _Device>(buffer); }

    //----------------------------------------------------------------------------------//
    // execute the callback that may customize the thread before returning the object
    // that provides the measurement
    //
    void configure(uint64_t tid) { configure_callback(tid, *this); }

    //----------------------------------------------------------------------------------//
    // execute the callback that may customize the thread before returning the object
    // that provides the measurement
    //
    counter_type get_counter() const { return counter_type(); }

    //----------------------------------------------------------------------------------//
    // record the data from a thread/process. Extra exec_params (_itrp) should contain
    // the computed grid size for serialization
    //
    inline void record(counter_type& _counter, int n, int t, uint64_t nops,
                       const exec_params& _itrp)
    {
        // std::cout << "Recording " << label << "..." << std::endl;
        uint64_t working_set_size = n * params.nthreads * params.nproc;
        uint64_t total_bytes =
            t * working_set_size * bytes_per_element * memory_accesses_per_element;
        uint64_t total_ops = t * working_set_size * nops;
        auto     seconds   = _counter.get() * counter_units;

        std::stringstream ss;
        ss << label;
        if(label.length() == 0)
        {
            if(nops > 1)
                ss << "vector_op";
            else
                ss << "scalar_op";
        }
        data->operator+=(data_type(ss.str(), working_set_size * bytes_per_element, t,
                                   seconds, total_bytes, total_ops, total_bytes / seconds,
                                   total_ops / seconds, nops, _Device::name(),
                                   tim::demangle(typeid(_Tp).name()), _itrp));
    }

    //----------------------------------------------------------------------------------//
    //
    template <typename _Func>
    void set_callback(_Func&& _f)
    {
        configure_callback = std::forward<_Func>(_f);
    }

    /*
    template <typename _Func>
    void set_callback(_Func&& f)
    {
        configure_callback = std::bind(f, _1, _2);
    }
    */

    //----------------------------------------------------------------------------------//
    //      provide ability to write to JSON/XML
    //
    template <typename Archive>
    void serialize(Archive& ar, const unsigned int)
    {
        if(!data.get())  // for input
            data = data_ptr_t(new _ExecData());
        ar(serializer::make_nvp("params", params), serializer::make_nvp("data", *data));
    }

    //----------------------------------------------------------------------------------//
    //      write to stream
    //
    friend std::ostream& operator<<(std::ostream& os, const counter& obj)
    {
        std::stringstream ss;
        ss << obj.params << ", "
           << "bytes_per_element = " << obj.bytes_per_element << ", "
           << "memory_accesses_per_element = " << obj.memory_accesses_per_element << ", "
           << "alignment = " << obj.align << ", "
           << "nsize = " << obj.nsize << ", "
           << "label = " << obj.label << ", "
           << "data entries = " << ((obj.data.get()) ? obj.data->size() : 0);
        os << ss.str();
        return os;
    }

    //----------------------------------------------------------------------------------//
    //  Get the data pointer
    //
    data_ptr_t&       get_data() { return data; }
    const data_ptr_t& get_data() const { return data; }

public:
    //----------------------------------------------------------------------------------//
    //  public data members, modify as needed
    //
    exec_params params                      = exec_params();
    int         bytes_per_element           = 0;
    int         memory_accesses_per_element = 0;
    uint64_t    align                       = sizeof(_Tp);
    uint64_t    nsize                       = 0;
    units_type  counter_units               = tim::units::sec;
    data_ptr_t  data                        = data_ptr_t(new data_ptr_t);
    std::string label                       = "";

protected:
    callback_type configure_callback = [](uint64_t, this_type&) {};

private:
    //----------------------------------------------------------------------------------//
    //  compute the data size
    //
    void compute_internal()
    {
        if(device::is_cpu<_Device>::value)
            params.nstreams = 1;
        nsize = params.memory_max / params.nproc / params.nthreads;
        nsize = nsize & (~(align - 1));
        nsize = nsize / sizeof(_Tp);
        nsize = std::max<uint64_t>(nsize, 1);
    }
};

//--------------------------------------------------------------------------------------//

inline void
serialize(std::string fname, const exec_data& obj)
{
    bool                  _init   = mpi::is_initialized();
    auto                  _rank   = mpi::rank();
    static constexpr auto spacing = cereal::JSONOutputArchive::Options::IndentChar::space;
    std::stringstream     ss;
    {
        // ensure json write final block during destruction before the file is closed
        //                                  args: precision, spacing, indent size
        cereal::JSONOutputArchive::Options opts(12, spacing, 4);
        cereal::JSONOutputArchive          oa(ss, opts);
        oa.setNextName("rank");
        oa.startNode();
        oa(cereal::make_nvp("rank_id", _rank));
        oa(cereal::make_nvp("data", obj));
        oa.finishNode();
    }
    fname = settings::compose_output_filename(fname, ".json", _init, &_rank);
    std::ofstream ofs(fname.c_str());
    if(ofs)
        ofs << ss.str() << std::endl;
    else
    {
        throw std::runtime_error(std::string("Error opening output file: " + fname));
    }
}

//--------------------------------------------------------------------------------------//

}  // namespace ert
}  // namespace tim
