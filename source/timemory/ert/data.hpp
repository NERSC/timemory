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

#include "timemory/components/timing.hpp"
#include "timemory/macros.hpp"
#include "timemory/mpi.hpp"

#include <cstdint>
#include <string>

#if defined(__INTEL_COMPILER)
#    define ASSUME_ALIGNED_ARRAY(ARRAY, WIDTH) __assume_aligned(ARRAY, WIDTH);
#elif defined(__xlC__)
#    define ASSUME_ALIGNED_ARRAY(ARRAY, WIDTH) __alignx(WIDTH, ARRAY);
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

namespace tim
{
namespace ert
{
using std::size_t;

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
    posix_memalign(&ptr, alignment, size * sizeof(_Tp));
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
    exec_params() {}
    exec_params(uint64_t _work_set, uint64_t _min_try, uint64_t mem_max, int _nthread = 1)
    : working_set_min(_work_set)
    , min_trials(_min_try)
    , memory_max(mem_max)
    , nthreads(_nthread)
    {
    }

    uint64_t  working_set_min = 1;
    uint64_t  min_trials      = 1;
    uint64_t  memory_max      = 64 * 64;
    const int nthreads        = 1;
    const int nrank           = tim::mpi_rank();
    const int nproc           = tim::mpi_size();
};

//--------------------------------------------------------------------------------------//
//  measure CPU floating-point or integer operations
//
namespace cpu
{
template <typename _Tp>
class operation_counter
{
public:
    using string_t = std::string;
    using timer_t  = tim::component::real_clock;
    using result_type =
        std::tuple<uint64_t, uint64_t, float, uint64_t, uint64_t, float, float>;
    using result_array = std::vector<result_type>;
    using labels_type  = std::array<string_t, 7>;

public:
    operation_counter() = default;
    operation_counter(const exec_params& _params, size_t _align = sizeof(_Tp))
    : params(_params)
    , align(_align)
    {
    }
    ~operation_counter()
    {
        delete rc;
        free_aligned(buffer);
    }

public:
    _Tp* initialize()
    {
        nsize  = params.memory_max / params.nproc / params.nthreads;
        nsize  = nsize & (~(align - 1));
        nsize  = nsize / sizeof(_Tp);
        buffer = allocate_aligned<_Tp>(nsize, align);
        for(uint64_t i = 0; i < nsize; ++i)
            buffer[i] = _Tp(1);
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
            t * working_set_size * bytes_per_elem * mem_accesses_per_elem;
        uint64_t total_ops = t * working_set_size * nops;
        auto     seconds   = rc->get() * tim::units::sec;
        data.push_back(result_type(working_set_size * bytes_per_elem, t, seconds,
                                   total_bytes, total_ops, total_ops / seconds,
                                   total_ops / static_cast<float>(total_bytes)));
        delete rc;
        rc = nullptr;
    }

public:
    exec_params  params                = exec_params();
    int          bytes_per_elem        = 0;
    int          mem_accesses_per_elem = 0;
    size_t       align                 = 32;
    uint64_t     nsize                 = 0;
    timer_t*     rc                    = nullptr;
    result_array data;
    labels_type  labels = labels_type({ "working-set", "trials", "seconds", "total-bytes",
                                       "total-ops", "ops-per-sec", "intensity" });
    RESTRICT(_Tp*) buffer = nullptr;

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
            obj.write<6>(os, itr, "\n");
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

}  // namespace ert
}  // namespace tim
