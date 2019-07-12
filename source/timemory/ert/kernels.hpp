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

#include "timemory/apply.hpp"
#include "timemory/backends/mpi.hpp"
#include "timemory/ert/data.hpp"
#include "timemory/macros.hpp"

#include <cstdint>
#include <functional>
#include <future>
#include <iomanip>
#include <sstream>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

#if defined(_OPENMP)
#    include <omp.h>
#    define OMP_BARRIER _Pragma("omp barrier")
#    define OMP_MASTER _Pragma("omp master")
#    define OMP_PARALLEL _Pragma("omp parallel")
#    define OMP_GET_THREAD_NUM() omp_get_thread_num()
#    define OMP_GET_NUM_THREADS() omp_get_num_threads()
#else
#    define OMP_BARRIER
#    define OMP_MASTER
#    define OMP_PARALLEL
#    define OMP_GET_THREAD_NUM() 0
#    define OMP_GET_NUM_THREADS() 1
#endif

namespace tim
{
namespace ert
{
//--------------------------------------------------------------------------------------//

template <size_t _Nrep, typename _Func, typename _Tp, typename _Intp = int32_t,
          tim::enable_if_t<(_Nrep == 1), int> = 0>
void
cpu_ops_kernel(_Intp ntrials, _Func&& func, _Intp nsize, _Tp* A, int& bytes_per_elem,
               int& mem_accesses_per_elem)
{
    bytes_per_elem        = sizeof(_Tp);
    mem_accesses_per_elem = 2;

    ASSUME_ALIGNED_ARRAY(A, ERT_ALIGN);

    _Tp alpha = 0.5;
    for(_Intp j = 0; j < ntrials; ++j)
    {
        for(_Intp i = 0; i < nsize; ++i)
        {
            _Tp beta = 0.8;
            func(beta, A[i], alpha);
            A[i] = beta;
        }
        alpha *= (1.0 - 1.0e-8);
    }
}

//--------------------------------------------------------------------------------------//

template <size_t _Nrep, typename _Func, typename _Tp, typename _Intp = int32_t,
          tim::enable_if_t<(_Nrep > 1), int> = 0>
void
cpu_ops_kernel(_Intp ntrials, _Func&& func, _Intp nsize, _Tp* A, int& bytes_per_elem,
               int& mem_accesses_per_elem)
{
    bytes_per_elem        = sizeof(_Tp);
    mem_accesses_per_elem = 2;

    ASSUME_ALIGNED_ARRAY(A, ERT_ALIGN);
    // divide by two here because macros halve, e.g. ERT_FLOP == 4 means 2 calls
    constexpr size_t NUM_REP = _Nrep / 2;

    _Tp alpha = 0.5;
    for(_Intp j = 0; j < ntrials; ++j)
    {
        for(_Intp i = 0; i < nsize; ++i)
        {
            _Tp beta = 0.8;
            apply<void>::unroll<NUM_REP>(std::forward<_Func>(func), beta, A[i], alpha);
            A[i] = beta;
        }
        alpha *= (1.0 - 1.0e-8);
    }
}

//--------------------------------------------------------------------------------------//

template <size_t _Nops, size_t... _Nextra, typename _Tp, typename _Counter,
          typename _Func, tim::enable_if_t<(sizeof...(_Nextra) == 0), int> = 0>
void
cpu_ops_main(cpu::operation_counter<_Tp, _Counter>& counter, _Func&& func)
{
    using thread_list_t = std::vector<std::thread>;

    auto _cpu_op = [&](int tid, thread_barrier* fbarrier, thread_barrier* lbarrier) {
        auto     buf = counter.get_buffer();
        uint64_t n   = counter.params.working_set_min;
        while(n <= counter.nsize)
        {
            // working set - nsize
            uint64_t ntrials = counter.nsize / n;
            if(ntrials < 1)
                ntrials = 1;

            // wait master thread notifies to proceed
            if(fbarrier)
                fbarrier->wait();

            // get instance of object measuring something during the calculation
            _Counter ct = counter.get_counter();
            // start the timer or anything else being recorded
            ct.start();

            cpu_ops_kernel<_Nops>(ntrials, std::forward<_Func>(func), n, buf,
                                  counter.bytes_per_elem, counter.mem_accesses_per_elem);

            // wait master thread notifies to proceed
            if(lbarrier)
                lbarrier->wait();
            // stop the timer or anything else being recorded
            ct.stop();
            // store the result
            counter.record(ct, n, ntrials, _Nops);

            n = ((1.1 * n) == n) ? (n + 1) : (1.1 * n);
        }
        printf("[%i]> terminating...\n", tid);
        counter.destroy_buffer(buf);
    };

    tim::mpi_barrier();  // i.e. OMP_MASTER

    if(counter.params.nthreads > 1)
    {
        // create synchronization barriers for the threads
        thread_barrier fbarrier(counter.params.nthreads);
        thread_barrier lbarrier(counter.params.nthreads);

        // launch the threads
        thread_list_t threads;
        for(uint64_t i = 0; i < counter.params.nthreads; ++i)
            threads.push_back(std::thread(_cpu_op, i, &fbarrier, &lbarrier));

        // wait for threads to finish
        for(auto& itr : threads)
            itr.join();
    }
    else
    {
        _cpu_op(0, nullptr, nullptr);
    }

    tim::mpi_barrier();  // i.e. OMP_MASTER
    // end the recursive loop
}

//--------------------------------------------------------------------------------------//

template <size_t _Nops, size_t... _Nextra, typename _Tp, typename _Counter,
          typename _Func, tim::enable_if_t<(sizeof...(_Nextra) > 0), int> = 0>
void
cpu_ops_main(cpu::operation_counter<_Tp, _Counter>& counter, _Func&& func)
{
    using thread_list_t = std::vector<std::thread>;

    auto _cpu_op = [&](int tid, thread_barrier* fbarrier, thread_barrier* lbarrier) {
        auto     buf = counter.get_buffer();
        uint64_t n   = counter.params.working_set_min;
        while(n <= counter.nsize)
        {
            // working set - nsize
            uint64_t ntrials = counter.nsize / n;
            if(ntrials < 1)
                ntrials = 1;

            // wait master thread notifies to proceed
            if(fbarrier)
                fbarrier->wait();

            // get instance of object measuring something during the calculation
            _Counter ct = counter.get_counter();
            // start the timer or anything else being recorded
            ct.start();

            cpu_ops_kernel<_Nops>(ntrials, std::forward<_Func>(func), n, buf,
                                  counter.bytes_per_elem, counter.mem_accesses_per_elem);

            // wait master thread notifies to proceed
            if(lbarrier)
                lbarrier->wait();
            // stop the timer or anything else being recorded
            ct.stop();
            // store the result
            counter.record(ct, n, ntrials, _Nops);

            n = ((1.1 * n) == n) ? (n + 1) : (1.1 * n);
        }
        printf("[%i]> terminating...\n", tid);
        counter.destroy_buffer(buf);
    };

    tim::mpi_barrier();  // i.e. OMP_MASTER

    if(counter.params.nthreads > 1)
    {
        // create synchronization barriers for the threads
        thread_barrier fbarrier(counter.params.nthreads);
        thread_barrier lbarrier(counter.params.nthreads);

        // launch the threads
        thread_list_t threads;
        for(uint64_t i = 0; i < counter.params.nthreads; ++i)
            threads.push_back(std::thread(_cpu_op, i, &fbarrier, &lbarrier));

        // wait for threads to finish
        for(auto& itr : threads)
            itr.join();
    }
    else
    {
        _cpu_op(0, nullptr, nullptr);
    }

    tim::mpi_barrier();  // i.e. OMP_MASTER

    // continue the recursive loop
    cpu_ops_main<_Nextra...>(counter, func);
}

//--------------------------------------------------------------------------------------//
#if defined(__NVCC__)

template <size_t _Nrep, typename _Func, typename _Tp, typename _Intp = int32_t,
          tim::enable_if_t<(_Nrep == 1), int> = 0>
__global__ void
gpu_ops_kernel(_Intp ntrials, _Func&& func, _Intp nsize, _Tp* A, int* bytes_per_elem,
               int* mem_accesses_per_elem)
{
    _Intp i0      = blockIdx.x * blockDim.x + threadIdx.x;
    _Intp istride = blockDim.x * gridDim.x;

    if(i0 == 0)
    {
        *bytes_per_elem        = sizeof(_Tp);
        *mem_accesses_per_elem = 2;
    }

    for(_Intp j = 0; j < ntrials; ++j)
    {
        _Tp alpha = 0.5;
        for(_Intp i = i0; i < nsize; i += istride)
        {
            _Tp beta = 0.8;
            func(beta, A[i], alpha);
            A[i] = beta;
        }
        alpha *= (1.0 - 1.0e-8);
    }
}

//--------------------------------------------------------------------------------------//

template <size_t _Nrep, typename _Func, typename _Tp, typename _Intp = int32_t,
          tim::enable_if_t<(_Nrep > 1), int> = 0>
__global__ void
gpu_ops_kernel(_Intp ntrials, _Func&& func, _Intp nsize, _Tp* A, int* bytes_per_elem,
               int* mem_accesses_per_elem)
{
    // divide by two here because macros halve, e.g. ERT_FLOP == 4 means 2 calls
    constexpr size_t NUM_REP = _Nrep / 2;

    _Intp i0      = blockIdx.x * blockDim.x + threadIdx.x;
    _Intp istride = blockDim.x * gridDim.x;

    if(i0 == 0)
    {
        *bytes_per_elem        = sizeof(_Tp);
        *mem_accesses_per_elem = 2;
    }

    for(_Intp j = 0; j < ntrials; ++j)
    {
        _Tp alpha = 0.5;
        for(_Intp i = i0; i < nsize; i += istride)
        {
            _Tp beta = 0.8;
            apply<void>::unroll<NUM_REP>(std::forward<_Func>(func), beta, A[i], alpha);
            A[i] = beta;
        }
        alpha *= (1.0 - 1.0e-8);
    }
}

//--------------------------------------------------------------------------------------//

template <size_t _Nops, size_t... _Nextra, typename _Tp, typename _Func,
          tim::enable_if_t<(sizeof...(_Nextra) == 0), int> = 0>
void
gpu_ops_main(gpu::operation_counter<_Tp>& counter, _Func&& func)
{
    OMP_PARALLEL
    {
        auto     buf          = counter.initialize();
        uint64_t n            = counter.params.working_set_min;
        int*     counter_data = cuda::malloc<int>(2);
        while(n <= counter.nsize)
        {
            // working set - nsize
            uint64_t ntrials = counter.nsize / n;
            if(ntrials < 1)
                ntrials = 1;

            OMP_MASTER { tim::mpi_barrier(); }
            OMP_BARRIER
            counter.start();
            gpu_ops_kernel<_Nops><<<counter.grid_size, counter.block_size, counter.shmem,
                                    counter.stream>>>(ntrials, std::forward<_Func>(func),
                                                      n, buf, &counter_data[0],
                                                      &counter_data[1]);
            cuda::stream_sync(counter.stream);
            OMP_BARRIER
            OMP_MASTER { tim::mpi_barrier(); }
            counter.stop(n, ntrials, _Nops);

            n = ((1.1 * n) == n) ? (n + 1) : (1.1 * n);
        }
        cuda::memcpy<int>(&counter.bytes_per_elem, counter_data + 0, 1,
                          cuda::device_to_host_v, counter.stream);
        cuda::memcpy<int>(&counter.mem_accesses_per_elem, counter_data + 1, 1,
                          cuda::device_to_host_v, counter.stream);
        cuda::stream_sync(counter.stream);
        cuda::free(counter_data);
    }
    tim::mpi_barrier();
}

//--------------------------------------------------------------------------------------//

template <size_t _Nops, size_t... _Nextra, typename _Tp, typename _Func,
          tim::enable_if_t<(sizeof...(_Nextra) > 0), int> = 0>
void
gpu_ops_main(gpu::operation_counter<_Tp>& counter, _Func&& func)
{
    OMP_PARALLEL
    {
        auto     buf          = counter.initialize();
        uint64_t n            = counter.params.working_set_min;
        int*     counter_data = cuda::malloc<int>(2);
        while(n <= counter.nsize)
        {
            // working set - nsize
            uint64_t ntrials = counter.nsize / n;
            if(ntrials < 1)
                ntrials = 1;

            OMP_MASTER { tim::mpi_barrier(); }
            OMP_BARRIER
            counter.start();
            gpu_ops_kernel<_Nops><<<counter.grid_size, counter.block_size, counter.shmem,
                                    counter.stream>>>(ntrials, std::forward<_Func>(func),
                                                      n, buf, &counter_data[0],
                                                      &counter_data[1]);
            cuda::stream_sync(counter.stream);
            OMP_BARRIER
            OMP_MASTER { tim::mpi_barrier(); }
            counter.stop(n, ntrials, _Nops);

            n = ((1.1 * n) == n) ? (n + 1) : (1.1 * n);
        }
        cuda::memcpy<int>(&counter.bytes_per_elem, counter_data + 0, 1,
                          cuda::device_to_host_v, counter.stream);
        cuda::memcpy<int>(&counter.mem_accesses_per_elem, counter_data + 1, 1,
                          cuda::device_to_host_v, counter.stream);
        cuda::stream_sync(counter.stream);
        cuda::free(counter_data);
    }
    tim::mpi_barrier();
    gpu_ops_main<_Nextra...>(counter, std::forward<_Func>(func));
}

#endif

}  // namespace ert
}  // namespace tim
