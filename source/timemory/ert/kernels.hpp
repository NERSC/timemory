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
#include "timemory/ert/data.hpp"
#include "timemory/mpl/apply.hpp"
#include "timemory/utility/macros.hpp"
#include "timemory/utility/utility.hpp"

#include <cstdint>
#include <functional>
#include <future>
#include <iomanip>
#include <sstream>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

namespace tim
{
namespace ert
{
//--------------------------------------------------------------------------------------//
//
//      CPU -- multiple trial
//
//--------------------------------------------------------------------------------------//

template <size_t _Nrep, typename _Device, typename _Intp, typename _Tp, typename _FuncOps,
          typename _FuncStore, device::enable_if_cpu_t<_Device> = 0>
void
ops_kernel(const _Intp& ntrials, const _Intp& nsize, _Tp* A, _FuncOps&& ops_func,
           _FuncStore&& store_func)
{
    // divide by two here because macros halve, e.g. ERT_FLOP == 4 means 2 calls
    constexpr size_t NUM_REP = _Nrep / 2;
    constexpr size_t MOD_REP = _Nrep % 2;
    auto             range   = device::grid_strided_range<_Device, 0, _Intp>(nsize);

    _Tp alpha = static_cast<_Tp>(0.5);
    for(_Intp j = 0; j < ntrials; ++j)
    {
        for(auto i = range.begin(); i < range.end(); i += range.stride())
        {
            _Tp beta = static_cast<_Tp>(0.8);
            apply<void>::unroll<NUM_REP + MOD_REP, _Device>(ops_func, beta, A[i], alpha);
            store_func(A[i], beta);
        }
        alpha *= static_cast<_Tp>(1.0 - 1.0e-8);
    }
}

//--------------------------------------------------------------------------------------//
//
//      GPU -- multiple trial
//
//--------------------------------------------------------------------------------------//

template <size_t _Nrep, typename _Device, typename _Intp, typename _Tp, typename _FuncOps,
          typename _FuncStore, device::enable_if_gpu_t<_Device> = 0,
          enable_if_t<!(std::is_same<_Tp, cuda::fp16_t>::value)> = 0>
GLOBAL_CALLABLE void
ops_kernel(_Intp ntrials, _Intp nsize, _Tp* A, _FuncOps&& ops_func,
           _FuncStore&& store_func)
{
    // divide by two here because macros halve, e.g. ERT_FLOP == 4 means 2 calls
    constexpr size_t NUM_REP = _Nrep / 2;
    constexpr size_t MOD_REP = _Nrep % 2;
    auto             range   = device::grid_strided_range<_Device, 0, _Intp>(nsize);

    _Tp alpha = static_cast<_Tp>(0.5);
    for(_Intp j = 0; j < ntrials; ++j)
    {
        for(auto i = range.begin(); i < range.end(); i += range.stride())
        {
            _Tp beta = static_cast<_Tp>(0.8);
            apply<void>::unroll<NUM_REP + MOD_REP, _Device>(ops_func, beta, A[i], alpha);
            store_func(A[i], beta);
        }
        alpha *= static_cast<_Tp>(1.0 - 1.0e-8);
    }
}

//--------------------------------------------------------------------------------------//
//
//      GPU -- multiple trial -- packed (2) half-precision
//
//--------------------------------------------------------------------------------------//

template <size_t _Nrep, typename _Device, typename _Intp, typename _Tp, typename _FuncOps,
          typename _FuncStore, device::enable_if_gpu_t<_Device> = 0,
          enable_if_t<(std::is_same<_Tp, cuda::fp16_t>::value)> = 0>
GLOBAL_CALLABLE void
ops_kernel(_Intp ntrials, _Intp nsize, _Tp* A, _FuncOps&& ops_func,
           _FuncStore&& store_func)
{
    // divide by four instead of two here because fp16_t is a packed operation
    constexpr size_t NUM_REP = _Nrep / 4;
    constexpr size_t MOD_REP = _Nrep % 4;
    auto             range   = device::grid_strided_range<_Device, 0, int32_t>(nsize);

    _Tp alpha = { 0.5, 0.5 };
    for(int32_t j = 0; j < ntrials; ++j)
    {
        for(auto i = range.begin(); i < range.end(); i += range.stride())
        {
            _Tp beta = { 0.8, 0.8 };
            apply<void>::unroll<NUM_REP + MOD_REP, _Device>(ops_func, beta, A[i], alpha);
            store_func(A[i], beta);
        }
        alpha *= { 1.0 - 1.0e-8, 1.0 - 1.0e-8 };
    }
}

//--------------------------------------------------------------------------------------//
///
///     This is the "main" function for ERT
///
template <size_t _Nops, size_t... _Nextra, typename _Device, typename _Tp,
          typename _ExecData, typename _Counter, typename _FuncOps, typename _FuncStore,
          enable_if_t<(sizeof...(_Nextra) == 0), int> = 0>
void
ops_main(counter<_Device, _Tp, _ExecData, _Counter>& _counter, _FuncOps&& ops_func,
         _FuncStore&& store_func)
{
    using stream_list_t   = std::vector<cuda::stream_t>;
    using thread_list_t   = std::vector<std::thread>;
    using device_params_t = device::params<_Device>;
    using _Intp           = int32_t;
    using ull             = long long unsigned;

    constexpr bool is_gpu = std::is_same<_Device, device::gpu>::value;

    if(settings::verbose() > 0 || settings::debug())
        printf("[%s] Executing %li ops...\n", __FUNCTION__, (long int) _Nops);

    if(_counter.bytes_per_element == 0)
        fprintf(stderr, "[%s:%i]> bytes-per-element is not set!\n", __FUNCTION__,
                __LINE__);

    if(_counter.memory_accesses_per_element == 0)
        fprintf(stderr, "[%s:%i]> memory-accesses-per-element is not set!\n",
                __FUNCTION__, __LINE__);

    // list of streams
    stream_list_t streams;
    // generate async streams if multiple streams were requested
    if(_counter.params.nstreams > 1)
    {
        // fill with implicit stream
        streams.resize(_counter.params.nstreams, 0);
        for(auto& itr : streams)
            cuda::stream_create(itr);
    }

    auto _opfunc = [&](uint64_t tid, thread_barrier* fbarrier, thread_barrier* lbarrier) {
        // execute the callback
        _counter.configure(tid);
        // allocate buffer
        auto     buf = _counter.get_buffer();
        uint64_t n   = _counter.params.working_set_min;
        // cache this
        const uint64_t nstreams = std::max<uint64_t>(_counter.params.nstreams, 1);
        // create the launch parameters (ignored on CPU)
        //
        // if grid_size is zero (default), the launch command will calculate a grid-size
        // as follows:
        //
        //      grid_size = ((data_size + block_size - 1) / block_size)
        //
        device_params_t dev_params(_counter.params.grid_size, _counter.params.block_size,
                                   _counter.params.shmem_size, 0);
        //
        if(n > _counter.nsize)
        {
            fprintf(stderr,
                    "[%s@'%s':%i]> Warning! ERT not running any trials because working "
                    "set min > nsize: %llu > %llu\n",
                    TIMEMORY_ERROR_FUNCTION_MACRO, __FILE__, __LINE__, (ull) n,
                    (ull) _counter.nsize);
        }

        while(n <= _counter.nsize)
        {
            // working set - nsize
            uint64_t ntrials = _counter.nsize / n;
            if(ntrials < 1)
                ntrials = 1;

            if(settings::debug() && tid == 0)
            {
                printf(
                    "[tim::ert::ops_main<%llu>]> number of trials: %llu, n = %llu, nsize "
                    "= %llu\n",
                    (ull) _Nops, (ull) ntrials, (ull) n, (ull) _counter.nsize);
            }

            auto _itr_params = _counter.params;

            if(is_gpu)
            {
                // make sure all streams are synced
                for(auto& itr : streams)
                    cuda::stream_sync(itr);

                // sync the streams
                if(nstreams < 2)
                    cuda::device_sync();
            }

            // wait master thread notifies to proceed
            if(fbarrier)
                fbarrier->spin_wait();

            // get instance of object measuring something during the calculation
            _Counter ct = _counter.get_counter();
            // start the timer or anything else being recorded
            ct.start();

            // only do this more complicated mess if we need to
            if(nstreams > 1)
            {
                auto nchunk  = n / nstreams;
                auto nmodulo = n % nstreams;
                for(uint64_t i = 0; i < nstreams; ++i)
                {
                    // calculate the buffer offset
                    auto offset = i * nchunk;
                    // calculate the size of the subchunk
                    int32_t _n      = nchunk + ((i + 1 == nstreams) ? nmodulo : 0);
                    auto    _buf    = buf + offset;
                    auto    _stream = streams.at(i % streams.size());
                    auto    _params = dev_params;  // copy of the parameters
                    device::launch(
                        _n, _stream, _params,
                        ops_kernel<_Nops, _Device, _Intp, _Tp, _FuncOps, _FuncStore>,
                        ntrials, _n, _buf, std::forward<_FuncOps>(ops_func),
                        std::forward<_FuncStore>(store_func));
                    if(i == 0)
                        _itr_params.grid_size = _params.grid;
                    else
                        _itr_params.grid_size =
                            std::max<int64_t>(_itr_params.grid_size, _params.grid);
                }
            }
            else
            {
                device::launch(
                    n, dev_params,
                    ops_kernel<_Nops, _Device, _Intp, _Tp, _FuncOps, _FuncStore>, ntrials,
                    n, buf, std::forward<_FuncOps>(ops_func),
                    std::forward<_FuncStore>(store_func));

                _itr_params.grid_size = dev_params.grid;
            }

            if(is_gpu)
            {
                for(auto& itr : streams)
                    cuda::stream_sync(itr);

                // sync the streams
                if(nstreams < 2)
                    cuda::device_sync();
            }

            // wait master thread notifies to proceed
            if(lbarrier)
                lbarrier->spin_wait();

            // stop the timer or anything else being recorded
            ct.stop();

            // store the result
            if(tid == 0)
                _counter.record(ct, n, ntrials, _Nops, _itr_params);

            n = ((1.1 * n) == n) ? (n + 1) : (1.1 * n);
        }

        if(is_gpu)
            cuda::device_sync();
        _counter.destroy_buffer(buf);
    };

    mpi::barrier();  // synchronize MPI processes

    if(is_gpu)
        cuda::device_sync();

    if(_counter.params.nthreads > 1)
    {
        // create synchronization barriers for the threads
        thread_barrier fbarrier(_counter.params.nthreads);
        thread_barrier lbarrier(_counter.params.nthreads);

        // list of threads
        thread_list_t threads;
        // create the threads
        for(uint64_t i = 0; i < _counter.params.nthreads; ++i)
            threads.push_back(std::thread(_opfunc, i, &fbarrier, &lbarrier));

        // wait for threads to finish
        for(auto& itr : threads)
            itr.join();
    }
    else
    {
        _opfunc(0, nullptr, nullptr);
    }

    if(is_gpu)
        cuda::device_sync();

    mpi::barrier();  // synchronize MPI processes
}

//--------------------------------------------------------------------------------------//
///
///     This is invokes the "main" function for ERT for all the desired "FLOPs" that
///     are unrolled in the kernel
///
template <size_t _Nops, size_t... _Nextra, typename _Device, typename _Tp,
          typename _ExecData, typename _Counter, typename _FuncOps, typename _FuncStore,
          enable_if_t<(sizeof...(_Nextra) > 0), int> = 0>
void
ops_main(counter<_Device, _Tp, _ExecData, _Counter>& _counter, _FuncOps&& ops_func,
         _FuncStore&& store_func)
{
    // execute a single parameter
    ops_main<_Nops>(_counter, ops_func, store_func);
    // continue the recursive loop
    ops_main<_Nextra...>(_counter, ops_func, store_func);
}

//--------------------------------------------------------------------------------------//

}  // namespace ert
}  // namespace tim
