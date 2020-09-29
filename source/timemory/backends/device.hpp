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
//

/** \file backends/device.hpp
 * \headerfile backends/device.hpp "timemory/backends/device.hpp"
 * Defines device-generic structures and routines
 *
 */

#pragma once

#if defined(__NVCC__)
#    include <cuda.h>
#    include <cuda_runtime_api.h>

#    define HOST_DEVICE_CALLABLE __host__ __device__
#    define DEVICE_CALLABLE __device__
#    define GLOBAL_CALLABLE __global__
#    define HOST_DEVICE_CALLABLE_INLINE __host__ __device__ __inline__
#    define DEVICE_CALLABLE_INLINE __device__ __inline__
#    define GLOBAL_CALLABLE_INLINE __global__ __inline__
#else
#    define HOST_DEVICE_CALLABLE
#    define DEVICE_CALLABLE
#    define GLOBAL_CALLABLE
#    define HOST_DEVICE_CALLABLE_INLINE inline
#    define DEVICE_CALLABLE_INLINE inline
#    define GLOBAL_CALLABLE_INLINE inline
#endif

#include <cstdint>
#include <type_traits>

#include "timemory/components/cuda/backends.hpp"

namespace tim
{
namespace device
{
//--------------------------------------------------------------------------------------//

template <typename... ArgsT>
inline void
consume_parameters(ArgsT&&...)
{}

//--------------------------------------------------------------------------------------//
//
template <bool B, class T = int>
using enable_if_t = typename std::enable_if<B, T>::type;

//--------------------------------------------------------------------------------------//

struct cpu
{
    using stream_t = cuda::stream_t;
    using fp16_t   = float;

    template <typename Tp>
    static Tp* alloc(std::size_t nsize)
    {
        return new Tp[nsize];
    }

    template <typename Tp>
    static void free(Tp* ptr)
    {
        delete[] ptr;
    }

    static std::string name() { return "cpu"; }
};

//--------------------------------------------------------------------------------------//

struct gpu
{
#if defined(TIMEMORY_USE_CUDA)
    using stream_t = cuda::stream_t;
    using fp16_t   = cuda::fp16_t;

    template <typename Tp>
    static Tp* alloc(std::size_t nsize)
    {
        return cuda::malloc<Tp>(nsize);
    }

    template <typename Tp>
    static void free(Tp* ptr)
    {
        cuda::free(ptr);
    }

    static std::string name() { return "gpu"; }
#else
    using stream_t = int;
    using fp16_t   = float;

    template <typename Tp>
    static Tp* alloc(std::size_t nsize)
    {
        return new Tp[nsize];
    }

    template <typename Tp>
    static void free(Tp* ptr)
    {
        delete[] ptr;
    }

    static std::string name() { return "gpu_on_cpu"; }
#endif
};

//--------------------------------------------------------------------------------------//

#if defined(__NVCC__)
using default_device = gpu;
#else
using default_device = cpu;
#endif

//--------------------------------------------------------------------------------------//

template <typename Up>
struct is_gpu
{
    static constexpr bool value = std::is_same<Up, device::gpu>::value;
};

//--------------------------------------------------------------------------------------//

template <typename Up>
struct is_cpu
{
    static constexpr bool value = std::is_same<Up, device::cpu>::value;
};

//--------------------------------------------------------------------------------------//

template <typename T, typename U = int, bool B = true>
using enable_if_gpu_t = enable_if_t<(is_gpu<T>::value == B), U>;

//--------------------------------------------------------------------------------------//

template <typename T, typename U = int, bool B = true>
using enable_if_cpu_t = enable_if_t<(is_cpu<T>::value == B), U>;

//--------------------------------------------------------------------------------------//

namespace impl
{
template <typename Tp, typename Intp, bool Valv = true>
using enable_if_gpu_int_t =
    enable_if_t<(is_gpu<Tp>::value == Valv && std::is_integral<Intp>::value)>;

template <typename Tp, typename Intp, bool Valv = true>
using enable_if_cpu_int_t =
    enable_if_t<(is_cpu<Tp>::value == Valv && std::is_integral<Intp>::value)>;

//--------------------------------------------------------------------------------------//
//
template <typename Intp>
struct range
{
    using this_type = range<Intp>;

    HOST_DEVICE_CALLABLE range(Intp _begin, Intp _end, Intp _stride)
    : m_begin(_begin)
    , m_end(_end)
    , m_stride(_stride)
    {}

    HOST_DEVICE_CALLABLE Intp& begin() { return m_begin; }
    HOST_DEVICE_CALLABLE Intp begin() const { return m_begin; }
    HOST_DEVICE_CALLABLE Intp& end() { return m_end; }
    HOST_DEVICE_CALLABLE Intp end() const { return m_end; }
    HOST_DEVICE_CALLABLE Intp& stride() { return m_stride; }
    HOST_DEVICE_CALLABLE Intp stride() const { return m_stride; }

    HOST_DEVICE_CALLABLE const char* c_str() const
    {
        using llu = long long unsigned;
        char desc[512];
        sprintf(desc, "(%llu,%llu,%llu)", (llu) m_begin, (llu) m_end, (llu) m_stride);
        return std::move(desc);  // NOLINT
    }

protected:
    Intp m_begin;
    Intp m_end;
    Intp m_stride;
};
}  // namespace impl

//======================================================================================//
//
//  These provide the kernel launch parameters for the GPU
//
//======================================================================================//

template <typename DeviceT>
struct params
{
    using stream_t = typename DeviceT::stream_t;
    params()       = default;

    explicit params(uint32_t _grid, uint32_t _block, uint32_t _shmem = 0,
                    stream_t _stream = 0)
    : block(_block)
    , grid(_grid)
    , shmem(_shmem)
    , stream(_stream)
    , dynamic((_grid > 0) ? false : true)
    {}

    int compute(const int& size)
    {
        if(grid == 0 || dynamic)
            grid = ((size + block - 1) / block);
        return grid;
    }

    static int compute(const int& size, const int& block_size)
    {
        return ((size + block_size - 1) / block_size);
    }

    uint32_t block   = 32;
    uint32_t grid    = 0;  // 0 == compute
    uint32_t shmem   = 0;
    stream_t stream  = 0;
    bool     dynamic = true;  // allow the grid size to be dynamically computed

    friend std::ostream& operator<<(std::ostream& os, const params& obj)
    {
        std::stringstream ss;
        ss << "<<<" << obj.grid << ", " << obj.block << ", " << obj.shmem << ", "
           << ((obj.stream == 0) ? "default_stream" : "stream") << ">>>";
        os << ss.str();
        return os;
    }
};

//======================================================================================//
//
//
//              CPU LAUNCH
//
//
//======================================================================================//

// execute a function with provided device parameters
//
template <typename DeviceT, typename FuncT, typename... ArgsT,
          enable_if_cpu_t<DeviceT> = 0>
void
launch(params<DeviceT>&, FuncT&& _func, ArgsT&&... _args)
{
    std::forward<FuncT>(_func)(std::forward<ArgsT>(_args)...);
}

//--------------------------------------------------------------------------------------//
// overload that passes size
//
template <typename Intp, typename DeviceT, typename FuncT, typename... ArgsT,
          impl::enable_if_cpu_int_t<DeviceT, Intp> = 0>
void
launch(const Intp&, params<DeviceT>&, FuncT&& _func, ArgsT&&... _args)
{
    std::forward<FuncT>(_func)(std::forward<ArgsT>(_args)...);
}

//--------------------------------------------------------------------------------------//
// overload that passes size
//
template <typename Intp, typename DeviceT, typename FuncT, typename... ArgsT,
          typename StreamT                         = typename DeviceT::stream_t,
          impl::enable_if_cpu_int_t<DeviceT, Intp> = 0>
void
launch(const Intp&, StreamT, params<DeviceT>&, FuncT&& _func, ArgsT&&... _args)
{
    std::forward<FuncT>(_func)(std::forward<ArgsT>(_args)...);
}

//======================================================================================//
//
//
//              GPU LAUNCH
//
//
//======================================================================================//

// execute a function with provided device parameters
//
template <typename DeviceT, typename FuncT, typename... ArgsT,
          enable_if_gpu_t<DeviceT> = 0>
void
launch(params<DeviceT>& _p, FuncT&& _func, ArgsT&&... _args)
{
#if defined(__NVCC__)
    if(_p.grid == 0)
        _p.grid = 1;
    std::forward<FuncT>(_func)<<<_p.grid, _p.block, _p.shmem, _p.stream>>>(
        std::forward<ArgsT>(_args)...);
    TIMEMORY_CUDA_RUNTIME_CHECK_ERROR(tim::cuda::get_last_error());
#else
    // static_assert(false, "Checking");
    consume_parameters(_p, _func, _args...);
    throw std::runtime_error(
        "Launch specified on GPU but not compiled with GPU support!");
#endif
}

//--------------------------------------------------------------------------------------//
// overload that passes size
//
template <typename Intp, typename DeviceT, typename FuncT, typename... ArgsT,
          impl::enable_if_gpu_int_t<DeviceT, Intp> = 0>
void
launch(const Intp& _nsize, params<DeviceT>& _p, FuncT&& _func, ArgsT&&... _args)
{
#if defined(__NVCC__)
    if(_p.grid == 0 && _nsize > 0)
        _p.grid = _p.compute(_nsize);
    else if(_p.grid == 0)
        _p.grid = 1;
    std::forward<FuncT>(_func)<<<_p.grid, _p.block, _p.shmem, _p.stream>>>(
        std::forward<ArgsT>(_args)...);
    TIMEMORY_CUDA_RUNTIME_CHECK_ERROR(tim::cuda::get_last_error());
#else
    // static_assert(false, "Checking");
    consume_parameters(_p, _func, _args..., _nsize);
    throw std::runtime_error(
        "Launch specified on GPU but not compiled with GPU support!");
#endif
}

//--------------------------------------------------------------------------------------//
// overload that passes size and stream
//
template <typename Intp, typename DeviceT, typename FuncT, typename... ArgsT,
          typename StreamT                               = typename DeviceT::stream_t,
          impl::enable_if_gpu_int_t<DeviceT, Intp, true> = 0>
void
launch(const Intp& _nsize, StreamT _stream, params<DeviceT>& _p, FuncT&& _func,
       ArgsT&&... _args)
{
#if defined(__NVCC__)
    if(_p.grid == 0 && _nsize > 0)
        _p.grid = _p.compute(_nsize);
    else if(_p.grid == 0)
        _p.grid = 1;
    std::forward<FuncT>(_func)<<<_p.grid, _p.block, _p.shmem, _stream>>>(
        std::forward<ArgsT>(_args)...);
    TIMEMORY_CUDA_RUNTIME_CHECK_ERROR(tim::cuda::get_last_error());
#else
    // static_assert(false, "Checking");
    consume_parameters(_p, _func, _args..., _nsize, _stream);
    throw std::runtime_error(
        "Launch specified on GPU but not compiled with GPU support!");
#endif
}

//======================================================================================//
//
//  These provide loop parameters for grid-strided loops on GPU or traditional loops on
//  CPU
//
//======================================================================================//

template <typename DeviceT, size_t DIM = 0, typename Intp = int32_t>
struct grid_strided_range : impl::range<Intp>
{
    using base_type = impl::range<Intp>;
    static_assert(DIM < 0 || DIM > 2,
                  "Error DIM parameter must be 0 (x), 1 (y), or 2 (z)");
};

//--------------------------------------------------------------------------------------//
// overload for 0/x
//
template <typename DeviceT, typename Intp>
struct grid_strided_range<DeviceT, 0, Intp> : impl::range<Intp>
{
    using base_type = impl::range<Intp>;

#if defined(TIMEMORY_USE_CUDA) && defined(__NVCC__)
    template <typename DevT                                       = DeviceT,
              enable_if_t<std::is_same<DevT, device::gpu>::value> = 0>
    DEVICE_CALLABLE explicit grid_strided_range(Intp max_iter)
    : base_type(blockIdx.x * blockDim.x + threadIdx.x, max_iter, blockDim.x * gridDim.x)
    {}

    template <typename DevT                                       = DeviceT,
              enable_if_t<std::is_same<DevT, device::cpu>::value> = 0>
#endif
    explicit grid_strided_range(Intp max_iter)
    : base_type(0, max_iter, 1)
    {}

    using base_type::begin;
    using base_type::end;
    using base_type::stride;
};

//--------------------------------------------------------------------------------------//
// overload for 1/y
//
template <typename DeviceT, typename Intp>
struct grid_strided_range<DeviceT, 1, Intp> : impl::range<Intp>
{
    using base_type = impl::range<Intp>;

#if defined(TIMEMORY_USE_CUDA) && defined(__NVCC__)
    template <typename DevT                                       = DeviceT,
              enable_if_t<std::is_same<DevT, device::gpu>::value> = 0>
    DEVICE_CALLABLE explicit grid_strided_range(Intp max_iter)
    : base_type(blockIdx.y * blockDim.y + threadIdx.y, max_iter, blockDim.y * gridDim.y)
    {}

    template <typename DevT                                       = DeviceT,
              enable_if_t<std::is_same<DevT, device::cpu>::value> = 0>
#endif
    explicit grid_strided_range(Intp max_iter)
    : base_type(0, max_iter, 1)
    {}

    using base_type::begin;
    using base_type::end;
    using base_type::stride;
};

//--------------------------------------------------------------------------------------//
// overload for 2/z
//
template <typename DeviceT, typename Intp>
struct grid_strided_range<DeviceT, 2, Intp> : impl::range<Intp>
{
    using base_type = impl::range<Intp>;

#if defined(TIMEMORY_USE_CUDA) && defined(__NVCC__)
    template <typename DevT                                       = DeviceT,
              enable_if_t<std::is_same<DevT, device::gpu>::value> = 0>
    DEVICE_CALLABLE explicit grid_strided_range(Intp max_iter)
    : base_type(blockIdx.z * blockDim.z + threadIdx.z, max_iter, blockDim.z * gridDim.z)
    {}

    template <typename DevT                                       = DeviceT,
              enable_if_t<std::is_same<DevT, device::cpu>::value> = 0>
#endif
    explicit grid_strided_range(Intp max_iter)
    : base_type(0, max_iter, 1)
    {}

    using base_type::begin;
    using base_type::end;
    using base_type::stride;
};

}  // namespace device
}  // namespace tim
