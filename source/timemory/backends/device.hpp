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

#pragma once

#if defined(__NVCC__)
#    include <cuda.h>
#    include <cuda_runtime_api.h>

#    define HOST_DEVICE_CALLABLE __host__ __device__
#    define DEVICE_CALLABLE __device__
#    define GLOBAL_CALLABLE __global__
#else
#    define HOST_DEVICE_CALLABLE
#    define DEVICE_CALLABLE
#    define GLOBAL_CALLABLE
#endif

#include <cstdint>
#include <timemory/backends/cuda.hpp>
#include <type_traits>

namespace tim
{
namespace device
{
struct cpu
{
    using stream_t = int;

    template <typename _Tp>
    static _Tp* alloc(std::size_t nsize)
    {
        return new _Tp[nsize];
    }

    template <typename _Tp>
    static void free(_Tp* ptr)
    {
        delete[] ptr;
    }

    static std::string name() { return "cpu"; }
};

struct gpu
{
#if defined(TIMEMORY_USE_CUDA)
    using stream_t = cudaStream_t;

    template <typename _Tp>
    static _Tp* alloc(std::size_t nsize)
    {
        return cuda::malloc<_Tp>(nsize);
    }

    template <typename _Tp>
    static void free(_Tp* ptr)
    {
        cuda::free(ptr);
    }

    static std::string name() { return "gpu"; }
#else
    using stream_t = int;
    template <typename _Tp>
    static _Tp* alloc(std::size_t nsize)
    {
        return new _Tp[nsize];
    }

    template <typename _Tp>
    static void free(_Tp* ptr)
    {
        delete[] ptr;
    }

    static std::string name() { return "cpu"; }
#endif
};

#if defined(__NVCC__)
using default_device = gpu;
#else
using default_device = cpu;
#endif

namespace impl
{
//--------------------------------------------------------------------------------------//
//
template <bool B, class T = char>
using enable_if_t = typename std::enable_if<B, T>::type;

template <typename _Tp, bool _Val = true>
using is_gpu_device = enable_if_t<(std::is_same<gpu, _Tp>::value == _Val)>;

//--------------------------------------------------------------------------------------//
//
template <typename _Intp>
struct range
{
    using this_type = range<_Intp>;

    HOST_DEVICE_CALLABLE range(_Intp _begin, _Intp _end, _Intp _stride)
    : m_begin(_begin)
    , m_end(_end)
    , m_stride(_stride)
    {
    }

    HOST_DEVICE_CALLABLE _Intp& begin() { return m_begin; }
    HOST_DEVICE_CALLABLE _Intp begin() const { return m_begin; }
    HOST_DEVICE_CALLABLE _Intp& end() { return m_end; }
    HOST_DEVICE_CALLABLE _Intp end() const { return m_end; }
    HOST_DEVICE_CALLABLE _Intp& stride() { return m_stride; }
    HOST_DEVICE_CALLABLE _Intp stride() const { return m_stride; }

protected:
    _Intp m_begin;
    _Intp m_end;
    _Intp m_stride;
};
}  // namespace impl

//======================================================================================//
//
//  These provide the kernel launch parameters for the GPU
//
//======================================================================================//

template <typename _Device>
struct params
{
    using stream_t = typename _Device::stream_t;
    params()       = default;

    explicit params(uint32_t _grid, uint32_t _block, uint32_t _shmem = 0,
                    stream_t _stream = 0)
    : block(_block)
    , grid(_grid)
    , shmem(_shmem)
    , stream(_stream)
    {
    }

    int compute(const int& size)
    {
        return (grid == 0) ? ((size + block - 1) / block) : grid;
    }

    static int compute(const int& size, const int& block_size)
    {
        return ((size + block_size - 1) / block_size);
    }

    uint32_t block  = 32;
    uint32_t grid   = 0;  // 0 == compute
    uint32_t shmem  = 0;
    stream_t stream = 0;

    friend std::ostream& operator<<(std::ostream& os, const params& obj)
    {
        std::stringstream ss;
        ss << "<<<" << obj.grid << ", " << obj.block << ", " << obj.shmem << ", "
           << ((obj.stream == 0) ? "default_stream" : "stream") << ">>>";
        os << ss.str();
        return os;
    }
};

#if defined(__NVCC__)
template <typename _Device, typename _Func, typename... _Args,
          impl::is_gpu_device<_Device, true> = 0>
void
launch(const params<_Device>& p, _Func&& _func, _Args&&... _args)
{
    std::forward<_Func>(_func)<<<p.grid, p.block, p.shmem, p.stream>>>(
        std::forward<_Args>(_args)...);
    CUDA_RUNTIME_CHECK_ERROR(tim::cuda::get_last_error());
}
#endif

template <typename _Device, typename _Func, typename... _Args,
          impl::is_gpu_device<_Device, false> = 0>
void
launch(const params<_Device>&, _Func&& _func, _Args&&... _args)
{
    std::forward<_Func>(_func)(std::forward<_Args>(_args)...);
}

//======================================================================================//
//
//  These provide loop parameters for grid-strided loops on GPU or traditional loops on
//  CPU
//
//======================================================================================//

template <typename _Device, size_t DIM = 0, typename _Intp = int32_t>
struct grid_strided_range : impl::range<_Intp>
{
    using base_type = impl::range<_Intp>;
    static_assert(DIM < 0 || DIM > 2,
                  "Error DIM parameter must be 0 (x), 1 (y), or 2 (z)");
};

//--------------------------------------------------------------------------------------//
// overload for 0/x
//
template <typename _Device, typename _Intp>
struct grid_strided_range<_Device, 0, _Intp> : impl::range<_Intp>
{
    using base_type = impl::range<_Intp>;
    template <bool B, typename T = char>
    using enable_if_t = impl::enable_if_t<B, T>;

#if defined(TIMEMORY_USE_CUDA) && defined(__NVCC__)
    template <typename _Dev                                       = _Device,
              enable_if_t<std::is_same<_Dev, device::gpu>::value> = 0>
    DEVICE_CALLABLE explicit grid_strided_range(_Intp max_iter)
    : base_type(blockIdx.x * blockDim.x + threadIdx.x, max_iter, blockDim.x * gridDim.x)
    {
    }

    template <typename _Dev                                       = _Device,
              enable_if_t<std::is_same<_Dev, device::cpu>::value> = 0>
#endif
    explicit grid_strided_range(_Intp max_iter)
    : base_type(0, max_iter, 1)
    {
    }

    using base_type::begin;
    using base_type::end;
    using base_type::stride;
};

//--------------------------------------------------------------------------------------//
// overload for 1/y
//
template <typename _Device, typename _Intp>
struct grid_strided_range<_Device, 1, _Intp> : impl::range<_Intp>
{
    using base_type = impl::range<_Intp>;
    template <bool B, typename T = char>
    using enable_if_t = impl::enable_if_t<B, T>;

#if defined(TIMEMORY_USE_CUDA) && defined(__NVCC__)
    template <typename _Dev                                       = _Device,
              enable_if_t<std::is_same<_Dev, device::gpu>::value> = 0>
    DEVICE_CALLABLE explicit grid_strided_range(_Intp max_iter)
    : base_type(blockIdx.y * blockDim.y + threadIdx.y, max_iter, blockDim.y * gridDim.y)
    {
    }

    template <typename _Dev                                       = _Device,
              enable_if_t<std::is_same<_Dev, device::cpu>::value> = 0>
#endif
    explicit grid_strided_range(_Intp max_iter)
    : base_type(0, max_iter, 1)
    {
    }

    using base_type::begin;
    using base_type::end;
    using base_type::stride;
};

//--------------------------------------------------------------------------------------//
// overload for 2/z
//
template <typename _Device, typename _Intp>
struct grid_strided_range<_Device, 2, _Intp> : impl::range<_Intp>
{
    using base_type = impl::range<_Intp>;
    template <bool B, typename T = char>
    using enable_if_t = impl::enable_if_t<B, T>;

#if defined(TIMEMORY_USE_CUDA) && defined(__NVCC__)
    template <typename _Dev                                       = _Device,
              enable_if_t<std::is_same<_Dev, device::gpu>::value> = 0>
    DEVICE_CALLABLE explicit grid_strided_range(_Intp max_iter)
    : base_type(blockIdx.z * blockDim.z + threadIdx.z, max_iter, blockDim.z * gridDim.z)
    {
    }

    template <typename _Dev                                       = _Device,
              enable_if_t<std::is_same<_Dev, device::cpu>::value> = 0>
#endif
    explicit grid_strided_range(_Intp max_iter)
    : base_type(0, max_iter, 1)
    {
    }

    using base_type::begin;
    using base_type::end;
    using base_type::stride;
};

}  // namespace device
}  // namespace tim
