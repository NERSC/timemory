//  MIT License
//
//  Copyright (c) 2019, The Regents of the University of California,
// through Lawrence Berkeley National Laboratory (subject to receipt of any
// required approvals from the U.S. Dept. of Energy).  All rights reserved.
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to
//  deal in the Software without restriction, including without limitation the
//  rights to use, copy, modify, merge, publish, distribute, sublicense, and
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in
//  all copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
//  IN THE SOFTWARE.
//

/** \file cuda.hpp
 * \headerfile cuda.hpp "timemory/backends/cuda.hpp"
 * Defines cuda functions and dummy functions when compiled without CUDA
 *
 */

#pragma once

#include "timemory/utility/macros.hpp"
#include "timemory/utility/utility.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>

#if defined(TIMEMORY_USE_CUDA)
#    include <cuda.h>
#    include <cuda_fp16.h>
#    include <cuda_fp16.hpp>
#    include <cuda_runtime_api.h>
#endif

#if defined(TIMEMORY_USE_CUPTI)
#    include <cupti.h>
#endif

#if defined(TIMEMORY_USE_CUDA) && (defined(__NVCC__) || defined(__CUDACC__)) &&          \
    (__CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__))
#    if !defined(TIMEMORY_CUDA_FP16) && !defined(TIMEMORY_DISABLE_CUDA_HALF2)
#        define TIMEMORY_CUDA_FP16
#    endif
#endif

#include "timemory/utility/utility.hpp"

// this a macro so we can pre-declare
#define CUDA_RUNTIME_API_CALL(apiFuncCall)                                               \
    {                                                                                    \
        ::tim::cuda::error_t err = apiFuncCall;                                          \
        if(err != ::tim::cuda::success_v && (int) err != 0)                              \
        {                                                                                \
            fprintf(stderr, "%s:%d: error: function '%s' failed with error: %s.\n",      \
                    __FILE__, __LINE__, #apiFuncCall,                                    \
                    ::tim::cuda::get_error_string(err));                                 \
        }                                                                                \
    }

// this a macro so we can pre-declare
#define CUDA_RUNTIME_API_CALL_THROW(apiFuncCall)                                         \
    {                                                                                    \
        ::tim::cuda::error_t err = apiFuncCall;                                          \
        if(err != ::tim::cuda::success_v && (int) err != 0)                              \
        {                                                                                \
            char errmsg[std::numeric_limits<uint16_t>::max()];                           \
            sprintf(errmsg, "%s:%d: error: function '%s' failed with error: %s.\n",      \
                    __FILE__, __LINE__, #apiFuncCall,                                    \
                    ::tim::cuda::get_error_string(err));                                 \
            throw std::runtime_error(errmsg);                                            \
        }                                                                                \
    }

// this a macro so we can pre-declare
#define CUDA_RUNTIME_CHECK_ERROR(err)                                                    \
    {                                                                                    \
        if(err != ::tim::cuda::success_v && (int) err != 0)                              \
        {                                                                                \
            fprintf(stderr, "%s:%d: error check failed with: code %i -- %s.\n",          \
                    __FILE__, __LINE__, (int) err, ::tim::cuda::get_error_string(err));  \
        }                                                                                \
    }

//--------------------------------------------------------------------------------------//
namespace tim
{
//--------------------------------------------------------------------------------------//
namespace cuda
{
//--------------------------------------------------------------------------------------//
//
#if defined(TIMEMORY_USE_CUDA)
// define some types for when CUDA is enabled
using stream_t = cudaStream_t;
using event_t  = cudaEvent_t;
using error_t  = cudaError_t;
using memcpy_t = cudaMemcpyKind;
// define some values for when CUDA is enabled
static const decltype(cudaSuccess)       success_v          = cudaSuccess;
static const decltype(cudaErrorNotReady) err_not_ready_v    = cudaErrorNotReady;
static const cudaMemcpyKind              host_to_host_v     = cudaMemcpyHostToHost;
static const cudaMemcpyKind              host_to_device_v   = cudaMemcpyHostToDevice;
static const cudaMemcpyKind              device_to_host_v   = cudaMemcpyDeviceToHost;
static const cudaMemcpyKind              device_to_device_v = cudaMemcpyDeviceToDevice;
#else
// define some types for when CUDA is disabled
using stream_t = int;
using event_t  = int;
using error_t  = int;
using memcpy_t = int;
// define some values for when CUDA is disabled
static const int success_v          = 0;
static const int err_not_ready_v    = 0;
static const int host_to_host_v     = 0;
static const int host_to_device_v   = 1;
static const int device_to_host_v   = 2;
static const int device_to_device_v = 3;
#endif

// half-precision floating point
#if !defined(TIMEMORY_CUDA_FP16)

// make a different type
struct half2
{
    half2()             = default;
    ~half2()            = default;
    half2(const half2&) = default;
    half2(half2&&)      = default;
    half2& operator=(const half2&) = default;
    half2& operator=(half2&&) = default;

    half2(float val)
    : value{ { val, val } }
    {}
    half2(float lhs, float rhs)
    : value{ { lhs, rhs } }
    {}
    half2& operator+=(const float& rhs)
    {
        value[0] += rhs;
        value[1] += rhs;
        return *this;
    }
    half2& operator-=(const float& rhs)
    {
        value[0] -= rhs;
        value[1] -= rhs;
        return *this;
    }
    half2& operator*=(const float& rhs)
    {
        value[0] *= rhs;
        value[1] *= rhs;
        return *this;
    }
    half2& operator/=(const float& rhs)
    {
        value[0] /= rhs;
        value[1] /= rhs;
        return *this;
    }

    half2& operator+=(const half2& rhs)
    {
        value[0] += rhs[0];
        value[1] += rhs[1];
        return *this;
    }
    half2& operator-=(const half2& rhs)
    {
        value[0] -= rhs[0];
        value[1] -= rhs[1];
        return *this;
    }
    half2& operator*=(const half2& rhs)
    {
        value[0] *= rhs[0];
        value[1] *= rhs[1];
        return *this;
    }
    half2& operator/=(const half2& rhs)
    {
        value[0] /= rhs[0];
        value[1] /= rhs[1];
        return *this;
    }
    friend half2 operator+(const half2& lhs, const half2& rhs)
    {
        return half2(lhs) += rhs;
    }
    friend half2 operator-(const half2& lhs, const half2& rhs)
    {
        return half2(lhs) -= rhs;
    }
    friend half2 operator*(const half2& lhs, const half2& rhs)
    {
        return half2(lhs) *= rhs;
    }
    friend half2 operator/(const half2& lhs, const half2& rhs)
    {
        return half2(lhs) /= rhs;
    }

    float&       operator[](int idx) { return value[idx % 2]; }
    const float& operator[](int idx) const { return value[idx % 2]; }

private:
    using value_type = std::array<float, 2>;
    value_type value;
};

using __half2 = half2;
using fp16_t  = half2;

#else
using fp16_t                        = half2;
#endif

//--------------------------------------------------------------------------------------//
//
//      functions dealing with cuda errors
//
//--------------------------------------------------------------------------------------//

/// check the success of a cudaError_t
inline bool
check(error_t err)
{
    return (err == success_v);
}

//--------------------------------------------------------------------------------------//
/// get last error but don't reset last error to cudaSuccess
inline error_t
peek_at_last_error()
{
#if defined(TIMEMORY_USE_CUDA)
    return cudaPeekAtLastError();
#else
    return success_v;
#endif
}

//--------------------------------------------------------------------------------------//
/// get last error and reset to cudaSuccess
inline error_t
get_last_error()
{
#if defined(TIMEMORY_USE_CUDA)
    return cudaGetLastError();
#else
    return success_v;
#endif
}

//--------------------------------------------------------------------------------------//
/// get the error string
inline const char*
get_error_string(error_t err)
{
#if defined(TIMEMORY_USE_CUDA)
    return cudaGetErrorString(err);
#else
    consume_parameters(err);
    return "";
#endif
}

//--------------------------------------------------------------------------------------//
//
//      functions dealing with the device
//
//--------------------------------------------------------------------------------------//

/// print info about devices available (only does this once per process)
inline void
device_query();

//--------------------------------------------------------------------------------------//
/// get the number of devices available
inline int
device_count()
{
#if defined(TIMEMORY_USE_CUDA)
    int dc = 0;
    if(cudaGetDeviceCount(&dc) != success_v)
        return 0;
    return dc;
#else
    return 0;
#endif
}

//--------------------------------------------------------------------------------------//
/// sets the thread to a specific device
inline void
set_device(int device)
{
#if defined(TIMEMORY_USE_CUDA)
    CUDA_RUNTIME_API_CALL(cudaSetDevice(device));
#else
    consume_parameters(device);
#endif
}

//--------------------------------------------------------------------------------------//
/// sync the device
inline void
device_sync()
{
#if defined(TIMEMORY_USE_CUDA)
    CUDA_RUNTIME_API_CALL_THROW(cudaDeviceSynchronize());
#endif
}

//--------------------------------------------------------------------------------------//
/// reset the device
inline void
device_reset()
{
#if defined(TIMEMORY_USE_CUDA)
    CUDA_RUNTIME_API_CALL(cudaDeviceReset());
#endif
}

//--------------------------------------------------------------------------------------//
/// get the size of the L2 cache (in bytes)
inline int
device_l2_cache_size(int dev = 0)
{
#if defined(TIMEMORY_USE_CUDA)
    if(device_count() == 0)
        return 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    return deviceProp.l2CacheSize;
#else
    consume_parameters(dev);
    return 0;
#endif
}
//--------------------------------------------------------------------------------------//
//
//      functions dealing with cuda streams
//
//--------------------------------------------------------------------------------------//

/// create a cuda stream
inline bool
stream_create(stream_t& stream)
{
#if defined(TIMEMORY_USE_CUDA)
    return check(cudaStreamCreate(&stream));
#else
    consume_parameters(stream);
    return true;
#endif
}

//--------------------------------------------------------------------------------------//
/// destroy a cuda stream
inline void
stream_destroy(stream_t& stream)
{
#if defined(TIMEMORY_USE_CUDA)
    cudaStreamDestroy(stream);
#else
    consume_parameters(stream);
#endif
}

//--------------------------------------------------------------------------------------//
/// sync the cuda stream
inline void
stream_sync(stream_t stream)
{
#if defined(TIMEMORY_USE_CUDA)
    auto ret = cudaStreamSynchronize(stream);
    consume_parameters(ret);
#else
    consume_parameters(stream);
#endif
}

//--------------------------------------------------------------------------------------//
//
//      functions dealing with cuda events
//
//--------------------------------------------------------------------------------------//

/// create a cuda event
inline bool
event_create(event_t& evt)
{
#if defined(TIMEMORY_USE_CUDA)
    return check(cudaEventCreate(&evt));
#else
    consume_parameters(evt);
    return true;
#endif
}

//--------------------------------------------------------------------------------------//
/// destroy a cuda event
inline void
event_destroy(event_t& evt)
{
#if defined(TIMEMORY_USE_CUDA)
    cudaEventDestroy(evt);
#else
    consume_parameters(evt);
#endif
}

//--------------------------------------------------------------------------------------//
/// record a cuda event
inline void
event_record(event_t& evt, stream_t& stream)
{
#if defined(TIMEMORY_USE_CUDA)
    CUDA_RUNTIME_API_CALL(cudaEventRecord(evt, stream));
#else
    consume_parameters(evt, stream);
#endif
}

//--------------------------------------------------------------------------------------//
/// wait for a cuda event to complete
inline void
event_sync(event_t& evt)
{
#if defined(TIMEMORY_USE_CUDA)
    CUDA_RUNTIME_API_CALL(cudaEventSynchronize(evt));
#else
    consume_parameters(evt);
#endif
}

//--------------------------------------------------------------------------------------//
/// get the elapsed time
inline float
event_elapsed_time(event_t& _start, event_t& _stop)
{
#if defined(TIMEMORY_USE_CUDA)
    float tmp = 0.0f;
    cudaEventElapsedTime(&tmp, _start, _stop);
    return tmp;
#else
    consume_parameters(_start, _stop);
    return 0.0f;
#endif
}

//--------------------------------------------------------------------------------------//
/// query whether an event is finished
inline error_t
event_query(event_t evt)
{
#if defined(TIMEMORY_USE_CUDA)
    return cudaEventQuery(evt);
#else
    consume_parameters(evt);
    return success_v;
#endif
}

//--------------------------------------------------------------------------------------//
//
//      functions dealing with cuda allocations
//
//--------------------------------------------------------------------------------------//

/// cuda malloc
template <typename _Tp>
inline _Tp*
malloc(size_t n)
{
#if defined(TIMEMORY_USE_CUDA)
    void* arr;
    CUDA_RUNTIME_API_CALL(cudaMalloc(&arr, n * sizeof(_Tp)));
    if(!arr)
    {
        unsigned long long sz = n * sizeof(_Tp);
        fprintf(stderr, "cudaMalloc unable to allocate %llu bytes\n", sz);
        throw std::bad_alloc();
    }
    return static_cast<_Tp*>(arr);
#else
    consume_parameters(n);
    return nullptr;
#endif
}

//--------------------------------------------------------------------------------------//

/// cuda malloc host (pinned memory)
template <typename _Tp>
inline _Tp*
malloc_host(size_t n)
{
#if defined(TIMEMORY_USE_CUDA)
    void* arr;
    CUDA_RUNTIME_API_CALL(cudaMallocHost(&arr, n * sizeof(_Tp)));
    if(!arr)
    {
        unsigned long long sz = n * sizeof(_Tp);
        fprintf(stderr, "cudaMallocHost unable to allocate %llu bytes\n", sz);
        throw std::bad_alloc();
    }
    return static_cast<_Tp*>(arr);
#else
    return new _Tp[n];
#endif
}

//--------------------------------------------------------------------------------------//

/// cuda malloc
template <typename _Tp>
inline void
free(_Tp*& arr)
{
#if defined(TIMEMORY_USE_CUDA)
    cudaFree(arr);
    arr = nullptr;
#else
    consume_parameters(arr);
    arr = nullptr;
#endif
}

//--------------------------------------------------------------------------------------//

/// cuda malloc
template <typename _Tp>
inline void
free_host(_Tp*& arr)
{
#if defined(TIMEMORY_USE_CUDA)
    cudaFreeHost(arr);
    arr = nullptr;
#else
    delete[] arr;
    arr = nullptr;
#endif
}

//--------------------------------------------------------------------------------------//

/// cuda memcpy
template <typename _Tp>
inline error_t
memcpy(_Tp* dst, const _Tp* src, size_t n, memcpy_t from_to)
{
#if defined(TIMEMORY_USE_CUDA)
    return cudaMemcpy(dst, src, n * sizeof(_Tp), from_to);
#else
    consume_parameters(from_to);
    std::memcpy(dst, src, n * sizeof(_Tp));
    return success_v;
#endif
}

//--------------------------------------------------------------------------------------//

/// cuda memcpy
template <typename _Tp>
inline error_t
memcpy(_Tp* dst, const _Tp* src, size_t n, memcpy_t from_to, stream_t stream)
{
#if defined(TIMEMORY_USE_CUDA)
    return cudaMemcpyAsync(dst, src, n * sizeof(_Tp), from_to, stream);
#else
    consume_parameters(from_to, stream);
    std::memcpy(dst, src, n * sizeof(_Tp));
    return success_v;
#endif
}

//--------------------------------------------------------------------------------------//

/// cuda memset
template <typename _Tp>
inline error_t
memset(_Tp* dst, const int& value, size_t n)
{
#if defined(TIMEMORY_USE_CUDA)
    return cudaMemset(dst, value, n * sizeof(_Tp));
#else
    return std::memset(dst, value, n * sizeof(_Tp));
#endif
}

//--------------------------------------------------------------------------------------//

/// cuda memset
template <typename _Tp>
inline error_t
memset(_Tp* dst, const int& value, size_t n, stream_t stream)
{
#if defined(TIMEMORY_USE_CUDA)
    return cudaMemsetAsync(dst, value, n * sizeof(_Tp), stream);
#else
    consume_parameters(stream);
    return std::memset(dst, value, n * sizeof(_Tp));
#endif
}

//--------------------------------------------------------------------------------------//
}  // namespace cuda
//--------------------------------------------------------------------------------------//
}  // namespace tim
//--------------------------------------------------------------------------------------//

// definition of cuda device query
#include "timemory/backends/bits/cuda.hpp"
