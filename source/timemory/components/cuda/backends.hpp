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
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//

/**
 * \file timemory/components/cuda/backends.hpp
 * \brief Implementation of the cuda functions/utilities
 */

#pragma once

#include "timemory/backends/cuda.hpp"
#include "timemory/backends/nvtx.hpp"
#include "timemory/macros.hpp"
#include "timemory/utility/macros.hpp"
#include "timemory/utility/types.hpp"
#include "timemory/utility/utility.hpp"

#include <array>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <string>
#include <vector>

#if defined(TIMEMORY_USE_CUDA)
#    include <cuda.h>
#    include <cuda_fp16.h>
#    include <cuda_fp16.hpp>
#    include <cuda_profiler_api.h>
#    include <cuda_runtime_api.h>
#endif

#if defined(TIMEMORY_USE_CUDA) && (defined(__NVCC__) || defined(__CUDACC__)) &&          \
    defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
#    if defined(TIMEMORY_USE_CUDA_HALF)
#        undef TIMEMORY_USE_CUDA_HALF
#    endif
#endif

//======================================================================================//
//
namespace tim
{
namespace cuda
{
//
const char*
get_error_string(error_t err);

// half-precision floating point
#if !defined(TIMEMORY_USE_CUDA_HALF)

// make a different type
struct half2
{
    half2()             = default;
    ~half2()            = default;
    half2(const half2&) = default;
    half2(half2&&)      = default;
    half2& operator=(const half2&) = default;
    half2& operator=(half2&&) = default;

    template <typename... Args>
    half2(float val, Args&&...)
    : value{ val }
    {}

    template <typename Tp>
    half2& operator+=(const Tp&)
    {
        return *this;
    }
    template <typename Tp>
    half2& operator-=(const Tp&)
    {
        return *this;
    }
    template <typename Tp>
    half2& operator*=(const Tp&)
    {
        return *this;
    }
    template <typename Tp>
    half2& operator/=(const Tp&)
    {
        return *this;
    }

    friend half2 operator+(half2 lhs, const half2& rhs) { return lhs += rhs; }
    friend half2 operator-(half2 lhs, const half2& rhs) { return lhs -= rhs; }
    friend half2 operator*(half2 lhs, const half2& rhs) { return lhs *= rhs; }
    friend half2 operator/(half2 lhs, const half2& rhs) { return lhs /= rhs; }

    float&       operator[](int) { return value; }
    const float& operator[](int) const { return value; }

private:
    using value_type = float;
    value_type value = 0.0f;
};

using __half2 = half2;
using fp16_t  = half2;

#else
using fp16_t = half2;
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
    TIMEMORY_CUDA_RUNTIME_CHECK_ERROR(err);
    // print_demangled_backtrace<32, 1>();
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
template <typename T = void>
static inline void
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
    TIMEMORY_CUDA_RUNTIME_API_CALL(cudaSetDevice(device));
#else
    consume_parameters(device);
#endif
}

//--------------------------------------------------------------------------------------//
/// sync the device
inline void
device_sync()
{
    TIMEMORY_CUDA_RUNTIME_API_CALL_THROW(cudaDeviceSynchronize());
}

//--------------------------------------------------------------------------------------//
/// reset the device
inline void
device_reset()
{
    TIMEMORY_CUDA_RUNTIME_API_CALL(cudaDeviceReset());
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
    TIMEMORY_CUDA_RUNTIME_API_CALL(cudaEventRecord(evt, stream));
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
    TIMEMORY_CUDA_RUNTIME_API_CALL(cudaEventSynchronize(evt));
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
template <typename Tp>
inline Tp*
malloc(size_t n)
{
#if defined(TIMEMORY_USE_CUDA)
    void* arr;
    TIMEMORY_CUDA_RUNTIME_API_CALL(cudaMalloc(&arr, n * sizeof(Tp)));
    if(!arr)
    {
        unsigned long long sz = n * sizeof(Tp);
        fprintf(stderr, "cudaMalloc unable to allocate %llu bytes\n", sz);
        throw std::bad_alloc();
    }
    return static_cast<Tp*>(arr);
#else
    consume_parameters(n);
    return nullptr;
#endif
}

//--------------------------------------------------------------------------------------//

/// cuda malloc host (pinned memory)
template <typename Tp>
inline Tp*
malloc_host(size_t n)
{
#if defined(TIMEMORY_USE_CUDA)
    void* arr;
    TIMEMORY_CUDA_RUNTIME_API_CALL(cudaMallocHost(&arr, n * sizeof(Tp)));
    if(!arr)
    {
        unsigned long long sz = n * sizeof(Tp);
        fprintf(stderr, "cudaMallocHost unable to allocate %llu bytes\n", sz);
        throw std::bad_alloc();
    }
    return static_cast<Tp*>(arr);
#else
    return new Tp[n];
#endif
}

//--------------------------------------------------------------------------------------//

/// cuda malloc
template <typename Tp>
inline void
free(Tp*& arr)
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
template <typename Tp>
inline void
free_host(Tp*& arr)
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
template <typename Tp>
inline error_t
memcpy(Tp* dst, const Tp* src, size_t n, memcpy_t from_to)
{
#if defined(TIMEMORY_USE_CUDA)
    return cudaMemcpy(dst, src, n * sizeof(Tp), from_to);
#else
    consume_parameters(from_to);
    std::memcpy(dst, src, n * sizeof(Tp));
    return success_v;
#endif
}

//--------------------------------------------------------------------------------------//

/// cuda memcpy
template <typename Tp>
inline error_t
memcpy(Tp* dst, const Tp* src, size_t n, memcpy_t from_to, stream_t stream)
{
#if defined(TIMEMORY_USE_CUDA)
    return cudaMemcpyAsync(dst, src, n * sizeof(Tp), from_to, stream);
#else
    consume_parameters(from_to, stream);
    std::memcpy(dst, src, n * sizeof(Tp));
    return success_v;
#endif
}

//--------------------------------------------------------------------------------------//

/// cuda memset
template <typename Tp>
inline error_t
memset(Tp* dst, int value, size_t n)
{
#if defined(TIMEMORY_USE_CUDA)
    return cudaMemset(dst, value, n * sizeof(Tp));
#else
    return std::memset(dst, value, n * sizeof(Tp));
#endif
}

//--------------------------------------------------------------------------------------//

/// cuda memset
template <typename Tp>
inline error_t
memset(Tp* dst, int value, size_t n, stream_t stream)
{
#if defined(TIMEMORY_USE_CUDA)
    return cudaMemsetAsync(dst, value, n * sizeof(Tp), stream);
#else
    consume_parameters(stream);
    return std::memset(dst, value, n * sizeof(Tp));
#endif
}

//--------------------------------------------------------------------------------------//
//
}  // namespace cuda
}  // namespace tim
//
//======================================================================================//

template <typename T>
inline void
tim::cuda::device_query()
{
#if defined(TIMEMORY_USE_CUDA)

    static std::atomic<int16_t> _once(0);
    auto                        _count = _once++;
    if(_count > 0)
        return;

    int         deviceCount    = 0;
    int         driverVersion  = 0;
    int         runtimeVersion = 0;
    cudaError_t error_id       = cudaGetDeviceCount(&deviceCount);

    if(error_id != cudaSuccess)
    {
        printf("cudaGetDeviceCount returned error code %d\n--> %s\n",
               static_cast<int>(error_id), cudaGetErrorString(error_id));

        if(deviceCount > 0)
        {
            cudaSetDevice(0);
            cudaDeviceProp deviceProp;
            cudaGetDeviceProperties(&deviceProp, 0);
            printf("\nDevice %d: \"%s\"\n", 0, deviceProp.name);

            // Console log
            cudaDriverGetVersion(&driverVersion);
            cudaRuntimeGetVersion(&runtimeVersion);
            printf("  CUDA Driver Version / Runtime Version          %d.%d / "
                   "%d.%d\n",
                   driverVersion / 1000, (driverVersion % 100) / 10,
                   runtimeVersion / 1000, (runtimeVersion % 100) / 10);
            printf("  CUDA Capability Major/Minor version number:    %d.%d\n",
                   deviceProp.major, deviceProp.minor);
        }

        return;
    }

    if(deviceCount == 0)
        printf("No available CUDA device(s) detected\n");
    else
        printf("Detected %d CUDA capable devices\n", deviceCount);

    for(int dev = 0; dev < deviceCount; ++dev)
    {
        cudaSetDevice(dev);
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);
        printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);

        // Console log
        cudaDriverGetVersion(&driverVersion);
        cudaRuntimeGetVersion(&runtimeVersion);

        printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n",
               driverVersion / 1000, (driverVersion % 100) / 10, runtimeVersion / 1000,
               (runtimeVersion % 100) / 10);

        printf("  CUDA Capability Major/Minor version number:    %d.%d\n",
               deviceProp.major, deviceProp.minor);

        char msg[256];
#    if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
        sprintf_s(msg, sizeof(msg),
                  "  Total amount of global memory:                 %.0f MBytes "
                  "(%llu bytes)\n",
                  static_cast<double>(deviceProp.totalGlobalMem / 1048576.0),
                  (unsigned long long) deviceProp.totalGlobalMem);
#    else
        snprintf(msg, sizeof(msg),
                 "  Total amount of global memory:                 %.0f MBytes "
                 "(%llu bytes)\n",
                 static_cast<double>(deviceProp.totalGlobalMem / 1048576.0),
                 (unsigned long long) deviceProp.totalGlobalMem);
#    endif
        printf("%s", msg);

        printf("  GPU Max Clock rate:                            %.0f MHz (%0.2f "
               "GHz)\n",
               deviceProp.clockRate * 1e-3, deviceProp.clockRate * 1.0e-6);

#    if CUDART_VERSION >= 5000
        // This is supported in CUDA 5.0 (runtime API device properties)
        printf("  Memory Clock rate:                             %.0f Mhz\n",
               deviceProp.memoryClockRate * 1.0e-3);
        printf("  Memory Bus Width:                              %d-bit\n",
               deviceProp.memoryBusWidth);

        if(deviceProp.l2CacheSize)
            printf("  L2 Cache Size:                                 %d bytes\n",
                   deviceProp.l2CacheSize);

#    else
        // This only available in CUDA 4.0-4.2 (but these were only exposed in
        // the CUDA Driver API)
        int memoryClock;
        int memBusWidth;
        int L2CacheSize;
        getCudaAttribute<int>(&memoryClock, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, dev);
        printf("  Memory Clock rate:                             %.0f Mhz\n",
               memoryClock * 1e-3f);
        getCudaAttribute<int>(&memBusWidth, CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH,
                              dev);
        printf("  Memory Bus Width:                              %d-bit\n", memBusWidth);
        getCudaAttribute<int>(&L2CacheSize, CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, dev);

        if(L2CacheSize)
            printf("  L2 Cache Size:                                 %d bytes\n",
                   L2CacheSize);
#    endif

        printf("  Maximum Texture Dimension Size (x,y,z)         1D=(%d), 2D=(%d, "
               "%d), 3D=(%d, %d, %d)\n",
               deviceProp.maxTexture1D, deviceProp.maxTexture2D[0],
               deviceProp.maxTexture2D[1], deviceProp.maxTexture3D[0],
               deviceProp.maxTexture3D[1], deviceProp.maxTexture3D[2]);
        printf("  Maximum Layered 1D Texture Size, (num) layers  1D=(%d), %d "
               "layers\n",
               deviceProp.maxTexture1DLayered[0], deviceProp.maxTexture1DLayered[1]);
        printf("  Maximum Layered 2D Texture Size, (num) layers  2D=(%d, %d), %d "
               "layers\n",
               deviceProp.maxTexture2DLayered[0], deviceProp.maxTexture2DLayered[1],
               deviceProp.maxTexture2DLayered[2]);

        printf("  Total amount of constant memory:               %lu bytes\n",
               deviceProp.totalConstMem);
        printf("  Total amount of shared memory per block:       %lu bytes\n",
               deviceProp.sharedMemPerBlock);
        printf("  Total number of registers available per block: %d\n",
               deviceProp.regsPerBlock);
        printf("  Warp size:                                     %d\n",
               deviceProp.warpSize);
        printf("  Multiprocessor count:                          %d\n",
               deviceProp.multiProcessorCount);
        printf("  Maximum number of threads per multiprocessor:  %d\n",
               deviceProp.maxThreadsPerMultiProcessor);
        printf("  Maximum number of threads per block:           %d\n",
               deviceProp.maxThreadsPerBlock);
        printf("  Max dimension size of a thread block (x,y,z): (%d, %d, %d)\n",
               deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1],
               deviceProp.maxThreadsDim[2]);
        printf("  Max dimension size of a grid size    (x,y,z): (%d, %d, %d)\n",
               deviceProp.maxGridSize[0], deviceProp.maxGridSize[1],
               deviceProp.maxGridSize[2]);
        printf("  Maximum memory pitch:                          %lu bytes\n",
               deviceProp.memPitch);
        printf("  Texture alignment:                             %lu bytes\n",
               deviceProp.textureAlignment);
        printf("  Concurrent copy and kernel execution:          %s with %d copy "
               "engine(s)\n",
               (deviceProp.deviceOverlap ? "Yes" : "No"), deviceProp.asyncEngineCount);
        printf("  Run time limit on kernels:                     %s\n",
               deviceProp.kernelExecTimeoutEnabled ? "Yes" : "No");
        printf("  Integrated GPU sharing Host Memory:            %s\n",
               deviceProp.integrated ? "Yes" : "No");
        printf("  Support host page-locked memory mapping:       %s\n",
               deviceProp.canMapHostMemory ? "Yes" : "No");
        printf("  Alignment requirement for Surfaces:            %s\n",
               deviceProp.surfaceAlignment ? "Yes" : "No");
        printf("  Device has ECC support:                        %s\n",
               deviceProp.ECCEnabled ? "Enabled" : "Disabled");
#    if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
        printf("  CUDA Device Driver Mode (TCC or WDDM):         %s\n",
               deviceProp.tccDriver ? "TCC (Tesla Compute Cluster Driver)"
                                    : "WDDM (Windows Display Driver Model)");
#    endif
        printf("  Device supports Unified Addressing (UVA):      %s\n",
               deviceProp.unifiedAddressing ? "Yes" : "No");
        printf("  Device supports Compute Preemption:            %s\n",
               deviceProp.computePreemptionSupported ? "Yes" : "No");
        printf("  Supports Cooperative Kernel Launch:            %s\n",
               deviceProp.cooperativeLaunch ? "Yes" : "No");
        printf("  Supports MultiDevice Co-op Kernel Launch:      %s\n",
               deviceProp.cooperativeMultiDeviceLaunch ? "Yes" : "No");
        printf("  Device PCI Domain ID / Bus ID / location ID:   %d / %d / %d\n",
               deviceProp.pciDomainID, deviceProp.pciBusID, deviceProp.pciDeviceID);

        const char* sComputeMode[] = {
            "Default (multiple host threads can use ::cudaSetDevice() with "
            "device "
            "simultaneously)",
            "Exclusive (only one host thread in one process is able to use "
            "::cudaSetDevice() with this device)",
            "Prohibited (no host thread can use ::cudaSetDevice() with this "
            "device)",
            "Exclusive Process (many threads in one process is able to use "
            "::cudaSetDevice() with this device)",
            "Unknown",
            nullptr
        };
        printf("  Compute Mode:\n");
        printf("     < %s >\n", sComputeMode[deviceProp.computeMode]);
    }

    printf("\n\n");
// CUDA_CHECK_LAST_ERROR();
#endif
}

//======================================================================================//
