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
 * \file timemory/components/hip/backends.hpp
 * \brief Implementation of the hip functions/utilities
 */

#pragma once

#include "timemory/backends/hip.hpp"
#include "timemory/components/cuda/backends.hpp"
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

#if defined(TIMEMORY_USE_HIP)
#    include <hip/hip_runtime.h>
#    include <hip/hip_runtime_api.h>
#endif

//======================================================================================//
//
namespace tim
{
namespace hip
{
//
const char*
get_error_string(error_t err);

// half-precision floating point
using __half2 = cuda::half2;
using fp16_t  = cuda::half2;

//--------------------------------------------------------------------------------------//
//
//      functions dealing with hip errors
//
//--------------------------------------------------------------------------------------//

/// check the success of a hipError_t
inline bool
check(error_t err)
{
    TIMEMORY_HIP_RUNTIME_CHECK_ERROR(err);
    // print_demangled_backtrace<32, 1>();
    return (err == success_v);
}

//--------------------------------------------------------------------------------------//
/// get last error but don't reset last error to hipSuccess
inline error_t
peek_at_last_error()
{
#if defined(TIMEMORY_USE_HIP)
    return hipPeekAtLastError();
#else
    return success_v;
#endif
}

//--------------------------------------------------------------------------------------//
/// get last error and reset to hipSuccess
inline error_t
get_last_error()
{
#if defined(TIMEMORY_USE_HIP)
    return hipGetLastError();
#else
    return success_v;
#endif
}

//--------------------------------------------------------------------------------------//
/// get the error string
inline const char*
get_error_string(error_t err)
{
#if defined(TIMEMORY_USE_HIP)
    return hipGetErrorString(err);
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
#if defined(TIMEMORY_USE_HIP)
    int dc = 0;
    if(hipGetDeviceCount(&dc) != success_v)
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
#if defined(TIMEMORY_USE_HIP)
    TIMEMORY_HIP_RUNTIME_API_CALL(hipSetDevice(device));
#else
    consume_parameters(device);
#endif
}

//--------------------------------------------------------------------------------------//
/// get the current device
inline int
get_device()
{
    int _device = 0;
#if defined(TIMEMORY_USE_HIP)
    TIMEMORY_HIP_RUNTIME_API_CALL(hipGetDevice(&_device));
#endif
    return _device;
}

//--------------------------------------------------------------------------------------//
/// sync the device
inline void
device_sync()
{
    TIMEMORY_HIP_RUNTIME_API_CALL_THROW(hipDeviceSynchronize());
}

//--------------------------------------------------------------------------------------//
/// reset the device
inline void
device_reset()
{
    TIMEMORY_HIP_RUNTIME_API_CALL(hipDeviceReset());
}

//--------------------------------------------------------------------------------------//
/// get the size of the L2 cache (in bytes)
inline int
device_l2_cache_size(int dev = 0)
{
#if defined(TIMEMORY_USE_HIP)
    if(device_count() == 0)
        return 0;
    hipDeviceProp_t deviceProp;
    TIMEMORY_HIP_RUNTIME_API_CALL(hipGetDeviceProperties(&deviceProp, dev));
    return deviceProp.l2CacheSize;
#else
    consume_parameters(dev);
    return 0;
#endif
}

//--------------------------------------------------------------------------------------//
/// get the clock rate (kilohertz)
inline int
get_device_clock_rate(int _dev = -1)
{
#if defined(TIMEMORY_USE_HIP)
    if(device_count() < 0)
        _dev = get_device();
    hipDeviceProp_t _device_prop{};
    TIMEMORY_HIP_RUNTIME_API_CALL(hipGetDeviceProperties(&_device_prop, _dev));
    return _device_prop.clockRate;
#else
    consume_parameters(_dev);
    return 1;
#endif
}

//--------------------------------------------------------------------------------------//
//
//      functions dealing with hip streams
//
//--------------------------------------------------------------------------------------//

/// create a hip stream
inline bool
stream_create(stream_t& stream)
{
#if defined(TIMEMORY_USE_HIP)
    return check(hipStreamCreate(&stream));
#else
    consume_parameters(stream);
    return true;
#endif
}

//--------------------------------------------------------------------------------------//
/// destroy a hip stream
inline void
stream_destroy(stream_t& stream)
{
#if defined(TIMEMORY_USE_HIP)
    if(stream != default_stream_v)
    {
        TIMEMORY_HIP_RUNTIME_API_CALL(hipStreamDestroy(stream));
    }
#else
    consume_parameters(stream);
#endif
}

//--------------------------------------------------------------------------------------//
/// sync the hip stream
inline void
stream_sync(stream_t stream)
{
#if defined(TIMEMORY_USE_HIP)
    TIMEMORY_HIP_RUNTIME_API_CALL(hipStreamSynchronize(stream));
#else
    consume_parameters(stream);
#endif
}

//--------------------------------------------------------------------------------------//
//
//      functions dealing with hip events
//
//--------------------------------------------------------------------------------------//

/// create a hip event
inline bool
event_create(event_t& evt)
{
#if defined(TIMEMORY_USE_HIP)
    return check(hipEventCreate(&evt));
#else
    consume_parameters(evt);
    return true;
#endif
}

//--------------------------------------------------------------------------------------//
/// destroy a hip event
inline void
event_destroy(event_t& evt)
{
#if defined(TIMEMORY_USE_HIP)
    TIMEMORY_HIP_RUNTIME_API_CALL(hipEventDestroy(evt));
#else
    consume_parameters(evt);
#endif
}

//--------------------------------------------------------------------------------------//
/// record a hip event
inline void
event_record(event_t& evt, stream_t& stream)
{
#if defined(TIMEMORY_USE_HIP)
    TIMEMORY_HIP_RUNTIME_API_CALL(hipEventRecord(evt, stream));
#else
    consume_parameters(evt, stream);
#endif
}

//--------------------------------------------------------------------------------------//
/// wait for a hip event to complete
inline void
event_sync(event_t& evt)
{
#if defined(TIMEMORY_USE_HIP)
    TIMEMORY_HIP_RUNTIME_API_CALL(hipEventSynchronize(evt));
#else
    consume_parameters(evt);
#endif
}

//--------------------------------------------------------------------------------------//
/// get the elapsed time
inline float
event_elapsed_time(event_t& _start, event_t& _stop)
{
#if defined(TIMEMORY_USE_HIP)
    float tmp = 0.0f;
    TIMEMORY_HIP_RUNTIME_API_CALL(hipEventElapsedTime(&tmp, _start, _stop));
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
#if defined(TIMEMORY_USE_HIP)
    return hipEventQuery(evt);
#else
    consume_parameters(evt);
    return success_v;
#endif
}

//--------------------------------------------------------------------------------------//
//
//      functions dealing with hip allocations
//
//--------------------------------------------------------------------------------------//

/// hip malloc
template <typename Tp>
inline Tp*
malloc(size_t n)
{
#if defined(TIMEMORY_USE_HIP)
    void* arr;
    TIMEMORY_HIP_RUNTIME_API_CALL(hipMalloc(&arr, n * sizeof(Tp)));
    if(!arr)
    {
        unsigned long long sz = n * sizeof(Tp);
        fprintf(stderr, "hipMalloc unable to allocate %llu bytes\n", sz);
        throw std::bad_alloc();
    }
    return static_cast<Tp*>(arr);
#else
    consume_parameters(n);
    return nullptr;
#endif
}

//--------------------------------------------------------------------------------------//

/// hip malloc host (pinned memory)
template <typename Tp>
inline Tp*
malloc_host(size_t n)
{
#if defined(TIMEMORY_USE_HIP)
    void* arr;
    TIMEMORY_HIP_RUNTIME_API_CALL(hipHostMalloc(&arr, n * sizeof(Tp)));
    if(!arr)
    {
        unsigned long long sz = n * sizeof(Tp);
        fprintf(stderr, "hipHostMalloc unable to allocate %llu bytes\n", sz);
        throw std::bad_alloc();
    }
    return static_cast<Tp*>(arr);
#else
    return new Tp[n];
#endif
}

//--------------------------------------------------------------------------------------//

/// hip malloc
template <typename Tp>
inline void
free(Tp*& arr)
{
#if defined(TIMEMORY_USE_HIP)
    TIMEMORY_HIP_RUNTIME_API_CALL(hipFree(arr));
    arr = nullptr;
#else
    consume_parameters(arr);
    arr = nullptr;
#endif
}

//--------------------------------------------------------------------------------------//

/// hip malloc
template <typename Tp>
inline void
free_host(Tp*& arr)
{
#if defined(TIMEMORY_USE_HIP)
    TIMEMORY_HIP_RUNTIME_API_CALL(hipHostFree(arr));
    arr = nullptr;
#else
    delete[] arr;
    arr = nullptr;
#endif
}

//--------------------------------------------------------------------------------------//

/// hip memcpy
template <typename Tp>
inline error_t
memcpy(Tp* dst, const Tp* src, size_t n, memcpy_t from_to)
{
#if defined(TIMEMORY_USE_HIP)
    return hipMemcpy(dst, src, n * sizeof(Tp), from_to);
#else
    consume_parameters(from_to);
    std::memcpy(dst, src, n * sizeof(Tp));
    return success_v;
#endif
}

//--------------------------------------------------------------------------------------//

/// hip memcpy
template <typename Tp>
inline error_t
memcpy(Tp* dst, const Tp* src, size_t n, memcpy_t from_to, stream_t stream)
{
#if defined(TIMEMORY_USE_HIP)
    return hipMemcpyAsync(dst, src, n * sizeof(Tp), from_to, stream);
#else
    consume_parameters(from_to, stream);
    std::memcpy(dst, src, n * sizeof(Tp));
    return success_v;
#endif
}

//--------------------------------------------------------------------------------------//

/// hip memset
template <typename Tp>
inline error_t
memset(Tp* dst, int value, size_t n)
{
#if defined(TIMEMORY_USE_HIP)
    return hipMemset(dst, value, n * sizeof(Tp));
#else
    return std::memset(dst, value, n * sizeof(Tp));
#endif
}

//--------------------------------------------------------------------------------------//

/// hip memset
template <typename Tp>
inline error_t
memset(Tp* dst, int value, size_t n, stream_t stream)
{
#if defined(TIMEMORY_USE_HIP)
    return hipMemsetAsync(dst, value, n * sizeof(Tp), stream);
#else
    consume_parameters(stream);
    return std::memset(dst, value, n * sizeof(Tp));
#endif
}

//--------------------------------------------------------------------------------------//
//
}  // namespace hip
}  // namespace tim
//
//======================================================================================//

template <typename T>
inline void
tim::hip::device_query()
{
#if defined(TIMEMORY_USE_HIP)

    static std::atomic<int16_t> _once(0);
    auto                        _count = _once++;
    if(_count > 0)
        return;

    int        deviceCount    = 0;
    int        driverVersion  = 0;
    int        runtimeVersion = 0;
    hipError_t error_id       = hipGetDeviceCount(&deviceCount);

    if(error_id != hipSuccess)
    {
        printf("hipGetDeviceCount returned error code %d\n--> %s\n",
               static_cast<int>(error_id), hipGetErrorString(error_id));

        if(deviceCount > 0)
        {
            TIMEMORY_HIP_RUNTIME_API_CALL(hipSetDevice(0));
            hipDeviceProp_t deviceProp;
            TIMEMORY_HIP_RUNTIME_API_CALL(hipGetDeviceProperties(&deviceProp, 0));
            printf("\nDevice %d: \"%s\"\n", 0, deviceProp.name);

            // Console log
            TIMEMORY_HIP_RUNTIME_API_CALL(hipDriverGetVersion(&driverVersion));
            TIMEMORY_HIP_RUNTIME_API_CALL(hipRuntimeGetVersion(&runtimeVersion));
            printf("  HIP Driver Version / Runtime Version          %d.%d / "
                   "%d.%d\n",
                   driverVersion / 1000, (driverVersion % 100) / 10,
                   runtimeVersion / 1000, (runtimeVersion % 100) / 10);
            printf("  HIP Capability Major/Minor version number:    %d.%d\n",
                   deviceProp.major, deviceProp.minor);
        }

        return;
    }

    if(deviceCount == 0)
        printf("No available HIP device(s) detected\n");
    else
        printf("Detected %d HIP capable devices\n", deviceCount);

    for(int dev = 0; dev < deviceCount; ++dev)
    {
        TIMEMORY_HIP_RUNTIME_API_CALL(hipSetDevice(dev));
        hipDeviceProp_t deviceProp;
        TIMEMORY_HIP_RUNTIME_API_CALL(hipGetDeviceProperties(&deviceProp, dev));
        printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);

        // Console log
        TIMEMORY_HIP_RUNTIME_API_CALL(hipDriverGetVersion(&driverVersion));
        TIMEMORY_HIP_RUNTIME_API_CALL(hipRuntimeGetVersion(&runtimeVersion));

        printf("  HIP Driver Version / Runtime Version          %d.%d / %d.%d\n",
               driverVersion / 1000, (driverVersion % 100) / 10, runtimeVersion / 1000,
               (runtimeVersion % 100) / 10);

        printf("  HIP Capability Major/Minor version number:    %d.%d\n",
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

#    if HIPRT_VERSION >= 5000
        // This is supported in HIP 5.0 (runtime API device properties)
        printf("  Memory Clock rate:                             %.0f Mhz\n",
               deviceProp.memoryClockRate * 1.0e-3);
        printf("  Memory Bus Width:                              %d-bit\n",
               deviceProp.memoryBusWidth);

        if(deviceProp.l2CacheSize)
            printf("  L2 Cache Size:                                 %d bytes\n",
                   deviceProp.l2CacheSize);
#    endif

        printf("  Maximum Texture Dimension Size (x,y,z)         1D=(%d), 2D=(%d, "
               "%d), 3D=(%d, %d, %d)\n",
               deviceProp.maxTexture1D, deviceProp.maxTexture2D[0],
               deviceProp.maxTexture2D[1], deviceProp.maxTexture3D[0],
               deviceProp.maxTexture3D[1], deviceProp.maxTexture3D[2]);

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
        printf("  Run time limit on kernels:                     %s\n",
               deviceProp.kernelExecTimeoutEnabled ? "Yes" : "No");
        printf("  Integrated GPU sharing Host Memory:            %s\n",
               deviceProp.integrated ? "Yes" : "No");
        printf("  Support host page-locked memory mapping:       %s\n",
               deviceProp.canMapHostMemory ? "Yes" : "No");
        printf("  Device has ECC support:                        %s\n",
               deviceProp.ECCEnabled ? "Enabled" : "Disabled");
#    if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
        printf("  HIP Device Driver Mode (TCC or WDDM):         %s\n",
               deviceProp.tccDriver ? "TCC (Tesla Compute Cluster Driver)"
                                    : "WDDM (Windows Display Driver Model)");
#    endif
        printf("  Supports Cooperative Kernel Launch:            %s\n",
               deviceProp.cooperativeLaunch ? "Yes" : "No");
        printf("  Supports MultiDevice Co-op Kernel Launch:      %s\n",
               deviceProp.cooperativeMultiDeviceLaunch ? "Yes" : "No");
        printf("  Device PCI Domain ID / Bus ID / location ID:   %d / %d / %d\n",
               deviceProp.pciDomainID, deviceProp.pciBusID, deviceProp.pciDeviceID);

        const char* sComputeMode[] = {
            "Default (multiple host threads can use ::hipSetDevice() with "
            "device "
            "simultaneously)",
            "Exclusive (only one host thread in one process is able to use "
            "::hipSetDevice() with this device)",
            "Prohibited (no host thread can use ::hipSetDevice() with this "
            "device)",
            "Exclusive Process (many threads in one process is able to use "
            "::hipSetDevice() with this device)",
            "Unknown",
            nullptr
        };
        printf("  Compute Mode:\n");
        printf("     < %s >\n", sComputeMode[deviceProp.computeMode]);
    }

    printf("\n\n");
// HIP_CHECK_LAST_ERROR();
#endif
}

//======================================================================================//
