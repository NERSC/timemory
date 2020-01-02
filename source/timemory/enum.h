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

/** \file timemory/enum.h
 * \headerfile timemory/enum.h "timemory/enum.h"
 * This provides the core enumeration for components
 *
 */

#pragma once

//--------------------------------------------------------------------------------------//
// base operating system

#if defined(_WIN32) || defined(_WIN64)
#    if !defined(_WINDOWS)
#        define _WINDOWS
#    endif
//--------------------------------------------------------------------------------------//

#elif defined(__APPLE__) || defined(__MACH__)
#    if !defined(_MACOS)
#        define _MACOS
#    endif
#    if !defined(_UNIX)
#        define _UNIX
#    endif
//--------------------------------------------------------------------------------------//

#elif defined(__linux__) || defined(__linux) || defined(linux) || defined(__gnu_linux__)
#    if !defined(_LINUX)
#        define _LINUX
#    endif
#    if !defined(_UNIX)
#        define _UNIX
#    endif
//--------------------------------------------------------------------------------------//

#elif defined(__unix__) || defined(__unix) || defined(unix) || defined(_)
#    if !defined(_UNIX)
#        define _UNIX
#    endif
#endif

//======================================================================================//
//
//      Windows DLL settings
//
//======================================================================================//

// Define macros for WIN32 for importing/exporting external symbols to DLLs
#if !defined(tim_api)
#    if defined(_WINDOWS) && !defined(_TIMEMORY_ARCHIVE)
#        if defined(_TIMEMORY_DLL)
#            define tim_api __declspec(dllexport)
#        else
#            define tim_api __declspec(dllimport)
#        endif
#    else
#        define tim_api
#    endif
#endif

//======================================================================================//
//
//      Symbol override
//
//======================================================================================//

#if !defined(_WINDOWS)
#    if defined(__clang__) && defined(__APPLE__)
#        define TIMEMORY_WEAK_PREFIX
#        define TIMEMORY_WEAK_POSTFIX __attribute__((weak_import))
#    else
#        define TIMEMORY_WEAK_PREFIX __attribute__((weak))
#        define TIMEMORY_WEAK_POSTFIX
#    endif
#endif

//======================================================================================//
//
//      Enumeration
//
//======================================================================================//

enum TIMEMORY_COMPONENT
{
    CALIPER                  = 0,
    CPU_CLOCK                = 1,
    CPU_ROOFLINE_DP_FLOPS    = 2,
    CPU_ROOFLINE_FLOPS       = 3,
    CPU_ROOFLINE_SP_FLOPS    = 4,
    CPU_UTIL                 = 5,
    CUDA_EVENT               = 6,
    CUDA_PROFILER            = 7,
    CUPTI_ACTIVITY           = 8,
    CUPTI_COUNTERS           = 9,
    DATA_RSS                 = 10,
    GPERF_CPU_PROFILER       = 11,
    GPERF_HEAP_PROFILER      = 12,
    GPU_ROOFLINE_DP_FLOPS    = 13,
    GPU_ROOFLINE_FLOPS       = 14,
    GPU_ROOFLINE_HP_FLOPS    = 15,
    GPU_ROOFLINE_SP_FLOPS    = 16,
    LIKWID_NVMON             = 17,
    LIKWID_PERFMON           = 18,
    MONOTONIC_CLOCK          = 19,
    MONOTONIC_RAW_CLOCK      = 20,
    NUM_IO_IN                = 21,
    NUM_IO_OUT               = 22,
    NUM_MAJOR_PAGE_FAULTS    = 23,
    NUM_MINOR_PAGE_FAULTS    = 24,
    NUM_MSG_RECV             = 25,
    NUM_MSG_SENT             = 26,
    NUM_SIGNALS              = 27,
    NUM_SWAP                 = 28,
    NVTX_MARKER              = 29,
    PAGE_RSS                 = 30,
    PAPI_ARRAY               = 31,
    PEAK_RSS                 = 32,
    PRIORITY_CONTEXT_SWITCH  = 33,
    PROCESS_CPU_CLOCK        = 34,
    PROCESS_CPU_UTIL         = 35,
    READ_BYTES               = 36,
    STACK_RSS                = 37,
    SYS_CLOCK                = 38,
    TAU_MARKER               = 39,
    THREAD_CPU_CLOCK         = 40,
    THREAD_CPU_UTIL          = 41,
    TRIP_COUNT               = 42,
    USER_CLOCK               = 43,
    USER_LIST_BUNDLE         = 44,
    USER_TUPLE_BUNDLE        = 45,
    VIRTUAL_MEMORY           = 46,
    VOLUNTARY_CONTEXT_SWITCH = 47,
    VTUNE_EVENT              = 48,
    VTUNE_FRAME              = 49,
    WALL_CLOCK               = 50,
    WRITTEN_BYTES            = 51,
    TIMEMORY_COMPONENTS_END  = 52
};
