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
//
/// \macro TIMEMORY_COMPONENT_ENUM_SIZE
/// \brief The number of enumerated components defined by timemory
//
#if !defined(TIMEMORY_COMPONENT_ENUM_SIZE)
#    define TIMEMORY_COMPONENT_ENUM_SIZE 60
#endif
//
/// \macro TIMEMORY_USER_COMPONENT_ENUM
/// \brief Extra enumerated components provided by a downstream application. If this
/// macro is used, be sure to end the list with a comma
///
/// \code
/// #define TIMEMORY_USER_COMPONENT_ENUM MY_COMPONENT = TIMEMORY_COMPONENT_ENUM_SIZE + 1,
//
#if !defined(TIMEMORY_USER_COMPONENT_ENUM)
#    define TIMEMORY_USER_COMPONENT_ENUM
#endif

#if !defined(TIMEMORY_USER_COMPONENT_ENUM_SIZE)
#    define TIMEMORY_USER_COMPONENT_ENUM_SIZE 0
#endif
//
/// \enum TIMEMORY_COMPONENT_ENUM
/// \brief Enumerated identifiers for timemory-provided components. If the user wishes
/// to add to the enumerated components, use \ref TIMEMORY_USER_COMPONENT_ENUM
//
enum TIMEMORY_NATIVE_COMPONENT
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
    CURRENT_PEAK_RSS         = 10,
    DATA_RSS                 = 11,
    GPERFTOOLS_CPU_PROFILER  = 12,
    GPERFTOOLS_HEAP_PROFILER = 13,
    GPU_ROOFLINE_DP_FLOPS    = 14,
    GPU_ROOFLINE_FLOPS       = 15,
    GPU_ROOFLINE_HP_FLOPS    = 16,
    GPU_ROOFLINE_SP_FLOPS    = 17,
    KERNEL_MODE_TIME         = 18,
    LIKWID_MARKER            = 19,
    LIKWID_NVMARKER          = 20,
    MALLOC_GOTCHA            = 21,
    MONOTONIC_CLOCK          = 22,
    MONOTONIC_RAW_CLOCK      = 23,
    USER_MPIP_BUNDLE         = 24,
    NUM_IO_IN                = 25,
    NUM_IO_OUT               = 26,
    NUM_MAJOR_PAGE_FAULTS    = 27,
    NUM_MINOR_PAGE_FAULTS    = 28,
    NUM_MSG_RECV             = 29,
    NUM_MSG_SENT             = 30,
    NUM_SIGNALS              = 31,
    NUM_SWAP                 = 32,
    NVTX_MARKER              = 33,
    USER_OMPT_BUNDLE         = 34,
    PAGE_RSS                 = 35,
    PAPI_ARRAY               = 36,
    PAPI_VECTOR              = 37,
    PEAK_RSS                 = 38,
    PRIORITY_CONTEXT_SWITCH  = 39,
    PROCESS_CPU_CLOCK        = 40,
    PROCESS_CPU_UTIL         = 41,
    READ_BYTES               = 42,
    STACK_RSS                = 43,
    SYS_CLOCK                = 44,
    TAU_MARKER               = 45,
    THREAD_CPU_CLOCK         = 46,
    THREAD_CPU_UTIL          = 47,
    TRIP_COUNT               = 48,
    USER_CLOCK               = 49,
    USER_GLOBAL_BUNDLE       = 50,
    USER_LIST_BUNDLE         = 51,
    USER_MODE_TIME           = 52,
    USER_TUPLE_BUNDLE        = 53,
    VIRTUAL_MEMORY           = 54,
    VOLUNTARY_CONTEXT_SWITCH = 55,
    VTUNE_EVENT              = 56,
    VTUNE_FRAME              = 57,
    VTUNE_PROFILER           = 58,
    WALL_CLOCK               = 59,
    WRITTEN_BYTES            = 60,
    TIMEMORY_USER_COMPONENT_ENUM TIMEMORY_COMPONENTS_END =
        (TIMEMORY_COMPONENT_ENUM_SIZE + TIMEMORY_USER_COMPONENT_ENUM_SIZE)
};

typedef int TIMEMORY_COMPONENT;
