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

#include "timemory/compat/macros.h"

//======================================================================================//
//
//      Enumeration
//
//======================================================================================//
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
/// \macro TIMEMORY_COMPONENT_ENUM_SIZE
/// \brief The number of enumerated components defined by timemory
//
#if !defined(TIMEMORY_COMPONENT_ENUM_SIZE)
#    define TIMEMORY_COMPONENT_ENUM_SIZE 68
#endif
//
/// \enum TIMEMORY_NATIVE_COMPONENT
/// \brief Enumerated identifiers for timemory-provided components. If the user wishes
/// to add to the enumerated components, use \ref TIMEMORY_USER_COMPONENT_ENUM
//
enum TIMEMORY_NATIVE_COMPONENT
{
    ALLINEA_MAP              = 0,
    CALIPER                  = 1,
    CPU_CLOCK                = 2,
    CPU_ROOFLINE_DP_FLOPS    = 3,
    CPU_ROOFLINE_FLOPS       = 4,
    CPU_ROOFLINE_SP_FLOPS    = 5,
    CPU_UTIL                 = 6,
    CRAYPAT_COUNTERS         = 7,
    CRAYPAT_FLUSH_BUFFER     = 8,
    CRAYPAT_HEAP_STATS       = 9,
    CRAYPAT_RECORD           = 10,
    CRAYPAT_REGION           = 11,
    CUDA_EVENT               = 12,
    CUDA_PROFILER            = 13,
    CUPTI_ACTIVITY           = 14,
    CUPTI_COUNTERS           = 15,
    CURRENT_PEAK_RSS         = 16,
    DATA_RSS                 = 17,
    GPERFTOOLS_CPU_PROFILER  = 18,
    GPERFTOOLS_HEAP_PROFILER = 19,
    GPU_ROOFLINE_DP_FLOPS    = 20,
    GPU_ROOFLINE_FLOPS       = 21,
    GPU_ROOFLINE_HP_FLOPS    = 22,
    GPU_ROOFLINE_SP_FLOPS    = 23,
    KERNEL_MODE_TIME         = 24,
    LIKWID_MARKER            = 25,
    LIKWID_NVMARKER          = 26,
    MALLOC_GOTCHA            = 27,
    MONOTONIC_CLOCK          = 28,
    MONOTONIC_RAW_CLOCK      = 29,
    NUM_IO_IN                = 30,
    NUM_IO_OUT               = 31,
    NUM_MAJOR_PAGE_FAULTS    = 32,
    NUM_MINOR_PAGE_FAULTS    = 33,
    NUM_MSG_RECV             = 34,
    NUM_MSG_SENT             = 35,
    NUM_SIGNALS              = 36,
    NUM_SWAP                 = 37,
    NVTX_MARKER              = 38,
    OMPT_HANDLE              = 39,
    PAGE_RSS                 = 40,
    PAPI_ARRAY               = 41,
    PAPI_VECTOR              = 42,
    PEAK_RSS                 = 43,
    PRIORITY_CONTEXT_SWITCH  = 44,
    PROCESS_CPU_CLOCK        = 45,
    PROCESS_CPU_UTIL         = 46,
    READ_BYTES               = 47,
    STACK_RSS                = 48,
    SYS_CLOCK                = 49,
    TAU_MARKER               = 50,
    THREAD_CPU_CLOCK         = 51,
    THREAD_CPU_UTIL          = 52,
    TRIP_COUNT               = 53,
    USER_CLOCK               = 54,
    USER_GLOBAL_BUNDLE       = 55,
    USER_LIST_BUNDLE         = 56,
    USER_MODE_TIME           = 57,
    USER_MPIP_BUNDLE         = 58,
    USER_OMPT_BUNDLE         = 59,
    USER_TUPLE_BUNDLE        = 60,
    VIRTUAL_MEMORY           = 61,
    VOLUNTARY_CONTEXT_SWITCH = 62,
    VTUNE_EVENT              = 63,
    VTUNE_FRAME              = 64,
    VTUNE_PROFILER           = 65,
    WALL_CLOCK               = 66,
    WRITTEN_BYTES            = 67,
    TIMEMORY_USER_COMPONENT_ENUM TIMEMORY_COMPONENTS_END =
        (TIMEMORY_COMPONENT_ENUM_SIZE + TIMEMORY_USER_COMPONENT_ENUM_SIZE)
};

typedef int TIMEMORY_COMPONENT;
