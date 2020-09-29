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
/// \code{.cpp}
/// #define TIMEMORY_USER_COMPONENT_ENUM MY_COMPONENT,
/// \endcode
//
#if !defined(TIMEMORY_USER_COMPONENT_ENUM)
#    define TIMEMORY_USER_COMPONENT_ENUM
#endif

/// \enum TIMEMORY_USER_COMPONENT_ENUM_SIZE
/// \brief Macro specifying how many user component enumerations are provided
#if !defined(TIMEMORY_USER_COMPONENT_ENUM_SIZE)
#    define TIMEMORY_USER_COMPONENT_ENUM_SIZE 16
#endif
//
/// \enum TIMEMORY_NATIVE_COMPONENT
/// \brief Enumerated identifiers for timemory-provided components. If the user wishes
/// to add to the enumerated components, use \ref TIMEMORY_USER_COMPONENT_ENUM
//
enum TIMEMORY_NATIVE_COMPONENT
{
    ALLINEA_MAP = 0,
    CALIPER_MARKER,
    CALIPER_CONFIG,
    CALIPER_LOOP_MARKER,
    CPU_CLOCK,
    CPU_ROOFLINE_DP_FLOPS,
    CPU_ROOFLINE_FLOPS,
    CPU_ROOFLINE_SP_FLOPS,
    CPU_UTIL,
    CRAYPAT_COUNTERS,
    CRAYPAT_FLUSH_BUFFER,
    CRAYPAT_HEAP_STATS,
    CRAYPAT_RECORD,
    CRAYPAT_REGION,
    CUDA_EVENT,
    CUDA_PROFILER,
    CUPTI_ACTIVITY,
    CUPTI_COUNTERS,
    CURRENT_PEAK_RSS,
    GPERFTOOLS_CPU_PROFILER,
    GPERFTOOLS_HEAP_PROFILER,
    GPU_ROOFLINE_DP_FLOPS,
    GPU_ROOFLINE_FLOPS,
    GPU_ROOFLINE_HP_FLOPS,
    GPU_ROOFLINE_SP_FLOPS,
    KERNEL_MODE_TIME,
    LIKWID_MARKER,
    LIKWID_NVMARKER,
    MALLOC_GOTCHA,
    MONOTONIC_CLOCK,
    MONOTONIC_RAW_CLOCK,
    NUM_IO_IN,
    NUM_IO_OUT,
    NUM_MAJOR_PAGE_FAULTS,
    NUM_MINOR_PAGE_FAULTS,
    NVTX_MARKER,
    OMPT_HANDLE,
    PAGE_RSS,
    PAPI_ARRAY,
    PAPI_VECTOR,
    PEAK_RSS,
    PRIORITY_CONTEXT_SWITCH,
    PROCESS_CPU_CLOCK,
    PROCESS_CPU_UTIL,
    READ_BYTES,
    READ_CHAR,
    SYS_CLOCK,
    TAU_MARKER,
    THREAD_CPU_CLOCK,
    THREAD_CPU_UTIL,
    TRIP_COUNT,
    USER_CLOCK,
    USER_GLOBAL_BUNDLE,
    USER_LIST_BUNDLE,
    USER_MODE_TIME,
    USER_MPIP_BUNDLE,
    USER_NCCLP_BUNDLE,
    USER_OMPT_BUNDLE,
    USER_TUPLE_BUNDLE,
    VIRTUAL_MEMORY,
    VOLUNTARY_CONTEXT_SWITCH,
    VTUNE_EVENT,
    VTUNE_FRAME,
    VTUNE_PROFILER,
    WALL_CLOCK,
    WRITTEN_BYTES,
    WRITTEN_CHAR,
    TIMEMORY_NATIVE_COMPONENTS_END,
    TIMEMORY_USER_COMPONENT_ENUM TIMEMORY_COMPONENTS_END =
        (TIMEMORY_NATIVE_COMPONENTS_END + TIMEMORY_USER_COMPONENT_ENUM_SIZE)
};
//
/// \macro TIMEMORY_NATIVE_COMPONENT_ENUM_SIZE
/// \brief The number of enumerated components natively defined by timemory
//
#if !defined(TIMEMORY_NATIVE_COMPONENT_ENUM_SIZE)
#    define TIMEMORY_NATIVE_COMPONENT_ENUM_SIZE TIMEMORY_NATIVE_COMPONENTS_END
#endif
//
//--------------------------------------------------------------------------------------//
//
typedef int TIMEMORY_COMPONENT;
//
#if !defined(CALIPER)
#    define CALIPER CALIPER_MARKER
#endif
//
//--------------------------------------------------------------------------------------//
//
/// \enum TIMEMORY_OPERATION
/// \brief Enumerated identifiers for subset of common operations for usage in C code
/// and specializations of \ref tim::trait::python_args.
enum TIMEMORY_OPERATION
{
    TIMEMORY_CONSTRUCT = 0,
    TIMEMORY_START,
    TIMEMORY_STOP,
    TIMEMORY_STORE,
    TIMEMORY_RECORD,
    TIMEMORY_MEASURE,
    TIMEMORY_MARK_BEGIN,
    TIMEMORY_MARK_END,
    TIMEMORY_OPERATION_END
};
//
//--------------------------------------------------------------------------------------//
//
