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

/** \file bits/ctimemory.h
 * \headerfile bits/ctimemory.h "timemory/bits/ctimemory.h"
 * This provides the core enumeration for components
 *
 */

#pragma once

enum TIMEMORY_COMPONENT
{
    CALIPER                  = 0,
    CPU_CLOCK                = 1,
    CPU_ROOFLINE_DP_FLOPS    = 2,
    CPU_ROOFLINE_FLOPS       = 3,
    CPU_ROOFLINE_SP_FLOPS    = 4,
    CPU_UTIL                 = 5,
    CUDA_EVENT               = 6,
    CUPTI_ACTIVITY           = 7,
    CUPTI_COUNTERS           = 8,
    DATA_RSS                 = 9,
    GPERF_CPU_PROFILER       = 10,
    GPERF_HEAP_PROFILER      = 11,
    GPU_ROOFLINE_DP_FLOPS    = 12,
    GPU_ROOFLINE_FLOPS       = 13,
    GPU_ROOFLINE_HP_FLOPS    = 14,
    GPU_ROOFLINE_SP_FLOPS    = 15,
    MONOTONIC_CLOCK          = 16,
    MONOTONIC_RAW_CLOCK      = 17,
    NUM_IO_IN                = 18,
    NUM_IO_OUT               = 19,
    NUM_MAJOR_PAGE_FAULTS    = 20,
    NUM_MINOR_PAGE_FAULTS    = 21,
    NUM_MSG_RECV             = 22,
    NUM_MSG_SENT             = 23,
    NUM_SIGNALS              = 24,
    NUM_SWAP                 = 25,
    NVTX_MARKER              = 26,
    PAGE_RSS                 = 27,
    PAPI_ARRAY               = 28,
    PEAK_RSS                 = 29,
    PRIORITY_CONTEXT_SWITCH  = 30,
    PROCESS_CPU_CLOCK        = 31,
    PROCESS_CPU_UTIL         = 32,
    READ_BYTES               = 33,
    WALL_CLOCK               = 34,
    STACK_RSS                = 35,
    SYS_CLOCK                = 36,
    THREAD_CPU_CLOCK         = 37,
    THREAD_CPU_UTIL          = 38,
    TRIP_COUNT               = 39,
    USER_CLOCK               = 40,
    VIRTUAL_MEMORY           = 41,
    VOLUNTARY_CONTEXT_SWITCH = 42,
    WRITTEN_BYTES            = 43,
    TIMEMORY_COMPONENTS_END  = 44
};
