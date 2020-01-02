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

/** \file timemory/backends.hpp
 * \headerfile timemory/backends.hpp "timemory/backends.hpp"
 * Generic header for backends
 *
 */

#pragma once

#include "timemory/backends/clocks.hpp"
#include "timemory/backends/device.hpp"
#include "timemory/backends/dmp.hpp"
#include "timemory/backends/rusage.hpp"
#include "timemory/backends/signals.hpp"

#if defined(TIMEMORY_USE_CALIPER)
#    include "timemory/backends/caliper.hpp"
#endif

#if defined(TIMEMORY_USE_GOTCHA)
#    include "timemory/backends/gotcha.hpp"
#endif

#if defined(TIMEMORY_USE_MPI)
#    include "timemory/backends/mpi.hpp"
#endif

#if defined(TIMEMORY_USE_UPCXX)
#    include "timemory/backends/upcxx.hpp"
#endif

#if defined(TIMEMORY_USE_PAPI)
#    include "timemory/backends/papi.hpp"
#endif

#if defined(TIMEMORY_USE_GPERF) || defined(TIMEMORY_USE_GPERF_HEAP_PROFILER) ||          \
    defined(TIMEMORY_USE_GPERF_CPU_PROFILER)
#    include "timemory/backends/gperf.hpp"
#endif

//--------------------------------------------------------------------------------------//
//
//      GPU backends
//
//--------------------------------------------------------------------------------------//

#if defined(TIMEMORY_USE_CUDA)
#    include "timemory/backends/cuda.hpp"
#endif

#if defined(TIMEMORY_USE_CUPTI)
#    include "timemory/backends/cupti.hpp"
#endif

#if defined(TIMEMORY_USE_NVTX)
#    include "timemory/backends/nvtx.hpp"
#endif
