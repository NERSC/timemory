//  MIT License
//
//  Copyright (c) 2020, The Regents of the University of California,
//  through Lawrence Berkeley National Laboratory (subject to receipt of any
//  required approvals from the U.S. Dept. of Energy).  All rights reserved.
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to deal
//  in the Software without restriction, including without limitation the rights
//  to use, copy, modify, merge, publish, distribute, sublicense, and
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in all
//  copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//  SOFTWARE.

/** \file timemory/components.hpp
 * \headerfile timemory/components.hpp "timemory/components.hpp"
 * These are core tools provided by TiMemory. These tools can be used individually
 * or bundled together in a component_tuple (C++) or component_list (C, Python)
 *
 */

#pragma once

// forward declare any types
#include "timemory/components/types.hpp"
#include "timemory/ert/types.hpp"
#include "timemory/mpl/types.hpp"
#include "timemory/variadic/types.hpp"

// general components
#include "timemory/components/general.hpp"
#include "timemory/components/rusage.hpp"
#include "timemory/components/timing.hpp"
#include "timemory/components/user_bundle.hpp"

// caliper components
#if defined(TIMEMORY_USE_CALIPER)
#    include "timemory/components/caliper.hpp"
#endif

// gotcha components
#if defined(TIMEMORY_USE_GOTCHA)
#    include "timemory/components/derived/malloc_gotcha.hpp"
#    include "timemory/components/gotcha.hpp"
#endif

// cuda event
#if defined(TIMEMORY_USE_CUDA)
#    include "timemory/components/cuda/event.hpp"
#    include "timemory/components/cuda/profiler.hpp"
#endif

// nvtx marker
#if defined(TIMEMORY_USE_NVTX)
#    include "timemory/components/cuda/nvtx_marker.hpp"
#endif

// likwid
#if defined(TIMEMORY_USE_LIKWID)
#    include "timemory/components/likwid.hpp"
#endif

// GPU hardware counter components
#if defined(TIMEMORY_USE_CUPTI)
#    include "timemory/components/cupti/activity.hpp"
#    include "timemory/components/cupti/counters.hpp"
#    include "timemory/components/roofline/gpu.hpp"
#endif

// CPU/GPU hardware counter components
#if defined(TIMEMORY_USE_PAPI)
#    include "timemory/components/papi/array.hpp"
#    include "timemory/components/papi/tuple.hpp"
#    include "timemory/components/roofline/cpu.hpp"
#endif

// TAU component
#if defined(TIMEMORY_USE_TAU)
#    include "timemory/components/tau.hpp"
#endif

// VTune components
#if defined(TIMEMORY_USE_VTUNE)
#    include "timemory/components/vtune/event.hpp"
#    include "timemory/components/vtune/frame.hpp"
#    include "timemory/components/vtune/profiler.hpp"
#endif

// OpenMP components
#if defined(TIMEMORY_USE_OPENMP)
#    include "timemory/components/openmp.hpp"
#endif

#include "timemory/backends/cuda.hpp"

// device backend
#include "timemory/backends/device.hpp"

//======================================================================================//
//
//      helpers for generating components
//
//======================================================================================//

// #include "timemory/runtime/enumerate.hpp"
// #include "timemory/runtime/configure.hpp"
// #include "timemory/runtime/insert.hpp"
// #include "timemory/runtime/initialize.hpp"

//======================================================================================//
//
//      default statistics (requires the components to be defined before implementing)
//
//======================================================================================//

#include "timemory/mpl/policy.hpp"

namespace tim
{
namespace policy
{
//--------------------------------------------------------------------------------------//
//
template <typename _Comp, typename _Tp>
inline void
record_statistics<_Comp, _Tp>::apply(statistics<_Tp>& _stat, const _Comp& _obj)
{
    using result_type = decltype(std::declval<_Comp>().get());
    static_assert(std::is_same<result_type, _Tp>::value,
                  "Error! The default implementation of "
                  "'policy::record_statistics<Component, T>::apply' requires 'T' to be "
                  "the same type as the return type from 'Component::get()'");

    _stat += _obj.get();
}

//--------------------------------------------------------------------------------------//
}  // namespace policy
}  // namespace tim
