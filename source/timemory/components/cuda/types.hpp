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

/**
 * \file timemory/components/cuda/types.hpp
 * \brief Declare the cuda component types
 */

#pragma once

#include "timemory/components/macros.hpp"
#include "timemory/enum.h"
#include "timemory/mpl/type_traits.hpp"
#include "timemory/mpl/types.hpp"

//======================================================================================//
//
TIMEMORY_DECLARE_COMPONENT(cuda_event)
TIMEMORY_DECLARE_COMPONENT(cuda_profiler)
TIMEMORY_DECLARE_COMPONENT(nvtx_marker)
TIMEMORY_COMPONENT_ALIAS(cuda_nvtx, nvtx_marker)
//
//======================================================================================//
//
//                              TYPE-TRAITS
//
//======================================================================================//
//
//                              STATISTICS
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_STATISTICS_TYPE(component::cuda_event, float)
//
//--------------------------------------------------------------------------------------//
//
//                              IS TIMING CATEGORY
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_timing_category, component::cuda_event, true_type)
//
//--------------------------------------------------------------------------------------//
//
//                              USES TIMING UNITS
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_DEFINE_CONCRETE_TRAIT(uses_timing_units, component::cuda_event, true_type)
//
//--------------------------------------------------------------------------------------//
//
//                              START PRIORITY
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_DEFINE_CONCRETE_TRAIT(start_priority, component::cuda_event,
                               priority_constant<128>)
//
//--------------------------------------------------------------------------------------//
//
//                              STOP PRIORITY
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_DEFINE_CONCRETE_TRAIT(stop_priority, component::cuda_event,
                               priority_constant<-128>)
//
//--------------------------------------------------------------------------------------//
//
//                              REQUIRES PREFIX
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_DEFINE_CONCRETE_TRAIT(requires_prefix, component::nvtx_marker, true_type)
//
//--------------------------------------------------------------------------------------//
//
//                              IS AVAILABLE
//
//--------------------------------------------------------------------------------------//
//
#if !defined(TIMEMORY_USE_CUDA)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::cuda_event, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::cuda_profiler, false_type)
#endif
//
#if !defined(TIMEMORY_USE_NVTX) || !defined(TIMEMORY_USE_CUDA)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::nvtx_marker, false_type)
#endif
//
//--------------------------------------------------------------------------------------//
//
//                              MISCELLANEOUS
//
//--------------------------------------------------------------------------------------//
//
namespace tim
{
namespace trait
{
//
//--------------------------------------------------------------------------------------//
//
template <>
struct data<component::cuda_event>
{
    using value_type = float;
};
//
//--------------------------------------------------------------------------------------//
//
template <>
struct data<component::cuda_profiler>
{
    using value_type = void;
};
//
//--------------------------------------------------------------------------------------//
//
template <>
struct data<component::nvtx_marker>
{
    using value_type = void;
};
//
//--------------------------------------------------------------------------------------//
//
template <>
struct collects_data<component::cuda_event>
{
    using type                  = component::cuda_event;
    using value_type            = float;
    static constexpr bool value = is_available<component::cuda_event>::value;
};
//
//--------------------------------------------------------------------------------------//
//
template <>
struct collects_data<component::cuda_profiler>
{
    using type                  = component::cuda_profiler;
    using value_type            = void;
    static constexpr bool value = false;
};
//
//--------------------------------------------------------------------------------------//
//
template <>
struct collects_data<component::nvtx_marker>
{
    using type                  = component::nvtx_marker;
    using value_type            = void;
    static constexpr bool value = false;
};
//
//--------------------------------------------------------------------------------------//
//
}  // namespace trait
}  // namespace tim
//
//--------------------------------------------------------------------------------------//
//
//
//======================================================================================//
//
TIMEMORY_PROPERTY_SPECIALIZATION(cuda_event, CUDA_EVENT, "cuda_event", "")
//
TIMEMORY_PROPERTY_SPECIALIZATION(cuda_profiler, CUDA_PROFILER, "cuda_profiler", "")
//
TIMEMORY_PROPERTY_SPECIALIZATION(nvtx_marker, NVTX_MARKER, "nvtx_marker", "nvtx")
