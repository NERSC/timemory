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
 * \file timemory/components/cupti/types.hpp
 * \brief Declare the cupti component types
 */

#pragma once

#include "timemory/components/macros.hpp"
#include "timemory/enum.h"
#include "timemory/mpl/type_traits.hpp"
#include "timemory/mpl/types.hpp"

//======================================================================================//
//
TIMEMORY_DECLARE_COMPONENT(cupti_activity)
TIMEMORY_DECLARE_COMPONENT(cupti_counters)
TIMEMORY_DECLARE_COMPONENT(cupti_profiler)
//
//======================================================================================//
//
//                              STATISTICS
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_STATISTICS_TYPE(component::cupti_activity, double)
TIMEMORY_STATISTICS_TYPE(component::cupti_counters, std::vector<double>)
TIMEMORY_STATISTICS_TYPE(component::cupti_profiler, std::vector<double>)
//
//--------------------------------------------------------------------------------------//
//
//                              IS AVAILABLE
//
//--------------------------------------------------------------------------------------//
//
#if !defined(TIMEMORY_USE_CUPTI) || !defined(TIMEMORY_USE_CUDA)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::cupti_counters, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::cupti_activity, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::cupti_profiler, false_type)
#endif
//
//--------------------------------------------------------------------------------------//
//
//                              IS TIMING CATEGORY
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_timing_category, component::cupti_activity, true_type)
//
//--------------------------------------------------------------------------------------//
//
//                              USES TIMING UNITS
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_DEFINE_CONCRETE_TRAIT(uses_timing_units, component::cupti_activity, true_type)
//
//--------------------------------------------------------------------------------------//
//
//                              SECONDARY DATA
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_DEFINE_CONCRETE_TRAIT(secondary_data, component::cupti_activity, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(secondary_data, component::cupti_counters, true_type)
//
//--------------------------------------------------------------------------------------//
//
//                              CUSTOM UNIT PRINTING
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_DEFINE_CONCRETE_TRAIT(custom_unit_printing, component::cupti_counters, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(custom_unit_printing, component::cupti_profiler, true_type)
//
//--------------------------------------------------------------------------------------//
//
//                              CUSTOM LABEL PRINTING
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_DEFINE_CONCRETE_TRAIT(custom_label_printing, component::cupti_counters,
                               true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(custom_label_printing, component::cupti_profiler,
                               true_type)
//
//--------------------------------------------------------------------------------------//
//
//                              ARRAY SERIALIZATION
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_DEFINE_CONCRETE_TRAIT(array_serialization, component::cupti_counters, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(array_serialization, component::cupti_profiler, true_type)
//
//--------------------------------------------------------------------------------------//
//
//                              CUSTOM SERIALIZATION
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_DEFINE_CONCRETE_TRAIT(custom_serialization, component::cupti_counters, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(custom_serialization, component::cupti_profiler, true_type)
//
//--------------------------------------------------------------------------------------//
//
//
//======================================================================================//
//
TIMEMORY_PROPERTY_SPECIALIZATION(cupti_activity, CUPTI_ACTIVITY, "cupti_activity", "")
//
TIMEMORY_PROPERTY_SPECIALIZATION(cupti_counters, CUPTI_COUNTERS, "cupti_counters", "")
//
// TIMEMORY_PROPERTY_SPECIALIZATION(cupti_profiler, CUPTI_PROFILER, "cupti_profiler", "")
