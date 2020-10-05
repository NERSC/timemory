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
 * \file timemory/components/trip_count/types.hpp
 * \brief Declare the trip_count component types
 */

#pragma once

#include "timemory/components/macros.hpp"
#include "timemory/enum.h"
#include "timemory/mpl/type_traits.hpp"
#include "timemory/mpl/types.hpp"

/// \struct trip_count
/// \brief Records the number of invocations. This is the most lightweight metric
/// available since it only increments an integer and never records any statistics.
/// If dynamic instrumentation is used and the overhead is significant, it is recommended
/// to set this as the only component (-d trip_count) and then use the regex exclude
/// option (-E) to remove any non-critical function calls which have very high
/// trip-counts.
TIMEMORY_DECLARE_COMPONENT(trip_count)
//
TIMEMORY_SET_COMPONENT_API(component::trip_count, project::timemory, os::agnostic)
//
TIMEMORY_DEFINE_CONCRETE_TRAIT(custom_laps_printing, component::trip_count, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(report_mean, component::trip_count, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(report_self, component::trip_count, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(report_units, component::trip_count, false_type)
//
TIMEMORY_PROPERTY_SPECIALIZATION(trip_count, TRIP_COUNT, "trip_count", "")
