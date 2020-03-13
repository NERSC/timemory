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
 * \file timemory/components/timing/properties.hpp
 * \brief Specialization of the properties for the components
 *
 */

#pragma once

#include "timemory/components/macros.hpp"
#include "timemory/components/timing/types.hpp"
#include "timemory/enum.h"

//======================================================================================//
//
TIMEMORY_PROPERTY_SPECIALIZATION(wall_clock, WALL_CLOCK, "wall_clock", "real_clock",
                                 "virtual_clock")

TIMEMORY_PROPERTY_SPECIALIZATION(system_clock, SYS_CLOCK, "system_clock", "sys_clock")

TIMEMORY_PROPERTY_SPECIALIZATION(user_clock, USER_CLOCK, "user_clock", "")

TIMEMORY_PROPERTY_SPECIALIZATION(cpu_clock, CPU_CLOCK, "cpu_clock", "")

TIMEMORY_PROPERTY_SPECIALIZATION(monotonic_clock, MONOTONIC_CLOCK, "monotonic_clock", "")

TIMEMORY_PROPERTY_SPECIALIZATION(monotonic_raw_clock, MONOTONIC_RAW_CLOCK,
                                 "monotonic_raw_clock", "")

TIMEMORY_PROPERTY_SPECIALIZATION(thread_cpu_clock, THREAD_CPU_CLOCK, "thread_cpu_clock",
                                 "")
TIMEMORY_PROPERTY_SPECIALIZATION(process_cpu_clock, PROCESS_CPU_CLOCK,
                                 "process_cpu_clock", "")

TIMEMORY_PROPERTY_SPECIALIZATION(cpu_util, CPU_UTIL, "cpu_util", "")

TIMEMORY_PROPERTY_SPECIALIZATION(process_cpu_util, PROCESS_CPU_UTIL, "process_cpu_util",
                                 "")

TIMEMORY_PROPERTY_SPECIALIZATION(thread_cpu_util, THREAD_CPU_UTIL, "thread_cpu_util", "")
//
//======================================================================================//
