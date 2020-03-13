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

#pragma once

#include "timemory/components/macros.hpp"
#include "timemory/components/timing/components.hpp"
#include "timemory/storage/definition.hpp"
//
#include "timemory/environment/declaration.hpp"
#include "timemory/settings/declaration.hpp"
#include "timemory/storage/declaration.hpp"

//======================================================================================//
//
TIMEMORY_EXTERN_STORAGE(component::real_clock, real_clock)
TIMEMORY_EXTERN_STORAGE(component::system_clock, system_clock)
TIMEMORY_EXTERN_STORAGE(component::user_clock, user_clock)
TIMEMORY_EXTERN_STORAGE(component::cpu_clock, cpu_clock)
TIMEMORY_EXTERN_STORAGE(component::cpu_util, cpu_util)
TIMEMORY_EXTERN_STORAGE(component::monotonic_clock, monotonic_clock)
TIMEMORY_EXTERN_STORAGE(component::monotonic_raw_clock, monotonic_raw_clock)
TIMEMORY_EXTERN_STORAGE(component::thread_cpu_clock, thread_cpu_clock)
TIMEMORY_EXTERN_STORAGE(component::thread_cpu_util, thread_cpu_util)
TIMEMORY_EXTERN_STORAGE(component::process_cpu_clock, process_cpu_clock)
TIMEMORY_EXTERN_STORAGE(component::process_cpu_util, process_cpu_util)
//
//======================================================================================//
