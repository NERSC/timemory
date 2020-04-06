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
 * \file timemory/components/timing/extern.hpp
 * \brief Include the extern declarations for timing components
 */

#pragma once

//======================================================================================//
//
#include "timemory/components/base.hpp"
#include "timemory/components/macros.hpp"
//
#include "timemory/components/timing/components.hpp"
#include "timemory/components/timing/types.hpp"
//
#if defined(TIMEMORY_COMPONENT_SOURCE) ||                                                \
    (!defined(TIMEMORY_USE_EXTERN) && !defined(TIMEMORY_USE_COMPONENT_EXTERN))
// source/header-only requirements
#    include "timemory/environment/declaration.hpp"
#    include "timemory/operations/definition.hpp"
#    include "timemory/plotting/definition.hpp"
#    include "timemory/settings/declaration.hpp"
#    include "timemory/storage/definition.hpp"
#else
// extern requirements
#    include "timemory/environment/declaration.hpp"
#    include "timemory/operations/definition.hpp"
#    include "timemory/plotting/declaration.hpp"
#    include "timemory/settings/declaration.hpp"
#    include "timemory/storage/declaration.hpp"
#endif
//
//======================================================================================//
//
namespace tim
{
namespace component
{
//
TIMEMORY_EXTERN_TEMPLATE(struct base<wall_clock>)
TIMEMORY_EXTERN_TEMPLATE(struct base<system_clock>)
TIMEMORY_EXTERN_TEMPLATE(struct base<user_clock>)
TIMEMORY_EXTERN_TEMPLATE(struct base<cpu_clock>)
TIMEMORY_EXTERN_TEMPLATE(struct base<monotonic_clock>)
TIMEMORY_EXTERN_TEMPLATE(struct base<monotonic_raw_clock>)
TIMEMORY_EXTERN_TEMPLATE(struct base<thread_cpu_clock>)
TIMEMORY_EXTERN_TEMPLATE(struct base<process_cpu_clock>)
TIMEMORY_EXTERN_TEMPLATE(struct base<cpu_util, std::pair<int64_t, int64_t>>)
TIMEMORY_EXTERN_TEMPLATE(struct base<process_cpu_util, std::pair<int64_t, int64_t>>)
TIMEMORY_EXTERN_TEMPLATE(struct base<thread_cpu_util, std::pair<int64_t, int64_t>>)
//
}  // namespace component
}  // namespace tim
//
//======================================================================================//
//
TIMEMORY_EXTERN_OPERATIONS(component::wall_clock, true)
TIMEMORY_EXTERN_OPERATIONS(component::system_clock, true)
TIMEMORY_EXTERN_OPERATIONS(component::user_clock, true)
TIMEMORY_EXTERN_OPERATIONS(component::cpu_clock, true)
TIMEMORY_EXTERN_OPERATIONS(component::monotonic_clock, true)
TIMEMORY_EXTERN_OPERATIONS(component::monotonic_raw_clock, true)
TIMEMORY_EXTERN_OPERATIONS(component::thread_cpu_clock, true)
TIMEMORY_EXTERN_OPERATIONS(component::process_cpu_clock, true)
TIMEMORY_EXTERN_OPERATIONS(component::cpu_util, true)
TIMEMORY_EXTERN_OPERATIONS(component::process_cpu_util, true)
TIMEMORY_EXTERN_OPERATIONS(component::thread_cpu_util, true)
//
//======================================================================================//
//
TIMEMORY_EXTERN_STORAGE(component::wall_clock, wall_clock)
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
