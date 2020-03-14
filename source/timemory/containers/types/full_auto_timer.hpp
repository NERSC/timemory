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
 * \file timemory/containers/types/full_auto_timer.hpp
 * \brief Include the extern declarations for full_auto_timer in containers
 */

#pragma once

//======================================================================================//
//
#include "timemory/containers/macros.hpp"
//
#include "timemory/containers/types.hpp"
//
#include "timemory/containers/declaration.hpp"
//
#include "timemory/components/types.hpp"
//
#include "timemory/components.hpp"
//
#include "timemory/variadic/component_hybrid.hpp"
#include "timemory/variadic/component_list.hpp"
#include "timemory/variadic/component_tuple.hpp"
//
#include "timemory/variadic/auto_hybrid.hpp"
#include "timemory/variadic/auto_list.hpp"
#include "timemory/variadic/auto_tuple.hpp"
//
#include "timemory/runtime/configure.hpp"
#include "timemory/runtime/enumerate.hpp"
#include "timemory/runtime/initialize.hpp"
#include "timemory/runtime/properties.hpp"
//
//======================================================================================//
//
// clang-format off
//
TIMEMORY_EXTERN_TUPLE(full_auto_timer_t,
                      component::real_clock,
                      component::system_clock,
                      component::user_clock,
                      component::cpu_util,
                      component::peak_rss,
                      component::user_tuple_bundle)
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_EXTERN_LIST(full_auto_timer_t,
                     component::user_list_bundle,
                     component::gperf_cpu_profiler,
                     component::gperf_heap_profiler,
                     component::caliper,
                     component::tau_marker,
                     component::papi_array_t,
                     component::cpu_roofline_sp_flops,
                     component::cpu_roofline_dp_flops,
                     component::cuda_event,
                     component::nvtx_marker,
                     component::cupti_activity,
                     component::cupti_counters,
                     component::gpu_roofline_flops,
                     component::gpu_roofline_hp_flops,
                     component::gpu_roofline_sp_flops,
                     component::gpu_roofline_dp_flops)
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_EXTERN_HYBRID(full_auto_timer_t)
//
// clang-format on
//
//--------------------------------------------------------------------------------------//
//
