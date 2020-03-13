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
 * \file timemory/components/roofline/types.hpp
 * \brief Declare the roofline component types
 */

#pragma once

#include "timemory/components/cuda/backends.hpp"
#include "timemory/components/macros.hpp"

//======================================================================================//
//
TIMEMORY_DECLARE_TEMPLATE_COMPONENT(cpu_roofline, typename... Types)
TIMEMORY_DECLARE_TEMPLATE_COMPONENT(gpu_roofline, typename... Types)

TIMEMORY_COMPONENT_ALIAS(cpu_roofline_sp_flops, cpu_roofline<float>)
TIMEMORY_COMPONENT_ALIAS(cpu_roofline_dp_flops, cpu_roofline<double>)
TIMEMORY_COMPONENT_ALIAS(cpu_roofline_flops, cpu_roofline<float, double>)

TIMEMORY_COMPONENT_ALIAS(gpu_roofline_hp_flops, gpu_roofline<cuda::fp16_t>)
TIMEMORY_COMPONENT_ALIAS(gpu_roofline_sp_flops, gpu_roofline<float>)
TIMEMORY_COMPONENT_ALIAS(gpu_roofline_dp_flops, gpu_roofline<double>)

#if !defined(TIMEMORY_DISABLE_CUDA_HALF)
TIMEMORY_COMPONENT_ALIAS(gpu_roofline_flops, gpu_roofline<cuda::fp16_t, float, double>)
#else
TIMEMORY_COMPONENT_ALIAS(gpu_roofline_flops, gpu_roofline<float, double>)
#endif
//
//======================================================================================//

#include "timemory/components/roofline/properties.hpp"
#include "timemory/components/roofline/traits.hpp"
