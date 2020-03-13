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
 * \file timemory/components/roofline/properties.hpp
 * \brief Specialization of the properties for the roofline components
 */

#pragma once

#include "timemory/components/macros.hpp"
#include "timemory/enum.h"

//======================================================================================//
//
TIMEMORY_PROPERTY_SPECIALIZATION(cpu_roofline_dp_flops, CPU_ROOFLINE_DP_FLOPS,
                                 "cpu_roofline_dp_flops", "cpu_roofline_dp",
                                 "cpu_roofline_double")

TIMEMORY_PROPERTY_SPECIALIZATION(cpu_roofline_flops, CPU_ROOFLINE_FLOPS,
                                 "cpu_roofline_flops", "cpu_roofline")

TIMEMORY_PROPERTY_SPECIALIZATION(cpu_roofline_sp_flops, CPU_ROOFLINE_SP_FLOPS,
                                 "cpu_roofline_sp_flops", "cpu_roofline_sp",
                                 "cpu_roofline_single")

TIMEMORY_PROPERTY_SPECIALIZATION(gpu_roofline_dp_flops, GPU_ROOFLINE_DP_FLOPS,
                                 "gpu_roofline_dp_flops", "gpu_roofline_dp",
                                 "gpu_roofline_double")

TIMEMORY_PROPERTY_SPECIALIZATION(gpu_roofline_flops, GPU_ROOFLINE_FLOPS,
                                 "gpu_roofline_flops", "gpu_roofline")

TIMEMORY_PROPERTY_SPECIALIZATION(gpu_roofline_hp_flops, GPU_ROOFLINE_HP_FLOPS,
                                 "gpu_roofline_hp_flops", "gpu_roofline_hp",
                                 "gpu_roofline_half")

TIMEMORY_PROPERTY_SPECIALIZATION(gpu_roofline_sp_flops, GPU_ROOFLINE_SP_FLOPS,
                                 "gpu_roofline_sp_flops", "gpu_roofline_sp",
                                 "gpu_roofline_single")
//
//======================================================================================//
