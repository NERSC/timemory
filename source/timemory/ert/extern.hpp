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
 * \headerfile "timemory/ert/extern.hpp"
 * Extern template declarations
 *
 */

#pragma once

#include "timemory/backends/device.hpp"
#include "timemory/components/timing/wall_clock.hpp"
#include "timemory/ert/configuration.hpp"
#include "timemory/ert/counter.hpp"
#include "timemory/ert/data.hpp"
#include "timemory/ert/macros.hpp"

#if defined(TIMEMORY_USE_CUDA)
#    include "timemory/components/cuda/backends.hpp"
#endif

namespace tim
{
namespace ert
{
//
//--------------------------------------------------------------------------------------//
//
//                   Don't enter this block while compiling CUDA
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_ERT_EXTERN_TEMPLATE_CXX(class exec_data<component::wall_clock>)
//
TIMEMORY_ERT_EXTERN_TEMPLATE_CXX(class counter<device::cpu, float, component::wall_clock>)
TIMEMORY_ERT_EXTERN_TEMPLATE_CXX(
    class counter<device::cpu, double, component::wall_clock>)
//
TIMEMORY_ERT_EXTERN_TEMPLATE_CXX(
    struct configuration<device::cpu, float, component::wall_clock>)
TIMEMORY_ERT_EXTERN_TEMPLATE_CXX(
    struct configuration<device::cpu, double, component::wall_clock>)
//
TIMEMORY_ERT_EXTERN_TEMPLATE_CXX(
    struct executor<device::cpu, float, component::wall_clock>)
TIMEMORY_ERT_EXTERN_TEMPLATE_CXX(
    struct executor<device::cpu, double, component::wall_clock>)
//
//--------------------------------------------------------------------------------------//
//
//                   Don't enter this block while compiling C++
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_ERT_EXTERN_TEMPLATE_CUDA(
    class counter<device::gpu, float, component::wall_clock>)
TIMEMORY_ERT_EXTERN_TEMPLATE_CUDA(
    class counter<device::gpu, double, component::wall_clock>)
//
#if defined(TIMEMORY_USE_CUDA_HALF)
TIMEMORY_ERT_EXTERN_TEMPLATE_CUDA(
    class counter<device::gpu, cuda::fp16_t, component::wall_clock>)
#endif
//
TIMEMORY_ERT_EXTERN_TEMPLATE_CUDA(
    struct configuration<device::gpu, float, component::wall_clock>)
TIMEMORY_ERT_EXTERN_TEMPLATE_CUDA(
    struct configuration<device::gpu, double, component::wall_clock>)
//
TIMEMORY_ERT_EXTERN_TEMPLATE_CUDA(
    struct executor<device::gpu, float, component::wall_clock>)
TIMEMORY_ERT_EXTERN_TEMPLATE_CUDA(
    struct executor<device::gpu, double, component::wall_clock>)
//
#if defined(TIMEMORY_USE_CUDA_HALF)
TIMEMORY_ERT_EXTERN_TEMPLATE_CUDA(
    struct configuration<device::gpu, cuda::fp16_t, component::wall_clock>)
//
TIMEMORY_ERT_EXTERN_TEMPLATE_CUDA(
    struct executor<device::gpu, cuda::fp16_t, component::wall_clock>)
#endif
//
}  // namespace ert
}  // namespace tim
