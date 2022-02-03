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

#include "timemory/defines.h"
#include "timemory/macros/compiler.hpp"

#if !defined(TIMEMORY_USE_CUDA) || !defined(TIMEMORY_USE_CUDA_HALF)
#    include "timemory/backends/bfloat16.hpp"
#    include "timemory/backends/fp16.hpp"
#endif

#if defined(TIMEMORY_USE_CUDA)
#    include <cuda.h>
#    include <cuda_runtime_api.h>
#    if defined(TIMEMORY_USE_CUDA_HALF)
#        include <cuda_bf16.h>
#        include <cuda_bf16.hpp>
#        include <cuda_fp16.h>
#        include <cuda_fp16.hpp>
#    endif
#endif

#include <cstdint>

namespace tim
{
namespace cuda
{
//
#if defined(TIMEMORY_USE_CUDA)
// define some types for when CUDA is enabled
using stream_t = cudaStream_t;
using event_t  = cudaEvent_t;
using error_t  = cudaError_t;
using memcpy_t = cudaMemcpyKind;
// define some values for when CUDA is enabled
static constexpr stream_t                default_stream_v   = 0;
static const decltype(cudaSuccess)       success_v          = cudaSuccess;
static const decltype(cudaErrorNotReady) err_not_ready_v    = cudaErrorNotReady;
static const cudaMemcpyKind              host_to_host_v     = cudaMemcpyHostToHost;
static const cudaMemcpyKind              host_to_device_v   = cudaMemcpyHostToDevice;
static const cudaMemcpyKind              device_to_host_v   = cudaMemcpyDeviceToHost;
static const cudaMemcpyKind              device_to_device_v = cudaMemcpyDeviceToDevice;
//
#else
//
// define some types for when CUDA is disabled
//
using stream_t = int;
using event_t  = int;
using error_t  = int;
using memcpy_t = int;
// define some values for when CUDA is disabled
static constexpr stream_t default_stream_v   = 0;
static const int          success_v          = 0;
static const int          err_not_ready_v    = 0;
static const int          host_to_host_v     = 0;
static const int          host_to_device_v   = 1;
static const int          device_to_host_v   = 2;
static const int          device_to_device_v = 3;
#endif

// half-precision floating point
#if defined(TIMEMORY_USE_CUDA) && defined(TIMEMORY_USE_CUDA_HALF)
//
using fp16_t = half2;
using bf16_t = nv_bfloat16;
//
#else
//
using fp16_t = numeric::half2;
using bf16_t = numeric::bfloat16;
//
#endif
//
}  // namespace cuda
}  // namespace tim
