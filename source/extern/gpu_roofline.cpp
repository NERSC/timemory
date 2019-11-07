// MIT License
//
// Copyright (c) 2019, The Regents of the University of California,
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

#define TIMEMORY_BUILD_EXTERN_INIT
#define TIMEMORY_BUILD_EXTERN_TEMPLATE

#include "timemory/components.hpp"
#include "timemory/manager.hpp"
#include "timemory/utility/macros.hpp"
#include "timemory/utility/serializer.hpp"
#include "timemory/utility/singleton.hpp"
#include "timemory/utility/utility.hpp"

#if defined(TIMEMORY_USE_CUPTI)

namespace tim
{
TIMEMORY_INSTANTIATE_EXTERN_INIT(gpu_roofline_flops)
TIMEMORY_INSTANTIATE_EXTERN_INIT(gpu_roofline_hp_flops)
TIMEMORY_INSTANTIATE_EXTERN_INIT(gpu_roofline_sp_flops)
TIMEMORY_INSTANTIATE_EXTERN_INIT(gpu_roofline_dp_flops)

namespace component
{
//
//
template struct base<
    gpu_roofline<float, double>,
    std::tuple<typename cupti_activity::value_type, typename cupti_counters::value_type>,
    policy::global_init, policy::global_finalize, policy::thread_init,
    policy::thread_finalize, policy::global_finalize, policy::serialization>;

template struct base<
    gpu_roofline<float>,
    std::tuple<typename cupti_activity::value_type, typename cupti_counters::value_type>,
    policy::global_init, policy::global_finalize, policy::thread_init,
    policy::thread_finalize, policy::global_finalize, policy::serialization>;

template struct base<
    gpu_roofline<double>,
    std::tuple<typename cupti_activity::value_type, typename cupti_counters::value_type>,
    policy::global_init, policy::global_finalize, policy::thread_init,
    policy::thread_finalize, policy::global_finalize, policy::serialization>;

template struct gpu_roofline<float, double>;

template struct gpu_roofline<float>;

template struct gpu_roofline<double>;

#    if defined(TIMEMORY_CUDA_FP16)

template struct base<
    gpu_roofline<cuda::fp16_t, float, double>,
    std::tuple<typename cupti_activity::value_type, typename cupti_counters::value_type>,
    policy::global_init, policy::global_finalize, policy::thread_init,
    policy::thread_finalize, policy::global_finalize, policy::serialization>;

template struct base<
    gpu_roofline<cuda::fp16_t>,
    std::tuple<typename cupti_activity::value_type, typename cupti_counters::value_type>,
    policy::global_init, policy::global_finalize, policy::thread_init,
    policy::thread_finalize, policy::global_finalize, policy::serialization>;

template struct gpu_roofline<cuda::fp16_t, float, double>;

template struct gpu_roofline<cuda::fp16_t>;

#    endif
//
//
}  // namespace component
}  // namespace tim

#endif
