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

namespace tim
{
TIMEMORY_INSTANTIATE_EXTERN_INIT(real_clock)
TIMEMORY_INSTANTIATE_EXTERN_INIT(system_clock)
TIMEMORY_INSTANTIATE_EXTERN_INIT(user_clock)
TIMEMORY_INSTANTIATE_EXTERN_INIT(cpu_clock)
TIMEMORY_INSTANTIATE_EXTERN_INIT(cpu_util)
TIMEMORY_INSTANTIATE_EXTERN_INIT(monotonic_clock)
TIMEMORY_INSTANTIATE_EXTERN_INIT(monotonic_raw_clock)
TIMEMORY_INSTANTIATE_EXTERN_INIT(thread_cpu_clock)
TIMEMORY_INSTANTIATE_EXTERN_INIT(thread_cpu_util)
TIMEMORY_INSTANTIATE_EXTERN_INIT(process_cpu_clock)
TIMEMORY_INSTANTIATE_EXTERN_INIT(process_cpu_util)

namespace component
{
//
//
template struct base<real_clock>;
template struct base<system_clock>;
template struct base<user_clock>;
template struct base<cpu_clock>;
template struct base<monotonic_clock>;
template struct base<monotonic_raw_clock>;
template struct base<thread_cpu_clock>;
template struct base<process_cpu_clock>;
template struct base<cpu_util, std::pair<int64_t, int64_t>>;
template struct base<process_cpu_util, std::pair<int64_t, int64_t>>;
template struct base<thread_cpu_util, std::pair<int64_t, int64_t>>;
//
//
}  // namespace component
}  // namespace tim
