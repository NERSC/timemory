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

#define TIMEMORY_BUILD_EXTERN_INIT
#define TIMEMORY_BUILD_EXTERN_TEMPLATE

#include "timemory/components.hpp"
#include "timemory/manager.hpp"
#include "timemory/mpl/operations.hpp"
#include "timemory/plotting.hpp"
#include "timemory/utility/bits/storage.hpp"
#include "timemory/utility/macros.hpp"
#include "timemory/utility/serializer.hpp"
#include "timemory/utility/singleton.hpp"
#include "timemory/utility/utility.hpp"

namespace tim
{
TIMEMORY_INSTANTIATE_EXTERN_INIT(trip_count)
TIMEMORY_INSTANTIATE_EXTERN_INIT(gperf_cpu_profiler)
TIMEMORY_INSTANTIATE_EXTERN_INIT(gperf_heap_profiler)

namespace component
{
//
//
template struct base<trip_count>;
template struct base<gperf_cpu_profiler, void>;
template struct base<gperf_heap_profiler, void>;
//
//
}  // namespace component

namespace operation
{
//
//
template struct init_storage<component::trip_count>;
template struct construct<component::trip_count>;
template struct set_prefix<component::trip_count>;
template struct insert_node<component::trip_count>;
template struct pop_node<component::trip_count>;
template struct record<component::trip_count>;
template struct reset<component::trip_count>;
template struct measure<component::trip_count>;
template struct sample<component::trip_count>;
template struct start<component::trip_count>;
template struct priority_start<component::trip_count>;
template struct standard_start<component::trip_count>;
template struct delayed_start<component::trip_count>;
template struct stop<component::trip_count>;
template struct priority_stop<component::trip_count>;
template struct standard_stop<component::trip_count>;
template struct delayed_stop<component::trip_count>;
template struct mark_begin<component::trip_count>;
template struct mark_end<component::trip_count>;
template struct audit<component::trip_count>;
template struct plus<component::trip_count>;
template struct minus<component::trip_count>;
template struct multiply<component::trip_count>;
template struct divide<component::trip_count>;
template struct get_data<component::trip_count>;
template struct copy<component::trip_count>;
//
//
}  // namespace operation

}  // namespace tim
