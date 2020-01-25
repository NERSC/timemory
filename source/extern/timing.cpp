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

// #define TIMEMORY_BUILD_EXTERN_INIT
// #define TIMEMORY_BUILD_EXTERN_TEMPLATE

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

namespace operation
{
//
//
template struct init_storage<component::wall_clock>;
template struct construct<component::wall_clock>;
template struct set_prefix<component::wall_clock>;
template struct insert_node<component::wall_clock>;
template struct pop_node<component::wall_clock>;
template struct record<component::wall_clock>;
template struct reset<component::wall_clock>;
template struct measure<component::wall_clock>;
template struct sample<component::wall_clock>;
template struct start<component::wall_clock>;
template struct priority_start<component::wall_clock>;
template struct standard_start<component::wall_clock>;
template struct delayed_start<component::wall_clock>;
template struct stop<component::wall_clock>;
template struct priority_stop<component::wall_clock>;
template struct standard_stop<component::wall_clock>;
template struct delayed_stop<component::wall_clock>;
template struct mark_begin<component::wall_clock>;
template struct mark_end<component::wall_clock>;
template struct audit<component::wall_clock>;
template struct plus<component::wall_clock>;
template struct minus<component::wall_clock>;
template struct multiply<component::wall_clock>;
template struct divide<component::wall_clock>;
template struct get_data<component::wall_clock>;
template struct copy<component::wall_clock>;
template struct print_statistics<component::wall_clock>;
template struct print_header<component::wall_clock>;
template struct print<component::wall_clock>;
template struct print_storage<component::wall_clock>;
template struct echo_measurement<component::wall_clock, true>;
template struct finalize::storage::get<component::wall_clock, true>;
template struct finalize::storage::mpi_get<component::wall_clock, true>;
template struct finalize::storage::upc_get<component::wall_clock, true>;
template struct finalize::storage::dmp_get<component::wall_clock, true>;

template struct init_storage<component::cpu_clock>;
template struct construct<component::cpu_clock>;
template struct set_prefix<component::cpu_clock>;
template struct insert_node<component::cpu_clock>;
template struct pop_node<component::cpu_clock>;
template struct record<component::cpu_clock>;
template struct reset<component::cpu_clock>;
template struct measure<component::cpu_clock>;
template struct sample<component::cpu_clock>;
template struct start<component::cpu_clock>;
template struct priority_start<component::cpu_clock>;
template struct standard_start<component::cpu_clock>;
template struct delayed_start<component::cpu_clock>;
template struct stop<component::cpu_clock>;
template struct priority_stop<component::cpu_clock>;
template struct standard_stop<component::cpu_clock>;
template struct delayed_stop<component::cpu_clock>;
template struct mark_begin<component::cpu_clock>;
template struct mark_end<component::cpu_clock>;
template struct audit<component::cpu_clock>;
template struct plus<component::cpu_clock>;
template struct minus<component::cpu_clock>;
template struct multiply<component::cpu_clock>;
template struct divide<component::cpu_clock>;
template struct get_data<component::cpu_clock>;
template struct copy<component::cpu_clock>;
template struct print_statistics<component::cpu_clock>;
template struct print_header<component::cpu_clock>;
template struct print<component::cpu_clock>;
template struct print_storage<component::cpu_clock>;
template struct echo_measurement<component::cpu_clock, true>;
template struct finalize::storage::get<component::cpu_clock, true>;
template struct finalize::storage::mpi_get<component::cpu_clock, true>;
template struct finalize::storage::upc_get<component::cpu_clock, true>;
template struct finalize::storage::dmp_get<component::cpu_clock, true>;
//
//
}  // namespace operation

}  // namespace tim
