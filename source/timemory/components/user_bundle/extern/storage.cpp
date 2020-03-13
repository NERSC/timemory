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

#include "timemory/components/types.hpp"
#include "timemory/manager/declaration.hpp"
#include "timemory/mpl/operations.hpp"
#include "timemory/mpl/types.hpp"
#include "timemory/plotting/definition.hpp"
#include "timemory/storage/definition.hpp"

//======================================================================================//
// clang-format off
//
#include "timemory/components/user_bundle/components.hpp"
#include "timemory/components/user_bundle/extern/storage.hpp"
//
// clang-format on
//======================================================================================//

/*
#define TIMEMORY_INSTANTIATE_INSERT(TYPE)                                                \
    namespace tim                                                                        \
    {                                                                                    \
    namespace impl                                                                       \
    {                                                                                    \
    template typename storage<TYPE, true>::iterator                                      \
    storage<TYPE, true>::insert<scope::flat>(uint64_t, const TYPE&, uint64_t);           \
    template typename storage<TYPE, true>::iterator                                      \
    storage<TYPE, true>::insert<scope::tree>(uint64_t, const TYPE&, uint64_t);           \
    template typename storage<TYPE, true>::iterator                                      \
    storage<TYPE, true>::insert<scope::timeline>(uint64_t, const TYPE&, uint64_t);       \
    }                                                                                    \
    }

TIMEMORY_INSTANTIATE_INSERT(component::peak_rss)
TIMEMORY_INSTANTIATE_INSERT(component::page_rss)
TIMEMORY_INSTANTIATE_INSERT(component::stack_rss)
TIMEMORY_INSTANTIATE_INSERT(component::data_rss)
TIMEMORY_INSTANTIATE_INSERT(component::num_io_in)
TIMEMORY_INSTANTIATE_INSERT(component::num_io_out)
TIMEMORY_INSTANTIATE_INSERT(component::num_major_page_faults)
TIMEMORY_INSTANTIATE_INSERT(component::num_minor_page_faults)
TIMEMORY_INSTANTIATE_INSERT(component::num_msg_recv)
TIMEMORY_INSTANTIATE_INSERT(component::num_msg_sent)
TIMEMORY_INSTANTIATE_INSERT(component::num_signals)
TIMEMORY_INSTANTIATE_INSERT(component::num_swap)
TIMEMORY_INSTANTIATE_INSERT(component::voluntary_context_switch)
TIMEMORY_INSTANTIATE_INSERT(component::priority_context_switch)
TIMEMORY_INSTANTIATE_INSERT(component::read_bytes)
TIMEMORY_INSTANTIATE_INSERT(component::written_bytes)
TIMEMORY_INSTANTIATE_INSERT(component::virtual_memory)
TIMEMORY_INSTANTIATE_INSERT(component::user_mode_time)
TIMEMORY_INSTANTIATE_INSERT(component::kernel_mode_time)
TIMEMORY_INSTANTIATE_INSERT(component::current_peak_rss)

TIMEMORY_INSTANTIATE_INSERT(component::real_clock)
TIMEMORY_INSTANTIATE_INSERT(component::system_clock)
TIMEMORY_INSTANTIATE_INSERT(component::user_clock)
TIMEMORY_INSTANTIATE_INSERT(component::cpu_clock)
TIMEMORY_INSTANTIATE_INSERT(component::cpu_util)
TIMEMORY_INSTANTIATE_INSERT(component::monotonic_clock)
TIMEMORY_INSTANTIATE_INSERT(component::monotonic_raw_clock)
TIMEMORY_INSTANTIATE_INSERT(component::thread_cpu_clock)
TIMEMORY_INSTANTIATE_INSERT(component::thread_cpu_util)
TIMEMORY_INSTANTIATE_INSERT(component::process_cpu_clock)
TIMEMORY_INSTANTIATE_INSERT(component::process_cpu_util)
*/
