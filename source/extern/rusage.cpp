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
#define TIMEMORY_USE_UNMAINTAINED_RUSAGE

#include "timemory/components.hpp"
#include "timemory/manager.hpp"
#include "timemory/mpl/operations.hpp"
#include "timemory/plotting.hpp"
#include "timemory/utility/bits/storage.hpp"
#include "timemory/utility/macros.hpp"
#include "timemory/utility/serializer.hpp"
#include "timemory/utility/singleton.hpp"
#include "timemory/utility/utility.hpp"

//======================================================================================//

TIMEMORY_INSTANTIATE_EXTERN_OPERATIONS(component::peak_rss, true)
TIMEMORY_INSTANTIATE_EXTERN_OPERATIONS(component::page_rss, true)

#if defined(_UNIX)
TIMEMORY_INSTANTIATE_EXTERN_OPERATIONS(component::stack_rss, true)
TIMEMORY_INSTANTIATE_EXTERN_OPERATIONS(component::data_rss, true)
TIMEMORY_INSTANTIATE_EXTERN_OPERATIONS(component::num_io_in, true)
TIMEMORY_INSTANTIATE_EXTERN_OPERATIONS(component::num_io_out, true)
TIMEMORY_INSTANTIATE_EXTERN_OPERATIONS(component::num_major_page_faults, true)
TIMEMORY_INSTANTIATE_EXTERN_OPERATIONS(component::num_minor_page_faults, true)
TIMEMORY_INSTANTIATE_EXTERN_OPERATIONS(component::num_msg_recv, true)
TIMEMORY_INSTANTIATE_EXTERN_OPERATIONS(component::num_msg_sent, true)
TIMEMORY_INSTANTIATE_EXTERN_OPERATIONS(component::num_signals, true)
TIMEMORY_INSTANTIATE_EXTERN_OPERATIONS(component::num_swap, true)
TIMEMORY_INSTANTIATE_EXTERN_OPERATIONS(component::voluntary_context_switch, true)
TIMEMORY_INSTANTIATE_EXTERN_OPERATIONS(component::priority_context_switch, true)
TIMEMORY_INSTANTIATE_EXTERN_OPERATIONS(component::read_bytes, true)
TIMEMORY_INSTANTIATE_EXTERN_OPERATIONS(component::written_bytes, true)
TIMEMORY_INSTANTIATE_EXTERN_OPERATIONS(component::virtual_memory, true)
TIMEMORY_INSTANTIATE_EXTERN_OPERATIONS(component::user_mode_time, true)
TIMEMORY_INSTANTIATE_EXTERN_OPERATIONS(component::kernel_mode_time, true)
TIMEMORY_INSTANTIATE_EXTERN_OPERATIONS(component::current_peak_rss, true)
#endif

//======================================================================================//

namespace tim
{
TIMEMORY_INSTANTIATE_EXTERN_INIT(peak_rss)
TIMEMORY_INSTANTIATE_EXTERN_INIT(page_rss)
TIMEMORY_INSTANTIATE_EXTERN_INIT(stack_rss)
TIMEMORY_INSTANTIATE_EXTERN_INIT(data_rss)
TIMEMORY_INSTANTIATE_EXTERN_INIT(num_io_in)
TIMEMORY_INSTANTIATE_EXTERN_INIT(num_io_out)
TIMEMORY_INSTANTIATE_EXTERN_INIT(num_major_page_faults)
TIMEMORY_INSTANTIATE_EXTERN_INIT(num_minor_page_faults)
TIMEMORY_INSTANTIATE_EXTERN_INIT(num_msg_recv)
TIMEMORY_INSTANTIATE_EXTERN_INIT(num_msg_sent)
TIMEMORY_INSTANTIATE_EXTERN_INIT(num_signals)
TIMEMORY_INSTANTIATE_EXTERN_INIT(num_swap)
TIMEMORY_INSTANTIATE_EXTERN_INIT(voluntary_context_switch)
TIMEMORY_INSTANTIATE_EXTERN_INIT(priority_context_switch)
TIMEMORY_INSTANTIATE_EXTERN_INIT(read_bytes)
TIMEMORY_INSTANTIATE_EXTERN_INIT(written_bytes)
TIMEMORY_INSTANTIATE_EXTERN_INIT(virtual_memory)
TIMEMORY_INSTANTIATE_EXTERN_INIT(user_mode_time)
TIMEMORY_INSTANTIATE_EXTERN_INIT(kernel_mode_time)
TIMEMORY_INSTANTIATE_EXTERN_INIT(current_peak_rss)

namespace component
{
//
//
template struct base<peak_rss>;
template struct base<page_rss>;
template struct base<stack_rss>;
template struct base<data_rss>;
template struct base<num_swap>;
template struct base<num_io_in>;
template struct base<num_io_out>;
template struct base<num_minor_page_faults>;
template struct base<num_major_page_faults>;
template struct base<num_msg_sent>;
template struct base<num_msg_recv>;
template struct base<num_signals>;
template struct base<voluntary_context_switch>;
template struct base<priority_context_switch>;
template struct base<read_bytes, std::tuple<int64_t, int64_t>>;
template struct base<written_bytes, std::array<int64_t, 2>>;
template struct base<virtual_memory>;
template struct base<user_mode_time>;
template struct base<kernel_mode_time>;
template struct base<current_peak_rss, std::pair<int64_t, int64_t>>;
//
//
}  // namespace component
}  // namespace tim
