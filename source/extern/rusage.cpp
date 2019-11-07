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
template struct base<written_bytes, std::tuple<int64_t, int64_t>>;
template struct base<virtual_memory>;
//
//
}  // namespace component
}  // namespace tim
