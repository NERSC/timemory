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

#include "timemory/components/base.hpp"
#include "timemory/components/rusage/components.hpp"
//
#include "timemory/mpl/types.hpp"
//
#include "timemory/manager/declaration.hpp"
#include "timemory/environment/declaration.hpp"
#include "timemory/settings/declaration.hpp"
#include "timemory/storage/declaration.hpp"
#include "timemory/operations/declaration.hpp"
#include "timemory/plotting/declaration.hpp"
#include "timemory/storage/declaration.hpp"
//
#include "timemory/operations/types/record.hpp"

//======================================================================================//
//
namespace tim
{
namespace component
{
//
TIMEMORY_EXTERN_TEMPLATE(struct base<peak_rss>)
TIMEMORY_EXTERN_TEMPLATE(struct base<page_rss>)
TIMEMORY_EXTERN_TEMPLATE(struct base<stack_rss>)
TIMEMORY_EXTERN_TEMPLATE(struct base<data_rss>)
TIMEMORY_EXTERN_TEMPLATE(struct base<num_swap>)
TIMEMORY_EXTERN_TEMPLATE(struct base<num_io_in>)
TIMEMORY_EXTERN_TEMPLATE(struct base<num_io_out>)
TIMEMORY_EXTERN_TEMPLATE(struct base<num_minor_page_faults>)
TIMEMORY_EXTERN_TEMPLATE(struct base<num_major_page_faults>)
TIMEMORY_EXTERN_TEMPLATE(struct base<num_msg_sent>)
TIMEMORY_EXTERN_TEMPLATE(struct base<num_msg_recv>)
TIMEMORY_EXTERN_TEMPLATE(struct base<num_signals>)
TIMEMORY_EXTERN_TEMPLATE(struct base<voluntary_context_switch>)
TIMEMORY_EXTERN_TEMPLATE(struct base<priority_context_switch>)
TIMEMORY_EXTERN_TEMPLATE(struct base<read_bytes, std::tuple<int64_t, int64_t>>)
TIMEMORY_EXTERN_TEMPLATE(struct base<written_bytes, std::array<int64_t, 2>>)
TIMEMORY_EXTERN_TEMPLATE(struct base<virtual_memory>)
TIMEMORY_EXTERN_TEMPLATE(struct base<user_mode_time>)
TIMEMORY_EXTERN_TEMPLATE(struct base<kernel_mode_time>)
TIMEMORY_EXTERN_TEMPLATE(struct base<current_peak_rss, std::pair<int64_t, int64_t>>)
//
}  // namespace component
}  // namespace tim
//
//======================================================================================//
