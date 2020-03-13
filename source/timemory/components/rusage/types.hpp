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
 * \file timemory/components/rusage/types.hpp
 * \brief Declare the rusage component types
 */

#pragma once

#include "timemory/components/macros.hpp"
// #include "timemory/components/timing/types.hpp"

//======================================================================================//
//
TIMEMORY_DECLARE_COMPONENT(peak_rss)
TIMEMORY_DECLARE_COMPONENT(page_rss)
TIMEMORY_DECLARE_COMPONENT(stack_rss)
TIMEMORY_DECLARE_COMPONENT(data_rss)
TIMEMORY_DECLARE_COMPONENT(num_swap)
TIMEMORY_DECLARE_COMPONENT(num_io_in)
TIMEMORY_DECLARE_COMPONENT(num_io_out)
TIMEMORY_DECLARE_COMPONENT(num_minor_page_faults)
TIMEMORY_DECLARE_COMPONENT(num_major_page_faults)
TIMEMORY_DECLARE_COMPONENT(num_msg_sent)
TIMEMORY_DECLARE_COMPONENT(num_msg_recv)
TIMEMORY_DECLARE_COMPONENT(num_signals)
TIMEMORY_DECLARE_COMPONENT(voluntary_context_switch)
TIMEMORY_DECLARE_COMPONENT(priority_context_switch)
TIMEMORY_DECLARE_COMPONENT(virtual_memory)
TIMEMORY_DECLARE_COMPONENT(read_bytes)
TIMEMORY_DECLARE_COMPONENT(written_bytes)
TIMEMORY_DECLARE_COMPONENT(user_mode_time)
TIMEMORY_DECLARE_COMPONENT(kernel_mode_time)
TIMEMORY_DECLARE_COMPONENT(current_peak_rss)
//
//======================================================================================//

#include "timemory/components/rusage/properties.hpp"
#include "timemory/components/rusage/traits.hpp"
