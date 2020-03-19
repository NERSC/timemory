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

#include "timemory/components/macros.hpp"
#include "timemory/components/rusage/components.hpp"
#include "timemory/operations/definition.hpp"
#include "timemory/plotting/definition.hpp"

//======================================================================================//
//
TIMEMORY_EXTERN_OPERATIONS(component::peak_rss, true)
TIMEMORY_EXTERN_OPERATIONS(component::page_rss, true)

// #if defined(_UNIX)
TIMEMORY_EXTERN_OPERATIONS(component::stack_rss, true)
TIMEMORY_EXTERN_OPERATIONS(component::data_rss, true)
TIMEMORY_EXTERN_OPERATIONS(component::num_io_in, true)
TIMEMORY_EXTERN_OPERATIONS(component::num_io_out, true)
TIMEMORY_EXTERN_OPERATIONS(component::num_major_page_faults, true)
TIMEMORY_EXTERN_OPERATIONS(component::num_minor_page_faults, true)
TIMEMORY_EXTERN_OPERATIONS(component::num_msg_recv, true)
TIMEMORY_EXTERN_OPERATIONS(component::num_msg_sent, true)
TIMEMORY_EXTERN_OPERATIONS(component::num_signals, true)
TIMEMORY_EXTERN_OPERATIONS(component::num_swap, true)
TIMEMORY_EXTERN_OPERATIONS(component::voluntary_context_switch, true)
TIMEMORY_EXTERN_OPERATIONS(component::priority_context_switch, true)
TIMEMORY_EXTERN_OPERATIONS(component::read_bytes, true)
TIMEMORY_EXTERN_OPERATIONS(component::written_bytes, true)
TIMEMORY_EXTERN_OPERATIONS(component::virtual_memory, true)
TIMEMORY_EXTERN_OPERATIONS(component::user_mode_time, true)
TIMEMORY_EXTERN_OPERATIONS(component::kernel_mode_time, true)
TIMEMORY_EXTERN_OPERATIONS(component::current_peak_rss, true)
// #endif
//
//======================================================================================//
