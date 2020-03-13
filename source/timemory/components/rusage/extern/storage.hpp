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
//
#include "timemory/environment/declaration.hpp"
#include "timemory/settings/declaration.hpp"
#include "timemory/storage/declaration.hpp"

//======================================================================================//
//
TIMEMORY_EXTERN_STORAGE(component::peak_rss, peak_rss)
TIMEMORY_EXTERN_STORAGE(component::page_rss, page_rss)
TIMEMORY_EXTERN_STORAGE(component::stack_rss, stack_rss)
TIMEMORY_EXTERN_STORAGE(component::data_rss, data_rss)
TIMEMORY_EXTERN_STORAGE(component::num_io_in, num_io_in)
TIMEMORY_EXTERN_STORAGE(component::num_io_out, num_io_out)
TIMEMORY_EXTERN_STORAGE(component::num_major_page_faults, num_major_page_faults)
TIMEMORY_EXTERN_STORAGE(component::num_minor_page_faults, num_minor_page_faults)
TIMEMORY_EXTERN_STORAGE(component::num_msg_recv, num_msg_recv)
TIMEMORY_EXTERN_STORAGE(component::num_msg_sent, num_msg_sent)
TIMEMORY_EXTERN_STORAGE(component::num_signals, num_signals)
TIMEMORY_EXTERN_STORAGE(component::num_swap, num_swap)
TIMEMORY_EXTERN_STORAGE(component::voluntary_context_switch, voluntary_context_switch)
TIMEMORY_EXTERN_STORAGE(component::priority_context_switch, priority_context_switch)
TIMEMORY_EXTERN_STORAGE(component::read_bytes, read_bytes)
TIMEMORY_EXTERN_STORAGE(component::written_bytes, written_bytes)
TIMEMORY_EXTERN_STORAGE(component::virtual_memory, virtual_memory)
TIMEMORY_EXTERN_STORAGE(component::user_mode_time, user_mode_time)
TIMEMORY_EXTERN_STORAGE(component::kernel_mode_time, kernel_mode_time)
TIMEMORY_EXTERN_STORAGE(component::current_peak_rss, current_peak_rss)
//
//======================================================================================//
