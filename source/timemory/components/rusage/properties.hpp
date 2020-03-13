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
 * \file timemory/components/rusage/properties.hpp
 * \brief Specialization of the properties for the rusage components
 *
 */

#pragma once

#include "timemory/components/macros.hpp"
#include "timemory/components/rusage/types.hpp"
#include "timemory/enum.h"

//======================================================================================//
//
// TIMEMORY_PROPERTY_SPECIALIZATION(example, EXAMPLE, "timemory_example", ...)
//--------------------------------------------------------------------------------------//

TIMEMORY_PROPERTY_SPECIALIZATION(peak_rss, PEAK_RSS, "peak_rss", "")

TIMEMORY_PROPERTY_SPECIALIZATION(page_rss, PAGE_RSS, "page_rss", "")

TIMEMORY_PROPERTY_SPECIALIZATION(stack_rss, STACK_RSS, "stack_rss", "")

TIMEMORY_PROPERTY_SPECIALIZATION(data_rss, DATA_RSS, "data_rss", "")

TIMEMORY_PROPERTY_SPECIALIZATION(num_swap, NUM_SWAP, "num_swap", "")

TIMEMORY_PROPERTY_SPECIALIZATION(num_io_in, NUM_IO_IN, "num_io_in", "")

TIMEMORY_PROPERTY_SPECIALIZATION(num_io_out, NUM_IO_OUT, "num_io_out", "")

TIMEMORY_PROPERTY_SPECIALIZATION(num_minor_page_faults, NUM_MINOR_PAGE_FAULTS,
                                 "num_minor_page_faults", "")

TIMEMORY_PROPERTY_SPECIALIZATION(num_major_page_faults, NUM_MAJOR_PAGE_FAULTS,
                                 "num_major_page_faults", "")

TIMEMORY_PROPERTY_SPECIALIZATION(num_msg_sent, NUM_MSG_SENT, "num_msg_sent", "")

TIMEMORY_PROPERTY_SPECIALIZATION(num_msg_recv, NUM_MSG_RECV, "num_msg_recv", "")

TIMEMORY_PROPERTY_SPECIALIZATION(num_signals, NUM_SIGNALS, "num_signals", "")

TIMEMORY_PROPERTY_SPECIALIZATION(voluntary_context_switch, VOLUNTARY_CONTEXT_SWITCH,
                                 "voluntary_context_switch", "")

TIMEMORY_PROPERTY_SPECIALIZATION(priority_context_switch, PRIORITY_CONTEXT_SWITCH,
                                 "priority_context_switch", "")

TIMEMORY_PROPERTY_SPECIALIZATION(virtual_memory, VIRTUAL_MEMORY, "virtual_memory", "")

TIMEMORY_PROPERTY_SPECIALIZATION(read_bytes, READ_BYTES, "read_bytes", "")

TIMEMORY_PROPERTY_SPECIALIZATION(written_bytes, WRITTEN_BYTES, "written_bytes",
                                 "write_bytes")

TIMEMORY_PROPERTY_SPECIALIZATION(user_mode_time, USER_MODE_TIME, "user_mode_time", "")

TIMEMORY_PROPERTY_SPECIALIZATION(kernel_mode_time, KERNEL_MODE_TIME, "kernel_mode_time",
                                 "")

TIMEMORY_PROPERTY_SPECIALIZATION(current_peak_rss, CURRENT_PEAK_RSS, "current_peak_rss",
                                 "")
//
//======================================================================================//
