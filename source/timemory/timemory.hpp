//  MIT License
//
//  Copyright (c) 2019, The Regents of the University of California,
// through Lawrence Berkeley National Laboratory (subject to receipt of any
// required approvals from the U.S. Dept. of Energy).  All rights reserved.
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to
//  deal in the Software without restriction, including without limitation the
//  rights to use, copy, modify, merge, publish, distribute, sublicense, and
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in
//  all copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
//  IN THE SOFTWARE.

/** \file timemory.hpp
 * \headerfile timemory.hpp "timemory/timemory.hpp"
 * All-inclusive timemory header
 *
 */

#pragma once

#include "timemory/components.hpp"
#include "timemory/manager.hpp"
#include "timemory/settings.hpp"
#include "timemory/units.hpp"
#include "timemory/utility/macros.hpp"
#include "timemory/utility/utility.hpp"
#include "timemory/variadic/auto_list.hpp"
#include "timemory/variadic/auto_timer.hpp"
#include "timemory/variadic/macros.hpp"

#if defined(TIMEMORY_EXTERN_INIT)
#    include "timemory/utility/storage.hpp"
#endif

#include "timemory/ctimemory.h"

//======================================================================================//

#if defined(TIMEMORY_EXTERN_TEMPLATES)
#    include "timemory/templates/native_extern.hpp"
#endif

//======================================================================================//

#if defined(TIMEMORY_EXTERN_TEMPLATES) && defined(TIMEMORY_USE_CUDA)
// not yet implemented
// #    include "timemory/templates/cuda_extern.hpp"
#endif

//======================================================================================//

#if defined(TIMEMORY_EXTERN_INIT)

TIMEMORY_DECLARE_EXTERN_STORAGE(real_clock)
TIMEMORY_DECLARE_EXTERN_STORAGE(system_clock)
TIMEMORY_DECLARE_EXTERN_STORAGE(user_clock)
TIMEMORY_DECLARE_EXTERN_STORAGE(cpu_clock)
TIMEMORY_DECLARE_EXTERN_STORAGE(monotonic_clock)
TIMEMORY_DECLARE_EXTERN_STORAGE(monotonic_raw_clock)
TIMEMORY_DECLARE_EXTERN_STORAGE(thread_cpu_clock)
TIMEMORY_DECLARE_EXTERN_STORAGE(process_cpu_clock)
TIMEMORY_DECLARE_EXTERN_STORAGE(cpu_util)
TIMEMORY_DECLARE_EXTERN_STORAGE(thread_cpu_util)
TIMEMORY_DECLARE_EXTERN_STORAGE(process_cpu_util)
TIMEMORY_DECLARE_EXTERN_STORAGE(current_rss)
TIMEMORY_DECLARE_EXTERN_STORAGE(peak_rss)
TIMEMORY_DECLARE_EXTERN_STORAGE(stack_rss)
TIMEMORY_DECLARE_EXTERN_STORAGE(data_rss)
TIMEMORY_DECLARE_EXTERN_STORAGE(num_swap)
TIMEMORY_DECLARE_EXTERN_STORAGE(num_io_in)
TIMEMORY_DECLARE_EXTERN_STORAGE(num_io_out)
TIMEMORY_DECLARE_EXTERN_STORAGE(num_minor_page_faults)
TIMEMORY_DECLARE_EXTERN_STORAGE(num_major_page_faults)
TIMEMORY_DECLARE_EXTERN_STORAGE(num_msg_sent)
TIMEMORY_DECLARE_EXTERN_STORAGE(num_msg_recv)
TIMEMORY_DECLARE_EXTERN_STORAGE(num_signals)
TIMEMORY_DECLARE_EXTERN_STORAGE(voluntary_context_switch)
TIMEMORY_DECLARE_EXTERN_STORAGE(priority_context_switch)

#endif

//======================================================================================//
