//  MIT License
//
//  Copyright (c) 2019, The Regents of the University of California,
//  through Lawrence Berkeley National Laboratory (subject to receipt of any
//  required approvals from the U.S. Dept. of Energy).  All rights reserved.
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to deal
//  in the Software without restriction, including without limitation the rights
//  to use, copy, modify, merge, publish, distribute, sublicense, and
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in all
//  copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//  SOFTWARE.

/** \file rusage.hpp
 * \headerfile rusage.hpp "timemory/rusage.hpp"
 * This headerfile provides the implementation for reading resource usage (rusage)
 * metrics
 *
 */

#pragma once

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <ios>
#include <iostream>
#include <stdio.h>
#include <string>

#include "timemory/macros.hpp"

//======================================================================================//

#if defined(_UNIX)
#    include <sys/resource.h>
#    include <unistd.h>
#    if defined(_MACOS)
#        include <mach/mach.h>
#    endif
#elif defined(_WINDOWS)
#    if !defined(NOMINMAX)
#        define NOMINMAX
#    endif
// currently, this is causing a bunch of errors, need to disable
// #    include <psapi.h>
#    include <stdio.h>
#    include <windows.h>
#else
#    error "Cannot define get_peak_rss() or get_current_rss() for an unknown OS."
#endif

// RSS - Resident set size (physical memory use, not in swap)

//--------------------------------------------------------------------------------------//

namespace tim
{
//--------------------------------------------------------------------------------------//

#if defined(_UNIX)
using rusage_type_t = decltype(RUSAGE_SELF);
inline rusage_type_t&
get_rusage_type()
{
    static auto instance = RUSAGE_SELF;
    return instance;
}
#endif

int64_t
get_peak_rss();
int64_t
get_current_rss();
int64_t
get_stack_rss();
int64_t
get_data_rss();
int64_t
get_num_swap();
int64_t
get_num_io_in();
int64_t
get_num_io_out();
int64_t
get_num_minor_page_faults();
int64_t
get_num_major_page_faults();
int64_t
get_num_messages_sent();
int64_t
get_num_messages_received();
int64_t
get_num_signals();
int64_t
get_num_voluntary_context_switch();
int64_t
get_num_priority_context_switch();

//--------------------------------------------------------------------------------------//

}  // namespace tim

//--------------------------------------------------------------------------------------//

#include "timemory/impl/rusage.icpp"
