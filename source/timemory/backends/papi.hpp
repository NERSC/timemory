//  MIT License
//
//  Copyright (c) 2020, The Regents of the University of California,
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

#pragma once

#include "timemory/defines.h"

#if defined(TIMEMORY_USE_EXTERN) && !defined(TIMEMORY_USE_CORE_EXTERN)
#    define TIMEMORY_USE_CORE_EXTERN
#endif

#include "timemory/backends/hardware_counters.hpp"
#include "timemory/backends/types/papi.hpp"
#include "timemory/utility/macros.hpp"
#include "timemory/utility/utility.hpp"

#include <cstdint>
#include <string>
#include <thread>
#include <vector>

#if defined(TIMEMORY_UNIX)
#    include <pthread.h>
#    include <unistd.h>
#endif

#if defined(TIMEMORY_USE_PAPI)
#    include <papiStdEventDefs.h>
//
#    include <papi.h>
#    if defined(TIMEMORY_UNIX)
#        include <pthread.h>
#    endif
#endif

namespace tim
{
namespace papi
{
using event_info_t     = PAPI_event_info_t;
using ulong_t          = unsigned long int;
using hwcounter_info_t = std::vector<hardware_counters::info>;

ulong_t
get_thread_index();

uint64_t
get_papi_thread_num();

void
check_papi_thread();

std::thread::id
get_tid();

std::thread::id
get_master_tid();

bool
is_master_thread();

bool&
working();

bool
check(int retval, const std::string& mesg, bool quiet = false);

void
set_debug(int level);

std::string
as_string(int* events, long long* values, int num_events, const std::string& indent);

void
register_thread();

void
unregister_thread();

int
get_event_code(const std::string& event_code_str);

std::string
get_event_code_name(int event_code);

event_info_t
get_event_info(int evt_type);

void
init();

void
shutdown();

void
print_hw_info();

void
enable_multiplexing(int event_set, int component = 0);

void
create_event_set(int* event_set, bool enable_multiplex);

void
destroy_event_set(int event_set);

void
start(int event_set);

void
stop(int event_set, long long* values);

void
read(int event_set, long long* values);

void
write(int event_set, long long* values);

void
accum(int event_set, long long* values);

void
reset(int event_set);

bool
add_event(int event_set, int event);

bool
remove_event(int event_set, int event);

void
add_events(int event_set, int* events, int number);

void
remove_events(int event_set, int* events, int number);

void
assign_event_set_component(int event_set, int cidx);

void
attach(int event_set, unsigned long tid);

void
detach(int event_set);

bool
query_event(int event);

hwcounter_info_t
available_events_info();

hardware_counters::info
get_hwcounter_info(const std::string& event_code_str);

namespace details
{
//
void
init_threading();

void
init_multiplexing();

void
init_library();
//
}  // namespace details

template <typename Tp>
inline void
attach(int event_set, Tp pid_or_tid)
{
    // inform PAPI that a previously registered thread is disappearing
#if defined(TIMEMORY_USE_PAPI)
    int retval = PAPI_attach(event_set, pid_or_tid);
    working()  = check(retval, "Warning!! Failure attaching to event set");
#else
    consume_parameters(event_set, pid_or_tid);
#endif
}

template <typename Func>
inline bool
overflow(int evt_set, int evt_code, int threshold, int flags, Func&& handler)
{
#if defined(TIMEMORY_USE_PAPI)
    return (PAPI_overflow(evt_set, evt_code, threshold, flags,
                          std::forward<Func>(handler)) == PAPI_OK);
#else
    consume_parameters(evt_set, evt_code, threshold, flags, handler);
    return false;
#endif
}
}  // namespace papi
}  // namespace tim

#if !defined(TIMEMORY_CORE_SOURCE) && !defined(TIMEMORY_USE_CORE_EXTERN)
#    include "timemory/backends/papi.cpp"
#endif
