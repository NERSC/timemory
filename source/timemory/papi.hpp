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

/** \file papi.hpp
 * \headerfile papi.hpp "timemory/papi.hpp"
 * Provides implementation of PAPI routines.
 *
 */

#pragma once

#include "timemory/macros.hpp"
#include "timemory/utility.hpp"

#include <cassert>
#include <cstdint>
#include <thread>

#if defined(_UNIX)
#    include <pthread.h>
#    include <unistd.h>
#endif

#if defined(TIMEMORY_USE_PAPI)
#    include <papi.h>
#    include <papiStdEventDefs.h>
#    if defined(_UNIX)
#        include <pthread.h>
#    endif
#else
#    include "timemory/impl/papi_defs.icpp"
#endif

//--------------------------------------------------------------------------------------//

namespace tim
{
//--------------------------------------------------------------------------------------//

namespace papi
{
//--------------------------------------------------------------------------------------//

using tid_t    = std::thread::id;
using string_t = std::string;

//--------------------------------------------------------------------------------------//

inline tid_t
get_tid()
{
    return std::this_thread::get_id();
}

//--------------------------------------------------------------------------------------//

inline tid_t
get_master_tid()
{
    static tid_t _instance = get_tid();
    return _instance;
}

//--------------------------------------------------------------------------------------//

inline bool&
working()
{
    static thread_local bool _instance = false;
    return _instance;
}

//--------------------------------------------------------------------------------------//

inline bool
check(int retval, const std::string& mesg)
{
    bool success = (retval == PAPI_OK);
    if(!success)
        std::cerr << mesg << " (error code = " << retval << ")" << std::endl;
    return success;
}

//--------------------------------------------------------------------------------------//

inline void
set_debug(int level)
{
    // set the current debug level for PAPI
#if defined(TIMEMORY_USE_PAPI)
    int retval = PAPI_set_debug(level);
    check(retval, "Warning!! Failure to set debug level");
#else
    consume_parameters(level);
#endif
}

//--------------------------------------------------------------------------------------//

inline string_t
as_string(int* events, long long* values, int num_events, const string_t& indent)
{
    std::stringstream ss;
#if defined(TIMEMORY_USE_PAPI)
    for(int i = 0; i < num_events; ++i)
    {
        PAPI_event_info_t evt_info;
        PAPI_get_event_info(events[i], &evt_info);
        char* description = evt_info.long_descr;
        ss << indent << description << " : " << values[i] << std::endl;
    }
#else
    consume_parameters(events, values, num_events, indent);
#endif
    return ss.str();
}

//--------------------------------------------------------------------------------------//

inline void
register_thread()
{
    // inform PAPI of the existence of a new thread
#if defined(TIMEMORY_USE_PAPI)
    // std::stringstream ss;
    // ss << std::this_thread::get_id();
    // PRINT_HERE(ss.str().c_str());
    int retval = PAPI_register_thread();
    working()  = check(retval, "Warning!! Failure registering thread");
#endif
}

//--------------------------------------------------------------------------------------//

inline void
unregister_thread()
{
    // inform PAPI that a previously registered thread is disappearing
#if defined(TIMEMORY_USE_PAPI)
    int retval = PAPI_unregister_thread();
    working()  = check(retval, "Warning!! Failure unregistering thread");
#endif
}

//--------------------------------------------------------------------------------------//

inline void
attach(int event_set)
{
    // inform PAPI that a previously registered thread is disappearing
#if defined(TIMEMORY_USE_PAPI) && defined(_UNIX)
    // int retval = PAPI_attach(event_set, getpid());
    int retval = PAPI_attach(event_set, pthread_self());
    working()  = check(retval, "Warning!! Failure attaching to event set");
#else
    consume_parameters(event_set);
#endif
}

//--------------------------------------------------------------------------------------//

inline void
init()
{
    // initialize the PAPI library
#if defined(TIMEMORY_USE_PAPI)
    if(!PAPI_is_initialized())
    {
        get_master_tid();
        {
            int events       = PAPI_L1_TCM;
            int num_events   = 1;
            int retval       = PAPI_start_counters(&events, num_events);
            working()        = (retval == PAPI_OK);
            long long values = 0;
            retval           = PAPI_stop_counters(&values, num_events);
            working()        = (retval == PAPI_OK);
        }
        if(!working())
        {
            int retval = PAPI_library_init(PAPI_VER_CURRENT);
            working()  = check(retval, "Warning!! Failure initializing PAPI");
        }
        // if(working())
        {
            int retval = PAPI_thread_init(pthread_self);
            working()  = check(retval, "Warning!! Failure thread init");
        }
    }
    register_thread();
#endif
}

//--------------------------------------------------------------------------------------//

inline void
shutdown()
{
    // finish using PAPI and free all related resources
#if defined(TIMEMORY_USE_PAPI)
    unregister_thread();
    if(get_tid() != get_master_tid())
        return;
    PAPI_shutdown();
    working() = false;
#endif
}

//--------------------------------------------------------------------------------------//

inline void
start(int event_set)
{
    // start counting hardware events in an event set
#if defined(TIMEMORY_USE_PAPI)
    int retval = PAPI_start(event_set);
    check(retval, "Warning!! Failure to start event set");
#else
    consume_parameters(event_set);
#endif
}

//--------------------------------------------------------------------------------------//

inline void
start_counters(int* events, int num_events)
{
    // start counting hardware events in an event set
#if defined(TIMEMORY_USE_PAPI)
    int retval = PAPI_start_counters(events, num_events);
    check(retval, "Warning!! Failure to start event counters");
#else
    consume_parameters(events, num_events);
#endif
}

//--------------------------------------------------------------------------------------//

inline void
stop(int event_set, long long* values)
{
    // stop counting hardware events in an event set and return current events
#if defined(TIMEMORY_USE_PAPI)
    int retval = PAPI_stop(event_set, values);
    check(retval, "Warning!! Failure to stop event set");
#else
    consume_parameters(event_set, values);
#endif
}

//--------------------------------------------------------------------------------------//

inline void
stop_counters(long long* values, int num_events)
{
    // stop counting hardware events in an event set and return current events
#if defined(TIMEMORY_USE_PAPI)
    int retval = PAPI_stop_counters(values, num_events);
    check(retval, "Warning!! Failure to stop counters");
#else
    consume_parameters(values, num_events);
#endif
}

//--------------------------------------------------------------------------------------//

inline void
read(int event_set, long long* values)
{
    // read hardware events from an event set with no reset
#if defined(TIMEMORY_USE_PAPI)
    int retval = PAPI_read(event_set, values);
    check(retval, "Warning!! Failure to read event set");
#else
    consume_parameters(event_set, values);
#endif
}

//--------------------------------------------------------------------------------------//

inline void
read_counters(long long* values, int num_events)
{
    // read hardware events from an event set with no reset
#if defined(TIMEMORY_USE_PAPI)
    int retval = PAPI_read_counters(values, num_events);
    check(retval, "Warning!! Failure to read event set");
#else
    consume_parameters(values, num_events);
#endif
}

//--------------------------------------------------------------------------------------//

inline void
write(int event_set, long long* values)
{
    // write counter values into counters
#if defined(TIMEMORY_USE_PAPI)
    int retval = PAPI_write(event_set, values);
    check(retval, "Warning!! Failure to write event set");
#else
    consume_parameters(event_set, values);
#endif
}

//--------------------------------------------------------------------------------------//

inline void
accum(int event_set, long long* values)
{
    // accumulate and reset hardware events from an event set
#if defined(TIMEMORY_USE_PAPI)
    int retval = PAPI_accum(event_set, values);
    check(retval, "Warning!! Failure to accum event set");
#else
    consume_parameters(event_set, values);
#endif
}

//--------------------------------------------------------------------------------------//

inline void
reset(int event_set)
{
    // reset the hardware event counts in an event set
#if defined(TIMEMORY_USE_PAPI)
    int retval = PAPI_reset(event_set);
    check(retval, "Warning!! Failure to reset event set");
#else
    consume_parameters(event_set);
#endif
}

//--------------------------------------------------------------------------------------//

inline void
create_event_set(int* event_set)
{
    // create a new empty PAPI event set
#if defined(TIMEMORY_USE_PAPI)
    init();
    int retval = PAPI_create_eventset(event_set);
    check(retval, "Warning!! Failure to create event set");
#else
    consume_parameters(event_set);
#endif
}

//--------------------------------------------------------------------------------------//

inline void
destroy_event_set(int event_set)
{
    // remove all PAPI events from an event set
#if defined(TIMEMORY_USE_PAPI)
    int retval = PAPI_cleanup_eventset(event_set);
    check(retval, "Warning!! Failure to cleanup event set");
#endif

    // deallocates memory associated with an empty PAPI event set
#if defined(TIMEMORY_USE_PAPI)
    retval = PAPI_destroy_eventset(&event_set);
    check(retval, "Warning!! Failure to destroy event set");
#else
    consume_parameters(event_set);
#endif
}

//--------------------------------------------------------------------------------------//

inline void
add_event(int event_set, int event)
{
    // add single PAPI preset or native hardware event to an event set
#if defined(TIMEMORY_USE_PAPI)
    init();
    int retval = PAPI_add_event(event_set, event);
    check(retval, "Warning!! Failure to add event to event set");
#else
    consume_parameters(event_set, event);
#endif
}

//--------------------------------------------------------------------------------------//

inline void
remove_event(int event_set, int event)
{
    // add single PAPI preset or native hardware event to an event set
#if defined(TIMEMORY_USE_PAPI)
    int retval = PAPI_remove_event(event_set, event);
    check(retval, "Warning!! Failure to remove event from event set");
#else
    consume_parameters(event_set, event);
#endif
}

//--------------------------------------------------------------------------------------//

inline void
add_events(int event_set, int* events, int number)
{
    // add array of PAPI preset or native hardware events to an event set
#if defined(TIMEMORY_USE_PAPI)
    init();
    int retval = PAPI_add_events(event_set, events, number);
    check(retval, "Warning!! Failure to add events to event set");
#else
    consume_parameters(event_set, events, number);
#endif
}

//--------------------------------------------------------------------------------------//

inline void
remove_events(int event_set, int* events, int number)
{
    // add array of PAPI preset or native hardware events to an event set
#if defined(TIMEMORY_USE_PAPI)
    init();
    int retval = PAPI_remove_events(event_set, events, number);
    check(retval, "Warning!! Failure to remove events from event set");
#else
    consume_parameters(event_set, events, number);
#endif
}

//--------------------------------------------------------------------------------------//

inline void
assign_event_set_component(int event_set, int cidx)
{
    // assign a component index to an existing but empty eventset
#if defined(TIMEMORY_USE_PAPI)
    int retval = PAPI_assign_eventset_component(event_set, cidx);
    check(retval, "Warning!! Failure to assign event set component");
#else
    consume_parameters(event_set, cidx);
#endif
}

//--------------------------------------------------------------------------------------//

inline void
attach(int event_set, unsigned long tid)
{
    // attach specified event set to a specific process or thread id
#if defined(TIMEMORY_USE_PAPI)
    int retval = PAPI_attach(event_set, tid);
    check(retval, "Warning!! Failure to attach event set");
#else
    consume_parameters(event_set, tid);
#endif
}

//--------------------------------------------------------------------------------------//

inline void
detach(int event_set)
{
    // detach specified event set from a previously specified process or thread id
#if defined(TIMEMORY_USE_PAPI)
    int retval = PAPI_detach(event_set);
    check(retval, "Warning!! Failure to detach event set");
#else
    consume_parameters(event_set);
#endif
}

//--------------------------------------------------------------------------------------//

}  // namespace papi

//--------------------------------------------------------------------------------------//

}  // namespace tim

//--------------------------------------------------------------------------------------//
