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

#include "timemory/utility/macros.hpp"
#include "timemory/utility/utility.hpp"

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
#    include "timemory/details/papi_defs.hpp"
#endif

// int EventSet = PAPI_NULL;
// unsigned int native = 0x0;
//
// if(PAPI_create_eventset(&EventSet) != PAPI_OK)
//     handle_error(1);
//
// Add Total Instructions Executed to our EventSet
// if(PAPI_add_event(EventSet, PAPI_TOT_INS) != PAPI_OK)
//     handle_error(1);
//
// Add native event PM_CYC to EventSet
// if(PAPI_event_name_to_code("PM_CYC",&native) != PAPI_OK)
//     handle_error(1);
//
// if(PAPI_add_event(EventSet, native) != PAPI_OK)
//     handle_error(1);

//--------------------------------------------------------------------------------------//

namespace tim
{
//--------------------------------------------------------------------------------------//

namespace papi
{
//--------------------------------------------------------------------------------------//

using tid_t        = std::thread::id;
using string_t     = std::string;
using event_info_t = PAPI_event_info_t;
using ulong_t      = unsigned long int;

//--------------------------------------------------------------------------------------//

inline ulong_t
get_thread_index()
{
    static std::atomic<ulong_t> thr_counter;
    static thread_local ulong_t thr_count = thr_counter++;
    return thr_count;
}

//--------------------------------------------------------------------------------------//

inline uint64_t
get_papi_thread_num()
{
#if defined(TIMEMORY_USE_PAPI)
    return PAPI_thread_id();
#else
    static std::atomic<uint64_t> thr_counter;
    static thread_local uint64_t thr_count = thr_counter++;
    return thr_count;
#endif
}

//--------------------------------------------------------------------------------------//

inline void
check_papi_thread()
{
#if defined(TIMEMORY_USE_PAPI)
    auto                               tid         = PAPI_thread_id();
    static constexpr unsigned long int invalid_tid = static_cast<unsigned long int>(-1);
    if(tid == invalid_tid)
        throw std::runtime_error("PAPI_thread_id() returned unknown thread");
#endif
}

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

inline bool
is_master_thread()
{
    return (get_tid() == get_master_tid());
}

//--------------------------------------------------------------------------------------//

inline bool&
working()
{
    static thread_local bool _instance = true;
    return _instance;
}

//--------------------------------------------------------------------------------------//

inline bool
check(int retval, const std::string& mesg, bool quiet = false)
{
    bool success = (retval == PAPI_OK);
    if(!success && !quiet)
    {
#if defined(TIMEMORY_USE_PAPI)
        auto error_str = PAPI_strerror(retval);
        fprintf(stderr, "%s : PAPI_error %d: %s\n", mesg.c_str(), retval, error_str);
#else
        fprintf(stderr, "%s (error code = %i)\n", mesg.c_str(), retval);
#endif
    }
    return (success && working());
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
    if(working())
    {
        int retval = PAPI_register_thread();
        working()  = check(retval, "Warning!! Failure registering thread");
        check_papi_thread();
    }
#endif
}

//--------------------------------------------------------------------------------------//

inline void
unregister_thread()
{
    // inform PAPI that a previously registered thread is disappearing
#if defined(TIMEMORY_USE_PAPI)
    if(working())
    {
        int retval = PAPI_unregister_thread();
        working()  = check(retval, "Warning!! Failure unregistering thread");
    }
#endif
}

//--------------------------------------------------------------------------------------//

inline int
get_event_code(const std::string& event_code_str)
{
#if defined(TIMEMORY_USE_PAPI) && defined(_UNIX)
    int               event_code;
    auto              event_code_char = const_cast<char*>(event_code_str.c_str());
    int               retval = PAPI_event_name_to_code(event_code_char, &event_code);
    std::stringstream ss;
    ss << "Warning!! Failure converting " << event_code_str << " to enum value";
    working() = check(retval, ss.str());
    return event_code;
#else
    consume_parameters(event_code_str);
    return -1;
#endif
}

//--------------------------------------------------------------------------------------//

inline event_info_t
get_event_info(int evt_type)
{
    PAPI_event_info_t evt_info;
#if defined(TIMEMORY_USE_PAPI)
    PAPI_get_event_info(evt_type, &evt_info);
#else
    consume_parameters(std::move(evt_type));
#endif
    return evt_info;
}

//--------------------------------------------------------------------------------------//

template <typename _Tp>
inline void
attach(int event_set, _Tp pid_or_tid)
{
    // inform PAPI that a previously registered thread is disappearing
#if defined(TIMEMORY_USE_PAPI)
    int retval = PAPI_attach(event_set, pid_or_tid);
    working()  = check(retval, "Warning!! Failure attaching to event set");
#else
    consume_parameters(event_set, pid_or_tid);
#endif
}

//--------------------------------------------------------------------------------------//

inline void
init_threading()
{
#if defined(TIMEMORY_USE_PAPI)
    static bool threading_initialized = false;
    if(is_master_thread() && !threading_initialized && working())
    {
        int retval = PAPI_thread_init(get_thread_index);
        working()  = check(retval, "Warning!! Failure initializing PAPI thread support");
        threading_initialized = true;
    }
    else if(!threading_initialized)
    {
        if(!is_master_thread())
        {
            fprintf(stderr,
                    "Warning!! Thread support is not enabled because it is not the "
                    "master thread\n");
        }
        else if(!working())
        {
            fprintf(stderr,
                    "Warning!! Thread support is not enabled because it is not currently "
                    "working\n");
        }
    }
#endif
}

//--------------------------------------------------------------------------------------//

inline void
init_multiplexing()
{
#if defined(TIMEMORY_USE_PAPI)
    static bool allow_multiplexing = get_env("TIMEMORY_PAPI_MULTIPLEXING", true);
    if(!allow_multiplexing)
        return;

    static bool thread_local multiplexing_initialized = false;
    if(!multiplexing_initialized && working())
    {
        int retval = PAPI_multiplex_init();
        working()  = check(retval, "Warning!! Failure initializing PAPI multiplexing");
        multiplexing_initialized = true;
    }
    else if(multiplexing_initialized)
    {
        if(!is_master_thread())
        {
            // fprintf(stderr,
            //        "Warning!! Multiplexing is not enabled because it is not the master"
            //        " thread\n");
        }
        else if(!working())
        {
            fprintf(stderr,
                    "Warning!! Multiplexing is not enabled because it is not currently "
                    "working\n");
        }
    }
#endif
}

//--------------------------------------------------------------------------------------//

inline void
init_library()
{
#if defined(TIMEMORY_USE_PAPI)
    get_master_tid();
    if(!PAPI_is_initialized())
    {
        int retval = PAPI_library_init(PAPI_VER_CURRENT);
        if(retval != PAPI_VER_CURRENT && retval > 0)
            fprintf(stderr, "PAPI library version mismatch!\n");
        working() = (retval == PAPI_VER_CURRENT && working());
    }
#endif
}

//--------------------------------------------------------------------------------------//

inline void
init()
{
    // initialize the PAPI library
#if defined(TIMEMORY_USE_PAPI)
    init_library();
    init_multiplexing();
    init_threading();
    if(!working())
    {
        fprintf(stderr, "Warning!! PAPI library not fully initialized!\n");
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
print_hw_info()
{
    init_library();
#if defined(TIMEMORY_USE_PAPI)
    const PAPI_hw_info_t* hwinfo = PAPI_get_hardware_info();
    const PAPI_mh_info_t* mh     = &hwinfo->mem_hierarchy;
    printf("\n");
    printf("                    Vendor :   %s\n", hwinfo->vendor_string);
    printf("                     Model :   %s\n", hwinfo->model_string);
    printf("                   CPU MHz :   %f\n", hwinfo->mhz);
    printf("               CPU Max MHz :   %i\n", hwinfo->cpu_max_mhz);
    printf("               CPU Min MHz :   %i\n", hwinfo->cpu_min_mhz);
    printf("          Total NUMA nodes :   %i\n", hwinfo->nnodes);
    printf("             Number of CPU :   %i\n", hwinfo->ncpu);
    printf("                 Total CPU :   %i\n", hwinfo->totalcpus);
    printf("                   Sockets :   %i\n", hwinfo->sockets);
    printf("                     Cores :   %i\n", hwinfo->cores);
    printf("                   Threads :   %i\n", hwinfo->threads);
    printf("    Memory Hierarch Levels :   %i\n", mh->levels);
    printf(" Max level of TLB or Cache :   %d\n", mh->levels);
    for(int i = 0; i < mh->levels; i++)
    {
        for(int j = 0; j < PAPI_MH_MAX_LEVELS; j++)
        {
            const PAPI_mh_cache_info_t* c = &mh->level[i].cache[j];
            const PAPI_mh_tlb_info_t*   t = &mh->level[i].tlb[j];
            printf("        Level %2d, TLB   %2d :  %2d, %8d, %8d\n", i, j, t->type,
                   t->num_entries, t->associativity);
            printf("        Level %2d, Cache %2d :  %2d, %8d, %8d, %8d, %8d\n", i, j,
                   c->type, c->size, c->line_size, c->num_lines, c->associativity);
        }
    }
    printf("\n");
#endif
}

//--------------------------------------------------------------------------------------//

inline void
create_event_set(int* event_set)
{
    // create a new empty PAPI event set
#if defined(TIMEMORY_USE_PAPI)
    int retval = PAPI_create_eventset(event_set);
    working()  = check(retval, "Warning!! Failure to create event set");
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
    working()  = check(retval, "Warning!! Failure to cleanup event set");
#endif

    // deallocates memory associated with an empty PAPI event set
#if defined(TIMEMORY_USE_PAPI)
    retval    = PAPI_destroy_eventset(&event_set);
    working() = check(retval, "Warning!! Failure to destroy event set");
#else
    consume_parameters(event_set);
#endif
}

//--------------------------------------------------------------------------------------//

inline void
enable_multiplexing(int event_set)
{
#if defined(TIMEMORY_USE_PAPI)
    int               retval = PAPI_set_multiplex(event_set);
    std::stringstream ss;
    ss << "Warning!! Failure to enable multiplex on EventSet " << event_set;
    working() = check(retval, ss.str());
#else
    consume_parameters(event_set);
#endif
}

//--------------------------------------------------------------------------------------//

inline void
start(int event_set, bool enable_multiplex = false)
{
    // start counting hardware events in an event set
#if defined(TIMEMORY_USE_PAPI)
    if(enable_multiplex)
        enable_multiplexing(event_set);
    int retval = PAPI_start(event_set);
    if(retval != PAPI_OK)
    {
        enable_multiplexing(event_set);
        retval = PAPI_start(event_set);
    }
    working() = check(retval, "Warning!! Failure to start event set");
#else
    consume_parameters(event_set, enable_multiplex);
#endif
}

//--------------------------------------------------------------------------------------//

inline void
stop(int event_set, long long* values)
{
    // stop counting hardware events in an event set and return current events
#if defined(TIMEMORY_USE_PAPI)
    int retval = PAPI_stop(event_set, values);
    working()  = check(retval, "Warning!! Failure to stop event set");
#else
    consume_parameters(event_set, values);
#endif
}

//--------------------------------------------------------------------------------------//

inline void
read(int event_set, long long* values)
{
    // read hardware events from an event set with no reset
#if defined(TIMEMORY_USE_PAPI)
    int retval = PAPI_read(event_set, values);
    working()  = check(retval, "Warning!! Failure to read event set");
#else
    consume_parameters(event_set, values);
#endif
}

//--------------------------------------------------------------------------------------//

inline void
write(int event_set, long long* values)
{
    // write counter values into counters
#if defined(TIMEMORY_USE_PAPI)
    int retval = PAPI_write(event_set, values);
    working()  = check(retval, "Warning!! Failure to write event set");
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
    working()  = check(retval, "Warning!! Failure to accum event set");
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
    working()  = check(retval, "Warning!! Failure to reset event set");
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
    working()  = check(retval, "Warning!! Failure to add event to event set");
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
    working()  = check(retval, "Warning!! Failure to remove event from event set");
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
    working()  = check(retval, "Warning!! Failure to add events to event set");
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
    int retval = PAPI_remove_events(event_set, events, number);
    working()  = check(retval, "Warning!! Failure to remove events from event set");
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
    working()  = check(retval, "Warning!! Failure to assign event set component");
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
    init();
    int retval = PAPI_attach(event_set, tid);
    working()  = check(retval, "Warning!! Failure to attach event set");
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
    working()  = check(retval, "Warning!! Failure to detach event set");
#else
    consume_parameters(event_set);
#endif
}

//--------------------------------------------------------------------------------------//

}  // namespace papi

//--------------------------------------------------------------------------------------//

}  // namespace tim

//--------------------------------------------------------------------------------------//
