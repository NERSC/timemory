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

#ifndef TIMEMORY_BACKENDS_PAPI_CPP_
#define TIMEMORY_BACKENDS_PAPI_CPP_ 1

#if defined(TIMEMORY_CORE_SOURCE)
#    include "timemory/backends/papi.hpp"
#    define TIMEMORY_PAPI_INLINE
#else
#    define TIMEMORY_PAPI_INLINE inline
#endif

// define TIMEMORY_EXTERNAL_PAPI_DEFINITIONS if these enumerations/defs in bits/papi.hpp
// cause problems

#include "timemory/backends/hardware_counters.hpp"
#include "timemory/backends/process.hpp"
#include "timemory/backends/threading.hpp"
#include "timemory/backends/types/papi.hpp"
#include "timemory/settings/declaration.hpp"
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
#    include <papiStdEventDefs.h>
//
#    include <papi.h>
#    if defined(_UNIX)
#        include <pthread.h>
#    endif
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

namespace tim
{
namespace papi
{
//--------------------------------------------------------------------------------------//

TIMEMORY_PAPI_INLINE
ulong_t
get_thread_index()
{
    return static_cast<ulong_t>(threading::get_id());
}

//--------------------------------------------------------------------------------------//

TIMEMORY_PAPI_INLINE
uint64_t
get_papi_thread_num()
{
#if defined(TIMEMORY_USE_PAPI)
    return PAPI_thread_id();
#else
    return static_cast<ulong_t>(threading::get_id());
#endif
}

//--------------------------------------------------------------------------------------//

TIMEMORY_PAPI_INLINE
void
check_papi_thread()
{
#if defined(TIMEMORY_USE_PAPI)
    auto                               tid         = PAPI_thread_id();
    static constexpr unsigned long int invalid_tid = static_cast<unsigned long int>(-1);
    if(tid == invalid_tid)
    {
        TIMEMORY_EXCEPTION("PAPI_thread_id() returned unknown thread");
    }
#endif
}

//--------------------------------------------------------------------------------------//

TIMEMORY_PAPI_INLINE
std::thread::id
get_tid()
{
    return threading::get_tid();
}

//--------------------------------------------------------------------------------------//

TIMEMORY_PAPI_INLINE
std::thread::id
get_master_tid()
{
    return threading::get_master_tid();
}

//--------------------------------------------------------------------------------------//

TIMEMORY_PAPI_INLINE
bool
is_master_thread()
{
    return threading::is_master_thread();
}

//--------------------------------------------------------------------------------------//

TIMEMORY_PAPI_INLINE
bool&
working()
{
    static thread_local bool _instance = true;
    return _instance;
}

//--------------------------------------------------------------------------------------//

TIMEMORY_PAPI_INLINE
bool
check(int retval, const std::string& mesg, bool quiet)
{
    bool success = (retval == PAPI_OK);
    if(!success && !quiet)
    {
#if defined(TIMEMORY_USE_PAPI)
        auto              error_str   = PAPI_strerror(retval);
        static const auto BUFFER_SIZE = 1024;
        static char       buf[BUFFER_SIZE];
        sprintf(buf, "%s : PAPI_error %d: %s\n", mesg.c_str(), retval, error_str);
        if(settings::papi_fail_on_error())
        {
            TIMEMORY_EXCEPTION(buf);
        }
        else
        {
            if(working() && !settings::papi_quiet())
                fprintf(stderr, "%s", buf);
        }
#else
        fprintf(stderr, "%s (error code = %i)\n", mesg.c_str(), retval);
#endif
    }
    return (success && working());
}

//--------------------------------------------------------------------------------------//

TIMEMORY_PAPI_INLINE
void
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

TIMEMORY_PAPI_INLINE
std::string
as_string(int* events, long long* values, int num_events, const std::string& indent)
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

TIMEMORY_PAPI_INLINE
void
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

TIMEMORY_PAPI_INLINE
void
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

TIMEMORY_PAPI_INLINE
int
get_event_code(const std::string& event_code_str)
{
#if defined(TIMEMORY_USE_PAPI) && defined(_UNIX)
    static const uint64_t BUFFER_SIZE = 1024;
    int                   event_code  = -1;
    char                  event_code_char[BUFFER_SIZE];
    sprintf(event_code_char, "%s", event_code_str.c_str());
    int               retval = PAPI_event_name_to_code(event_code_char, &event_code);
    std::stringstream ss;
    ss << "Warning!! Failure converting " << event_code_str << " to enum value";
    working() = check(retval, ss.str());
    return (retval == PAPI_OK) ? event_code : PAPI_NOT_INITED;
#else
    consume_parameters(event_code_str);
    return PAPI_NOT_INITED;
#endif
}

//--------------------------------------------------------------------------------------//

TIMEMORY_PAPI_INLINE
std::string
get_event_code_name(int event_code)
{
#if defined(TIMEMORY_USE_PAPI) && defined(_UNIX)
    static const uint64_t BUFFER_SIZE = 1024;
    char                  event_code_char[BUFFER_SIZE];
    int                   retval = PAPI_event_code_to_name(event_code, event_code_char);
    std::stringstream     ss;
    ss << "Warning!! Failure converting event code " << event_code << " to a name";
    working() = check(retval, ss.str());
    return (retval == PAPI_OK) ? std::string(event_code_char) : "";
#else
    consume_parameters(event_code);
    return "";
#endif
}

//--------------------------------------------------------------------------------------//

TIMEMORY_PAPI_INLINE
event_info_t
get_event_info(int evt_type)
{
    PAPI_event_info_t evt_info;
#if defined(TIMEMORY_USE_PAPI)
    PAPI_get_event_info(evt_type, &evt_info);
#else
    consume_parameters(evt_type);
#endif
    return evt_info;
}

//--------------------------------------------------------------------------------------//

namespace details
{
TIMEMORY_PAPI_INLINE
void
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
        if(is_master_thread() && !working())
        {
            static std::atomic<int> _once(0);
            if(_once++ == 0)
                fprintf(stderr,
                        "Warning!! Thread support is not enabled because it is not "
                        "currently working\n");
        }
    }
#endif
}

//--------------------------------------------------------------------------------------//

TIMEMORY_PAPI_INLINE
void
init_multiplexing()
{
#if defined(TIMEMORY_USE_PAPI)
    static bool allow_multiplexing = settings::papi_multiplexing();
    if(!allow_multiplexing)
        return;

    static bool multiplexing_initialized = false;
    if(!multiplexing_initialized)
    {
        int retval = PAPI_multiplex_init();
        working()  = check(retval, "Warning!! Failure initializing PAPI multiplexing");
        multiplexing_initialized = true;
    }
    else if(multiplexing_initialized)
    {
        if(!working())
        {
            static std::atomic<int32_t> _once(0);
            if(_once++ == 0)
                fprintf(stderr,
                        "Warning!! Multiplexing is not enabled because of previous PAPI "
                        "errors\n");
        }
    }
#endif
}

//--------------------------------------------------------------------------------------//

TIMEMORY_PAPI_INLINE
void
init_library()
{
#if defined(TIMEMORY_USE_PAPI)
    get_master_tid();
    if(!PAPI_is_initialized())
    {
        int retval = PAPI_library_init(PAPI_VER_CURRENT);
        if(retval != PAPI_VER_CURRENT && retval > 0)
            fprintf(stderr, "PAPI library version mismatch!\n");
        working() = (retval == PAPI_VER_CURRENT);
    }
#endif
}
}  // namespace details

//--------------------------------------------------------------------------------------//

TIMEMORY_PAPI_INLINE
void
init()
{
    // initialize the PAPI library
#if defined(TIMEMORY_USE_PAPI)
    if(!PAPI_is_initialized())
    {
        details::init_library();
        details::init_multiplexing();
        if(!working())
            fprintf(stderr, "Warning!! PAPI library not fully initialized!\n");
    }
#else
    working() = false;
#endif
}

//--------------------------------------------------------------------------------------//

TIMEMORY_PAPI_INLINE
void
shutdown()
{
    // finish using PAPI and free all related resources
#if defined(TIMEMORY_USE_PAPI)
    if(PAPI_is_initialized())
    {
        unregister_thread();
        if(get_tid() == get_master_tid())
        {
            PAPI_shutdown();
            working() = false;
        }
    }
#endif
}

//--------------------------------------------------------------------------------------//

TIMEMORY_PAPI_INLINE
void
print_hw_info()
{
    init();
#if defined(TIMEMORY_USE_PAPI)
    const PAPI_hw_info_t* hwinfo = PAPI_get_hardware_info();
    const PAPI_mh_info_t* mh     = &hwinfo->mem_hierarchy;
    printf("\n");
    printf("                    Vendor :   %s\n", hwinfo->vendor_string);
    printf("                     Model :   %s\n", hwinfo->model_string);
    printf("                   CPU MHz :   %f\n", static_cast<double>(hwinfo->mhz));
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

TIMEMORY_PAPI_INLINE
void
enable_multiplexing(int event_set, int component)
{
#if defined(TIMEMORY_USE_PAPI)
    bool working_assign = true;
    if(working())
    {
        auto              retval = PAPI_assign_eventset_component(event_set, component);
        std::stringstream ss;
        ss << "Warning!! Failure to assign event set component. event set: " << event_set
           << ", component: " << component;
        working_assign = check(retval, ss.str());
    }
    if(working_assign)
    {
        auto              retval = PAPI_set_multiplex(event_set);
        std::stringstream ss;
        ss << "Warning!! Failure to enable multiplex on EventSet " << event_set;
        check(retval, ss.str());
    }
#else
    consume_parameters(event_set, component);
#endif
}

//--------------------------------------------------------------------------------------//

TIMEMORY_PAPI_INLINE
void
create_event_set(int* event_set, bool enable_multiplex)
{
    // create a new empty PAPI event set
#if defined(TIMEMORY_USE_PAPI)
    if(working())
    {
        int retval = PAPI_create_eventset(event_set);
        working()  = check(retval, "Warning!! Failure to create event set");
        if(working() && enable_multiplex)
            enable_multiplexing(*event_set);
        if(working())
            details::init_threading();
    }
#else
    consume_parameters(event_set, enable_multiplex);
#endif
}

//--------------------------------------------------------------------------------------//

TIMEMORY_PAPI_INLINE
void
destroy_event_set(int event_set)
{
    // remove all PAPI events from an event set
#if defined(TIMEMORY_USE_PAPI)
    if(working())
    {
        int retval = PAPI_cleanup_eventset(event_set);
        working()  = check(retval, "Warning!! Failure to cleanup event set");
    }
#endif

    // deallocates memory associated with an empty PAPI event set
#if defined(TIMEMORY_USE_PAPI)
    if(working())
    {
        int retval = PAPI_destroy_eventset(&event_set);
        working()  = check(retval, "Warning!! Failure to destroy event set");
    }
#else
    consume_parameters(event_set);
#endif
}

//--------------------------------------------------------------------------------------//

TIMEMORY_PAPI_INLINE
void
start(int event_set)
{
    // start counting hardware events in an event set
#if defined(TIMEMORY_USE_PAPI)
    if(working())
    {
        int retval = PAPI_start(event_set);
        working()  = check(retval, "Warning!! Failure to start event set");
    }
#else
    consume_parameters(event_set);
#endif
}

//--------------------------------------------------------------------------------------//

TIMEMORY_PAPI_INLINE
void
stop(int event_set, long long* values)
{
    // stop counting hardware events in an event set and return current events
#if defined(TIMEMORY_USE_PAPI)
    if(working())
    {
        int retval = PAPI_stop(event_set, values);
        working()  = check(retval, "Warning!! Failure to stop event set");
    }
#else
    consume_parameters(event_set, values);
#endif
}

//--------------------------------------------------------------------------------------//

TIMEMORY_PAPI_INLINE
void
read(int event_set, long long* values)
{
    // read hardware events from an event set with no reset
#if defined(TIMEMORY_USE_PAPI)
    if(working())
    {
        int retval = PAPI_read(event_set, values);
        working()  = check(retval, "Warning!! Failure to read event set");
    }
#else
    consume_parameters(event_set, values);
#endif
}

//--------------------------------------------------------------------------------------//

TIMEMORY_PAPI_INLINE
void
write(int event_set, long long* values)
{
    // write counter values into counters
#if defined(TIMEMORY_USE_PAPI)
    if(working())
    {
        int retval = PAPI_write(event_set, values);
        working()  = check(retval, "Warning!! Failure to write event set");
    }
#else
    consume_parameters(event_set, values);
#endif
}

//--------------------------------------------------------------------------------------//

TIMEMORY_PAPI_INLINE
void
accum(int event_set, long long* values)
{
    // accumulate and reset hardware events from an event set
#if defined(TIMEMORY_USE_PAPI)
    if(working())
    {
        int retval = PAPI_accum(event_set, values);
        working()  = check(retval, "Warning!! Failure to accum event set");
    }
#else
    consume_parameters(event_set, values);
#endif
}

//--------------------------------------------------------------------------------------//

TIMEMORY_PAPI_INLINE
void
reset(int event_set)
{
    // reset the hardware event counts in an event set
#if defined(TIMEMORY_USE_PAPI)
    if(working())
    {
        int retval = PAPI_reset(event_set);
        working()  = check(retval, "Warning!! Failure to reset event set");
    }
#else
    consume_parameters(event_set);
#endif
}

//--------------------------------------------------------------------------------------//

TIMEMORY_PAPI_INLINE
bool
add_event(int event_set, int event)
{
    // add single PAPI preset or native hardware event to an event set
#if defined(TIMEMORY_USE_PAPI)
    init();
    if(working())
    {
        int  retval   = PAPI_add_event(event_set, event);
        bool _working = check(retval, "Warning!! Failure to add event to event set");
        working()     = _working;
        return _working;
    }
#else
    consume_parameters(event_set, event);
#endif
    return false;
}

//--------------------------------------------------------------------------------------//

TIMEMORY_PAPI_INLINE
bool
remove_event(int event_set, int event)
{
    // add single PAPI preset or native hardware event to an event set
#if defined(TIMEMORY_USE_PAPI)
    if(working())
    {
        int  retval   = PAPI_remove_event(event_set, event);
        bool _working = check(retval, "Warning!! Failure to remove event from event set");
        working()     = _working;
        return _working;
    }
#else
    consume_parameters(event_set, event);
#endif
    return false;
}

//--------------------------------------------------------------------------------------//

TIMEMORY_PAPI_INLINE
void
add_events(int event_set, int* events, int number)
{
    // add array of PAPI preset or native hardware events to an event set
#if defined(TIMEMORY_USE_PAPI)
    init();
    if(working())
    {
        int retval = PAPI_add_events(event_set, events, number);
        working()  = check(retval, "Warning!! Failure to add events to event set");
    }
#else
    consume_parameters(event_set, events, number);
#endif
}

//--------------------------------------------------------------------------------------//

TIMEMORY_PAPI_INLINE
void
remove_events(int event_set, int* events, int number)
{
    // add array of PAPI preset or native hardware events to an event set
#if defined(TIMEMORY_USE_PAPI)
    if(working())
    {
        int retval = PAPI_remove_events(event_set, events, number);
        working()  = check(retval, "Warning!! Failure to remove events from event set");
    }
#else
    consume_parameters(event_set, events, number);
#endif
}

//--------------------------------------------------------------------------------------//

TIMEMORY_PAPI_INLINE
void
assign_event_set_component(int event_set, int cidx)
{
    // assign a component index to an existing but empty eventset
#if defined(TIMEMORY_USE_PAPI)
    if(working())
    {
        int retval = PAPI_assign_eventset_component(event_set, cidx);
        working()  = check(retval, "Warning!! Failure to assign event set component");
    }
#else
    consume_parameters(event_set, cidx);
#endif
}

//--------------------------------------------------------------------------------------//

TIMEMORY_PAPI_INLINE
void
attach(int event_set, unsigned long tid)
{
    // attach specified event set to a specific process or thread id
#if defined(TIMEMORY_USE_PAPI)
    init();
    if(working())
    {
        int retval = PAPI_attach(event_set, tid);
        working()  = check(retval, "Warning!! Failure to attach event set");
    }
#else
    consume_parameters(event_set, tid);
#endif
}

//--------------------------------------------------------------------------------------//

TIMEMORY_PAPI_INLINE
void
detach(int event_set)
{
    // detach specified event set from a previously specified process or thread id
#if defined(TIMEMORY_USE_PAPI)
    if(working())
    {
        int retval = PAPI_detach(event_set);
        working()  = check(retval, "Warning!! Failure to detach event set");
    }
#else
    consume_parameters(event_set);
#endif
}

//--------------------------------------------------------------------------------------//

TIMEMORY_PAPI_INLINE
bool
query_event(int event)
{
#if defined(TIMEMORY_USE_PAPI)
    return (PAPI_query_event(event) == PAPI_OK);
#else
    consume_parameters(event);
    return false;
#endif
}

//--------------------------------------------------------------------------------------//

TIMEMORY_PAPI_INLINE
hwcounter_info_t
available_events_info()
{
    hwcounter_info_t evts{};

#if defined(TIMEMORY_USE_PAPI)
    details::init_library();
    if(working())
    {
        for(int i = 0; i < PAPI_END_idx; ++i)
        {
            PAPI_event_info_t info;
            auto              ret = PAPI_get_event_info((i | PAPI_PRESET_MASK), &info);
            if(ret != PAPI_OK)
                continue;

            bool        _avail = query_event((i | PAPI_PRESET_MASK));
            std::string _sym   = get_timemory_papi_presets()[i].symbol;
            std::string _pysym = _sym;
            for(auto& itr : _pysym)
                itr = tolower(itr);
            std::string _rm = "papi_";
            auto        idx = _pysym.find(_rm);
            if(idx != std::string::npos)
                _pysym.substr(idx + _rm.length());
            evts.push_back(hardware_counters::info(
                _avail, hardware_counters::api::papi, i, PAPI_PRESET_MASK, _sym, _pysym,
                get_timemory_papi_presets()[i].short_descr,
                get_timemory_papi_presets()[i].long_descr));
        }
    }
#endif
    if(!working() || evts.empty())
    {
        for(int i = 0; i < TIMEMORY_PAPI_PRESET_EVENTS; ++i)
        {
            if(get_timemory_papi_presets()[i].symbol == nullptr)
                continue;

            std::string _sym   = get_timemory_papi_presets()[i].symbol;
            std::string _pysym = _sym;
            for(auto& itr : _pysym)
                itr = tolower(itr);
            evts.push_back(hardware_counters::info(
                false, hardware_counters::api::papi, i, PAPI_PRESET_MASK, _sym, _pysym,
                get_timemory_papi_presets()[i].short_descr,
                get_timemory_papi_presets()[i].long_descr));
        }
    }

    return evts;
}

//--------------------------------------------------------------------------------------//

TIMEMORY_PAPI_INLINE
tim::hardware_counters::info
get_hwcounter_info(const std::string& event_code_str)
{
#if defined(TIMEMORY_USE_PAPI)
    details::init_library();
#endif

    if(working())
    {
        auto         idx         = get_event_code(event_code_str);
        event_info_t _info       = get_event_info(idx);
        bool         _avail      = query_event(idx);
        std::string  _sym        = _info.symbol;
        std::string  _short_desc = _info.short_descr;
        std::string  _long_desc  = _info.long_descr;
        std::string  _pysym      = _sym;
        for(auto& itr : _pysym)
            itr = tolower(itr);
        // int32_t _off = _info.event_code;
        int32_t _off = 0;
        return hardware_counters::info(_avail, hardware_counters::api::papi, idx, _off,
                                       _sym, _pysym, _short_desc, _long_desc);
    }

    return hardware_counters::info(false, hardware_counters::api::papi, -1, 0,
                                   event_code_str, "", "", "");
}

//--------------------------------------------------------------------------------------//

}  // namespace papi

//--------------------------------------------------------------------------------------//

}  // namespace tim

//--------------------------------------------------------------------------------------//

#undef TIMEMORY_PAPI_INLINE

#endif
