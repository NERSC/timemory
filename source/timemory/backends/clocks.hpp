// MIT License
//
// Copyright (c) 2019, The Regents of the University of California,
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
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//

/** \file clocks.hpp
 * \headerfile clocks.hpp "timemory/clocks.hpp"
 * Implementation of the timing functions/utilities
 */

#pragma once

#include <chrono>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <ratio>
#include <sstream>
#include <stdexcept>
#include <time.h>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "timemory/macros.hpp"
#include "timemory/utility.hpp"

#if defined(_UNIX)

#    include <pthread.h>
#    include <sys/times.h>
#    include <unistd.h>

#elif defined(_WINDOWS)
//
//  Windows does not have tms definition
//

// without this, windows will define macros for min and max
#    ifndef NOMINMIX
#        define NOMINMAX
#    endif

#    include <sys/timeb.h>
#    include <sys/types.h>
#    include <winsock.h>

EXTERN_C inline int
gettimeofday(struct timeval* t, void* timezone)
{
    struct _timeb timebuffer;
#    if defined(_WIN64)
    _ftime64(&timebuffer);
#    elif defined(_WIN32)
    _ftime(&timebuffer);
#    endif
    t->tv_sec  = timebuffer.time;
    t->tv_usec = 1000 * timebuffer.millitm;
    return 0;
}

#    define __need_clock_t

#    include <Windows.h>
#    include <time.h>
#    include <windows.h>

// Structure describing CPU time used by a process and its children.
struct tms
{
    clock_t tms_utime;   // User CPU time
    clock_t tms_stime;   // System CPU time
    clock_t tms_cutime;  // User CPU time of dead children
    clock_t tms_cstime;  // System CPU time of dead children
};

// Store the CPU time used by this process and all its
// dead children (and their dead children) in BUFFER.
// Return the elapsed real time, or (clock_t) -1 for errors.
// All times are in CLK_TCKths of a second.

EXTERN_C inline clock_t
times(struct tms* __buffer)
{
    __buffer->tms_utime  = clock();
    __buffer->tms_stime  = 0;
    __buffer->tms_cstime = 0;
    __buffer->tms_cutime = 0;
    return __buffer->tms_utime;
}

using suseconds_t = long long;

enum clockid_t
{
    CLOCK_REALTIME,
    CLOCK_MONOTONIC,
    CLOCK_MONOTONIC_RAW,
    CLOCK_THREAD_CPUTIME_ID,
    CLOCK_PROCESS_CPUTIME_ID
};

EXTERN_C inline LARGE_INTEGER
get_filetime_offset()
{
    SYSTEMTIME    s;
    FILETIME      f;
    LARGE_INTEGER t;

    s.wYear         = 1970;
    s.wMonth        = 1;
    s.wDay          = 1;
    s.wHour         = 0;
    s.wMinute       = 0;
    s.wSecond       = 0;
    s.wMilliseconds = 0;
    SystemTimeToFileTime(&s, &f);
    t.QuadPart = f.dwHighDateTime;
    t.QuadPart <<= 32;
    t.QuadPart |= f.dwLowDateTime;
    return (t);
}

EXTERN_C inline int
clock_gettime(clockid_t, struct timespec* tv)
{
    LARGE_INTEGER        t;
    FILETIME             f;
    double               microseconds;
    static LARGE_INTEGER offset;
    static double        frequencyToMicroseconds;
    static int           initialized           = 0;
    static BOOL          usePerformanceCounter = 0;

    if(!initialized)
    {
        LARGE_INTEGER performanceFrequency;
        initialized           = 1;
        usePerformanceCounter = QueryPerformanceFrequency(&performanceFrequency);
        if(usePerformanceCounter)
        {
            QueryPerformanceCounter(&offset);
            frequencyToMicroseconds = (double) performanceFrequency.QuadPart / 1000000.;
        }
        else
        {
            offset                  = get_filetime_offset();
            frequencyToMicroseconds = 10.;
        }
    }
    if(usePerformanceCounter)
        QueryPerformanceCounter(&t);
    else
    {
        GetSystemTimeAsFileTime(&f);
        t.QuadPart = f.dwHighDateTime;
        t.QuadPart <<= 32;
        t.QuadPart |= f.dwLowDateTime;
    }

    t.QuadPart -= offset.QuadPart;
    microseconds = (double) t.QuadPart / frequencyToMicroseconds;
    t.QuadPart   = microseconds;
    tv->tv_sec   = t.QuadPart / 1000000;
    tv->tv_nsec  = t.QuadPart % 1000000;
    return (0);
}

#endif

namespace tim
{
//--------------------------------------------------------------------------------------//

template <typename Ratio>
struct time_units;

template <>
struct time_units<std::pico>
{
    static constexpr const char* str = "psec";
};
template <>
struct time_units<std::nano>
{
    static constexpr const char* str = "nsec";
};
template <>
struct time_units<std::micro>
{
    static constexpr const char* str = "usec";
};
template <>
struct time_units<std::milli>
{
    static constexpr const char* str = "msec";
};
template <>
struct time_units<std::centi>
{
    static constexpr const char* str = "csec";
};
template <>
struct time_units<std::deci>
{
    static constexpr const char* str = "dsec";
};
template <>
struct time_units<std::ratio<1>>
{
    static constexpr const char* str = "sec";
};
template <>
struct time_units<std::ratio<60, 1>>
{
    static constexpr const char* str = "min";
};
template <>
struct time_units<std::ratio<3600, 1>>
{
    static constexpr const char* str = "hr";
};
template <>
struct time_units<std::ratio<3600 * 24, 1>>
{
    static constexpr const char* str = "day";
};

//--------------------------------------------------------------------------------------//

template <typename Precision>
int64_t
clock_tick()
{
    auto _get_sys_tick = []() {
#if defined(_WINDOWS)
        return CLOCKS_PER_SEC;
#else
        return ::sysconf(_SC_CLK_TCK);
#endif
    };

    static int64_t result = 0;
    if(result == 0)
    {
        result = _get_sys_tick();
        if(result <= 0)
        {
            std::stringstream ss;
            ss << "Could not retrieve number of clock ticks "
               << "per second (_SC_CLK_TCK / CLOCKS_PER_SEC).";
            result = _get_sys_tick();
            throw std::runtime_error(ss.str().c_str());
        }
        else if(result > Precision::den)  // den == std::ratio::denominator
        {
            std::stringstream ss;
            ss << "Found more than 1 clock tick per " << time_units<Precision>::str
               << ". cpu_clock can't handle that.";
            result = _get_sys_tick();
            throw std::runtime_error(ss.str().c_str());
        }
        else
        {
            result = Precision::den / _get_sys_tick();
        }
    }
    return result;
}

//--------------------------------------------------------------------------------------//
// general struct for the differnt clock_gettime functions
template <typename _Tp = double, typename Precision = std::ratio<1>>
_Tp
get_clock_now(clockid_t clock_id)
{
    constexpr _Tp factor = static_cast<_Tp>(std::nano::den) / Precision::den;
#if defined(_MACOS)
    return clock_gettime_nsec_np(clock_id) / factor;
#else
    struct timespec ts;
    clock_gettime(clock_id, &ts);
    return (ts.tv_sec * std::nano::den + ts.tv_nsec) / factor;
#endif
}

//--------------------------------------------------------------------------------------//
// the system's real time (i.e. wall time) clock, expressed as the amount of time since
// the epoch.
template <typename _Tp = double, typename Precision = std::ratio<1>>
_Tp
get_clock_real_now()
{
    using clock_type    = std::chrono::high_resolution_clock;
    using duration_type = std::chrono::duration<clock_type::rep, Precision>;

    // return get_clock_now<_Tp, Precision>(CLOCK_REALTIME);
    return std::chrono::duration_cast<duration_type>(
               std::chrono::high_resolution_clock::now().time_since_epoch())
        .count();
}

//--------------------------------------------------------------------------------------//
// clock that increments monotonically, tracking the time since an arbitrary point,
// and will continue to increment while the system is asleep.
template <typename _Tp = double, typename Precision = std::ratio<1>>
_Tp
get_clock_monotonic_now()
{
    return get_clock_now<_Tp, Precision>(CLOCK_MONOTONIC);
}

//--------------------------------------------------------------------------------------//
// clock that increments monotonically, tracking the time since an arbitrary point like
// CLOCK_MONOTONIC.  However, this clock is unaffected by frequency or time adjustments.
// It should not be compared to other system time sources.
template <typename _Tp = double, typename Precision = std::ratio<1>>
_Tp
get_clock_monotonic_raw_now()
{
    return get_clock_now<_Tp, Precision>(CLOCK_MONOTONIC_RAW);
}

//--------------------------------------------------------------------------------------//
// this clock measures the CPU time within the current thread (excludes sibling/child
// threads)
// clock that tracks the amount of CPU (in user- or kernel-mode) used by the calling
// thread.
template <typename _Tp = double, typename Precision = std::ratio<1>>
_Tp
get_clock_thread_now()
{
    return get_clock_now<_Tp, Precision>(CLOCK_THREAD_CPUTIME_ID);
}

//--------------------------------------------------------------------------------------//
// this clock measures the CPU time within the current process (excludes child processes)
// clock that tracks the amount of CPU (in user- or kernel-mode) used by the calling
// process.
template <typename _Tp = double, typename Precision = std::ratio<1>>
_Tp
get_clock_process_now()
{
    return get_clock_now<_Tp, Precision>(CLOCK_PROCESS_CPUTIME_ID);
}

//--------------------------------------------------------------------------------------//
// uses clock() -- only relevant as a time when a different is computed
// Do not use a single CPU time as an amount of time; it doesn’t work that way.
// units are reported in number of clock ticks per second
//
// this function extracts only the CPU time spent in user-mode
template <typename _Tp = double, typename Precision = std::ratio<1>>
_Tp
get_clock_user_now()
{
    // return clock() / units::clocks_per_sec;
    struct tms _tms;
    ::times(&_tms);
    return (_tms.tms_utime + _tms.tms_cutime) * static_cast<_Tp>(clock_tick<Precision>());
}

//--------------------------------------------------------------------------------------//
// uses clock() -- only relevant as a time when a different is computed
// Do not use a single CPU time as an amount of time; it doesn’t work that way.
// units are reported in number of clock ticks per second
//
// this function extracts only the CPU time spent in kernel-mode
template <typename _Tp = double, typename Precision = std::ratio<1>>
_Tp
get_clock_system_now()
{
    tms _tms;
    ::times(&_tms);
    return (_tms.tms_stime + _tms.tms_cstime) * static_cast<_Tp>(clock_tick<Precision>());
}

//--------------------------------------------------------------------------------------//
// uses clock() -- only relevant as a time when a different is computed
// Do not use a single CPU time as an amount of time; it doesn’t work that way.
// units are reported in number of clock ticks per second
//
// this function extracts only the CPU time spent in both user- and kernel- mode
template <typename _Tp = double, typename Precision = std::ratio<1>>
_Tp
get_clock_cpu_now()
{
    tms _tms;
    ::times(&_tms);
    return (_tms.tms_utime + _tms.tms_cutime + _tms.tms_stime + _tms.tms_cstime) *
           static_cast<_Tp>(clock_tick<Precision>());
}

//--------------------------------------------------------------------------------------//

}  // namespace tim

//======================================================================================//
