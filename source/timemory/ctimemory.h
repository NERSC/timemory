//  MIT License
//
//  Copyright (c) 2019, The Regents of the University of California,
//  through Lawrence Berkeley National Laboratory (subject to receipt of any
//  required approvals from the U.S. Dept. of Energy).  All rights reserved.
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

/** \file ctimemory.h
 * \headerfile ctimemory.h "timemory/ctimemory.h"
 * This header file provides the C interface to TiMemory and generally just
 * redirects to functions implemented in ctimemory.cpp
 *
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(DISABLE_TIMEMORY)

#    define TIMEMORY_SETTINGS_INIT                                                       \
        {                                                                                \
        }
#    define TIMEMORY_INIT(...)
#    define TIMEMORY_BLANK_AUTO_TIMER(...) NULL
#    define TIMEMORY_BASIC_AUTO_TIMER(...) NULL
#    define TIMEMORY_AUTO_TIMER(...) NULL
#    define TIMEMORY_BASIC_AUTO_TUPLE(...) NULL
#    define TIMEMORY_BLANK_AUTO_TUPLE(...) NULL
#    define TIMEMORY_AUTO_TUPLE(...) NULL
#    define FREE_TIMEMORY_AUTO_TIMER(...)
#    define FREE_TIMEMORY_AUTO_TUPLE(...)

#else  // !defined(DISABLE_TIMEMORY)

//======================================================================================//
//
//      Operating System
//
//======================================================================================//

// machine bits
#    if defined(__x86_64__)
#        if !defined(_64BIT)
#            define _64BIT
#        endif
#    else
#        if !defined(_32BIT)
#            define _32BIT
#        endif
#    endif

//--------------------------------------------------------------------------------------//
// base operating system

#    if defined(_WIN32) || defined(_WIN64)
#        if !defined(_WINDOWS)
#            define _WINDOWS
#        endif
//--------------------------------------------------------------------------------------//

#    elif defined(__APPLE__) || defined(__MACH__)
#        if !defined(_MACOS)
#            define _MACOS
#        endif
#        if !defined(_UNIX)
#            define _UNIX
#        endif
//--------------------------------------------------------------------------------------//

#    elif defined(__linux__) || defined(__linux) || defined(linux) ||                    \
        defined(__gnu_linux__)
#        if !defined(_LINUX)
#            define _LINUX
#        endif
#        if !defined(_UNIX)
#            define _UNIX
#        endif
//--------------------------------------------------------------------------------------//

#    elif defined(__unix__) || defined(__unix) || defined(unix) || defined(_)
#        if !defined(_UNIX)
#            define _UNIX
#        endif
#    endif

//======================================================================================//
//
//      Windows DLL settings
//
//======================================================================================//

// Define macros for WIN32 for importing/exporting external symbols to DLLs
#    if defined(_WINDOWS) && !defined(_TIMEMORY_ARCHIVE)
#        if defined(_TIMEMORY_DLL)
#            define tim_api __declspec(dllexport)
#            define tim_api_static static __declspec(dllexport)
#        else
#            define tim_api __declspec(dllimport)
#            define tim_api_static static __declspec(dllimport)
#        endif
#    else
#        define tim_api
#        define tim_api_static static
#    endif

//======================================================================================//
//
//      C component enum
//
//======================================================================================//

enum TIMEMORY_COMPONENT
{
    WALL_CLOCK               = 0,
    SYS_CLOCK                = 1,
    USER_CLOCK               = 2,
    CPU_CLOCK                = 3,
    MONOTONIC_CLOCK          = 4,
    MONOTONIC_RAW_CLOCK      = 5,
    THREAD_CPU_CLOCK         = 6,
    PROCESS_CPU_CLOCK        = 7,
    CPU_UTIL                 = 8,
    THREAD_CPU_UTIL          = 9,
    PROCESS_CPU_UTIL         = 10,
    CURRENT_RSS              = 11,
    PEAK_RSS                 = 12,
    STACK_RSS                = 13,
    DATA_RSS                 = 14,
    NUM_SWAP                 = 15,
    NUM_IO_IN                = 16,
    NUM_IO_OUT               = 17,
    NUM_MINOR_PAGE_FAULTS    = 18,
    NUM_MAJOR_PAGE_FAULTS    = 19,
    NUM_MSG_SENT             = 20,
    NUM_MSG_RECV             = 21,
    NUM_SIGNALS              = 22,
    VOLUNTARY_CONTEXT_SWITCH = 23,
    PRIORITY_CONTEXT_SWITCH  = 24,
    CUDA_EVENT               = 25,
    PAPI_ARRAY               = 26,
    CPU_ROOFLINE_SP_FLOPS    = 27,  // single-precision cpu_roofline
    CPU_ROOFLINE_DP_FLOPS    = 28,  // double-precision cpu_roofline
    CALIPER                  = 29,
    TRIP_COUNT               = 30,
    READ_BYTES               = 31,
    WRITTEN_BYTES            = 32,
    CUPTI_EVENT              = 33,
    NVTX_MARKER              = 34,
    TIMEMORY_COMPONENTS_END  = 35
};

//======================================================================================//
//
//      C struct for settings
//
//======================================================================================//

#    if defined(__cplusplus)
extern "C"
{
#    endif

    typedef struct
    {
        int enabled;
        int auto_output;
        int file_output;
        int text_output;
        int json_output;
        int cout_output;
        int precision;
        int width;
        int scientific;
        // skipping remainder
    } timemory_settings;

#    if defined(__cplusplus)
}
#    endif

//======================================================================================//
//
//      C function declaration
//
//======================================================================================//

tim_api extern void
c_timemory_init(int argc, char** argv, timemory_settings);
tim_api extern int
c_timemory_enabled(void);
tim_api extern void*
c_timemory_create_auto_timer(const char*, int);
tim_api extern void
c_timemory_delete_auto_timer(void*);
tim_api extern void*
c_timemory_create_auto_tuple(const char*, int, ...);
tim_api extern void
c_timemory_delete_auto_tuple(void*);
tim_api extern const char*
c_timemory_string_combine(const char*, const char*);
tim_api extern const char*
c_timemory_auto_str(const char*, const char*, const char*, int);

//======================================================================================//
//
//      C timemory macros
//
//======================================================================================//

// only define for C
#    if !defined(__cplusplus)

#        if !defined(__FUNCTION__) && defined(__func__)
#            define __FUNCTION__ __func__
#        endif

#        if defined(TIMEMORY_PRETTY_FUNCTION) && !defined(_WINDOWS)
#            define __TIMEMORY_FUNCTION__ __PRETTY_FUNCTION__
#        else
#            define __TIMEMORY_FUNCTION__ __FUNCTION__
#        endif

// stringify some macro -- uses TIMEMORY_STRINGIFY2 which does the actual
//   "stringify-ing" after the macro has been substituted by it's result
#        if !defined(TIMEMORY_STRINGIZE)
#            define TIMEMORY_STRINGIZE(X) TIMEMORY_STRINGIZE2(X)
#        endif

// actual stringifying
#        if !defined(TIMEMORY_STRINGIZE2)
#            define TIMEMORY_STRINGIZE2(X) #            X
#        endif

// stringify the __LINE__ macro
#        if !defined(TIMEMORY_LINE_STRING)
#            define TIMEMORY_LINE_STRING TIMEMORY_STRINGIZE(__LINE__)
#        endif

//--------------------------------------------------------------------------------------//
//
#        if !defined(TIMEMORY_AUTO_LABEL)
#            define TIMEMORY_AUTO_LABEL(c_str)                                           \
                c_timemory_auto_str(__TIMEMORY_FUNCTION__, c_str, __FILE__, __LINE__)
#        endif

//--------------------------------------------------------------------------------------//
#        define TIMEMORY_SETTINGS_INIT { 1, -1, -1, -1, -1, -1, -1, -1, -1 };
#        define TIMEMORY_INIT(argc, argv, settings) c_timemory_init(argc, argv, settings)

//--------------------------------------------------------------------------------------//
/*! \def TIMEMORY_BLANK_AUTO_TIMER(c_str)
 *
 * Usage:
 *
 *      void some_func()
 *      {
 *          void* timer = new TIMEMORY_BLANK_AUTO_TIMER("label");
 *          ...
 *          FREE_TIMEMORY_AUTO_TIMER(timer);
 *      }
 */
#        define TIMEMORY_BLANK_AUTO_TIMER(c_str)                                         \
            c_timemory_create_auto_timer(c_str, __LINE__)

//--------------------------------------------------------------------------------------//
/*! \def TIMEMORY_BASIC_AUTO_TIMER(c_str)
 *
 * Usage:
 *
 *      void some_func()
 *      {
 *          void* timer = new TIMEMORY_BASIC_AUTO_TIMER("");
 *          ...
 *          FREE_TIMEMORY_AUTO_TIMER(timer);
 *      }
 */
#        define TIMEMORY_BASIC_AUTO_TIMER(c_str)                                         \
            c_timemory_create_auto_timer(                                                \
                c_timemory_string_combine(__TIMEMORY_FUNCTION__, c_str), __LINE__)

//--------------------------------------------------------------------------------------//
/*! \def TIMEMORY_AUTO_TIMER(str)
 *
 * Usage:
 *
 *      void some_func()
 *      {
 *          void* timer = new TIMEMORY_AUTO_TIMER("");
 *          ...
 *          FREE_TIMEMORY_AUTO_TIMER(timer);
 *      }
 *
 */
#        define TIMEMORY_AUTO_TIMER(c_str)                                               \
            c_timemory_create_auto_timer(                                                \
                c_timemory_auto_str(__TIMEMORY_FUNCTION__, c_str, __FILE__, __LINE__),   \
                __LINE__)

//--------------------------------------------------------------------------------------//
/*! \def TIMEMORY_BASIC_OBJECT(c_str, ...)
 *
 * Usage:
 *
 *      void some_func()
 *      {
 *          void* timer = new TIMEMORY_BASIC_OBJECT("", WALL_CLOCK, CPU_CLOCK);
 *          ...
 *          FREE_TIMEMORY_OBJECT(timer);
 *      }
 *
 */
#        define TIMEMORY_BASIC_OBJECT(c_str, ...)                                        \
            c_timemory_create_auto_tuple(                                                \
                c_timemory_auto_str(__TIMEMORY_FUNCTION__, c_str, __FILE__, __LINE__),   \
                __LINE__, __VA_ARGS__, TIMEMORY_COMPONENTS_END)

//--------------------------------------------------------------------------------------//
/*! \def TIMEMORY_BLANK_OBJECT(c_str, ...)
 *
 * Usage:
 *
 *      void some_func()
 *      {
 *          void* timer = new TIMEMORY_BLANK_OBJECT("id", WALL_CLOCK, CPU_CLOCK);
 *          ...
 *          FREE_TIMEMORY_OBJECT(timer);
 *      }
 *
 */
#        define TIMEMORY_BLANK_OBJECT(c_str, ...)                                        \
            c_timemory_create_auto_tuple(c_str, __LINE__, __VA_ARGS__,                   \
                                         TIMEMORY_COMPONENTS_END)

//--------------------------------------------------------------------------------------//
/*! \def TIMEMORY_OBJECT(c_str, ...)
 *
 * Usage:
 *
 *      void some_func()
 *      {
 *          void* timer = new TIMEMORY_OBJECT("", WALL_CLOCK, SYS_CLOCK, CPU_CLOCK);
 *          ...
 *          FREE_TIMEMORY_OBJECT(timer);
 *      }
 *
 */
#        define TIMEMORY_OBJECT(c_str, ...)                                              \
            c_timemory_create_auto_tuple(                                                \
                c_timemory_string_combine(__TIMEMORY_FUNCTION__, c_str), __LINE__,       \
                __VA_ARGS__, TIMEMORY_COMPONENTS_END)

//--------------------------------------------------------------------------------------//
/*! \def FREE_TIMEMORY_AUTO_TIMER(ctimer)
 *
 * Usage:
 *
 *      void some_func()
 *      {
 *          void* timer = new TIMEMORY_AUTO_TIMER("");
 *          ...
 *          FREE_TIMEMORY_AUTO_TIMER(timer);
 *      }
 */
#        define FREE_TIMEMORY_AUTO_TIMER(ctimer)                                         \
            c_timemory_delete_auto_timer((void*) ctimer)

//--------------------------------------------------------------------------------------//
/*! \def FREE_TIMEMORY_OBJECT(ctimer)
 *
 * Usage:
 *
 *      void some_func()
 *      {
 *          void* timer = new TIMEMORY_OBJECT("", WALL_CLOCK);
 *          ...
 *          FREE_TIMEMORY_OBJECT(timer);
 *      }
 */
#        define FREE_TIMEMORY_OBJECT(ctimer) c_timemory_delete_auto_tuple((void*) ctimer)

//--------------------------------------------------------------------------------------//

#    endif  // !defined(__cplusplus)

#endif  // !defined(DISABLE_TIMEMORY)
