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

//======================================================================================//
//
//      Operating System
//
//======================================================================================//

// machine bits
#if defined(__x86_64__)
#    if !defined(_64BIT)
#        define _64BIT
#    endif
#else
#    if !defined(_32BIT)
#        define _32BIT
#    endif
#endif

//--------------------------------------------------------------------------------------//
// base operating system

#if defined(_WIN32) || defined(_WIN64)
#    if !defined(_WINDOWS)
#        define _WINDOWS
#    endif
//--------------------------------------------------------------------------------------//

#elif defined(__APPLE__) || defined(__MACH__)
#    if !defined(_MACOS)
#        define _MACOS
#    endif
#    if !defined(_UNIX)
#        define _UNIX
#    endif
//--------------------------------------------------------------------------------------//

#elif defined(__linux__) || defined(__linux) || defined(linux) || defined(__gnu_linux__)
#    if !defined(_LINUX)
#        define _LINUX
#    endif
#    if !defined(_UNIX)
#        define _UNIX
#    endif
//--------------------------------------------------------------------------------------//

#elif defined(__unix__) || defined(__unix) || defined(unix) || defined(_)
#    if !defined(_UNIX)
#        define _UNIX
#    endif
#endif

//======================================================================================//
//
//      Windows DLL settings
//
//======================================================================================//

// Define macros for WIN32 for importing/exporting external symbols to DLLs
#if defined(_WINDOWS) && !defined(_TIMEMORY_ARCHIVE)
#    if defined(_TIMEMORY_DLL)
#        define tim_api __declspec(dllexport)
#        define tim_api_static static __declspec(dllexport)
#    else
#        define tim_api __declspec(dllimport)
#        define tim_api_static static __declspec(dllimport)
#    endif
#else
#    define tim_api
#    define tim_api_static static
#endif

//======================================================================================//
//
//      C component enum
//
//======================================================================================//

enum TIMEMORY_COMPONENT
{
    WALL_CLOCK,
    SYS_CLOCK,
    USER_CLOCK,
    CPU_CLOCK,
    MONOTONIC_CLOCK,
    MONOTONIC_RAW_CLOCK,
    THREAD_CPU_CLOCK,
    PROCESS_CPU_CLOCK,
    CPU_UTIL,
    THREAD_CPU_UTIL,
    PROCESS_CPU_UTIL,
    CURRENT_RSS,
    PEAK_RSS,
    STACK_RSS,
    DATA_RSS,
    NUM_SWAP,
    NUM_IO_IN,
    NUM_IO_OUT,
    NUM_MINOR_PAGE_FAULTS,
    NUM_MAJOR_PAGE_FAULTS,
    NUM_MSG_SENT,
    NUM_MSG_RECV,
    NUM_SIGNALS,
    VOLUNTARY_CONTEXT_SWITCH,
    PRIORITY_CONTEXT_SWITCH,
    CUDA_EVENT
};

//======================================================================================//
//
//      C function declaration
//
//======================================================================================//

tim_api extern int
c_timemory_enabled(void);
tim_api extern void*
c_timemory_create_auto_timer(const char*, int);
tim_api extern void
c_timemory_delete_auto_timer(void*);
tim_api extern void*
c_timemory_create_auto_tuple(const char*, int, int, ...);
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
// Count the number of __VA_ARGS__
//
#if !defined(__VA_NARG__)
#    define __VA_NARG__(...) (__VA_NARG_(_0, ##__VA_ARGS__, __RSEQ_N()))
#    define __VA_NARG_(...) __VA_ARG_N(__VA_ARGS__)
#    define __VA_ARG_N(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, \
                       _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28,  \
                       _29, _30, _31, _32, _33, _34, _35, _36, _37, _38, _39, _40, _41,  \
                       _42, _43, _44, _45, _46, _47, _48, _49, _50, _51, _52, _53, _54,  \
                       _55, _56, _57, _58, _59, _60, _61, _62, N, ...)                   \
        N
#    define __RSEQ_N()                                                                   \
        63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44,  \
            43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25,  \
            24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5,   \
            4, 3, 2, 1, 0
#endif

//--------------------------------------------------------------------------------------//

#if !defined(__FUNCTION__) && defined(__func__)
#    define __FUNCTION__ __func__
#endif

#if defined(TIMEMORY_PRETTY_FUNCTION) && !defined(_WINDOWS)
#    define __TIMEMORY_FUNCTION__ __PRETTY_FUNCTION__
#else
#    define __TIMEMORY_FUNCTION__ __FUNCTION__
#endif

// stringify some macro -- uses TIMEMORY_STRINGIFY2 which does the actual
//   "stringify-ing" after the macro has been substituted by it's result
#if !defined(TIMEMORY_STRINGIZE)
#    define TIMEMORY_STRINGIZE(X) TIMEMORY_STRINGIZE2(X)
#endif

// actual stringifying
#if !defined(TIMEMORY_STRINGIZE2)
#    define TIMEMORY_STRINGIZE2(X) #    X
#endif

// stringify the __LINE__ macro
#if !defined(TIMEMORY_LINE_STRING)
#    define TIMEMORY_LINE_STRING TIMEMORY_STRINGIZE(__LINE__)
#endif

//--------------------------------------------------------------------------------------//
//
#if !defined(TIMEMORY_AUTO_SIGN)
#    define TIMEMORY_AUTO_SIGN(c_str)                                                    \
        c_timemory_auto_str(__TIMEMORY_FUNCTION__, c_str, __FILE__, __LINE__)
#endif

//--------------------------------------------------------------------------------------//
// only define for C
#if !defined(__cplusplus)

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
#    if !defined(TIMEMORY_BASIC_AUTO_TIMER)
#        define TIMEMORY_BASIC_AUTO_TIMER(c_str)                                         \
            c_timemory_create_auto_timer(                                                \
                c_timemory_string_combine(__TIMEMORY_FUNCTION__, c_str), __LINE__)
#    endif

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
#    if !defined(TIMEMORY_AUTO_TIMER)
#        define TIMEMORY_AUTO_TIMER(c_str)                                               \
            c_timemory_create_auto_timer(                                                \
                c_timemory_auto_str(__TIMEMORY_FUNCTION__, c_str, __FILE__, __LINE__),   \
                __LINE__)
#    endif

//--------------------------------------------------------------------------------------//
/*! \def TIMEMORY_BASIC_AUTO_TUPLE(c_str, ...)
 *
 * Usage:
 *
 *      void some_func()
 *      {
 *          void* timer = new TIMEMORY_BASIC_AUTO_TUPLE("", WALL_CLOCK, CPU_CLOCK);
 *          ...
 *          FREE_TIMEMORY_AUTO_TUPLE(timer);
 *      }
 *
 */
#    if !defined(TIMEMORY_BASIC_AUTO_TUPLE)
#        define TIMEMORY_BASIC_AUTO_TUPLE(c_str, ...)                                    \
            c_timemory_create_auto_tuple(                                                \
                c_timemory_auto_str(__TIMEMORY_FUNCTION__, c_str, __FILE__, __LINE__),   \
                __LINE__, __VA_NARG__(__VA_ARGS__), __VA_ARGS__)
#    endif

//--------------------------------------------------------------------------------------//
/*! \def TIMEMORY_AUTO_TUPLE(c_str, ...)
 *
 * Usage:
 *
 *      void some_func()
 *      {
 *          void* timer = new TIMEMORY_AUTO_TUPLE("", WALL_CLOCK, SYS_CLOCK, CPU_CLOCK);
 *          ...
 *          FREE_TIMEMORY_AUTO_TUPLE(timer);
 *      }
 *
 */
#    if !defined(TIMEMORY_AUTO_TUPLE)
#        define TIMEMORY_AUTO_TUPLE(c_str, ...)                                          \
            c_timemory_create_auto_tuple(                                                \
                c_timemory_string_combine(__TIMEMORY_FUNCTION__, c_str), __LINE__,       \
                __VA_NARG__(__VA_ARGS__), __VA_ARGS__)
#    endif

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
#    if !defined(FREE_TIMEMORY_AUTO_TIMER)
#        define FREE_TIMEMORY_AUTO_TIMER(ctimer)                                         \
            c_timemory_delete_auto_timer((void*) ctimer);
#    endif

//--------------------------------------------------------------------------------------//
/*! \def FREE_TIMEMORY_AUTO_TUPLE(ctimer)
 *
 * Usage:
 *
 *      void some_func()
 *      {
 *          void* timer = new TIMEMORY_AUTO_TUPLE("", WALL_CLOCK);
 *          ...
 *          FREE_TIMEMORY_AUTO_TUPLE(timer);
 *      }
 */
#    if !defined(FREE_TIMEMORY_AUTO_TUPLE)
#        define FREE_TIMEMORY_AUTO_TUPLE(ctimer)                                         \
            c_timemory_delete_auto_tuple((void*) ctimer);
#    endif

//--------------------------------------------------------------------------------------//

#endif  // !defined(__cplusplus)
